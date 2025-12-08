//-----------------------------------------------------------------------------
// Title         : SNN Top-Level HLS Module with On-Chip Learning
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_top_hls.cpp
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Unified HLS top-level integrating:
//                 - AXI4-Lite control/status registers
//                 - AXI4-Stream spike I/O (AER: Address Event Representation)
//                 - STDP/R-STDP on-chip learning engine
//                 - Per-Neuron Trace (NOT Per-Synapse) for memory efficiency
//                 - Lazy Update with timestamp-based decay
//                 - Weight memory management
//                 - Direct interface to Verilog SNN core
//-----------------------------------------------------------------------------

#include "snn_top_hls.h"

//=============================================================================
// Weight Memory (On-Chip BRAM) - O(N x M)
//=============================================================================
static weight_t weight_memory[MAX_NEURONS][MAX_NEURONS];

//=============================================================================
// Per-Neuron Trace Storage - O(N + M) instead of O(N x M)
//=============================================================================

// Pre-synaptic trace (x_i): Updated when input neuron i fires
// Stores: 8-bit trace value + 16-bit last spike timestamp
typedef struct {
    ap_uint<8> trace;           // Exponential trace value (fixed-point 0.0-1.0)
    ap_uint<16> last_spike_time; // Timestamp for lazy update
} neuron_trace_t;

static neuron_trace_t pre_traces[MAX_NEURONS];   // Input neuron traces
static neuron_trace_t post_traces[MAX_NEURONS];  // Output neuron traces

//=============================================================================
// Exponential Decay Look-Up Table for Lazy Update
// exp(-dt/tau) approximated with 16 entries for dt=0..15
// After dt>=16, trace decays to near zero
//=============================================================================
static const ap_uint<8> EXP_DECAY_LUT[16] = {
    255,  // dt=0:  exp(0) = 1.0
    223,  // dt=1:  ~0.875
    195,  // dt=2:  ~0.765
    170,  // dt=3:  ~0.670
    149,  // dt=4:  ~0.585
    130,  // dt=5:  ~0.512
    114,  // dt=6:  ~0.448
    100,  // dt=7:  ~0.392
    87,   // dt=8:  ~0.343
    76,   // dt=9:  ~0.300
    67,   // dt=10: ~0.262
    58,   // dt=11: ~0.229
    51,   // dt=12: ~0.200
    45,   // dt=13: ~0.175
    39,   // dt=14: ~0.153
    34    // dt=15: ~0.134
};

//=============================================================================
// Lazy Update: Compute decayed trace on-demand
// Only calculates when spike occurs, avoiding per-step computation
//=============================================================================
static ap_uint<8> compute_decayed_trace(
    ap_uint<8> old_trace,
    ap_uint<16> last_time,
    ap_uint<16> current_time
) {
    #pragma HLS INLINE
    
    if (old_trace == 0) return 0;
    
    // Calculate elapsed time
    ap_uint<16> dt = current_time - last_time;
    
    // If too much time passed, trace is essentially zero
    if (dt >= 16) return 0;
    
    // Apply exponential decay from LUT
    // new_trace = old_trace * decay_factor / 256
    ap_uint<16> decayed = ((ap_uint<16>)old_trace * EXP_DECAY_LUT[dt]) >> 8;
    
    return (ap_uint<8>)decayed;
}

//=============================================================================
// Weight Clipping
//=============================================================================
static weight_t clip_weight(ap_int<16> w) {
    #pragma HLS INLINE
    if (w > MAX_WEIGHT) return MAX_WEIGHT;
    if (w < MIN_WEIGHT) return MIN_WEIGHT;
    return (weight_t)w;
}

//=============================================================================
// Process Pre-Synaptic Spike (Input Spike) - Per-Neuron Trace STDP
// When pre-neuron i fires:
//   1. Update pre_trace[i] with lazy decay + spike
//   2. For all post-neurons j: Apply LTD using post_trace[j]
//      W[i][j] -= A_neg * post_trace[j]
//=============================================================================
static void process_pre_spike_aer(
    neuron_id_t pre_id,
    ap_uint<16> current_time,
    const learning_params_t &params
) {
    #pragma HLS INLINE off
    
    // Step 1: Lazy update pre-trace and add new spike
    neuron_trace_t old_pre = pre_traces[pre_id];
    ap_uint<8> decayed_pre = compute_decayed_trace(old_pre.trace, old_pre.last_spike_time, current_time);
    
    // Add new spike contribution (saturating add)
    ap_uint<9> new_trace = (ap_uint<9>)decayed_pre + 128;  // Add 0.5 in 8-bit fixed point
    pre_traces[pre_id].trace = (new_trace > 255) ? (ap_uint<8>)255 : (ap_uint<8>)new_trace;
    pre_traces[pre_id].last_spike_time = current_time;
    
    // Step 2: Apply LTD to all synapses from this pre-neuron
    // W[pre_id][j] -= A_neg * post_trace[j]
    LTD_LOOP: for (int j = 0; j < MAX_NEURONS; j++) {
        #pragma HLS PIPELINE II=2
        #pragma HLS UNROLL factor=4
        
        // Get decayed post-trace (lazy update)
        neuron_trace_t post_t = post_traces[j];
        ap_uint<8> post_trace_val = compute_decayed_trace(post_t.trace, post_t.last_spike_time, current_time);
        
        // Skip if no recent post-synaptic activity
        if (post_trace_val == 0) continue;
        
        // Compute LTD: delta = -A_neg * post_trace (using shifts)
        // A_neg ≈ 0.01 -> post_trace >> 7 (approximately)
        ap_int<16> delta = -((ap_int<16>)post_trace_val >> 6);  // ~0.015
        
        // Read current weight
        weight_t current_w = weight_memory[pre_id][j];
        
        // Apply update
        weight_memory[pre_id][j] = clip_weight((ap_int<16>)current_w + delta);
    }
}

//=============================================================================
// Process Post-Synaptic Spike (Output Spike) - Per-Neuron Trace STDP
// When post-neuron j fires:
//   1. Update post_trace[j] with lazy decay + spike
//   2. For all pre-neurons i: Apply LTP using pre_trace[i]
//      W[i][j] += A_pos * pre_trace[i]
//=============================================================================
static void process_post_spike_aer(
    neuron_id_t post_id,
    ap_uint<16> current_time,
    const learning_params_t &params
) {
    #pragma HLS INLINE off
    
    // Step 1: Lazy update post-trace and add new spike
    neuron_trace_t old_post = post_traces[post_id];
    ap_uint<8> decayed_post = compute_decayed_trace(old_post.trace, old_post.last_spike_time, current_time);
    
    // Add new spike contribution (saturating add)
    ap_uint<9> new_trace = (ap_uint<9>)decayed_post + 128;
    post_traces[post_id].trace = (new_trace > 255) ? (ap_uint<8>)255 : (ap_uint<8>)new_trace;
    post_traces[post_id].last_spike_time = current_time;
    
    // Step 2: Apply LTP to all synapses to this post-neuron
    // W[i][post_id] += A_pos * pre_trace[i]
    LTP_LOOP: for (int i = 0; i < MAX_NEURONS; i++) {
        #pragma HLS PIPELINE II=2
        #pragma HLS UNROLL factor=4
        
        // Get decayed pre-trace (lazy update)
        neuron_trace_t pre_t = pre_traces[i];
        ap_uint<8> pre_trace_val = compute_decayed_trace(pre_t.trace, pre_t.last_spike_time, current_time);
        
        // Skip if no recent pre-synaptic activity
        if (pre_trace_val == 0) continue;
        
        // Compute LTP: delta = +A_pos * pre_trace (using shifts)
        // A_pos ≈ 0.01 -> pre_trace >> 6 (slightly larger than LTD)
        ap_int<16> delta = (ap_int<16>)pre_trace_val >> 5;  // ~0.03
        
        // Read current weight
        weight_t current_w = weight_memory[i][post_id];
        
        // Apply update
        weight_memory[i][post_id] = clip_weight((ap_int<16>)current_w + delta);
    }
}

//=============================================================================
// Apply Weight Updates with R-STDP Modulation
// Per-Neuron Eligibility Traces: O(N+M) storage only
//=============================================================================

// Per-Neuron Eligibility traces for R-STDP (NOT per-synapse!)
static ap_int<8> pre_eligibility[MAX_NEURONS];   // Pre-neuron eligibility
static ap_int<8> post_eligibility[MAX_NEURONS];  // Post-neuron eligibility

static void apply_rstdp_reward(
    ap_int<8> reward_signal,
    const learning_params_t &params,
    ap_uint<16> current_time
) {
    #pragma HLS INLINE off
    
    if (reward_signal == 0 || !params.rstdp_enable) return;
    
    bool reward_positive = (reward_signal >= 0);
    ap_uint<8> reward_mag = reward_positive ? (ap_uint<8>)reward_signal : (ap_uint<8>)(-reward_signal);
    
    // Determine shift based on reward magnitude
    ap_uint<2> shift_sel;
    if (reward_mag >= 64) shift_sel = 0;
    else if (reward_mag >= 32) shift_sel = 1;
    else if (reward_mag >= 16) shift_sel = 2;
    else shift_sel = 3;
    
    // Apply reward modulated by eligibility traces
    // W[i][j] += reward * pre_elig[i] * post_elig[j] (approximated with shifts)
    RSTDP_OUTER: for (int i = 0; i < MAX_NEURONS; i++) {
        #pragma HLS LOOP_FLATTEN off
        
        ap_int<8> pre_elig = pre_eligibility[i];
        if (pre_elig == 0) continue;  // Skip inactive pre-neurons
        
        RSTDP_INNER: for (int j = 0; j < MAX_NEURONS; j++) {
            #pragma HLS PIPELINE II=2
            #pragma HLS UNROLL factor=4
            
            ap_int<8> post_elig = post_eligibility[j];
            if (post_elig == 0) continue;  // Skip inactive post-neurons
            
            // Combine eligibilities: (pre_elig * post_elig) >> 8
            ap_int<16> combined_elig = ((ap_int<16>)pre_elig * (ap_int<16>)post_elig) >> 8;
            
            // Apply reward scaling
            ap_int<16> scaled;
            switch (shift_sel) {
                case 0: scaled = combined_elig >> 1; break;
                case 1: scaled = combined_elig >> 2; break;
                case 2: scaled = combined_elig >> 3; break;
                default: scaled = combined_elig >> 4; break;
            }
            
            // Apply sign
            ap_int<16> delta = reward_positive ? scaled : (ap_int<16>)(-scaled);
            
            // Update weight
            weight_t current_w = weight_memory[i][j];
            weight_memory[i][j] = clip_weight((ap_int<16>)current_w + delta);
        }
    }
}

//=============================================================================
// Decay Per-Neuron Eligibility Traces - O(N+M) operations only!
// Much faster than O(N*M) per-synapse decay
//=============================================================================
static void decay_eligibility_traces(const learning_params_t &params) {
    #pragma HLS INLINE off
    
    // Decay pre-neuron eligibility traces
    DECAY_PRE: for (int i = 0; i < MAX_NEURONS; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=8
        
        // Simple shift-based decay: trace = trace - (trace >> 3) ≈ 0.875
        ap_int<8> trace = pre_eligibility[i];
        pre_eligibility[i] = trace - (trace >> 3);
    }
    
    // Decay post-neuron eligibility traces
    DECAY_POST: for (int j = 0; j < MAX_NEURONS; j++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=8
        
        ap_int<8> trace = post_eligibility[j];
        post_eligibility[j] = trace - (trace >> 3);
    }
}

//=============================================================================
// Update Eligibility on Spike Events
// Called when pre/post spikes occur to mark "credit assignment"
//=============================================================================
static void update_eligibility_on_pre_spike(neuron_id_t pre_id) {
    #pragma HLS INLINE
    // Increase pre-neuron eligibility (saturating)
    ap_int<9> new_elig = (ap_int<9>)pre_eligibility[pre_id] + 32;
    pre_eligibility[pre_id] = (new_elig > 127) ? (ap_int<8>)127 : (ap_int<8>)new_elig;
}

static void update_eligibility_on_post_spike(neuron_id_t post_id) {
    #pragma HLS INLINE
    // Increase post-neuron eligibility (saturating)
    ap_int<9> new_elig = (ap_int<9>)post_eligibility[post_id] + 32;
    post_eligibility[post_id] = (new_elig > 127) ? (ap_int<8>)127 : (ap_int<8>)new_elig;
}

//=============================================================================
// Main Top-Level Function
//=============================================================================
void snn_top_hls(
    // AXI4-Lite Control Interface
    ap_uint<32> ctrl_reg,
    ap_uint<32> config_reg,
    learning_params_t learning_params,
    ap_uint<32> &status_reg,
    ap_uint<32> &spike_count_reg,
    ap_uint<32> &weight_sum_reg,
    ap_uint<32> &version_reg,
    
    // AXI4-Stream Spike Input (from PS)
    hls::stream<axis_spike_t> &s_axis_spikes,
    
    // AXI4-Stream Spike Output (to PS)
    hls::stream<axis_spike_t> &m_axis_spikes,
    
    // AXI4-Stream Weight Read (for debugging)
    hls::stream<axis_weight_t> &m_axis_weights,
    
    // Reward signal input (for R-STDP)
    ap_int<8> reward_signal,
    
    // Verilog Interface - Spike Input (to SNN core)
    ap_uint<1> &spike_in_valid,
    ap_uint<8> &spike_in_neuron_id,
    ap_int<8> &spike_in_weight,
    ap_uint<1> spike_in_ready,
    
    // Verilog Interface - Spike Output (from SNN core)
    ap_uint<1> spike_out_valid,
    ap_uint<8> spike_out_neuron_id,
    ap_int<8> spike_out_weight,
    ap_uint<1> &spike_out_ready,
    
    // Verilog Interface - Control signals
    ap_uint<1> &snn_enable,
    ap_uint<1> &snn_reset,
    ap_uint<16> &threshold_out,
    ap_uint<16> &leak_rate_out,
    
    // Verilog Interface - Status signals
    ap_uint<1> snn_ready,
    ap_uint<1> snn_busy
) {
    //=========================================================================
    // HLS Interface Pragmas
    //=========================================================================
    // AXI4-Lite slave interface
    #pragma HLS INTERFACE s_axilite port=ctrl_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=config_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=learning_params bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=status_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=spike_count_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=weight_sum_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=version_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=reward_signal bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return bundle=ctrl
    
    // AXI4-Stream interfaces
    #pragma HLS INTERFACE axis port=s_axis_spikes
    #pragma HLS INTERFACE axis port=m_axis_spikes
    #pragma HLS INTERFACE axis port=m_axis_weights
    
    // Direct wire interfaces to Verilog
    #pragma HLS INTERFACE ap_none port=spike_in_valid
    #pragma HLS INTERFACE ap_none port=spike_in_neuron_id
    #pragma HLS INTERFACE ap_none port=spike_in_weight
    #pragma HLS INTERFACE ap_none port=spike_in_ready
    #pragma HLS INTERFACE ap_none port=spike_out_valid
    #pragma HLS INTERFACE ap_none port=spike_out_neuron_id
    #pragma HLS INTERFACE ap_none port=spike_out_weight
    #pragma HLS INTERFACE ap_none port=spike_out_ready
    #pragma HLS INTERFACE ap_none port=snn_enable
    #pragma HLS INTERFACE ap_none port=snn_reset
    #pragma HLS INTERFACE ap_none port=threshold_out
    #pragma HLS INTERFACE ap_none port=leak_rate_out
    #pragma HLS INTERFACE ap_none port=snn_ready
    #pragma HLS INTERFACE ap_none port=snn_busy
    
    //=========================================================================
    // Static Array Storage Bindings (must be inside function scope)
    // Per-Neuron architecture: O(N+M) instead of O(N*M)
    //=========================================================================
    #pragma HLS BIND_STORAGE variable=weight_memory type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=weight_memory cyclic factor=4 dim=2
    
    // Per-Neuron Eligibility (O(N+M) vs O(N*M)) - Major memory savings!
    #pragma HLS BIND_STORAGE variable=pre_eligibility type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=pre_eligibility cyclic factor=8
    #pragma HLS BIND_STORAGE variable=post_eligibility type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=post_eligibility cyclic factor=8
    
    // Per-Neuron Traces for Lazy Update (includes timestamp)
    #pragma HLS BIND_STORAGE variable=pre_traces type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=pre_traces cyclic factor=8
    #pragma HLS BIND_STORAGE variable=post_traces type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=post_traces cyclic factor=8
    
    //=========================================================================
    // Internal State
    //=========================================================================
    static ap_uint<32> timestamp = 0;
    static ap_uint<32> spike_counter = 0;
    static ap_uint<32> update_counter = 0;
    static bool initialized = false;
    
    //=========================================================================
    // Control Signal Extraction
    //=========================================================================
    bool enable = ctrl_reg[0];
    bool reset = ctrl_reg[1];
    bool clear_counters = ctrl_reg[2];
    bool learning_enable = ctrl_reg[3];
    bool weight_read_mode = ctrl_reg[4];
    bool apply_reward = ctrl_reg[5];
    
    ap_uint<16> threshold = config_reg(15, 0);
    ap_uint<16> leak_rate = config_reg(31, 16);
    
    //=========================================================================
    // Reset Logic
    //=========================================================================
    if (reset || !initialized) {
        timestamp = 0;
        spike_counter = 0;
        update_counter = 0;
        
        // Clear Per-Neuron eligibility traces (O(N+M) - much faster than O(N*M)!)
        RESET_ELIG: for (int i = 0; i < MAX_NEURONS; i++) {
            #pragma HLS PIPELINE II=1
            pre_eligibility[i] = 0;
            post_eligibility[i] = 0;
        }
        
        // Clear Per-Neuron STDP traces
        RESET_TRACES: for (int i = 0; i < MAX_NEURONS; i++) {
            #pragma HLS PIPELINE II=1
            pre_traces[i].trace = 0;
            pre_traces[i].last_spike_time = 0;
            post_traces[i].trace = 0;
            post_traces[i].last_spike_time = 0;
        }
        
        // Initialize weights (optional: set to small random or zero)
        if (!initialized) {
            INIT_WEIGHT_OUTER: for (int i = 0; i < MAX_NEURONS; i++) {
                INIT_WEIGHT_INNER: for (int j = 0; j < MAX_NEURONS; j++) {
                    #pragma HLS PIPELINE II=1
                    weight_memory[i][j] = 0;
                }
            }
            initialized = true;
        }
    }
    
    if (clear_counters) {
        spike_counter = 0;
        update_counter = 0;
    }
    
    //=========================================================================
    // Route Control to Verilog Core
    //=========================================================================
    snn_enable = enable;
    snn_reset = reset;
    threshold_out = threshold;
    leak_rate_out = leak_rate;
    
    //=========================================================================
    // Input Spike Processing
    //=========================================================================
    spike_in_valid = 0;
    spike_in_neuron_id = 0;
    spike_in_weight = 0;
    
    if (enable && spike_in_ready && !s_axis_spikes.empty()) {
        axis_spike_t in_pkt = s_axis_spikes.read();
        
        neuron_id_t pre_id = in_pkt.data(7, 0);
        weight_t weight = in_pkt.data(15, 8);
        
        // Forward to Verilog core
        spike_in_valid = 1;
        spike_in_neuron_id = pre_id;
        spike_in_weight = weight;
        
        // Per-Neuron STDP: Update pre-trace and process LTD
        if (learning_enable) {
            process_pre_spike_aer(pre_id, timestamp, learning_params);
            update_eligibility_on_pre_spike(pre_id);
        }
        
        spike_counter++;
    }
    
    //=========================================================================
    // Output Spike Processing
    //=========================================================================
    spike_out_ready = !m_axis_spikes.full();
    
    if (spike_out_valid && !m_axis_spikes.full()) {
        neuron_id_t post_id = spike_out_neuron_id;
        weight_t weight = spike_out_weight;
        
        // Forward to PS
        axis_spike_t out_pkt;
        out_pkt.data = 0;
        out_pkt.data(7, 0) = post_id;
        out_pkt.data(15, 8) = weight;
        out_pkt.data(31, 16) = timestamp(15, 0);
        out_pkt.keep = 0xF;
        out_pkt.strb = 0xF;
        out_pkt.last = 1;
        out_pkt.id = 0;
        out_pkt.dest = 0;
        out_pkt.user = 0;
        m_axis_spikes.write(out_pkt);
        
        // Per-Neuron STDP: Update post-trace and process LTP
        if (learning_enable) {
            process_post_spike_aer(post_id, timestamp, learning_params);
            update_eligibility_on_post_spike(post_id);
        }
    }
    
    //=========================================================================
    // R-STDP Reward Application (uses Per-Neuron Eligibility Traces)
    //=========================================================================
    if (apply_reward && learning_params.rstdp_enable) {
        apply_rstdp_reward(reward_signal, learning_params, timestamp);
        decay_eligibility_traces(learning_params);
    }
    
    //=========================================================================
    // Weight Read Mode (for debugging/monitoring)
    //=========================================================================
    if (weight_read_mode && !m_axis_weights.full()) {
        static ap_uint<8> read_row = 0;
        static ap_uint<8> read_col = 0;
        
        axis_weight_t w_pkt;
        w_pkt.data = 0;
        w_pkt.data(7, 0) = read_row;
        w_pkt.data(15, 8) = read_col;
        w_pkt.data(23, 16) = weight_memory[read_row][read_col];
        w_pkt.keep = 0xF;
        w_pkt.strb = 0xF;
        w_pkt.last = (read_row == MAX_NEURONS-1 && read_col == MAX_NEURONS-1) ? 1 : 0;
        w_pkt.id = 0;
        w_pkt.dest = 0;
        w_pkt.user = 0;
        m_axis_weights.write(w_pkt);
        
        read_col++;
        if (read_col >= MAX_NEURONS) {
            read_col = 0;
            read_row++;
            if (read_row >= MAX_NEURONS) {
                read_row = 0;
            }
        }
    }
    
    //=========================================================================
    // Timestamp Update
    //=========================================================================
    if (enable) {
        timestamp++;
    }
    
    //=========================================================================
    // Status Register Assembly
    //=========================================================================
    ap_uint<32> status = 0;
    status[0] = snn_ready;
    status[1] = snn_busy;
    status[2] = learning_enable;
    status[3] = 0;  // Reserved
    status[4] = learning_params.rstdp_enable;
    status(15, 8) = update_counter(7, 0);
    
    status_reg = status;
    spike_count_reg = spike_counter;
    version_reg = VERSION_ID;
    
    // Calculate weight sum for monitoring (reduced sampling to save LUTs)
    ap_int<32> weight_sum = 0;
    WEIGHT_SUM: for (int i = 0; i < 8; i++) {  // Reduced sample subset
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < 8; j++) {
            weight_sum += weight_memory[i][j];
        }
    }
    weight_sum_reg = (ap_uint<32>)weight_sum;
}

//=============================================================================
// Weight Memory Access Functions (for external use)
//=============================================================================
void write_weight(
    neuron_id_t pre_id,
    neuron_id_t post_id,
    weight_t weight
) {
    #pragma HLS INLINE
    weight_memory[pre_id][post_id] = weight;
}

weight_t read_weight(
    neuron_id_t pre_id,
    neuron_id_t post_id
) {
    #pragma HLS INLINE
    return weight_memory[pre_id][post_id];
}

//=============================================================================
// Batch Weight Load (via AXI-Stream)
//=============================================================================
void load_weights_from_stream(
    hls::stream<axis_weight_t> &weight_stream,
    ap_uint<16> num_weights
) {
    #pragma HLS INLINE off
    
    LOAD_WEIGHTS: for (int i = 0; i < num_weights; i++) {
        #pragma HLS PIPELINE II=1
        
        if (weight_stream.empty()) break;
        
        axis_weight_t pkt = weight_stream.read();
        neuron_id_t pre_id = pkt.data(7, 0);
        neuron_id_t post_id = pkt.data(15, 8);
        weight_t weight = pkt.data(23, 16);
        
        weight_memory[pre_id][post_id] = weight;
    }
}
