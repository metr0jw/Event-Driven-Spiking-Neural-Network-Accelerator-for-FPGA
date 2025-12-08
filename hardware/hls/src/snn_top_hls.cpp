//-----------------------------------------------------------------------------
// Title         : SNN Top-Level HLS Module with On-Chip Learning
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_top_hls.cpp
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Unified HLS top-level integrating:
//                 - AXI4-Lite control/status registers
//                 - AXI4-Stream spike I/O
//                 - STDP/R-STDP on-chip learning engine
//                 - Weight memory management
//                 - Direct interface to Verilog SNN core
//-----------------------------------------------------------------------------

#include "snn_top_hls.h"

//=============================================================================
// Weight Memory (On-Chip BRAM)
// Note: HLS pragmas for these arrays are applied inside the top function
//=============================================================================
static weight_t weight_memory[MAX_NEURONS][MAX_NEURONS];

// Spike time tracking for STDP
static spike_time_t pre_spike_times[MAX_NEURONS];
static spike_time_t post_spike_times[MAX_NEURONS];

// Eligibility traces for R-STDP
static ap_fixed<16,8> eligibility_traces[MAX_NEURONS][MAX_NEURONS];

//=============================================================================
// STDP Weight Update Calculation (Shift/AC with DSP)
//=============================================================================
static weight_delta_t calc_stdp_update(
    spike_time_t pre_time,
    spike_time_t post_time,
    const learning_params_t &params
) {
    #pragma HLS INLINE
    #pragma HLS ALLOCATION operation instances=mul limit=0
    
    if (pre_time == 0 || post_time == 0) return 0;
    
    ap_int<32> dt = (ap_int<32>)post_time - (ap_int<32>)pre_time;
    weight_delta_t delta = 0;
    
    // Pure shift/add STDP implementation
    // Decay approximation: window - |dt| scaled by shifts only
    // a_plus/a_minus assumed to be power-of-2 friendly (0.5, 0.25, etc.)
    
    ap_int<16> abs_dt = (dt >= 0) ? (ap_int<16>)dt : (ap_int<16>)(-dt);
    ap_int<16> window = (ap_int<16>)params.stdp_window;
    
    if (abs_dt < window && abs_dt > 0) {
        // Linear decay: (window - abs_dt) >> 4 gives normalized decay factor
        ap_int<16> decay = (window - abs_dt) >> 4;
        
        // Scale by learning amplitude using shifts only
        // Approximate a_plus/a_minus as shift amounts (e.g., 0.5 = >>1, 0.25 = >>2)
        // decay >> 2 for base amplitude, then adjust
        ap_int<16> base_delta = decay >> 2;  // Base scaling
        
        // Add fractional components: x + (x>>1) + (x>>2) ≈ x*1.75
        ap_int<16> scaled_delta = base_delta + (base_delta >> 1);
        
        if (dt > 0) {
            // LTP: pre before post (positive delta)
            delta = (weight_delta_t)scaled_delta;
        } else {
            // LTD: post before pre (negative delta)
            // Slightly smaller magnitude for LTD
            delta = (weight_delta_t)(-(base_delta + (base_delta >> 2)));
        }
    }
    
    return delta;
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
// Process Pre-Synaptic Spike (Input Spike)
//=============================================================================
static void process_pre_spike(
    neuron_id_t pre_id,
    spike_time_t current_time,
    const learning_params_t &params,
    bool learning_enable,
    hls::stream<weight_update_t> &weight_updates
) {
    #pragma HLS INLINE off
    
    // Update pre-spike time
    pre_spike_times[pre_id] = current_time;
    
    if (!learning_enable) return;
    
    // STDP: Check all post-synaptic neurons for LTD
    STDP_LTD_LOOP: for (int post_id = 0; post_id < MAX_NEURONS; post_id++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=4
        
        spike_time_t post_time = post_spike_times[post_id];
        if (post_time == 0) continue;
        
        weight_delta_t delta = calc_stdp_update(current_time, post_time, params);
        
        if (delta != 0) {
            weight_update_t update;
            update.pre_id = pre_id;
            update.post_id = post_id;
            update.delta = delta;
            update.timestamp = current_time;
            
            if (!weight_updates.full()) {
                weight_updates.write(update);
            }
        }
    }
}

//=============================================================================
// Process Post-Synaptic Spike (Output Spike)
//=============================================================================
static void process_post_spike(
    neuron_id_t post_id,
    spike_time_t current_time,
    const learning_params_t &params,
    bool learning_enable,
    hls::stream<weight_update_t> &weight_updates
) {
    #pragma HLS INLINE off
    
    // Update post-spike time
    post_spike_times[post_id] = current_time;
    
    if (!learning_enable) return;
    
    // STDP: Check all pre-synaptic neurons for LTP
    STDP_LTP_LOOP: for (int pre_id = 0; pre_id < MAX_NEURONS; pre_id++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=4
        
        spike_time_t pre_time = pre_spike_times[pre_id];
        if (pre_time == 0) continue;
        
        weight_delta_t delta = calc_stdp_update(pre_time, current_time, params);
        
        if (delta != 0) {
            weight_update_t update;
            update.pre_id = pre_id;
            update.post_id = post_id;
            update.delta = delta;
            update.timestamp = current_time;
            
            if (!weight_updates.full()) {
                weight_updates.write(update);
            }
        }
    }
}

//=============================================================================
// Apply Weight Updates (Shift/Add only - minimizes LUT, uses BRAM/DSP)
//=============================================================================
static void apply_weight_updates(
    hls::stream<weight_update_t> &weight_updates,
    const learning_params_t &params,
    ap_int<8> reward_signal  // For R-STDP
) {
    #pragma HLS INLINE off
    #pragma HLS ALLOCATION operation instances=mul limit=1
    
    // Process limited updates per cycle to bound resource usage
    int update_count = 0;
    const int MAX_UPDATES_PER_CYCLE = 8;
    
    while (!weight_updates.empty() && update_count < MAX_UPDATES_PER_CYCLE) {
        #pragma HLS PIPELINE II=2
        #pragma HLS LOOP_TRIPCOUNT min=0 max=8
        
        weight_update_t update = weight_updates.read();
        update_count++;
        
        // Read current weight from BRAM
        weight_t current_weight = weight_memory[update.pre_id][update.post_id];
        
        // Apply R-STDP modulation using shifts only
        ap_int<16> modulated_delta = (ap_int<16>)update.delta;
        
        if (params.rstdp_enable && reward_signal != 0) {
            // Reward scaling: delta * (reward/128) ≈ (delta * reward) >> 7
            // But avoid multiply: use conditional shifts based on reward magnitude
            ap_int<8> abs_reward = (reward_signal >= 0) ? reward_signal : (ap_int<8>)(-reward_signal);
            
            // Approximate scaling by reward using shifts
            // reward ~64-127 -> delta >> 1, reward ~32-63 -> delta >> 2, etc.
            ap_int<16> reward_scaled;
            if (abs_reward >= 64) {
                reward_scaled = modulated_delta - (modulated_delta >> 2);  // ~0.75x
            } else if (abs_reward >= 32) {
                reward_scaled = modulated_delta >> 1;  // 0.5x
            } else if (abs_reward >= 16) {
                reward_scaled = modulated_delta >> 2;  // 0.25x
            } else {
                reward_scaled = modulated_delta >> 3;  // 0.125x
            }
            
            // Apply sign of reward
            modulated_delta = (reward_signal >= 0) ? reward_scaled : -reward_scaled;
            
            // Update eligibility trace using shift
            ap_fixed<16,8> trace_update = (ap_fixed<16,8>)update.delta >> 2;
            eligibility_traces[update.pre_id][update.post_id] += trace_update;
        }
        
        // Apply learning rate using shifts: lr * delta
        // Approximate learning_rate (0.01-1.0) with shift: >>4 for ~0.0625, >>3 for ~0.125
        ap_int<16> lr_scaled = (modulated_delta >> 3) + (modulated_delta >> 5);  // ~0.156
        
        // Update weight
        ap_int<16> new_weight = (ap_int<16>)current_weight + lr_scaled;
        
        // Clip and store to BRAM
        weight_memory[update.pre_id][update.post_id] = clip_weight(new_weight);
    }
}

//=============================================================================
// Decay Eligibility Traces (For R-STDP) - Shift-based decay
//=============================================================================
static void decay_eligibility_traces(const learning_params_t &params) {
    #pragma HLS INLINE off
    #pragma HLS ALLOCATION operation instances=mul limit=0
    
    DECAY_OUTER: for (int i = 0; i < MAX_NEURONS; i++) {
        DECAY_INNER: for (int j = 0; j < MAX_NEURONS; j++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=4
            
            // Decay using shift: trace = trace - (trace >> 3) ≈ trace * 0.875
            ap_fixed<16,8> trace = eligibility_traces[i][j];
            ap_fixed<16,8> decay_amount = trace >> 3;  // 12.5% decay
            eligibility_traces[i][j] = trace - decay_amount;
        }
    }
}

//=============================================================================
// Apply Reward Signal (R-STDP) - Pure Shift/Add Operations
//=============================================================================
static void apply_reward_signal(
    ap_int<8> reward_signal,
    const learning_params_t &params
) {
    #pragma HLS INLINE off
    #pragma HLS ALLOCATION operation instances=mul limit=0
    
    if (reward_signal == 0) return;
    
    // Precompute reward sign and magnitude
    bool reward_positive = (reward_signal >= 0);
    ap_uint<8> reward_mag = reward_positive ? (ap_uint<8>)reward_signal : (ap_uint<8>)(-reward_signal);
    
    // Determine shift amount based on reward magnitude
    ap_uint<3> reward_shift;
    if (reward_mag >= 64) reward_shift = 1;       // ~0.5x
    else if (reward_mag >= 32) reward_shift = 2;  // ~0.25x
    else if (reward_mag >= 16) reward_shift = 3;  // ~0.125x
    else reward_shift = 4;                         // ~0.0625x
    
    REWARD_OUTER: for (int i = 0; i < MAX_NEURONS; i++) {
        REWARD_INNER: for (int j = 0; j < MAX_NEURONS; j++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=4
            
            ap_fixed<16,8> trace = eligibility_traces[i][j];
            
            // Skip zero traces (most common case)
            if (trace == 0) continue;
            
            weight_t current = weight_memory[i][j];
            
            // Scale trace by reward using precomputed shift
            ap_int<16> trace_int = (ap_int<16>)trace;
            ap_int<16> scaled_trace;
            
            // Manual shift selection to avoid variable shift (saves LUT)
            switch (reward_shift) {
                case 1: scaled_trace = trace_int >> 1; break;
                case 2: scaled_trace = trace_int >> 2; break;
                case 3: scaled_trace = trace_int >> 3; break;
                default: scaled_trace = trace_int >> 4; break;
            }
            
            // Apply learning rate: another shift (>> 3 ≈ 0.125)
            ap_int<16> delta = scaled_trace >> 3;
            
            // Apply reward sign
            if (!reward_positive) delta = -delta;
            
            // Update weight
            weight_memory[i][j] = clip_weight((ap_int<16>)current + delta);
        }
    }
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
    // Increased BRAM usage to reduce LUT - factor=4 matches UNROLL
    //=========================================================================
    #pragma HLS BIND_STORAGE variable=weight_memory type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=weight_memory cyclic factor=4 dim=2
    #pragma HLS BIND_STORAGE variable=pre_spike_times type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=pre_spike_times cyclic factor=4
    #pragma HLS BIND_STORAGE variable=post_spike_times type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=post_spike_times cyclic factor=4
    #pragma HLS BIND_STORAGE variable=eligibility_traces type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=eligibility_traces cyclic factor=4 dim=2
    
    //=========================================================================
    // Internal State
    //=========================================================================
    static ap_uint<32> timestamp = 0;
    static ap_uint<32> spike_counter = 0;
    static ap_uint<32> update_counter = 0;
    static bool initialized = false;
    
    // Weight update FIFO
    static hls::stream<weight_update_t> weight_update_fifo;
    #pragma HLS STREAM variable=weight_update_fifo depth=64
    
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
        
        // Clear spike times
        RESET_PRE: for (int i = 0; i < MAX_NEURONS; i++) {
            #pragma HLS PIPELINE II=1
            pre_spike_times[i] = 0;
            post_spike_times[i] = 0;
        }
        
        // Clear eligibility traces
        RESET_TRACE_OUTER: for (int i = 0; i < MAX_NEURONS; i++) {
            RESET_TRACE_INNER: for (int j = 0; j < MAX_NEURONS; j++) {
                #pragma HLS PIPELINE II=1
                eligibility_traces[i][j] = 0;
            }
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
        
        // Process for STDP learning
        if (learning_enable) {
            process_pre_spike(pre_id, timestamp, learning_params, true, weight_update_fifo);
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
        
        // Process for STDP learning
        if (learning_enable) {
            process_post_spike(post_id, timestamp, learning_params, true, weight_update_fifo);
        }
    }
    
    //=========================================================================
    // Apply Weight Updates (STDP)
    //=========================================================================
    if (learning_enable) {
        apply_weight_updates(weight_update_fifo, learning_params, 
                            learning_params.rstdp_enable ? reward_signal : (ap_int<8>)0);
    }
    
    //=========================================================================
    // Apply Reward Signal (R-STDP)
    //=========================================================================
    if (apply_reward && learning_params.rstdp_enable) {
        apply_reward_signal(reward_signal, learning_params);
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
    status[3] = !weight_update_fifo.empty();
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
