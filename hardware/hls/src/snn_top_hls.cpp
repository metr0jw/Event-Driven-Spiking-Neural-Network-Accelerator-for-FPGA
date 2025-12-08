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
// STDP Weight Update Calculation
//=============================================================================
static weight_delta_t calc_stdp_update(
    spike_time_t pre_time,
    spike_time_t post_time,
    const learning_params_t &params
) {
    #pragma HLS INLINE
    
    if (pre_time == 0 || post_time == 0) return 0;
    
    ap_int<32> dt = (ap_int<32>)post_time - (ap_int<32>)pre_time;
    weight_delta_t delta = 0;
    
    if (dt > 0 && dt < (ap_int<32>)params.stdp_window) {
        // LTP: pre before post
        ap_fixed<16,8> dt_ratio = (ap_fixed<16,8>)dt / (ap_fixed<16,8>)params.tau_plus;
        ap_fixed<16,8> exp_decay = ap_fixed<16,8>(1.0) - dt_ratio;
        if (exp_decay > 0) {
            delta = (weight_delta_t)(params.a_plus * exp_decay * WEIGHT_SCALE);
        }
    } else if (dt < 0 && (-dt) < (ap_int<32>)params.stdp_window) {
        // LTD: post before pre
        ap_fixed<16,8> dt_ratio = (ap_fixed<16,8>)dt / (ap_fixed<16,8>)params.tau_minus;
        ap_fixed<16,8> exp_decay = ap_fixed<16,8>(1.0) + dt_ratio;
        if (exp_decay > 0) {
            delta = (weight_delta_t)(-params.a_minus * exp_decay * WEIGHT_SCALE);
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
        #pragma HLS UNROLL factor=8
        
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
        #pragma HLS UNROLL factor=8
        
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
// Apply Weight Updates
//=============================================================================
static void apply_weight_updates(
    hls::stream<weight_update_t> &weight_updates,
    const learning_params_t &params,
    ap_int<8> reward_signal  // For R-STDP
) {
    #pragma HLS INLINE off
    
    while (!weight_updates.empty()) {
        #pragma HLS PIPELINE II=2
        
        weight_update_t update = weight_updates.read();
        
        // Read current weight
        weight_t current_weight = weight_memory[update.pre_id][update.post_id];
        
        // Apply R-STDP modulation if reward signal provided
        weight_delta_t modulated_delta = update.delta;
        if (params.rstdp_enable && reward_signal != 0) {
            // Scale delta by reward signal
            ap_fixed<16,8> reward_factor = (ap_fixed<16,8>)reward_signal / ap_fixed<16,8>(128.0);
            modulated_delta = (weight_delta_t)((ap_fixed<16,8>)update.delta * reward_factor);
            
            // Update eligibility trace
            eligibility_traces[update.pre_id][update.post_id] += 
                (ap_fixed<16,8>)update.delta * params.trace_decay;
        }
        
        // Calculate new weight with learning rate
        ap_int<16> new_weight = (ap_int<16>)current_weight + 
            (ap_int<16>)(modulated_delta * params.learning_rate);
        
        // Clip and store
        weight_memory[update.pre_id][update.post_id] = clip_weight(new_weight);
    }
}

//=============================================================================
// Decay Eligibility Traces (For R-STDP)
//=============================================================================
static void decay_eligibility_traces(const learning_params_t &params) {
    #pragma HLS INLINE off
    
    DECAY_OUTER: for (int i = 0; i < MAX_NEURONS; i++) {
        DECAY_INNER: for (int j = 0; j < MAX_NEURONS; j++) {
            #pragma HLS PIPELINE II=1
            eligibility_traces[i][j] *= params.trace_decay;
        }
    }
}

//=============================================================================
// Apply Reward Signal (R-STDP)
//=============================================================================
static void apply_reward_signal(
    ap_int<8> reward_signal,
    const learning_params_t &params
) {
    #pragma HLS INLINE off
    
    if (reward_signal == 0) return;
    
    ap_fixed<16,8> reward_factor = (ap_fixed<16,8>)reward_signal / ap_fixed<16,8>(128.0);
    
    REWARD_OUTER: for (int i = 0; i < MAX_NEURONS; i++) {
        REWARD_INNER: for (int j = 0; j < MAX_NEURONS; j++) {
            #pragma HLS PIPELINE II=1
            
            ap_fixed<16,8> trace = eligibility_traces[i][j];
            if (trace != 0) {
                weight_t current = weight_memory[i][j];
                ap_int<16> delta = (ap_int<16>)(trace * reward_factor * params.learning_rate * WEIGHT_SCALE);
                weight_memory[i][j] = clip_weight((ap_int<16>)current + delta);
            }
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
    //=========================================================================
    #pragma HLS BIND_STORAGE variable=weight_memory type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=weight_memory cyclic factor=8 dim=2
    #pragma HLS BIND_STORAGE variable=pre_spike_times type=RAM_1P impl=BRAM
    #pragma HLS BIND_STORAGE variable=post_spike_times type=RAM_1P impl=BRAM
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
            #pragma HLS UNROLL factor=8
            pre_spike_times[i] = 0;
            post_spike_times[i] = 0;
        }
        
        // Clear eligibility traces
        RESET_TRACE_OUTER: for (int i = 0; i < MAX_NEURONS; i++) {
            RESET_TRACE_INNER: for (int j = 0; j < MAX_NEURONS; j++) {
                #pragma HLS UNROLL factor=8
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
    
    // Calculate weight sum for monitoring
    ap_int<32> weight_sum = 0;
    WEIGHT_SUM: for (int i = 0; i < 16; i++) {  // Sample subset
        #pragma HLS UNROLL
        for (int j = 0; j < 16; j++) {
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
