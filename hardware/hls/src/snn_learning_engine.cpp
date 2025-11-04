//-----------------------------------------------------------------------------
// Title         : SNN Learning Engine
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_learning_engine.cpp
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Description   : STDP learning algorithm implementation in HLS
//-----------------------------------------------------------------------------

#include "snn_learning_engine.h"
#include <hls_math.h>

// Top-level function for STDP learning
void snn_learning_engine(
    // Control interface
    bool enable,
    bool reset,
    learning_config_t config,
    
    // Spike input streams
    hls::stream<spike_event_t> &pre_spikes,
    hls::stream<spike_event_t> &post_spikes,
    
    // Weight update stream
    hls::stream<weight_update_t> &weight_updates,
    
    // Status output
    ap_uint<32> &status
) {
    #pragma HLS INTERFACE s_axilite port=enable
    #pragma HLS INTERFACE s_axilite port=reset
    #pragma HLS INTERFACE s_axilite port=config
    #pragma HLS INTERFACE s_axilite port=status
    #pragma HLS INTERFACE axis port=pre_spikes
    #pragma HLS INTERFACE axis port=post_spikes
    #pragma HLS INTERFACE axis port=weight_updates
    #pragma HLS INTERFACE s_axilite port=return
    
    // Internal state
    static spike_time_t pre_spike_times[MAX_NEURONS];
    static spike_time_t post_spike_times[MAX_NEURONS];
    static ap_uint<32> update_counter = 0;
    
    #pragma HLS ARRAY_PARTITION variable=pre_spike_times cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=post_spike_times cyclic factor=8
    
    if (reset) {
        RESET_LOOP: for (int i = 0; i < MAX_NEURONS; i++) {
            #pragma HLS PIPELINE II=1
            pre_spike_times[i] = 0;
            post_spike_times[i] = 0;
        }
        update_counter = 0;
        status = 0;
        return;
    }
    
    if (!enable) {
        status = 0x80000000; // Disabled flag
        return;
    }
    
    // Process pre-synaptic spikes
    if (!pre_spikes.empty()) {
        spike_event_t pre_event = pre_spikes.read();
        neuron_id_t pre_id = pre_event.neuron_id;
        spike_time_t pre_time = pre_event.timestamp;
        
        if (pre_id < MAX_NEURONS) {
            pre_spike_times[pre_id] = pre_time;
            
            // Check for post-pre spike pairs (LTD)
            LTD_LOOP: for (int post_id = 0; post_id < MAX_NEURONS; post_id++) {
                #pragma HLS PIPELINE II=2
                if (post_spike_times[post_id] > 0) {
                    ap_int<32> dt = pre_time - post_spike_times[post_id];
                    
                    if (dt > 0 && dt < config.stdp_window) {
                        // Calculate LTD weight change
                        weight_delta_t delta = calculate_ltd(dt, config);
                        
                        if (delta != 0) {
                            weight_update_t update;
                            update.pre_id = pre_id;
                            update.post_id = post_id;
                            update.delta = delta;
                            update.timestamp = pre_time;
                            
                            weight_updates.write(update);
                            update_counter++;
                        }
                    }
                }
            }
        }
    }
    
    // Process post-synaptic spikes
    if (!post_spikes.empty()) {
        spike_event_t post_event = post_spikes.read();
        neuron_id_t post_id = post_event.neuron_id;
        spike_time_t post_time = post_event.timestamp;
        
        if (post_id < MAX_NEURONS) {
            post_spike_times[post_id] = post_time;
            
            // Check for pre-post spike pairs (LTP)
            LTP_LOOP: for (int pre_id = 0; pre_id < MAX_NEURONS; pre_id++) {
                #pragma HLS PIPELINE II=2
                if (pre_spike_times[pre_id] > 0) {
                    ap_int<32> dt = post_time - pre_spike_times[pre_id];
                    
                    if (dt > 0 && dt < config.stdp_window) {
                        // Calculate LTP weight change
                        weight_delta_t delta = calculate_ltp(dt, config);
                        
                        if (delta != 0) {
                            weight_update_t update;
                            update.pre_id = pre_id;
                            update.post_id = post_id;
                            update.delta = delta;
                            update.timestamp = post_time;
                            
                            weight_updates.write(update);
                            update_counter++;
                        }
                    }
                }
            }
        }
    }
    
    // Update status
    status = update_counter;
}

// Calculate LTP (Long-Term Potentiation) weight change
weight_delta_t calculate_ltp(ap_int<32> dt, learning_config_t config) {
    #pragma HLS INLINE
    
    if (dt <= 0 || dt >= config.stdp_window) {
        return 0;
    }
    
    // Exponential STDP curve: A+ * exp(-dt/tau+)
    ap_fixed<16,8> exp_factor = hls::exp(-ap_fixed<16,8>(dt) / config.tau_plus);
    ap_fixed<16,8> delta_float = config.a_plus * exp_factor;
    
    // Convert to fixed-point weight delta
    weight_delta_t delta = delta_float * WEIGHT_SCALE;
    
    // Clamp to valid range
    if (delta > MAX_WEIGHT_DELTA) {
        delta = MAX_WEIGHT_DELTA;
    }
    
    return delta;
}

// Calculate LTD (Long-Term Depression) weight change
weight_delta_t calculate_ltd(ap_int<32> dt, learning_config_t config) {
    #pragma HLS INLINE
    
    if (dt <= 0 || dt >= config.stdp_window) {
        return 0;
    }
    
    // Exponential STDP curve: -A- * exp(-dt/tau-)
    ap_fixed<16,8> exp_factor = hls::exp(-ap_fixed<16,8>(dt) / config.tau_minus);
    ap_fixed<16,8> delta_float = -config.a_minus * exp_factor;
    
    // Convert to fixed-point weight delta
    weight_delta_t delta = delta_float * WEIGHT_SCALE;
    
    // Clamp to valid range
    if (delta < -MAX_WEIGHT_DELTA) {
        delta = -MAX_WEIGHT_DELTA;
    }
    
    return delta;
}
