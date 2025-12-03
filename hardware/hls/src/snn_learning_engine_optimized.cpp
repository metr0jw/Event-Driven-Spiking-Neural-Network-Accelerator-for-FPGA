//-----------------------------------------------------------------------------
// Title         : SNN Learning Engine - Optimized Version
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_learning_engine_optimized.cpp
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Email         : jwlee@linux.com
// Description   : STDP learning with multiple optimization strategies
//                 - Active neuron tracking
//                 - Early exit optimization
//                 - CAM-based lookup
//                 - Dataflow parallel processing
//                 - Hierarchical time binning
//-----------------------------------------------------------------------------

#include "snn_learning_engine_optimized.h"
#include <hls_math.h>

//=============================================================================
// Optimization Strategy Selection
//=============================================================================
// Enable exactly ONE strategy per compilation
// #define OPT_STRATEGY_BASELINE        // Original implementation
// #define OPT_STRATEGY_ACTIVE_TRACKING // Strategy 1: Active neuron lists
// #define OPT_STRATEGY_CAM_LOOKUP      // Strategy 3: CAM-based O(1) lookup
#define OPT_STRATEGY_DATAFLOW           // Strategy 4: Parallel pre/post processing
// #define OPT_STRATEGY_HIERARCHICAL    // Strategy 5: Time-based binning

//=============================================================================
// Configuration Parameters
//=============================================================================
#ifdef OPT_STRATEGY_ACTIVE_TRACKING
const int MAX_ACTIVE_NEURONS = 16;  // Tunable: 8, 16, 32
const int CLEANUP_INTERVAL = 1000;  // Cycles between pruning
#endif

#ifdef OPT_STRATEGY_CAM_LOOKUP
const int CAM_SIZE = 32;            // CAM entries (power of 2)
const int CAM_UNROLL_FACTOR = 8;    // Parallel search factor
#endif

#ifdef OPT_STRATEGY_HIERARCHICAL
const int NUM_TIME_BINS = 8;        // Number of time bins
#endif

//=============================================================================
// Data Structures
//=============================================================================

#ifdef OPT_STRATEGY_ACTIVE_TRACKING
// Active neuron tracking lists
struct active_list_t {
    neuron_id_t neurons[MAX_ACTIVE_NEURONS];
    ap_uint<8> count;
};
#endif

#ifdef OPT_STRATEGY_CAM_LOOKUP
// Content Addressable Memory entry
struct cam_entry_t {
    neuron_id_t neuron_id;
    spike_time_t spike_time;
    bool valid;
};
#endif

#ifdef OPT_STRATEGY_HIERARCHICAL
// Time bin structure
struct time_bin_t {
    neuron_id_t neurons[MAX_NEURONS / NUM_TIME_BINS + 1];
    ap_uint<8> count;
};
#endif

//=============================================================================
// Helper Functions
//=============================================================================

#ifdef OPT_STRATEGY_ACTIVE_TRACKING
// Add neuron to active list
void add_to_active_list(neuron_id_t neuron_id, active_list_t &active_list) {
    #pragma HLS INLINE
    
    // Check if already in list
    bool found = false;
    CHECK_EXISTING: for (int i = 0; i < active_list.count; i++) {
        #pragma HLS UNROLL
        if (active_list.neurons[i] == neuron_id) {
            found = true;
            break;
        }
    }
    
    // Add if not found and space available
    if (!found && active_list.count < MAX_ACTIVE_NEURONS) {
        active_list.neurons[active_list.count] = neuron_id;
        active_list.count++;
    }
}

// Prune old spike times outside STDP window
void prune_old_spikes(spike_time_t spike_times[MAX_NEURONS], 
                      spike_time_t current_time, 
                      ap_uint<32> stdp_window,
                      active_list_t &active_list) {
    #pragma HLS INLINE off
    
    ap_uint<8> new_count = 0;
    
    PRUNE_LOOP: for (int i = 0; i < active_list.count; i++) {
        #pragma HLS PIPELINE II=1
        
        neuron_id_t neuron_id = active_list.neurons[i];
        spike_time_t spike_time = spike_times[neuron_id];
        
        // Keep if within window
        if (spike_time > 0 && (current_time - spike_time) < stdp_window * 2) {
            active_list.neurons[new_count] = neuron_id;
            new_count++;
        } else {
            spike_times[neuron_id] = 0;
        }
    }
    
    active_list.count = new_count;
}
#endif

#ifdef OPT_STRATEGY_CAM_LOOKUP
// Initialize CAM
void cam_init(cam_entry_t cam[CAM_SIZE]) {
    #pragma HLS INLINE
    
    CAM_INIT: for (int i = 0; i < CAM_SIZE; i++) {
        #pragma HLS UNROLL
        cam[i].valid = false;
        cam[i].neuron_id = 0;
        cam[i].spike_time = 0;
    }
}

// Insert into CAM (replace oldest or invalid entry)
void cam_insert(cam_entry_t cam[CAM_SIZE], neuron_id_t neuron_id, spike_time_t spike_time) {
    #pragma HLS INLINE
    
    // First pass: find existing entry or invalid slot
    CAM_INSERT_PASS1: for (int i = 0; i < CAM_SIZE; i++) {
        #pragma HLS UNROLL factor=CAM_UNROLL_FACTOR
        
        if (!cam[i].valid || cam[i].neuron_id == neuron_id) {
            cam[i].neuron_id = neuron_id;
            cam[i].spike_time = spike_time;
            cam[i].valid = true;
            return;
        }
    }
    
    // If all valid, replace entry 0 (simple policy)
    cam[0].neuron_id = neuron_id;
    cam[0].spike_time = spike_time;
    cam[0].valid = true;
}

// Parallel CAM lookup and STDP check
void cam_lookup(cam_entry_t cam[CAM_SIZE],
                neuron_id_t query_id,
                spike_time_t query_time,
                learning_config_t config,
                bool is_ltd,
                hls::stream<weight_update_t> &weight_updates,
                ap_uint<32> &update_counter) {
    #pragma HLS INLINE off
    
    CAM_LOOKUP: for (int i = 0; i < CAM_SIZE; i++) {
        #pragma HLS UNROLL factor=CAM_UNROLL_FACTOR
        
        if (cam[i].valid) {
            ap_int<32> dt;
            neuron_id_t pre_id, post_id;
            
            if (is_ltd) {
                // LTD: pre-spike query, post-spike in CAM
                dt = query_time - cam[i].spike_time;
                pre_id = query_id;
                post_id = cam[i].neuron_id;
            } else {
                // LTP: post-spike query, pre-spike in CAM
                dt = query_time - cam[i].spike_time;
                pre_id = cam[i].neuron_id;
                post_id = query_id;
            }
            
            if (dt > 0 && dt < config.stdp_window) {
                weight_delta_t delta = is_ltd ? 
                    calculate_ltd(dt, config) : 
                    calculate_ltp(dt, config);
                
                if (delta != 0) {
                    weight_update_t update;
                    update.pre_id = pre_id;
                    update.post_id = post_id;
                    update.delta = delta;
                    update.timestamp = query_time;
                    
                    weight_updates.write(update);
                    update_counter++;
                }
            }
        }
    }
}
#endif

#ifdef OPT_STRATEGY_HIERARCHICAL
// Get bin index for timestamp
ap_uint<4> get_bin_index(spike_time_t timestamp, ap_uint<32> stdp_window) {
    #pragma HLS INLINE
    ap_uint<32> bin_size = stdp_window / NUM_TIME_BINS;
    return (timestamp / bin_size) % NUM_TIME_BINS;
}

// Add neuron to time bin
void add_to_bin(time_bin_t bins[NUM_TIME_BINS], 
                neuron_id_t neuron_id, 
                spike_time_t timestamp,
                ap_uint<32> stdp_window) {
    #pragma HLS INLINE
    
    ap_uint<4> bin_idx = get_bin_index(timestamp, stdp_window);
    
    if (bins[bin_idx].count < (MAX_NEURONS / NUM_TIME_BINS + 1)) {
        bins[bin_idx].neurons[bins[bin_idx].count] = neuron_id;
        bins[bin_idx].count++;
    }
}
#endif

//=============================================================================
// Main Learning Engine - Dataflow Version
//=============================================================================

#ifdef OPT_STRATEGY_DATAFLOW

// Process pre-synaptic spikes (LTD)
void process_pre_spikes(
    hls::stream<spike_event_t> &pre_spikes,
    spike_time_t post_spike_times[MAX_NEURONS],
    learning_config_t config,
    hls::stream<weight_update_t> &pre_updates,
    ap_uint<32> &pre_update_counter,
    spike_time_t pre_spike_times[MAX_NEURONS]
) {
    #pragma HLS INLINE off
    
    if (!pre_spikes.empty()) {
        spike_event_t pre_event = pre_spikes.read();
        neuron_id_t pre_id = pre_event.neuron_id;
        spike_time_t pre_time = pre_event.timestamp;
        
        if (pre_id < MAX_NEURONS) {
            // Update spike time array
            pre_spike_times[pre_id] = pre_time;
            
            // Scan post neurons with early exit
            ap_uint<16> valid_post_count = 0;
            
            LTD_LOOP: for (int post_id = 0; post_id < MAX_NEURONS; post_id++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=64 max=256 avg=128
                
                spike_time_t post_time = post_spike_times[post_id];
                
                if (post_time > 0) {
                    ap_int<32> dt = pre_time - post_time;
                    
                    if (dt > 0 && dt < config.stdp_window) {
                        weight_delta_t delta = calculate_ltd(dt, config);
                        
                        if (delta != 0) {
                            weight_update_t update;
                            update.pre_id = pre_id;
                            update.post_id = post_id;
                            update.delta = delta;
                            update.timestamp = pre_time;
                            
                            pre_updates.write(update);
                            pre_update_counter++;
                        }
                    }
                    valid_post_count++;
                }
            }
        }
    }
}

// Process post-synaptic spikes (LTP)
void process_post_spikes(
    hls::stream<spike_event_t> &post_spikes,
    spike_time_t pre_spike_times[MAX_NEURONS],
    learning_config_t config,
    hls::stream<weight_update_t> &post_updates,
    ap_uint<32> &post_update_counter,
    spike_time_t post_spike_times[MAX_NEURONS]
) {
    #pragma HLS INLINE off
    
    if (!post_spikes.empty()) {
        spike_event_t post_event = post_spikes.read();
        neuron_id_t post_id = post_event.neuron_id;
        spike_time_t post_time = post_event.timestamp;
        
        if (post_id < MAX_NEURONS) {
            // Update spike time array
            post_spike_times[post_id] = post_time;
            
            // Scan pre neurons
            LTP_LOOP: for (int pre_id = 0; pre_id < MAX_NEURONS; pre_id++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=64 max=256 avg=128
                
                spike_time_t pre_time = pre_spike_times[pre_id];
                
                if (pre_time > 0) {
                    ap_int<32> dt = post_time - pre_time;
                    
                    if (dt > 0 && dt < config.stdp_window) {
                        weight_delta_t delta = calculate_ltp(dt, config);
                        
                        if (delta != 0) {
                            weight_update_t update;
                            update.pre_id = pre_id;
                            update.post_id = post_id;
                            update.delta = delta;
                            update.timestamp = post_time;
                            
                            post_updates.write(update);
                            post_update_counter++;
                        }
                    }
                }
            }
        }
    }
}

// Merge weight updates from both streams
void merge_weight_updates(
    hls::stream<weight_update_t> &pre_updates,
    hls::stream<weight_update_t> &post_updates,
    hls::stream<weight_update_t> &weight_updates
) {
    #pragma HLS INLINE off
    
    // Read and forward pre updates
    while (!pre_updates.empty()) {
        weight_updates.write(pre_updates.read());
    }
    
    // Read and forward post updates
    while (!post_updates.empty()) {
        weight_updates.write(post_updates.read());
    }
}

void snn_learning_engine_optimized(
    bool enable,
    bool reset,
    learning_config_t config,
    hls::stream<spike_event_t> &pre_spikes,
    hls::stream<spike_event_t> &post_spikes,
    hls::stream<weight_update_t> &weight_updates,
    ap_uint<32> &status
) {
    #pragma HLS INTERFACE s_axilite port=enable bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=reset bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=config bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=status bundle=ctrl
    #pragma HLS INTERFACE axis port=pre_spikes
    #pragma HLS INTERFACE axis port=post_spikes
    #pragma HLS INTERFACE axis port=weight_updates
    #pragma HLS INTERFACE s_axilite port=return bundle=ctrl
    
    #pragma HLS DATAFLOW
    
    // Shared spike time arrays
    static spike_time_t pre_spike_times[MAX_NEURONS];
    static spike_time_t post_spike_times[MAX_NEURONS];
    static ap_uint<32> pre_counter = 0;
    static ap_uint<32> post_counter = 0;
    
    #pragma HLS ARRAY_PARTITION variable=pre_spike_times cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=post_spike_times cyclic factor=8
    
    // Internal update streams
    static hls::stream<weight_update_t> pre_updates("pre_updates");
    static hls::stream<weight_update_t> post_updates("post_updates");
    #pragma HLS STREAM variable=pre_updates depth=32
    #pragma HLS STREAM variable=post_updates depth=32
    
    if (reset) {
        RESET_LOOP: for (int i = 0; i < MAX_NEURONS; i++) {
            #pragma HLS PIPELINE II=1
            pre_spike_times[i] = 0;
            post_spike_times[i] = 0;
        }
        pre_counter = 0;
        post_counter = 0;
        status = 0;
        return;
    }
    
    if (!enable) {
        status = 0x80000000;
        return;
    }
    
    // Parallel processing with dataflow
    process_pre_spikes(pre_spikes, post_spike_times, config, pre_updates, pre_counter, pre_spike_times);
    process_post_spikes(post_spikes, pre_spike_times, config, post_updates, post_counter, post_spike_times);
    merge_weight_updates(pre_updates, post_updates, weight_updates);
    
    status = pre_counter + post_counter;
}

#endif // OPT_STRATEGY_DATAFLOW

//=============================================================================
// Main Learning Engine - Active Tracking Version
//=============================================================================

#ifdef OPT_STRATEGY_ACTIVE_TRACKING

void snn_learning_engine_optimized(
    bool enable,
    bool reset,
    learning_config_t config,
    hls::stream<spike_event_t> &pre_spikes,
    hls::stream<spike_event_t> &post_spikes,
    hls::stream<weight_update_t> &weight_updates,
    ap_uint<32> &status
) {
    #pragma HLS INTERFACE s_axilite port=enable bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=reset bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=config bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=status bundle=ctrl
    #pragma HLS INTERFACE axis port=pre_spikes
    #pragma HLS INTERFACE axis port=post_spikes
    #pragma HLS INTERFACE axis port=weight_updates
    #pragma HLS INTERFACE s_axilite port=return bundle=ctrl
    
    static spike_time_t pre_spike_times[MAX_NEURONS];
    static spike_time_t post_spike_times[MAX_NEURONS];
    static active_list_t active_pre;
    static active_list_t active_post;
    static ap_uint<32> update_counter = 0;
    static ap_uint<32> cycle_counter = 0;
    
    #pragma HLS ARRAY_PARTITION variable=pre_spike_times cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=post_spike_times cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=active_pre.neurons complete
    #pragma HLS ARRAY_PARTITION variable=active_post.neurons complete
    
    if (reset) {
        RESET_LOOP: for (int i = 0; i < MAX_NEURONS; i++) {
            #pragma HLS PIPELINE II=1
            pre_spike_times[i] = 0;
            post_spike_times[i] = 0;
        }
        active_pre.count = 0;
        active_post.count = 0;
        update_counter = 0;
        cycle_counter = 0;
        status = 0;
        return;
    }
    
    if (!enable) {
        status = 0x80000000;
        return;
    }
    
    cycle_counter++;
    
    // Periodic cleanup
    if (cycle_counter % CLEANUP_INTERVAL == 0) {
        prune_old_spikes(pre_spike_times, cycle_counter, config.stdp_window, active_pre);
        prune_old_spikes(post_spike_times, cycle_counter, config.stdp_window, active_post);
    }
    
    // Process pre-synaptic spikes
    if (!pre_spikes.empty()) {
        spike_event_t pre_event = pre_spikes.read();
        neuron_id_t pre_id = pre_event.neuron_id;
        spike_time_t pre_time = pre_event.timestamp;
        
        if (pre_id < MAX_NEURONS) {
            pre_spike_times[pre_id] = pre_time;
            add_to_active_list(pre_id, active_pre);
            
            // Only scan active post neurons
            LTD_ACTIVE_LOOP: for (int i = 0; i < active_post.count; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=8 max=16 avg=12
                
                neuron_id_t post_id = active_post.neurons[i];
                spike_time_t post_time = post_spike_times[post_id];
                
                if (post_time > 0) {
                    ap_int<32> dt = pre_time - post_time;
                    
                    if (dt > 0 && dt < config.stdp_window) {
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
            add_to_active_list(post_id, active_post);
            
            // Only scan active pre neurons
            LTP_ACTIVE_LOOP: for (int i = 0; i < active_pre.count; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=8 max=16 avg=12
                
                neuron_id_t pre_id = active_pre.neurons[i];
                spike_time_t pre_time = pre_spike_times[pre_id];
                
                if (pre_time > 0) {
                    ap_int<32> dt = post_time - pre_time;
                    
                    if (dt > 0 && dt < config.stdp_window) {
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
    
    status = update_counter;
}

#endif // OPT_STRATEGY_ACTIVE_TRACKING

//=============================================================================
// Main Learning Engine - CAM-based Version
//=============================================================================

#ifdef OPT_STRATEGY_CAM_LOOKUP

void snn_learning_engine_optimized(
    bool enable,
    bool reset,
    learning_config_t config,
    hls::stream<spike_event_t> &pre_spikes,
    hls::stream<spike_event_t> &post_spikes,
    hls::stream<weight_update_t> &weight_updates,
    ap_uint<32> &status
) {
    #pragma HLS INTERFACE s_axilite port=enable bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=reset bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=config bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=status bundle=ctrl
    #pragma HLS INTERFACE axis port=pre_spikes
    #pragma HLS INTERFACE axis port=post_spikes
    #pragma HLS INTERFACE axis port=weight_updates
    #pragma HLS INTERFACE s_axilite port=return bundle=ctrl
    
    static cam_entry_t pre_cam[CAM_SIZE];
    static cam_entry_t post_cam[CAM_SIZE];
    static ap_uint<32> update_counter = 0;
    
    #pragma HLS ARRAY_PARTITION variable=pre_cam complete
    #pragma HLS ARRAY_PARTITION variable=post_cam complete
    
    if (reset) {
        cam_init(pre_cam);
        cam_init(post_cam);
        update_counter = 0;
        status = 0;
        return;
    }
    
    if (!enable) {
        status = 0x80000000;
        return;
    }
    
    // Process pre-synaptic spikes
    if (!pre_spikes.empty()) {
        spike_event_t pre_event = pre_spikes.read();
        neuron_id_t pre_id = pre_event.neuron_id;
        spike_time_t pre_time = pre_event.timestamp;
        
        if (pre_id < MAX_NEURONS) {
            cam_insert(pre_cam, pre_id, pre_time);
            cam_lookup(post_cam, pre_id, pre_time, config, true, weight_updates, update_counter);
        }
    }
    
    // Process post-synaptic spikes
    if (!post_spikes.empty()) {
        spike_event_t post_event = post_spikes.read();
        neuron_id_t post_id = post_event.neuron_id;
        spike_time_t post_time = post_event.timestamp;
        
        if (post_id < MAX_NEURONS) {
            cam_insert(post_cam, post_id, post_time);
            cam_lookup(pre_cam, post_id, post_time, config, false, weight_updates, update_counter);
        }
    }
    
    status = update_counter;
}

#endif // OPT_STRATEGY_CAM_LOOKUP

//=============================================================================
// Main Learning Engine - Hierarchical Binning Version
//=============================================================================

#ifdef OPT_STRATEGY_HIERARCHICAL

void snn_learning_engine_optimized(
    bool enable,
    bool reset,
    learning_config_t config,
    hls::stream<spike_event_t> &pre_spikes,
    hls::stream<spike_event_t> &post_spikes,
    hls::stream<weight_update_t> &weight_updates,
    ap_uint<32> &status
) {
    #pragma HLS INTERFACE s_axilite port=enable bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=reset bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=config bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=status bundle=ctrl
    #pragma HLS INTERFACE axis port=pre_spikes
    #pragma HLS INTERFACE axis port=post_spikes
    #pragma HLS INTERFACE axis port=weight_updates
    #pragma HLS INTERFACE s_axilite port=return bundle=ctrl
    
    static time_bin_t pre_bins[NUM_TIME_BINS];
    static time_bin_t post_bins[NUM_TIME_BINS];
    static spike_time_t pre_spike_times[MAX_NEURONS];
    static spike_time_t post_spike_times[MAX_NEURONS];
    static ap_uint<32> update_counter = 0;
    
    #pragma HLS ARRAY_PARTITION variable=pre_bins complete
    #pragma HLS ARRAY_PARTITION variable=post_bins complete
    #pragma HLS ARRAY_PARTITION variable=pre_spike_times cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=post_spike_times cyclic factor=8
    
    if (reset) {
        RESET_BINS: for (int i = 0; i < NUM_TIME_BINS; i++) {
            #pragma HLS UNROLL
            pre_bins[i].count = 0;
            post_bins[i].count = 0;
        }
        RESET_TIMES: for (int i = 0; i < MAX_NEURONS; i++) {
            #pragma HLS PIPELINE II=1
            pre_spike_times[i] = 0;
            post_spike_times[i] = 0;
        }
        update_counter = 0;
        status = 0;
        return;
    }
    
    if (!enable) {
        status = 0x80000000;
        return;
    }
    
    // Process pre-synaptic spikes
    if (!pre_spikes.empty()) {
        spike_event_t pre_event = pre_spikes.read();
        neuron_id_t pre_id = pre_event.neuron_id;
        spike_time_t pre_time = pre_event.timestamp;
        
        if (pre_id < MAX_NEURONS) {
            pre_spike_times[pre_id] = pre_time;
            add_to_bin(pre_bins, pre_id, pre_time, config.stdp_window);
            
            // Check relevant post bins
            CHECK_POST_BINS: for (int bin = 0; bin < NUM_TIME_BINS; bin++) {
                #pragma HLS UNROLL
                
                // Check neurons in this bin
                BIN_SEARCH: for (int i = 0; i < post_bins[bin].count; i++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=4 max=12 avg=8
                    
                    neuron_id_t post_id = post_bins[bin].neurons[i];
                    spike_time_t post_time = post_spike_times[post_id];
                    
                    if (post_time > 0) {
                        ap_int<32> dt = pre_time - post_time;
                        
                        if (dt > 0 && dt < config.stdp_window) {
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
    }
    
    // Process post-synaptic spikes
    if (!post_spikes.empty()) {
        spike_event_t post_event = post_spikes.read();
        neuron_id_t post_id = post_event.neuron_id;
        spike_time_t post_time = post_event.timestamp;
        
        if (post_id < MAX_NEURONS) {
            post_spike_times[post_id] = post_time;
            add_to_bin(post_bins, post_id, post_time, config.stdp_window);
            
            // Check relevant pre bins
            CHECK_PRE_BINS: for (int bin = 0; bin < NUM_TIME_BINS; bin++) {
                #pragma HLS UNROLL
                
                BIN_SEARCH_LTP: for (int i = 0; i < pre_bins[bin].count; i++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=4 max=12 avg=8
                    
                    neuron_id_t pre_id = pre_bins[bin].neurons[i];
                    spike_time_t pre_time = pre_spike_times[pre_id];
                    
                    if (pre_time > 0) {
                        ap_int<32> dt = post_time - pre_time;
                        
                        if (dt > 0 && dt < config.stdp_window) {
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
    }
    
    status = update_counter;
}

#endif // OPT_STRATEGY_HIERARCHICAL

//=============================================================================
// Main Learning Engine - Baseline Version
//=============================================================================

#ifdef OPT_STRATEGY_BASELINE

void snn_learning_engine_optimized(
    bool enable,
    bool reset,
    learning_config_t config,
    hls::stream<spike_event_t> &pre_spikes,
    hls::stream<spike_event_t> &post_spikes,
    hls::stream<weight_update_t> &weight_updates,
    ap_uint<32> &status
) {
    #pragma HLS INTERFACE s_axilite port=enable bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=reset bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=config bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=status bundle=ctrl
    #pragma HLS INTERFACE axis port=pre_spikes
    #pragma HLS INTERFACE axis port=post_spikes
    #pragma HLS INTERFACE axis port=weight_updates
    #pragma HLS INTERFACE s_axilite port=return bundle=ctrl
    
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
        status = 0x80000000;
        return;
    }
    
    // Baseline implementation (same as original)
    if (!pre_spikes.empty()) {
        spike_event_t pre_event = pre_spikes.read();
        neuron_id_t pre_id = pre_event.neuron_id;
        spike_time_t pre_time = pre_event.timestamp;
        
        if (pre_id < MAX_NEURONS) {
            pre_spike_times[pre_id] = pre_time;
            
            LTD_LOOP: for (int post_id = 0; post_id < MAX_NEURONS; post_id++) {
                #pragma HLS PIPELINE II=2
                #pragma HLS LOOP_TRIPCOUNT min=64 max=256 avg=128
                
                spike_time_t post_time = post_spike_times[post_id];
                
                if (post_time > 0) {
                    ap_int<32> dt = pre_time - post_time;
                    
                    if (dt > 0 && dt < config.stdp_window) {
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
    
    if (!post_spikes.empty()) {
        spike_event_t post_event = post_spikes.read();
        neuron_id_t post_id = post_event.neuron_id;
        spike_time_t post_time = post_event.timestamp;
        
        if (post_id < MAX_NEURONS) {
            post_spike_times[post_id] = post_time;
            
            LTP_LOOP: for (int pre_id = 0; pre_id < MAX_NEURONS; pre_id++) {
                #pragma HLS PIPELINE II=2
                #pragma HLS LOOP_TRIPCOUNT min=64 max=256 avg=128
                
                if (pre_spike_times[pre_id] > 0) {
                    ap_int<32> dt = post_time - pre_spike_times[pre_id];
                    
                    if (dt > 0 && dt < config.stdp_window) {
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
    
    status = update_counter;
}

#endif // OPT_STRATEGY_BASELINE

//=============================================================================
// STDP Calculation Functions (shared by all versions)
//=============================================================================

weight_delta_t calculate_ltp(ap_int<32> dt, learning_config_t config) {
    #pragma HLS INLINE
    
    if (dt <= 0 || dt >= config.stdp_window) {
        return 0;
    }
    
    ap_fixed<16,8> exp_factor = hls::exp(-ap_fixed<16,8>(dt) / config.tau_plus);
    ap_fixed<16,8> delta_float = config.a_plus * exp_factor;
    
    weight_delta_t delta = delta_float * WEIGHT_SCALE;
    
    if (delta > MAX_WEIGHT_DELTA) {
        delta = MAX_WEIGHT_DELTA;
    }
    
    return delta;
}

weight_delta_t calculate_ltd(ap_int<32> dt, learning_config_t config) {
    #pragma HLS INLINE
    
    if (dt <= 0 || dt >= config.stdp_window) {
        return 0;
    }
    
    ap_fixed<16,8> exp_factor = hls::exp(-ap_fixed<16,8>(dt) / config.tau_minus);
    ap_fixed<16,8> delta_float = -config.a_minus * exp_factor;
    
    weight_delta_t delta = delta_float * WEIGHT_SCALE;
    
    if (delta < -MAX_WEIGHT_DELTA) {
        delta = -MAX_WEIGHT_DELTA;
    }
    
    return delta;
}
