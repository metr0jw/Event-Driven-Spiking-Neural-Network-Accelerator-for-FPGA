//-----------------------------------------------------------------------------
// Title         : Weight Updater
// Project       : PYNQ-Z2 SNN Accelerator
// File          : weight_updater.cpp
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Email         : jwlee@linux.com
// Description   : Applies weight updates to synaptic memory
//-----------------------------------------------------------------------------

#include "weight_updater.h"

void weight_updater(
    // Control
    bool enable,
    bool reset,
    
    // Weight update input
    hls::stream<weight_update_t> &updates_in,
    
    // Memory interface
    weight_t *weight_memory,
    
    // Configuration
    weight_config_t config,
    
    // Status
    ap_uint<32> &updates_applied
) {
    #pragma HLS INTERFACE s_axilite port=enable
    #pragma HLS INTERFACE s_axilite port=reset
    #pragma HLS INTERFACE s_axilite port=config
    #pragma HLS INTERFACE s_axilite port=updates_applied
    #pragma HLS INTERFACE axis port=updates_in
    #pragma HLS INTERFACE m_axi port=weight_memory offset=slave depth=65536
    #pragma HLS INTERFACE s_axilite port=return
    
    static ap_uint<32> update_counter = 0;
    
    if (reset) {
        update_counter = 0;
        updates_applied = 0;
        return;
    }
    
    if (!enable) {
        updates_applied = update_counter;
        return;
    }
    
    // Process weight updates
    if (!updates_in.empty()) {
        weight_update_t update = updates_in.read();
        
        // Validate neuron IDs
        if (update.pre_id < MAX_NEURONS && update.post_id < MAX_NEURONS) {
            // Calculate memory address
            ap_uint<32> addr = (update.pre_id * MAX_NEURONS) + update.post_id;
            
            if (addr < MAX_SYNAPSES) {
                // Read current weight
                weight_t current_weight = weight_memory[addr];
                
                // Apply update with bounds checking
                ap_int<16> new_weight = current_weight + update.delta;
                
                // Apply weight bounds
                if (new_weight > config.max_weight) {
                    new_weight = config.max_weight;
                } else if (new_weight < config.min_weight) {
                    new_weight = config.min_weight;
                }
                
                // Apply weight decay if enabled
                if (config.enable_decay) {
                    new_weight = apply_decay(new_weight, config.decay_rate);
                }
                
                // Write back updated weight
                weight_memory[addr] = new_weight;
                update_counter++;
            }
        }
    }
    
    updates_applied = update_counter;
}

// Apply exponential weight decay
weight_t apply_decay(weight_t weight, ap_uint<8> decay_rate) {
    #pragma HLS INLINE
    
    if (weight == 0) return 0;
    
    // Decay towards zero
    ap_int<16> decayed = weight;
    
    if (weight > 0) {
        // Positive weights decay toward zero
        ap_int<16> decay_amount = (weight * decay_rate) >> 8;
        decayed = weight - decay_amount;
        if (decayed < 0) decayed = 0;
    } else {
        // Negative weights decay toward zero (subtract from negative = move toward zero)
        ap_int<16> decay_amount = ((-weight) * decay_rate) >> 8;
        decayed = weight + decay_amount;  // Adding positive to negative moves toward zero
        if (decayed > 0) decayed = 0;
    }
    
    return decayed;
}
