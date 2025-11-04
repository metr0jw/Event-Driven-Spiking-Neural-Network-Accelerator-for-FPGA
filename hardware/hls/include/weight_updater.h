//-----------------------------------------------------------------------------
// Title         : Weight Updater Header
// Project       : PYNQ-Z2 SNN Accelerator
// File          : weight_updater.h
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Interface for synaptic weight management
//-----------------------------------------------------------------------------

#ifndef WEIGHT_UPDATER_H
#define WEIGHT_UPDATER_H

#include "snn_types.h"

// Weight configuration
struct weight_config_t {
    weight_t max_weight;
    weight_t min_weight;
    bool enable_decay;
    ap_uint<8> decay_rate;      // Decay rate (0-255, where 255 = no decay)
    bool enable_normalization;   // Enable synaptic normalization
    ap_uint<16> norm_target;     // Target sum of weights
};

// Function prototypes
void weight_updater(
    bool enable,
    bool reset,
    hls::stream<weight_update_t> &updates_in,
    weight_t *weight_memory,
    weight_config_t config,
    ap_uint<32> &updates_applied
);

// Utility functions
weight_t apply_decay(weight_t weight, ap_uint<8> decay_rate);
void normalize_weights(weight_t *weights, int num_weights, ap_uint<16> target_sum);

#endif // WEIGHT_UPDATER_H
