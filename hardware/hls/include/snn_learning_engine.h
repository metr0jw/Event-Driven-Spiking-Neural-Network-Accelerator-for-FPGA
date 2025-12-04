  //-----------------------------------------------------------------------------
// Title         : SNN Learning Engine Header
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_learning_engine.h
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Description   : STDP learning with multiple strategies
//-----------------------------------------------------------------------------

#ifndef SNN_LEARNING_ENGINE_H
#define SNN_LEARNING_ENGINE_H

#include "snn_types.h"

// Learning configuration structure
struct learning_config_t {
    ap_fixed<16,8> a_plus;        // LTP amplitude
    ap_fixed<16,8> a_minus;       // LTD amplitude
    ap_fixed<16,8> tau_plus;      // LTP time constant
    ap_fixed<16,8> tau_minus;     // LTD time constant
    ap_uint<32> stdp_window;      // STDP time window
    bool enable_homeostasis;      // Enable synaptic homeostasis
    ap_fixed<16,8> target_rate;   // Target firing rate for homeostasis
};

// Learning engine function
void snn_learning_engine(
    bool enable,
    bool reset,
    learning_config_t config,
    hls::stream<spike_event_t> &pre_spikes,
    hls::stream<spike_event_t> &post_spikes,
    hls::stream<weight_update_t> &weight_updates,
    ap_uint<32> &status
);

// STDP calculation functions
weight_delta_t calculate_ltp(ap_int<32> dt, learning_config_t config);
weight_delta_t calculate_ltd(ap_int<32> dt, learning_config_t config);

#endif // SNN_LEARNING_ENGINE_H
