//-----------------------------------------------------------------------------
// Title         : Spike Encoder Header
// Project       : PYNQ-Z2 SNN Accelerator
// File          : spike_encoder.h
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Interface for data to spike conversion
//-----------------------------------------------------------------------------

#ifndef SPIKE_ENCODER_H
#define SPIKE_ENCODER_H

#include "snn_types.h"

// Encoder configuration
struct encoder_config_t {
    encoding_type_t encoding_type;
    ap_uint<16> num_channels;
    ap_uint<16> time_window;      // Time window for temporal coding
    ap_uint<16> rate_scale;       // Scaling factor for rate coding
    ap_uint<16> phase_scale;      // Scaling factor for phase coding
    ap_uint<16> phase_threshold;  // Phase accumulator threshold
    weight_t default_weight;       // Default spike weight
};

// Function prototypes
void spike_encoder(
    bool enable,
    encoder_config_t config,
    hls::stream<input_data_t> &data_in,
    hls::stream<spike_event_t> &spikes_out,
    ap_uint<32> &spike_count
);

// Encoding functions
void encode_rate(
    int channel,
    pixel_t value,
    ap_uint<32> time,
    encoder_config_t config,
    hls::stream<spike_event_t> &spikes_out,
    ap_uint<32> &spike_counter
);

void encode_temporal(
    int channel,
    pixel_t value,
    ap_uint<32> time,
    encoder_config_t config,
    hls::stream<spike_event_t> &spikes_out,
    ap_uint<32> &spike_counter
);

void encode_phase(
    int channel,
    pixel_t value,
    ap_uint<32> time,
    encoder_config_t config,
    ap_uint<16> &phase_acc,
    hls::stream<spike_event_t> &spikes_out,
    ap_uint<32> &spike_counter
);

// Utility functions
ap_uint<16> lfsr_random();

#endif // SPIKE_ENCODER_H
