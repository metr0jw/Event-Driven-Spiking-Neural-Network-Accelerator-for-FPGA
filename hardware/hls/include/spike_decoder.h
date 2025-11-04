//-----------------------------------------------------------------------------
// Title         : Spike Decoder Header
// Project       : PYNQ-Z2 SNN Accelerator
// File          : spike_decoder.h
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Interface for spike to output conversion
//-----------------------------------------------------------------------------

#ifndef SPIKE_DECODER_H
#define SPIKE_DECODER_H

#include "snn_types.h"

// Decoder configuration
struct decoder_config_t {
    decoding_type_t decoding_type;
    ap_uint<16> num_outputs;
    ap_uint<32> window_size;     // Integration window size
    ap_fixed<8,4> rate_alpha;    // Exponential moving average factor
    bool enable_softmax;         // Apply softmax to outputs
    ap_fixed<16,8> temperature;  // Softmax temperature
};

// Function prototypes
void spike_decoder(
    bool enable,
    decoder_config_t config,
    hls::stream<spike_event_t> &spikes_in,
    hls::stream<output_data_t> &data_out,
    ap_uint<32> &status
);

// Decoding functions
void decode_spike_count(
    ap_uint<16> counts[MAX_OUTPUT_NEURONS],
    decoder_config_t config,
    output_data_t &output
);

void decode_spike_rate(
    ap_fixed<16,8> rates[MAX_OUTPUT_NEURONS],
    decoder_config_t config,
    output_data_t &output
);

void decode_first_spike(
    ap_uint<16> counts[MAX_OUTPUT_NEURONS],
    decoder_config_t config,
    output_data_t &output
);

// Utility functions
void apply_softmax(
    ap_fixed<16,8> inputs[MAX_OUTPUT_NEURONS],
    ap_fixed<16,8> outputs[MAX_OUTPUT_NEURONS],
    int num_outputs,
    ap_fixed<16,8> temperature
);

#endif // SPIKE_DECODER_H
