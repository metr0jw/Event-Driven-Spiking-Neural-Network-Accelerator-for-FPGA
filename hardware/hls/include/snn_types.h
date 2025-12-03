//-----------------------------------------------------------------------------
// Title         : Common SNN Data Types
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_types.h
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Common type definitions for SNN HLS modules
//-----------------------------------------------------------------------------

#ifndef SNN_TYPES_H
#define SNN_TYPES_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

// System parameters
const int MAX_NEURONS = 64;
const int MAX_SYNAPSES = 4096;  // 64x64
const int MAX_INPUT_CHANNELS = 784;  // For MNIST 28x28
const int MAX_OUTPUT_NEURONS = 10;   // For 10 classes

// Basic data types
typedef ap_uint<8> neuron_id_t;
typedef ap_uint<8> axon_id_t;
typedef ap_uint<32> spike_time_t;
typedef ap_int<8> weight_t;
typedef ap_int<16> weight_delta_t;
typedef ap_uint<8> pixel_t;
typedef ap_uint<16> membrane_t;

// Fixed-point types for learning
typedef ap_fixed<16,8> learning_rate_t;
typedef ap_fixed<16,8> decay_rate_t;

// Constants
const weight_t MAX_WEIGHT = 127;
const weight_t MIN_WEIGHT = -128;
const weight_delta_t MAX_WEIGHT_DELTA = 127;
const int WEIGHT_SCALE = 128;

// Spike event structure
struct spike_event_t {
    neuron_id_t neuron_id;
    spike_time_t timestamp;
    weight_t weight;
};

// Weight update structure
struct weight_update_t {
    neuron_id_t pre_id;
    neuron_id_t post_id;
    weight_delta_t delta;
    spike_time_t timestamp;
};

// Input data structure (e.g., for image processing)
struct input_data_t {
    pixel_t pixels[MAX_INPUT_CHANNELS];
    ap_uint<16> label;  // For supervised learning
    ap_uint<32> frame_id;
};

// Output data structure
struct output_data_t {
    ap_uint<8> class_id;
    ap_uint<8> confidence;
    ap_uint<16> values[MAX_OUTPUT_NEURONS];
    ap_uint<32> frame_id;
};

// Control packet for inter-module communication
struct control_packet_t {
    ap_uint<8> command;
    ap_uint<16> param1;
    ap_uint<16> param2;
    spike_time_t timestamp;
};

// Encoding types
enum encoding_type_t {
    RATE_CODING = 0,
    TEMPORAL_CODING = 1,
    PHASE_CODING = 2,
    BURST_CODING = 3
};

// Decoding types
enum decoding_type_t {
    SPIKE_COUNT = 0,
    SPIKE_RATE = 1,
    FIRST_SPIKE = 2,
    POPULATION_VECTOR = 3
};

// Network modes
enum network_mode_t {
    MODE_IDLE = 0,
    MODE_TRAINING = 1,
    MODE_INFERENCE = 2,
    MODE_VALIDATION = 3
};

// Control commands
enum control_command_t {
    CTRL_RESET = 0,
    CTRL_ENABLE = 1,
    CTRL_DISABLE = 2,
    CTRL_CONFIGURE = 3,
    CTRL_FLUSH = 4
};

#endif // SNN_TYPES_H
