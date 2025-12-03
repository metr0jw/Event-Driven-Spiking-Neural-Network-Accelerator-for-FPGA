//-----------------------------------------------------------------------------
// Title         : Network Controller Header
// Project       : PYNQ-Z2 SNN Accelerator
// File          : network_controller.h
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : High-level network control interface
//-----------------------------------------------------------------------------

#ifndef NETWORK_CONTROLLER_H
#define NETWORK_CONTROLLER_H

#include "snn_types.h"

// Network commands
enum network_command_t {
    CMD_IDLE = 0,
    CMD_START = 1,
    CMD_STOP = 2,
    CMD_PAUSE = 3,
    CMD_RESUME = 4,
    CMD_RESET = 5
};

// Network states
enum network_state_t {
    STATE_IDLE = 0,
    STATE_INIT = 1,
    STATE_RUNNING = 2,
    STATE_PAUSED = 3,
    STATE_COMPLETE = 4,
    STATE_ERROR = 5
};

// Network configuration
struct network_config_t {
    network_mode_t mode;
    encoding_type_t encoding_type;
    decoding_type_t decoding_type;
    ap_uint<16> batch_size;
    ap_uint<16> input_scale;
    ap_uint<16> learning_rate;
    ap_uint<32> stdp_window;
    bool enable_monitoring;
};

// Network status
struct network_status_t {
    network_state_t state;
    ap_uint<32> cycles_run;
    ap_uint<32> batches_processed;
    ap_uint<16> errors;
    ap_uint<16> warnings;
};

// Function prototype
void network_controller(
    network_command_t command,
    network_config_t config,
    hls::stream<input_data_t> &input_data,
    hls::stream<output_data_t> &output_data,
    hls::stream<control_packet_t> &encoder_ctrl,
    hls::stream<control_packet_t> &learning_ctrl,
    hls::stream<control_packet_t> &decoder_ctrl,
    network_status_t &status
);

// Helper functions
void send_control_packet(
    hls::stream<control_packet_t> &stream,
    control_command_t cmd,
    ap_uint<16> param
);

#endif // NETWORK_CONTROLLER_H
