//-----------------------------------------------------------------------------
// Title         : Network Controller
// Project       : PYNQ-Z2 SNN Accelerator
// File          : network_controller.cpp
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Description   : High-level network control and orchestration
//-----------------------------------------------------------------------------

#include "network_controller.h"

void network_controller(
    // Control interface
    network_command_t command,
    network_config_t config,
    
    // Data streams
    hls::stream<input_data_t> &input_data,
    hls::stream<output_data_t> &output_data,
    
    // Control streams to other modules
    hls::stream<control_packet_t> &encoder_ctrl,
    hls::stream<control_packet_t> &learning_ctrl,
    hls::stream<control_packet_t> &decoder_ctrl,
    
    // Status
    network_status_t &status
) {
    #pragma HLS INTERFACE s_axilite port=command
    #pragma HLS INTERFACE s_axilite port=config
    #pragma HLS INTERFACE s_axilite port=status
    #pragma HLS INTERFACE axis port=input_data
    #pragma HLS INTERFACE axis port=output_data
    #pragma HLS INTERFACE axis port=encoder_ctrl
    #pragma HLS INTERFACE axis port=learning_ctrl
    #pragma HLS INTERFACE axis port=decoder_ctrl
    #pragma HLS INTERFACE s_axilite port=return
    
    static network_state_t state = STATE_IDLE;
    static ap_uint<32> cycle_counter = 0;
    static ap_uint<32> batch_counter = 0;
    
    // Update cycle counter
    cycle_counter++;
    
    // State machine
    switch (state) {
        case STATE_IDLE:
            if (command == CMD_START) {
                state = STATE_INIT;
                batch_counter = 0;
            }
            break;
            
        case STATE_INIT:
            // Send initialization commands to all modules
            send_control_packet(encoder_ctrl, CTRL_RESET, 0);
            send_control_packet(learning_ctrl, CTRL_RESET, 0);
            send_control_packet(decoder_ctrl, CTRL_RESET, 0);
            
            state = STATE_RUNNING;
            break;
            
        case STATE_RUNNING:
            // Process input data
            if (!input_data.empty()) {
                input_data_t data = input_data.read();
                
                // Configure encoder for this batch
                control_packet_t enc_config;
                enc_config.command = CTRL_CONFIGURE;
                enc_config.param1 = config.encoding_type;
                enc_config.param2 = config.input_scale;
                encoder_ctrl.write(enc_config);
                
                // Enable learning if in training mode
                if (config.mode == MODE_TRAINING) {
                    control_packet_t learn_config;
                    learn_config.command = CTRL_ENABLE;
                    learn_config.param1 = config.learning_rate;
                    learn_config.param2 = config.stdp_window;
                    learning_ctrl.write(learn_config);
                }
                
                batch_counter++;
            }
            
            // Check for completion
            if (batch_counter >= config.batch_size) {
                state = STATE_COMPLETE;
            }
            
            // Handle pause/stop commands
            if (command == CMD_PAUSE) {
                state = STATE_PAUSED;
            } else if (command == CMD_STOP) {
                state = STATE_IDLE;
            }
            break;
            
        case STATE_PAUSED:
            if (command == CMD_RESUME) {
                state = STATE_RUNNING;
            } else if (command == CMD_STOP) {
                state = STATE_IDLE;
            }
            break;
            
        case STATE_COMPLETE:
            // Send completion signal
            send_control_packet(decoder_ctrl, CTRL_FLUSH, 0);
            state = STATE_IDLE;
            break;
    }
    
    // Update status
    status.state = state;
    status.cycles_run = cycle_counter;
    status.batches_processed = batch_counter;
    status.errors = 0;
}

void send_control_packet(
    hls::stream<control_packet_t> &stream,
    control_command_t cmd,
    ap_uint<16> param
) {
    #pragma HLS INLINE
    
    control_packet_t packet;
    packet.command = cmd;
    packet.param1 = param;
    packet.param2 = 0;
    packet.timestamp = 0; // Would use global timestamp in real implementation
    
    stream.write(packet);
}
