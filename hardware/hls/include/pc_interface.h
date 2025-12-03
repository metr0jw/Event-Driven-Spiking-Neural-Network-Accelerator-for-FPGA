//-----------------------------------------------------------------------------
// Title         : PC Communication Interface Header
// Project       : PYNQ-Z2 SNN Accelerator
// File          : pc_interface.h
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Header file for PC communication interface
//-----------------------------------------------------------------------------

#ifndef PC_INTERFACE_H
#define PC_INTERFACE_H

#include <ap_int.h>
#include <hls_stream.h>
#include "snn_types.h"

// Constants
#define SPIKE_BUFFER_SIZE 256
#define CONFIG_MEMORY_SIZE 1024
#define MAX_NEURONS 4096
#define MAX_TIMESTAMP 0xFFFFFFFF
#define MIN_WEIGHT 0
#define MAX_WEIGHT 255

// Interface version and features
#define INTERFACE_VERSION 0x01000000  // Version 1.0.0.0
#define INTERFACE_FEATURES 0x0000001F  // Features bitmap

// Packet types
#define PACKET_TYPE_SPIKE 0x01
#define PACKET_TYPE_CONFIG 0x02
#define PACKET_TYPE_WEIGHT_UPDATE 0x03
#define PACKET_TYPE_STATUS 0x04

// Status register bits
#define STATUS_ENABLED (1 << 0)
#define STATUS_DISABLED (1 << 1)
#define STATUS_ERROR (1 << 2)
#define STATUS_INPUT_BUFFER_FULL (1 << 3)
#define STATUS_OUTPUT_BUFFER_FULL (1 << 4)

// Configuration addresses
#define CONFIG_ADDR_NEURON_THRESHOLD 0x0000
#define CONFIG_ADDR_LEARNING_RATE 0x0001
#define CONFIG_ADDR_STDP_WINDOW 0x0002
#define CONFIG_ADDR_VERSION 0x1000
#define CONFIG_ADDR_FEATURES 0x1001
#define CONFIG_ADDR_MAX_NEURONS 0x1002

// Processing modes
#define PROCESS_MODE_PASSTHROUGH 0x00
#define PROCESS_MODE_FILTER 0x01
#define PROCESS_MODE_TRANSFORM 0x02

// Processing parameters
#define FILTER_THRESHOLD 128
#define TRANSFORM_SCALE 2

// Ethernet types
#define ETH_TYPE_SNN 0x8888
#define ETH_TYPE_SNN_ACK 0x8889

// Spike packet structure for PC communication
typedef struct {
    ap_uint<16> neuron_id;
    ap_uint<16> timestamp;
    ap_uint<8> weight;
    ap_uint<8> packet_type;
    ap_uint<16> sequence_num;
    ap_uint<32> payload;
} spike_packet_t;

// Spike event structure for SNN core
typedef struct {
    ap_uint<16> neuron_id;
    ap_uint<32> timestamp;
    ap_int<8> weight;
    bool valid;
} spike_event_t;

// Weight update structure
typedef struct {
    ap_uint<16> pre_neuron_id;
    ap_uint<16> post_neuron_id;
    ap_int<8> new_weight;
    ap_int<8> delta_weight;
    bool valid;
} weight_update_t;

// Spike buffer entry
typedef struct {
    spike_event_t spike;
    bool valid;
} spike_buffer_t;

// PC packet buffer entry
typedef struct {
    spike_packet_t packet;
    bool valid;
} pc_buffer_t;

// Ethernet packet structure
typedef struct {
    ap_uint<48> dest_mac;
    ap_uint<48> src_mac;
    ap_uint<16> eth_type;
    ap_uint<1500*8> payload;  // Max Ethernet payload
    ap_uint<32> crc;
} ethernet_packet_t;

// Function prototypes
void pc_interface(
    bool enable,
    bool reset,
    ap_uint<32> control_reg,
    ap_uint<32> &status_reg,
    hls::stream<spike_packet_t> &pc_spike_in,
    bool &pc_spike_in_ready,
    hls::stream<spike_packet_t> &pc_spike_out,
    bool pc_spike_out_ready,
    hls::stream<spike_event_t> &snn_spike_in,
    hls::stream<spike_event_t> &snn_spike_out,
    hls::stream<weight_update_t> &weight_updates,
    ap_uint<32> config_addr,
    ap_uint<32> config_data,
    bool config_write,
    ap_uint<32> &config_read_data,
    ap_uint<32> &input_spike_count,
    ap_uint<32> &output_spike_count,
    ap_uint<32> &cycle_count
);

bool validate_spike_packet(const spike_packet_t &packet);
bool is_buffer_full(ap_uint<8> head, ap_uint<8> tail);
bool is_buffer_empty(ap_uint<8> head, ap_uint<8> tail);
void handle_config_write(ap_uint<32> addr, ap_uint<32> data);
ap_uint<32> handle_config_read(ap_uint<32> addr);

void process_ethernet_packet(
    hls::stream<ethernet_packet_t> &eth_in,
    hls::stream<ethernet_packet_t> &eth_out,
    hls::stream<spike_packet_t> &spike_out
);

void spike_stream_processor(
    hls::stream<spike_event_t> &input_stream,
    hls::stream<spike_event_t> &output_stream,
    ap_uint<32> processing_mode,
    ap_uint<32> &processed_count
);

#endif // PC_INTERFACE_H
