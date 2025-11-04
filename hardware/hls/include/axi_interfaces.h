//-----------------------------------------------------------------------------
// Title         : AXI Interface Definitions
// Project       : PYNQ-Z2 SNN Accelerator
// File          : axi_interfaces.h
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : AXI protocol interfaces for HLS
//-----------------------------------------------------------------------------

#ifndef AXI_INTERFACES_H
#define AXI_INTERFACES_H

#include <ap_axi_sdata.h>
#include "snn_types.h"

// AXI-Stream packet definitions
typedef ap_axiu<32,1,1,1> axi_spike_packet_t;
typedef ap_axiu<64,2,1,1> axi_data_packet_t;
typedef ap_axiu<32,4,1,1> axi_control_packet_t;

// Convert spike event to AXI-Stream packet
inline axi_spike_packet_t spike_to_axi(spike_event_t spike) {
    #pragma HLS INLINE
    axi_spike_packet_t packet;
    packet.data = (spike.neuron_id) | 
                  (spike.weight << 8) | 
                  ((spike.timestamp & 0xFFFF) << 16);
    packet.keep = 0xF;
    packet.strb = 0xF;
    packet.last = 1;
    packet.id = 0;
    packet.dest = 0;
    packet.user = 1;  // Spike packet type
    return packet;
}

// Convert AXI-Stream packet to spike event
inline spike_event_t axi_to_spike(axi_spike_packet_t packet) {
    #pragma HLS INLINE
    spike_event_t spike;
    spike.neuron_id = packet.data & 0xFF;
    spike.weight = (packet.data >> 8) & 0xFF;
    spike.timestamp = (packet.data >> 16) & 0xFFFF;
    return spike;
}

// Memory-mapped register interface
struct snn_registers_t {
    ap_uint<32> control;      // 0x00: Control register
    ap_uint<32> status;       // 0x04: Status register
    ap_uint<32> config;       // 0x08: Configuration
    ap_uint<32> spike_count;  // 0x0C: Spike counter
    ap_uint<32> error_count;  // 0x10: Error counter
    ap_uint<32> version;      // 0x14: Version ID
    ap_uint<32> reserved[10]; // 0x18-0x3C: Reserved
};

// Control register bits
#define CTRL_ENABLE_BIT    0
#define CTRL_RESET_BIT     1
#define CTRL_CLEAR_BIT     2
#define CTRL_MODE_BIT      4  // 2 bits for mode
#define CTRL_IRQ_EN_BIT    8

// Status register bits
#define STAT_READY_BIT     0
#define STAT_BUSY_BIT      1
#define STAT_ERROR_BIT     2
#define STAT_DONE_BIT      3
#define STAT_OVERFLOW_BIT  4

#endif // AXI_INTERFACES_H
