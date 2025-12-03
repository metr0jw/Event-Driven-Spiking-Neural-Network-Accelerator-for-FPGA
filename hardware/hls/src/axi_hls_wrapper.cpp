//-----------------------------------------------------------------------------
// Title         : HLS AXI Wrapper Implementation
// Project       : PYNQ-Z2 SNN Accelerator
// File          : axi_hls_wrapper.cpp
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : HLS-based AXI wrapper for Hybrid Verilog/HLS integration
//                 
// Architecture:
//   PS (ARM) <--AXI4-Lite/Stream--> HLS Wrapper <--Wire--> Verilog SNN Core
//
// This module handles:
//   1. AXI4-Lite register access (control, status, configuration)
//   2. AXI4-Stream to simple valid/ready handshake conversion
//   3. Buffering between AXI and Verilog domains (optional)
//-----------------------------------------------------------------------------

#include "axi_hls_wrapper.h"

//=============================================================================
// Main HLS AXI Wrapper Function
//=============================================================================
void axi_hls_wrapper(
    // AXI4-Lite Control Interface (Control registers)
    ap_uint<32> ctrl_reg,
    ap_uint<32> config_reg,
    ap_uint<16> leak_rate,
    ap_uint<16> threshold,
    ap_uint<16> refractory_period,
    ap_uint<32> &status_reg,
    ap_uint<32> &spike_count_out,
    ap_uint<32> &version_reg,
    
    // AXI4-Stream Spike Input (from PS)
    hls::stream<axis_spike_t> &s_axis_spikes,
    
    // AXI4-Stream Spike Output (to PS)
    hls::stream<axis_output_t> &m_axis_spikes,
    
    // Verilog Interface - Spike Input (to SNN core)
    ap_uint<1> &spike_in_valid,
    ap_uint<NEURON_ID_WIDTH> &spike_in_neuron_id,
    ap_int<WEIGHT_WIDTH> &spike_in_weight,
    ap_uint<1> spike_in_ready,
    
    // Verilog Interface - Spike Output (from SNN core)
    ap_uint<1> spike_out_valid,
    ap_uint<NEURON_ID_WIDTH> spike_out_neuron_id,
    ap_int<WEIGHT_WIDTH> spike_out_weight,
    ap_uint<1> &spike_out_ready,
    
    // Verilog Interface - Control signals
    ap_uint<1> &snn_enable,
    ap_uint<1> &snn_reset,
    ap_uint<1> &clear_counters,
    ap_uint<16> &leak_rate_out,
    ap_uint<16> &threshold_out,
    ap_uint<16> &refractory_out,
    
    // Verilog Interface - Status signals
    ap_uint<1> snn_ready,
    ap_uint<1> snn_busy,
    ap_uint<1> snn_error,
    ap_uint<32> snn_spike_count
) {
    //=========================================================================
    // HLS Interface Pragmas
    //=========================================================================
    // AXI4-Lite slave interface for control registers
    #pragma HLS INTERFACE s_axilite port=ctrl_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=config_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=leak_rate bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=threshold bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=refractory_period bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=status_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=spike_count_out bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=version_reg bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return bundle=ctrl
    
    // AXI4-Stream interfaces
    #pragma HLS INTERFACE axis port=s_axis_spikes
    #pragma HLS INTERFACE axis port=m_axis_spikes
    
    // Direct wire interfaces to Verilog (ap_none = simple wire)
    #pragma HLS INTERFACE ap_none port=spike_in_valid
    #pragma HLS INTERFACE ap_none port=spike_in_neuron_id
    #pragma HLS INTERFACE ap_none port=spike_in_weight
    #pragma HLS INTERFACE ap_none port=spike_in_ready
    #pragma HLS INTERFACE ap_none port=spike_out_valid
    #pragma HLS INTERFACE ap_none port=spike_out_neuron_id
    #pragma HLS INTERFACE ap_none port=spike_out_weight
    #pragma HLS INTERFACE ap_none port=spike_out_ready
    #pragma HLS INTERFACE ap_none port=snn_enable
    #pragma HLS INTERFACE ap_none port=snn_reset
    #pragma HLS INTERFACE ap_none port=clear_counters
    #pragma HLS INTERFACE ap_none port=leak_rate_out
    #pragma HLS INTERFACE ap_none port=threshold_out
    #pragma HLS INTERFACE ap_none port=refractory_out
    #pragma HLS INTERFACE ap_none port=snn_ready
    #pragma HLS INTERFACE ap_none port=snn_busy
    #pragma HLS INTERFACE ap_none port=snn_error
    #pragma HLS INTERFACE ap_none port=snn_spike_count
    
    // Pipeline the entire function for high throughput
    #pragma HLS PIPELINE II=1
    
    //=========================================================================
    // Internal State
    //=========================================================================
    static ap_uint<32> internal_spike_count = 0;
    static ap_uint<16> timestamp_counter = 0;
    
    // Small FIFOs for crossing domains (optional buffering)
    static ap_uint<32> input_fifo[4];
    static ap_uint<2> input_fifo_head = 0;
    static ap_uint<2> input_fifo_tail = 0;
    
    static ap_uint<32> output_fifo[4];
    static ap_uint<2> output_fifo_head = 0;
    static ap_uint<2> output_fifo_tail = 0;
    
    #pragma HLS ARRAY_PARTITION variable=input_fifo complete
    #pragma HLS ARRAY_PARTITION variable=output_fifo complete
    
    //=========================================================================
    // Control Signal Routing
    //=========================================================================
    // Extract control bits and route to Verilog
    snn_enable = ctrl_reg[0];
    snn_reset = ctrl_reg[1];
    clear_counters = ctrl_reg[2];
    
    // Route configuration to Verilog
    leak_rate_out = leak_rate;
    threshold_out = threshold;
    refractory_out = refractory_period;
    
    //=========================================================================
    // Status Register Assembly
    //=========================================================================
    // Combine status signals into status register
    ap_uint<32> status_combined = 0;
    status_combined[0] = snn_ready;
    status_combined[1] = snn_busy;
    status_combined[2] = snn_error;
    status_combined[3] = (input_fifo_head != input_fifo_tail);  // Input FIFO not empty
    status_combined[4] = (output_fifo_head != output_fifo_tail); // Output FIFO not empty
    
    status_reg = status_combined;
    spike_count_out = snn_spike_count;
    version_reg = VERSION_ID;
    
    //=========================================================================
    // Timestamp Counter
    //=========================================================================
    if (ctrl_reg[1]) {  // Reset
        timestamp_counter = 0;
        internal_spike_count = 0;
    } else if (ctrl_reg[0]) {  // Enable
        timestamp_counter++;
    }
    
    //=========================================================================
    // Input Path: AXI4-Stream -> Verilog SNN Core
    //=========================================================================
    // Default: no spike
    spike_in_valid = 0;
    spike_in_neuron_id = 0;
    spike_in_weight = 0;
    
    // Check if Verilog is ready and we have data
    if (spike_in_ready == 1) {
        // First, try to drain from internal FIFO
        if (input_fifo_head != input_fifo_tail) {
            ap_uint<32> fifo_data = input_fifo[input_fifo_head];
            spike_in_neuron_id = fifo_data(7, 0);
            spike_in_weight = fifo_data(23, 16);
            spike_in_valid = 1;
            input_fifo_head++;
        }
        // Then, try to read from AXI-Stream
        else if (!s_axis_spikes.empty()) {
            axis_spike_t axis_pkt = s_axis_spikes.read();
            spike_in_neuron_id = axis_pkt.data(7, 0);
            spike_in_weight = axis_pkt.data(23, 16);
            spike_in_valid = 1;
            internal_spike_count++;
        }
    } else {
        // Verilog not ready, buffer incoming data
        if (!s_axis_spikes.empty()) {
            // Check if FIFO has space
            ap_uint<2> next_tail = input_fifo_tail + 1;
            if (next_tail != input_fifo_head) {
                axis_spike_t axis_pkt = s_axis_spikes.read();
                input_fifo[input_fifo_tail] = axis_pkt.data;
                input_fifo_tail = next_tail;
                internal_spike_count++;
            }
            // If FIFO full, packet is dropped (overflow)
        }
    }
    
    //=========================================================================
    // Output Path: Verilog SNN Core -> AXI4-Stream
    //=========================================================================
    // Always ready to accept from Verilog (we'll buffer if needed)
    spike_out_ready = 1;
    
    // First, try to drain output FIFO to AXI-Stream
    if (output_fifo_head != output_fifo_tail) {
        if (!m_axis_spikes.full()) {
            axis_output_t out_pkt;
            ap_uint<32> fifo_data = output_fifo[output_fifo_head];
            out_pkt.data = fifo_data;
            out_pkt.keep = 0xF;
            out_pkt.strb = 0xF;
            out_pkt.last = 1;
            out_pkt.id = 0;
            out_pkt.dest = 0;
            out_pkt.user = 0;
            m_axis_spikes.write(out_pkt);
            output_fifo_head++;
        }
    }
    
    // Accept new data from Verilog
    if (spike_out_valid == 1) {
        // Pack output spike data
        ap_uint<32> out_data;
        out_data(7, 0) = spike_out_neuron_id;
        out_data(15, 8) = spike_out_weight;
        out_data(31, 16) = timestamp_counter;
        
        // Try to send directly to AXI-Stream
        if (!m_axis_spikes.full() && (output_fifo_head == output_fifo_tail)) {
            axis_output_t out_pkt;
            out_pkt.data = out_data;
            out_pkt.keep = 0xF;
            out_pkt.strb = 0xF;
            out_pkt.last = 1;
            out_pkt.id = 0;
            out_pkt.dest = 0;
            out_pkt.user = 0;
            m_axis_spikes.write(out_pkt);
        } else {
            // Buffer in output FIFO
            ap_uint<2> next_tail = output_fifo_tail + 1;
            if (next_tail != output_fifo_head) {
                output_fifo[output_fifo_tail] = out_data;
                output_fifo_tail = next_tail;
            }
            // If FIFO full, we might need to stall (spike_out_ready = 0)
            // For now, we drop - in production, add back-pressure
        }
    }
}

//=============================================================================
// Simplified Wrapper (Alternative Implementation)
//=============================================================================
void axi_hls_wrapper_simple(
    // AXI4-Lite registers (all in one bundle)
    ctrl_regs_t ctrl_regs,
    status_regs_t &status_regs,
    
    // AXI4-Stream interfaces
    hls::stream<axis_spike_t> &s_axis_spikes,
    hls::stream<axis_output_t> &m_axis_spikes,
    
    // Direct wire interface to Verilog
    ap_uint<32> &spike_to_verilog,
    ap_uint<1> &spike_valid_to_verilog,
    ap_uint<1> spike_ready_from_verilog,
    ap_uint<32> spike_from_verilog,
    ap_uint<1> spike_valid_from_verilog,
    ap_uint<1> &spike_ready_to_verilog,
    ap_uint<32> &ctrl_to_verilog,
    ap_uint<32> status_from_verilog
) {
    #pragma HLS INTERFACE s_axilite port=ctrl_regs bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=status_regs bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return bundle=ctrl
    
    #pragma HLS INTERFACE axis port=s_axis_spikes
    #pragma HLS INTERFACE axis port=m_axis_spikes
    
    #pragma HLS INTERFACE ap_none port=spike_to_verilog
    #pragma HLS INTERFACE ap_none port=spike_valid_to_verilog
    #pragma HLS INTERFACE ap_none port=spike_ready_from_verilog
    #pragma HLS INTERFACE ap_none port=spike_from_verilog
    #pragma HLS INTERFACE ap_none port=spike_valid_from_verilog
    #pragma HLS INTERFACE ap_none port=spike_ready_to_verilog
    #pragma HLS INTERFACE ap_none port=ctrl_to_verilog
    #pragma HLS INTERFACE ap_none port=status_from_verilog
    
    #pragma HLS PIPELINE II=1
    
    // Pass control signals directly
    ctrl_to_verilog = ctrl_regs.ctrl;
    
    // Build status
    status_regs.status = status_from_verilog;
    status_regs.version = VERSION_ID;
    
    // Input path: AXI-Stream -> Verilog
    spike_valid_to_verilog = 0;
    spike_to_verilog = 0;
    
    if (spike_ready_from_verilog && !s_axis_spikes.empty()) {
        axis_spike_t pkt = s_axis_spikes.read();
        spike_to_verilog = pkt.data;
        spike_valid_to_verilog = 1;
    }
    
    // Output path: Verilog -> AXI-Stream
    spike_ready_to_verilog = !m_axis_spikes.full();
    
    if (spike_valid_from_verilog && !m_axis_spikes.full()) {
        axis_output_t out_pkt;
        out_pkt.data = spike_from_verilog;
        out_pkt.keep = 0xF;
        out_pkt.strb = 0xF;
        out_pkt.last = 1;
        out_pkt.id = 0;
        out_pkt.dest = 0;
        out_pkt.user = 0;
        m_axis_spikes.write(out_pkt);
    }
}
