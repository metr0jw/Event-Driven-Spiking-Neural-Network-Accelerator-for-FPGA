//-----------------------------------------------------------------------------
// Title         : HLS AXI Wrapper Testbench
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_axi_hls_wrapper.cpp
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : C++ testbench for HLS AXI wrapper simulation
//-----------------------------------------------------------------------------

#include <iostream>
#include <iomanip>
#include "axi_hls_wrapper.h"

// Test configuration
#define NUM_TEST_SPIKES 10
#define VERBOSE 1

// Error counter
int error_count = 0;

// Helper function to create AXI-Stream spike packet
axis_spike_t create_spike_packet(ap_uint<8> neuron_id, ap_int<8> weight) {
    axis_spike_t pkt;
    pkt.data = 0;
    pkt.data(7, 0) = neuron_id;
    pkt.data(23, 16) = weight;
    pkt.keep = 0xF;
    pkt.strb = 0xF;
    pkt.last = 1;
    pkt.id = 0;
    pkt.dest = 0;
    pkt.user = 0;
    return pkt;
}

// Helper function to print test result
void print_result(const char* test_name, bool passed) {
    std::cout << "  " << (passed ? "PASS" : "FAIL") << ": " << test_name << std::endl;
    if (!passed) error_count++;
}

//=============================================================================
// Test 1: Control Register Routing
//=============================================================================
void test_control_registers() {
    std::cout << "\n=== Test 1: Control Register Routing ===" << std::endl;
    
    // Input streams
    hls::stream<axis_spike_t> s_axis_spikes;
    hls::stream<axis_output_t> m_axis_spikes;
    
    // AXI-Lite registers
    ap_uint<32> ctrl_reg = 0x05;  // Enable + Clear counters
    ap_uint<32> config_reg = 0x100;
    ap_uint<16> leak_rate = 10;
    ap_uint<16> threshold = 1000;
    ap_uint<16> refractory = 20;
    ap_uint<32> status_reg;
    ap_uint<32> spike_count_out;
    ap_uint<32> version_reg;
    
    // Verilog interface signals
    ap_uint<1> spike_in_valid;
    ap_uint<8> spike_in_neuron_id;
    ap_int<8> spike_in_weight;
    ap_uint<1> spike_in_ready = 1;
    
    ap_uint<1> spike_out_valid = 0;
    ap_uint<8> spike_out_neuron_id = 0;
    ap_int<8> spike_out_weight = 0;
    ap_uint<1> spike_out_ready;
    
    ap_uint<1> snn_enable;
    ap_uint<1> snn_reset;
    ap_uint<1> clear_counters;
    ap_uint<16> leak_rate_out;
    ap_uint<16> threshold_out;
    ap_uint<16> refractory_out;
    
    ap_uint<1> snn_ready = 1;
    ap_uint<1> snn_busy = 0;
    ap_uint<1> snn_error = 0;
    ap_uint<32> snn_spike_count = 0;
    
    // Run wrapper
    axi_hls_wrapper(
        ctrl_reg, config_reg, leak_rate, threshold, refractory,
        status_reg, spike_count_out, version_reg,
        s_axis_spikes, m_axis_spikes,
        spike_in_valid, spike_in_neuron_id, spike_in_weight, spike_in_ready,
        spike_out_valid, spike_out_neuron_id, spike_out_weight, spike_out_ready,
        snn_enable, snn_reset, clear_counters,
        leak_rate_out, threshold_out, refractory_out,
        snn_ready, snn_busy, snn_error, snn_spike_count
    );
    
    // Verify control signals
    print_result("SNN Enable", snn_enable == 1);
    print_result("SNN Reset", snn_reset == 0);
    print_result("Clear Counters", clear_counters == 1);
    print_result("Leak Rate", leak_rate_out == 10);
    print_result("Threshold", threshold_out == 1000);
    print_result("Refractory", refractory_out == 20);
    print_result("Version", version_reg == VERSION_ID);
    
    if (VERBOSE) {
        std::cout << "  Control outputs: enable=" << snn_enable 
                  << " reset=" << snn_reset 
                  << " clear=" << clear_counters << std::endl;
        std::cout << "  Config outputs: leak=" << leak_rate_out 
                  << " thresh=" << threshold_out 
                  << " refrac=" << refractory_out << std::endl;
    }
}

//=============================================================================
// Test 2: Input Spike Path (AXI-Stream -> Verilog)
//=============================================================================
void test_input_spike_path() {
    std::cout << "\n=== Test 2: Input Spike Path ===" << std::endl;
    
    // Input streams
    hls::stream<axis_spike_t> s_axis_spikes;
    hls::stream<axis_output_t> m_axis_spikes;
    
    // AXI-Lite registers
    ap_uint<32> ctrl_reg = 0x01;  // Enable
    ap_uint<32> config_reg = 0;
    ap_uint<16> leak_rate = 10;
    ap_uint<16> threshold = 1000;
    ap_uint<16> refractory = 20;
    ap_uint<32> status_reg;
    ap_uint<32> spike_count_out;
    ap_uint<32> version_reg;
    
    // Verilog interface signals
    ap_uint<1> spike_in_valid;
    ap_uint<8> spike_in_neuron_id;
    ap_int<8> spike_in_weight;
    ap_uint<1> spike_in_ready = 1;  // Verilog ready
    
    ap_uint<1> spike_out_valid = 0;
    ap_uint<8> spike_out_neuron_id = 0;
    ap_int<8> spike_out_weight = 0;
    ap_uint<1> spike_out_ready;
    
    ap_uint<1> snn_enable, snn_reset, clear_counters;
    ap_uint<16> leak_rate_out, threshold_out, refractory_out;
    ap_uint<1> snn_ready = 1, snn_busy = 0, snn_error = 0;
    ap_uint<32> snn_spike_count = 0;
    
    // Send test spikes
    int passed_count = 0;
    for (int i = 0; i < NUM_TEST_SPIKES; i++) {
        // Create and send spike
        axis_spike_t pkt = create_spike_packet(i, 100 + i);
        s_axis_spikes.write(pkt);
        
        // Run wrapper
        axi_hls_wrapper(
            ctrl_reg, config_reg, leak_rate, threshold, refractory,
            status_reg, spike_count_out, version_reg,
            s_axis_spikes, m_axis_spikes,
            spike_in_valid, spike_in_neuron_id, spike_in_weight, spike_in_ready,
            spike_out_valid, spike_out_neuron_id, spike_out_weight, spike_out_ready,
            snn_enable, snn_reset, clear_counters,
            leak_rate_out, threshold_out, refractory_out,
            snn_ready, snn_busy, snn_error, snn_spike_count
        );
        
        // Verify output
        if (spike_in_valid && 
            spike_in_neuron_id == i && 
            spike_in_weight == (100 + i)) {
            passed_count++;
        }
        
        if (VERBOSE) {
            std::cout << "  Spike " << i << ": valid=" << spike_in_valid
                      << " neuron=" << spike_in_neuron_id
                      << " weight=" << spike_in_weight << std::endl;
        }
    }
    
    print_result("Input Spikes Passed", passed_count == NUM_TEST_SPIKES);
}

//=============================================================================
// Test 3: Output Spike Path (Verilog -> AXI-Stream)
//=============================================================================
void test_output_spike_path() {
    std::cout << "\n=== Test 3: Output Spike Path ===" << std::endl;
    
    // Input streams
    hls::stream<axis_spike_t> s_axis_spikes;
    hls::stream<axis_output_t> m_axis_spikes;
    
    // AXI-Lite registers
    ap_uint<32> ctrl_reg = 0x01;
    ap_uint<32> config_reg = 0;
    ap_uint<16> leak_rate = 10;
    ap_uint<16> threshold = 1000;
    ap_uint<16> refractory = 20;
    ap_uint<32> status_reg;
    ap_uint<32> spike_count_out;
    ap_uint<32> version_reg;
    
    // Verilog interface signals
    ap_uint<1> spike_in_valid;
    ap_uint<8> spike_in_neuron_id;
    ap_int<8> spike_in_weight;
    ap_uint<1> spike_in_ready = 1;
    
    ap_uint<1> spike_out_valid;
    ap_uint<8> spike_out_neuron_id;
    ap_int<8> spike_out_weight;
    ap_uint<1> spike_out_ready;
    
    ap_uint<1> snn_enable, snn_reset, clear_counters;
    ap_uint<16> leak_rate_out, threshold_out, refractory_out;
    ap_uint<1> snn_ready = 1, snn_busy = 0, snn_error = 0;
    ap_uint<32> snn_spike_count = 0;
    
    // Simulate output spikes from Verilog
    int received_count = 0;
    for (int i = 0; i < NUM_TEST_SPIKES; i++) {
        // Set Verilog outputs
        spike_out_valid = 1;
        spike_out_neuron_id = i + 10;
        spike_out_weight = 50 + i;
        
        // Run wrapper
        axi_hls_wrapper(
            ctrl_reg, config_reg, leak_rate, threshold, refractory,
            status_reg, spike_count_out, version_reg,
            s_axis_spikes, m_axis_spikes,
            spike_in_valid, spike_in_neuron_id, spike_in_weight, spike_in_ready,
            spike_out_valid, spike_out_neuron_id, spike_out_weight, spike_out_ready,
            snn_enable, snn_reset, clear_counters,
            leak_rate_out, threshold_out, refractory_out,
            snn_ready, snn_busy, snn_error, snn_spike_count
        );
        
        // Check if spike was accepted
        if (spike_out_ready == 1) {
            received_count++;
        }
    }
    
    // Read output packets
    int output_count = 0;
    while (!m_axis_spikes.empty()) {
        axis_output_t out_pkt = m_axis_spikes.read();
        ap_uint<8> neuron = out_pkt.data(7, 0);
        ap_uint<8> weight = out_pkt.data(15, 8);
        ap_uint<16> timestamp = out_pkt.data(31, 16);
        
        if (VERBOSE) {
            std::cout << "  Output " << output_count 
                      << ": neuron=" << neuron
                      << " weight=" << weight
                      << " timestamp=" << timestamp << std::endl;
        }
        output_count++;
    }
    
    print_result("Output Spikes Received", output_count > 0);
    print_result("Spike Ready Signal", received_count == NUM_TEST_SPIKES);
}

//=============================================================================
// Test 4: Buffering (Verilog not ready)
//=============================================================================
void test_input_buffering() {
    std::cout << "\n=== Test 4: Input Buffering ===" << std::endl;
    
    // Input streams
    hls::stream<axis_spike_t> s_axis_spikes;
    hls::stream<axis_output_t> m_axis_spikes;
    
    // AXI-Lite registers
    ap_uint<32> ctrl_reg = 0x01;
    ap_uint<32> config_reg = 0;
    ap_uint<16> leak_rate = 10;
    ap_uint<16> threshold = 1000;
    ap_uint<16> refractory = 20;
    ap_uint<32> status_reg;
    ap_uint<32> spike_count_out;
    ap_uint<32> version_reg;
    
    // Verilog interface signals
    ap_uint<1> spike_in_valid;
    ap_uint<8> spike_in_neuron_id;
    ap_int<8> spike_in_weight;
    ap_uint<1> spike_in_ready = 0;  // Verilog NOT ready
    
    ap_uint<1> spike_out_valid = 0;
    ap_uint<8> spike_out_neuron_id = 0;
    ap_int<8> spike_out_weight = 0;
    ap_uint<1> spike_out_ready;
    
    ap_uint<1> snn_enable, snn_reset, clear_counters;
    ap_uint<16> leak_rate_out, threshold_out, refractory_out;
    ap_uint<1> snn_ready = 1, snn_busy = 1, snn_error = 0;
    ap_uint<32> snn_spike_count = 0;
    
    // Send spikes while Verilog is not ready
    for (int i = 0; i < 3; i++) {
        axis_spike_t pkt = create_spike_packet(i, 100);
        s_axis_spikes.write(pkt);
        
        axi_hls_wrapper(
            ctrl_reg, config_reg, leak_rate, threshold, refractory,
            status_reg, spike_count_out, version_reg,
            s_axis_spikes, m_axis_spikes,
            spike_in_valid, spike_in_neuron_id, spike_in_weight, spike_in_ready,
            spike_out_valid, spike_out_neuron_id, spike_out_weight, spike_out_ready,
            snn_enable, snn_reset, clear_counters,
            leak_rate_out, threshold_out, refractory_out,
            snn_ready, snn_busy, snn_error, snn_spike_count
        );
    }
    
    // Now make Verilog ready and drain buffered spikes
    spike_in_ready = 1;
    int drained_count = 0;
    
    for (int i = 0; i < 5; i++) {
        axi_hls_wrapper(
            ctrl_reg, config_reg, leak_rate, threshold, refractory,
            status_reg, spike_count_out, version_reg,
            s_axis_spikes, m_axis_spikes,
            spike_in_valid, spike_in_neuron_id, spike_in_weight, spike_in_ready,
            spike_out_valid, spike_out_neuron_id, spike_out_weight, spike_out_ready,
            snn_enable, snn_reset, clear_counters,
            leak_rate_out, threshold_out, refractory_out,
            snn_ready, snn_busy, snn_error, snn_spike_count
        );
        
        if (spike_in_valid) {
            drained_count++;
            if (VERBOSE) {
                std::cout << "  Drained spike: neuron=" << spike_in_neuron_id
                          << " weight=" << spike_in_weight << std::endl;
            }
        }
    }
    
    print_result("Buffered spikes drained", drained_count >= 3);
}

//=============================================================================
// Test 5: Reset Functionality
//=============================================================================
void test_reset() {
    std::cout << "\n=== Test 5: Reset Functionality ===" << std::endl;
    
    // Input streams
    hls::stream<axis_spike_t> s_axis_spikes;
    hls::stream<axis_output_t> m_axis_spikes;
    
    // AXI-Lite registers - with reset
    ap_uint<32> ctrl_reg = 0x02;  // Reset bit
    ap_uint<32> config_reg = 0;
    ap_uint<16> leak_rate = 10;
    ap_uint<16> threshold = 1000;
    ap_uint<16> refractory = 20;
    ap_uint<32> status_reg;
    ap_uint<32> spike_count_out;
    ap_uint<32> version_reg;
    
    // Verilog interface signals
    ap_uint<1> spike_in_valid;
    ap_uint<8> spike_in_neuron_id;
    ap_int<8> spike_in_weight;
    ap_uint<1> spike_in_ready = 1;
    
    ap_uint<1> spike_out_valid = 0;
    ap_uint<8> spike_out_neuron_id = 0;
    ap_int<8> spike_out_weight = 0;
    ap_uint<1> spike_out_ready;
    
    ap_uint<1> snn_enable, snn_reset, clear_counters;
    ap_uint<16> leak_rate_out, threshold_out, refractory_out;
    ap_uint<1> snn_ready = 1, snn_busy = 0, snn_error = 0;
    ap_uint<32> snn_spike_count = 0;
    
    // Run with reset
    axi_hls_wrapper(
        ctrl_reg, config_reg, leak_rate, threshold, refractory,
        status_reg, spike_count_out, version_reg,
        s_axis_spikes, m_axis_spikes,
        spike_in_valid, spike_in_neuron_id, spike_in_weight, spike_in_ready,
        spike_out_valid, spike_out_neuron_id, spike_out_weight, spike_out_ready,
        snn_enable, snn_reset, clear_counters,
        leak_rate_out, threshold_out, refractory_out,
        snn_ready, snn_busy, snn_error, snn_spike_count
    );
    
    print_result("Reset signal", snn_reset == 1);
    print_result("Enable during reset", snn_enable == 0);
}

//=============================================================================
// Main
//=============================================================================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "HLS AXI Wrapper Testbench" << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_control_registers();
    test_input_spike_path();
    test_output_spike_path();
    test_input_buffering();
    test_reset();
    
    std::cout << "\n========================================" << std::endl;
    if (error_count == 0) {
        std::cout << "All tests PASSED!" << std::endl;
    } else {
        std::cout << "Tests completed with " << error_count << " errors" << std::endl;
    }
    std::cout << "========================================" << std::endl;
    
    return error_count;
}
