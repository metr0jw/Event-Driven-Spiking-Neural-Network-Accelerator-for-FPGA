//-----------------------------------------------------------------------------
// Title         : SNN Top-Level HLS Testbench
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_snn_top_hls.cpp
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : C++ testbench for SNN top-level with on-chip learning
//-----------------------------------------------------------------------------

#include <iostream>
#include <iomanip>
#include <cmath>
#include "snn_top_hls.h"

// Test configuration
#define NUM_TEST_SPIKES 20
#define NUM_TIMESTEPS 100
#define VERBOSE 1

// Error counter
int error_count = 0;

//=============================================================================
// Helper Functions
//=============================================================================
void print_result(const char* test_name, bool passed) {
    std::cout << "  " << (passed ? "PASS" : "FAIL") << ": " << test_name << std::endl;
    if (!passed) error_count++;
}

axis_spike_t create_spike(uint8_t neuron_id, int8_t weight, uint16_t timestamp) {
    axis_spike_t pkt;
    pkt.data = 0;
    pkt.data(7, 0) = neuron_id;
    pkt.data(15, 8) = weight;
    pkt.data(31, 16) = timestamp;
    pkt.keep = 0xF;
    pkt.strb = 0xF;
    pkt.last = 1;
    pkt.id = 0;
    pkt.dest = 0;
    pkt.user = 0;
    return pkt;
}

learning_params_t get_default_params() {
    learning_params_t params;
    params.a_plus = 0.1;
    params.a_minus = 0.12;
    params.tau_plus = 20;
    params.tau_minus = 20;
    params.stdp_window = 50;
    params.learning_rate = 0.01;
    params.rstdp_enable = false;
    params.trace_decay = 0.99;
    params.reward_scale = 1.0;
    return params;
}

encoder_config_t get_default_encoder_config() {
    encoder_config_t cfg;
    cfg.encoding_type = RATE_CODING;
    cfg.num_channels = 0;
    cfg.time_window = 32;
    cfg.rate_scale = 128;
    cfg.phase_scale = 1;
    cfg.phase_threshold = 1024;
    cfg.default_weight = 1;
    return cfg;
}

//=============================================================================
// Test 1: Basic Control Register Operations
//=============================================================================
void test_control_registers() {
    std::cout << "\n=== Test 1: Control Registers ===" << std::endl;
    
    // Streams
    hls::stream<axis_spike_t> s_axis_spikes;
    hls::stream<input_data_t> s_axis_data;
    hls::stream<axis_spike_t> m_axis_spikes;
    hls::stream<axis_weight_t> m_axis_weights;
    
    // Control/Status
    ap_uint<32> ctrl_reg = 0;
    ap_uint<32> config_reg = 0;
    ap_uint<32> mode_reg = 0;
    ap_uint<32> time_steps_reg = 1;
    encoder_config_t encoder_cfg = get_default_encoder_config();
    learning_params_t params = get_default_params();
    ap_uint<32> status_reg, spike_count_reg, weight_sum_reg, version_reg;
    ap_int<8> reward_signal = 0;
    
    // Verilog interface (simulated)
    ap_uint<1> spike_in_valid, spike_out_ready, snn_enable, snn_reset;
    ap_uint<8> spike_in_neuron_id;
    ap_int<8> spike_in_weight;
    ap_uint<16> threshold_out, leak_rate_out;
    
    ap_uint<1> spike_in_ready = 1;
    ap_uint<1> spike_out_valid = 0;
    ap_uint<8> spike_out_neuron_id = 0;
    ap_int<8> spike_out_weight = 0;
    ap_uint<1> snn_ready = 1;
    ap_uint<1> snn_busy = 0;
    
    // Test reset
    ctrl_reg = 0x02;  // Reset bit
    config_reg = (100 << 16) | 51;  // leak_rate=100, threshold=51
    
        snn_top_hls(ctrl_reg, config_reg, mode_reg, time_steps_reg, params, encoder_cfg,
                    status_reg, spike_count_reg,
                    weight_sum_reg, version_reg, s_axis_spikes, s_axis_data, m_axis_spikes,
                    m_axis_weights, reward_signal,
                spike_in_valid, spike_in_neuron_id, spike_in_weight, spike_in_ready,
                spike_out_valid, spike_out_neuron_id, spike_out_weight, spike_out_ready,
                snn_enable, snn_reset, threshold_out, leak_rate_out,
                snn_ready, snn_busy);
    
    print_result("Reset signal routed", snn_reset == 1);
    print_result("Version register set", version_reg == VERSION_ID);
    print_result("Threshold routed", threshold_out == 51);
    print_result("Leak rate routed", leak_rate_out == 100);
    
    // Test enable
    ctrl_reg = 0x01;  // Enable bit
    
        snn_top_hls(ctrl_reg, config_reg, mode_reg, time_steps_reg, params, encoder_cfg,
                    status_reg, spike_count_reg,
                    weight_sum_reg, version_reg, s_axis_spikes, s_axis_data, m_axis_spikes,
                    m_axis_weights, reward_signal,
                spike_in_valid, spike_in_neuron_id, spike_in_weight, spike_in_ready,
                spike_out_valid, spike_out_neuron_id, spike_out_weight, spike_out_ready,
                snn_enable, snn_reset, threshold_out, leak_rate_out,
                snn_ready, snn_busy);
    
    print_result("Enable signal routed", snn_enable == 1);
}

//=============================================================================
// Test 2: Spike Input Path
//=============================================================================
void test_spike_input() {
    std::cout << "\n=== Test 2: Spike Input Path ===" << std::endl;
    
    hls::stream<axis_spike_t> s_axis_spikes;
    hls::stream<input_data_t> s_axis_data;
    hls::stream<axis_spike_t> m_axis_spikes;
    hls::stream<axis_weight_t> m_axis_weights;
    
    ap_uint<32> ctrl_reg = 0x01;  // Enable
    ap_uint<32> config_reg = (100 << 16) | 51;
    ap_uint<32> mode_reg = 0;
    ap_uint<32> time_steps_reg = 1;
    encoder_config_t encoder_cfg = get_default_encoder_config();
    learning_params_t params = get_default_params();
    ap_uint<32> status_reg, spike_count_reg, weight_sum_reg, version_reg;
    ap_int<8> reward_signal = 0;
    
    ap_uint<1> spike_in_valid, spike_out_ready, snn_enable, snn_reset;
    ap_uint<8> spike_in_neuron_id;
    ap_int<8> spike_in_weight;
    ap_uint<16> threshold_out, leak_rate_out;
    
    ap_uint<1> spike_in_ready = 1;
    ap_uint<1> spike_out_valid = 0;
    ap_uint<8> spike_out_neuron_id = 0;
    ap_int<8> spike_out_weight = 0;
    ap_uint<1> snn_ready = 1;
    ap_uint<1> snn_busy = 0;
    
    // Send spikes
    for (int i = 0; i < 5; i++) {
        s_axis_spikes.write(create_spike(i, 10 + i, 100 + i));
    }
    
    int spikes_received = 0;
    for (int t = 0; t < 10; t++) {
        snn_top_hls(ctrl_reg, config_reg, mode_reg, time_steps_reg, params, encoder_cfg,
                status_reg, spike_count_reg,
                weight_sum_reg, version_reg, s_axis_spikes, s_axis_data, m_axis_spikes,
                m_axis_weights, reward_signal,
                    spike_in_valid, spike_in_neuron_id, spike_in_weight, spike_in_ready,
                    spike_out_valid, spike_out_neuron_id, spike_out_weight, spike_out_ready,
                    snn_enable, snn_reset, threshold_out, leak_rate_out,
                    snn_ready, snn_busy);
        
        if (spike_in_valid) {
            if (VERBOSE) {
                std::cout << "  Spike: neuron=" << (int)spike_in_neuron_id 
                          << " weight=" << (int)spike_in_weight << std::endl;
            }
            spikes_received++;
        }
    }
    
    print_result("All input spikes forwarded", spikes_received == 5);
    print_result("Spike counter updated", spike_count_reg == 5);
}

//=============================================================================
// Test 3: Spike Output Path
//=============================================================================
void test_spike_output() {
    std::cout << "\n=== Test 3: Spike Output Path ===" << std::endl;
    
    hls::stream<axis_spike_t> s_axis_spikes;
    hls::stream<input_data_t> s_axis_data;
    hls::stream<axis_spike_t> m_axis_spikes;
    hls::stream<axis_weight_t> m_axis_weights;
    
    ap_uint<32> ctrl_reg = 0x01;
    ap_uint<32> config_reg = (100 << 16) | 51;
    ap_uint<32> mode_reg = 0;
    ap_uint<32> time_steps_reg = 1;
    encoder_config_t encoder_cfg = get_default_encoder_config();
    learning_params_t params = get_default_params();
    ap_uint<32> status_reg, spike_count_reg, weight_sum_reg, version_reg;
    ap_int<8> reward_signal = 0;
    
    ap_uint<1> spike_in_valid, spike_out_ready, snn_enable, snn_reset;
    ap_uint<8> spike_in_neuron_id;
    ap_int<8> spike_in_weight;
    ap_uint<16> threshold_out, leak_rate_out;
    
    ap_uint<1> spike_in_ready = 1;
    ap_uint<1> spike_out_valid = 0;
    ap_uint<8> spike_out_neuron_id = 0;
    ap_int<8> spike_out_weight = 0;
    ap_uint<1> snn_ready = 1;
    ap_uint<1> snn_busy = 0;
    
    // Simulate output spikes from Verilog core
    int spikes_collected = 0;
    for (int t = 0; t < 10; t++) {
        // Simulate Verilog generating a spike
        if (t < 3) {
            spike_out_valid = 1;
            spike_out_neuron_id = t * 2;
            spike_out_weight = 20 + t;
        } else {
            spike_out_valid = 0;
        }
        
        snn_top_hls(ctrl_reg, config_reg, mode_reg, time_steps_reg, params, encoder_cfg,
                status_reg, spike_count_reg,
                weight_sum_reg, version_reg, s_axis_spikes, s_axis_data, m_axis_spikes,
                m_axis_weights, reward_signal,
                    spike_in_valid, spike_in_neuron_id, spike_in_weight, spike_in_ready,
                    spike_out_valid, spike_out_neuron_id, spike_out_weight, spike_out_ready,
                    snn_enable, snn_reset, threshold_out, leak_rate_out,
                    snn_ready, snn_busy);
    }
    
    // Check output stream
    while (!m_axis_spikes.empty()) {
        axis_spike_t pkt = m_axis_spikes.read();
        if (VERBOSE) {
            std::cout << "  Output: neuron=" << (int)(uint8_t)pkt.data(7,0)
                      << " weight=" << (int)(int8_t)pkt.data(15,8) << std::endl;
        }
        spikes_collected++;
    }
    
    print_result("Output spikes collected", spikes_collected == 3);
}

//=============================================================================
// Test 4: STDP Learning
//=============================================================================
void test_stdp_learning() {
    std::cout << "\n=== Test 4: STDP Learning ===" << std::endl;
    
    hls::stream<axis_spike_t> s_axis_spikes;
    hls::stream<input_data_t> s_axis_data;
    hls::stream<axis_spike_t> m_axis_spikes;
    hls::stream<axis_weight_t> m_axis_weights;
    
    ap_uint<32> ctrl_reg = 0x09;  // Enable + Learning enable (bit 0 and bit 3)
    ap_uint<32> config_reg = (100 << 16) | 51;
    ap_uint<32> mode_reg = MODE_TRAIN_STDP;
    ap_uint<32> time_steps_reg = 1;
    encoder_config_t encoder_cfg = get_default_encoder_config();
    learning_params_t params = get_default_params();
    params.a_plus = 0.5;
    params.a_minus = 0.5;
    params.learning_rate = 0.1;
    
    ap_uint<32> status_reg, spike_count_reg, weight_sum_reg, version_reg;
    ap_int<8> reward_signal = 0;
    
    ap_uint<1> spike_in_valid, spike_out_ready, snn_enable, snn_reset;
    ap_uint<8> spike_in_neuron_id;
    ap_int<8> spike_in_weight;
    ap_uint<16> threshold_out, leak_rate_out;
    
    ap_uint<1> spike_in_ready = 1;
    ap_uint<1> spike_out_valid = 0;
    ap_uint<8> spike_out_neuron_id = 0;
    ap_int<8> spike_out_weight = 0;
    ap_uint<1> snn_ready = 1;
    ap_uint<1> snn_busy = 0;
    
    // First, reset to initialize
    ctrl_reg = 0x02;
    snn_top_hls(ctrl_reg, config_reg, mode_reg, time_steps_reg, params, encoder_cfg,
                status_reg, spike_count_reg,
                weight_sum_reg, version_reg, s_axis_spikes, s_axis_data, m_axis_spikes,
                m_axis_weights, reward_signal,
                spike_in_valid, spike_in_neuron_id, spike_in_weight, spike_in_ready,
                spike_out_valid, spike_out_neuron_id, spike_out_weight, spike_out_ready,
                snn_enable, snn_reset, threshold_out, leak_rate_out,
                snn_ready, snn_busy);
    
    // Enable with learning
    ctrl_reg = 0x09;
    
    // Scenario: Pre-spike at t=5, Post-spike at t=10 (should cause LTP)
    // Send pre-synaptic spike from neuron 0
    s_axis_spikes.write(create_spike(0, 10, 5));
    
    for (int t = 0; t < 5; t++) {
        snn_top_hls(ctrl_reg, config_reg, mode_reg, time_steps_reg, params, encoder_cfg,
                status_reg, spike_count_reg,
                weight_sum_reg, version_reg, s_axis_spikes, s_axis_data, m_axis_spikes,
                m_axis_weights, reward_signal,
                    spike_in_valid, spike_in_neuron_id, spike_in_weight, spike_in_ready,
                    spike_out_valid, spike_out_neuron_id, spike_out_weight, spike_out_ready,
                    snn_enable, snn_reset, threshold_out, leak_rate_out,
                    snn_ready, snn_busy);
    }
    
    // Now simulate post-synaptic spike from neuron 1
    for (int t = 5; t < 15; t++) {
        if (t == 10) {
            spike_out_valid = 1;
            spike_out_neuron_id = 1;
            spike_out_weight = 20;
        } else {
            spike_out_valid = 0;
        }
        
        snn_top_hls(ctrl_reg, config_reg, mode_reg, time_steps_reg, params, encoder_cfg,
                status_reg, spike_count_reg,
                weight_sum_reg, version_reg, s_axis_spikes, s_axis_data, m_axis_spikes,
                m_axis_weights, reward_signal,
                    spike_in_valid, spike_in_neuron_id, spike_in_weight, spike_in_ready,
                    spike_out_valid, spike_out_neuron_id, spike_out_weight, spike_out_ready,
                    snn_enable, snn_reset, threshold_out, leak_rate_out,
                    snn_ready, snn_busy);
    }
    
    // Check learning status
    bool learning_active = status_reg[2];
    print_result("Learning enabled in status", learning_active);
    
    std::cout << "  Weight sum (sample): " << weight_sum_reg << std::endl;
}

//=============================================================================
// Test 5: R-STDP with Reward
//=============================================================================
void test_rstdp_learning() {
    std::cout << "\n=== Test 5: R-STDP Learning ===" << std::endl;
    
    hls::stream<axis_spike_t> s_axis_spikes;
    hls::stream<input_data_t> s_axis_data;
    hls::stream<axis_spike_t> m_axis_spikes;
    hls::stream<axis_weight_t> m_axis_weights;
    
    ap_uint<32> ctrl_reg = 0x09;
    ap_uint<32> config_reg = (100 << 16) | 51;
    ap_uint<32> mode_reg = MODE_TRAIN_STDP;
    ap_uint<32> time_steps_reg = 1;
    encoder_config_t encoder_cfg = get_default_encoder_config();
    learning_params_t params = get_default_params();
    params.rstdp_enable = true;
    params.trace_decay = 0.95;
    params.reward_scale = 1.0;
    
    ap_uint<32> status_reg, spike_count_reg, weight_sum_reg, version_reg;
    ap_int<8> reward_signal = 64;  // Positive reward
    
    ap_uint<1> spike_in_valid, spike_out_ready, snn_enable, snn_reset;
    ap_uint<8> spike_in_neuron_id;
    ap_int<8> spike_in_weight;
    ap_uint<16> threshold_out, leak_rate_out;
    
    ap_uint<1> spike_in_ready = 1;
    ap_uint<1> spike_out_valid = 0;
    ap_uint<8> spike_out_neuron_id = 0;
    ap_int<8> spike_out_weight = 0;
    ap_uint<1> snn_ready = 1;
    ap_uint<1> snn_busy = 0;
    
    // Reset
    ctrl_reg = 0x02;
    snn_top_hls(ctrl_reg, config_reg, mode_reg, time_steps_reg, params, encoder_cfg,
                status_reg, spike_count_reg,
                weight_sum_reg, version_reg, s_axis_spikes, s_axis_data, m_axis_spikes,
                m_axis_weights, reward_signal,
                spike_in_valid, spike_in_neuron_id, spike_in_weight, spike_in_ready,
                spike_out_valid, spike_out_neuron_id, spike_out_weight, spike_out_ready,
                snn_enable, snn_reset, threshold_out, leak_rate_out,
                snn_ready, snn_busy);
    
    // Run with R-STDP
    ctrl_reg = 0x29;  // Enable + Learning + Apply reward (bits 0, 3, 5)
    
    // Generate spike activity
    s_axis_spikes.write(create_spike(0, 10, 0));
    
    for (int t = 0; t < 20; t++) {
        if (t == 5) {
            spike_out_valid = 1;
            spike_out_neuron_id = 1;
            spike_out_weight = 20;
        } else {
            spike_out_valid = 0;
        }
        
        snn_top_hls(ctrl_reg, config_reg, mode_reg, time_steps_reg, params, encoder_cfg,
                status_reg, spike_count_reg,
                weight_sum_reg, version_reg, s_axis_spikes, s_axis_data, m_axis_spikes,
                m_axis_weights, reward_signal,
                    spike_in_valid, spike_in_neuron_id, spike_in_weight, spike_in_ready,
                    spike_out_valid, spike_out_neuron_id, spike_out_weight, spike_out_ready,
                    snn_enable, snn_reset, threshold_out, leak_rate_out,
                    snn_ready, snn_busy);
    }
    
    bool rstdp_enabled = status_reg[4];
    print_result("R-STDP enabled in status", rstdp_enabled);
    
    std::cout << "  Weight sum after R-STDP: " << weight_sum_reg << std::endl;
}

//=============================================================================
// Main
//=============================================================================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "SNN Top-Level HLS Testbench" << std::endl;
    std::cout << "On-Chip Learning Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_control_registers();
    test_spike_input();
    test_spike_output();
    test_stdp_learning();
    test_rstdp_learning();
    
    std::cout << "\n========================================" << std::endl;
    if (error_count == 0) {
        std::cout << "All tests PASSED!" << std::endl;
    } else {
        std::cout << "Tests completed with " << error_count << " errors" << std::endl;
    }
    std::cout << "========================================" << std::endl;
    
    return error_count;
}
