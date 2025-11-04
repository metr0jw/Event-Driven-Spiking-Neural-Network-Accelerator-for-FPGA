//-----------------------------------------------------------------------------
// Title         : Testbench for SNN Learning Engine
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_snn_learning_engine.cpp
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Comprehensive testbench for STDP learning
//-----------------------------------------------------------------------------

#include <iostream>
#include <iomanip>
#include <cmath>
#include "../include/snn_learning_engine.h"
#include "test_utils.h"

using namespace std;

// Test configuration
const int NUM_TEST_SPIKES = 100;
const int TEST_DURATION = 1000;

// Function to generate test spike patterns
void generate_spike_pattern(
    hls::stream<spike_event_t> &spike_stream,
    int neuron_id,
    int base_time,
    int num_spikes,
    int interval
) {
    for (int i = 0; i < num_spikes; i++) {
        spike_event_t spike;
        spike.neuron_id = neuron_id;
        spike.timestamp = base_time + i * interval;
        spike.weight = 50;
        spike_stream.write(spike);
    }
}

// Verify STDP weight updates
bool verify_stdp_update(
    weight_update_t update,
    int pre_id,
    int post_id,
    bool expect_ltp
) {
    if (update.pre_id != pre_id || update.post_id != post_id) {
        cout << "ERROR: Wrong neuron IDs. Expected (" << pre_id << "," << post_id 
             << "), got (" << update.pre_id << "," << update.post_id << ")" << endl;
        return false;
    }
    
    if (expect_ltp && update.delta <= 0) {
        cout << "ERROR: Expected LTP (positive delta), got " << update.delta << endl;
        return false;
    }
    
    if (!expect_ltp && update.delta >= 0) {
        cout << "ERROR: Expected LTD (negative delta), got " << update.delta << endl;
        return false;
    }
    
    return true;
}

int main() {
    cout << "==============================================\n";
    cout << "SNN Learning Engine Testbench\n";
    cout << "==============================================\n";
    
    // Test streams
    hls::stream<spike_event_t> pre_spikes("pre_spikes");
    hls::stream<spike_event_t> post_spikes("post_spikes");
    hls::stream<weight_update_t> weight_updates("weight_updates");
    
    // Configuration
    learning_config_t config;
    config.a_plus = 0.01;
    config.a_minus = 0.01;
    config.tau_plus = 20.0;
    config.tau_minus = 20.0;
    config.stdp_window = 100;
    config.enable_homeostasis = false;
    config.target_rate = 10.0;
    
    // Control signals
    bool enable = true;
    bool reset = false;
    ap_uint<32> status;
    
    int total_errors = 0;
    
    //-------------------------------------------------------------------------
    // Test 1: Reset functionality
    //-------------------------------------------------------------------------
    cout << "\nTest 1: Reset Functionality\n";
    cout << "----------------------------------------\n";
    
    reset = true;
    snn_learning_engine(enable, reset, config, pre_spikes, post_spikes, 
                       weight_updates, status);
    reset = false;
    
    if (status == 0) {
        cout << "PASS: Reset cleared status\n";
    } else {
        cout << "FAIL: Status not cleared after reset\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test 2: Basic LTP (Pre before Post)
    //-------------------------------------------------------------------------
    cout << "\nTest 2: LTP - Pre before Post\n";
    cout << "----------------------------------------\n";
    
    // Clear streams
    while (!weight_updates.empty()) weight_updates.read();
    
    // Pre spike at t=100
    spike_event_t pre_spike;
    pre_spike.neuron_id = 0;
    pre_spike.timestamp = 100;
    pre_spike.weight = 50;
    pre_spikes.write(pre_spike);
    
    // Process pre spike
    snn_learning_engine(enable, reset, config, pre_spikes, post_spikes, 
                       weight_updates, status);
    
    // Post spike at t=120 (dt = 20)
    spike_event_t post_spike;
    post_spike.neuron_id = 1;
    post_spike.timestamp = 120;
    post_spike.weight = 50;
    post_spikes.write(post_spike);
    
    // Process post spike
    snn_learning_engine(enable, reset, config, pre_spikes, post_spikes, 
                       weight_updates, status);
    
    // Check weight update
    if (!weight_updates.empty()) {
        weight_update_t update = weight_updates.read();
        if (verify_stdp_update(update, 0, 1, true)) {
            cout << "PASS: LTP update generated correctly\n";
            cout << "  Delta = " << update.delta << " (positive)\n";
        } else {
            cout << "FAIL: Incorrect LTP update\n";
            total_errors++;
        }
    } else {
        cout << "FAIL: No weight update generated\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test 3: Basic LTD (Post before Pre)
    //-------------------------------------------------------------------------
    cout << "\nTest 3: LTD - Post before Pre\n";
    cout << "----------------------------------------\n";
    
    // Clear streams
    while (!weight_updates.empty()) weight_updates.read();
    
    // Post spike at t=200
    post_spike.neuron_id = 2;
    post_spike.timestamp = 200;
    post_spikes.write(post_spike);
    
    // Process post spike
    snn_learning_engine(enable, reset, config, pre_spikes, post_spikes, 
                       weight_updates, status);
    
    // Pre spike at t=230 (dt = 30)
    pre_spike.neuron_id = 3;
    pre_spike.timestamp = 230;
    pre_spikes.write(pre_spike);
    
    // Process pre spike
    snn_learning_engine(enable, reset, config, pre_spikes, post_spikes, 
                       weight_updates, status);
    
    // Check weight update
    if (!weight_updates.empty()) {
        weight_update_t update = weight_updates.read();
        if (verify_stdp_update(update, 3, 2, false)) {
            cout << "PASS: LTD update generated correctly\n";
            cout << "  Delta = " << update.delta << " (negative)\n";
        } else {
            cout << "FAIL: Incorrect LTD update\n";
            total_errors++;
        }
    } else {
        cout << "FAIL: No weight update generated\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test 4: STDP Time Window
    //-------------------------------------------------------------------------
    cout << "\nTest 4: STDP Time Window\n";
    cout << "----------------------------------------\n";
    
    // Clear streams
    while (!weight_updates.empty()) weight_updates.read();
    
    // Pre spike at t=300
    pre_spike.neuron_id = 4;
    pre_spike.timestamp = 300;
    pre_spikes.write(pre_spike);
    
    snn_learning_engine(enable, reset, config, pre_spikes, post_spikes, 
                       weight_updates, status);
    
    // Post spike outside window (t=450, dt=150 > 100)
    post_spike.neuron_id = 5;
    post_spike.timestamp = 450;
    post_spikes.write(post_spike);
    
    snn_learning_engine(enable, reset, config, pre_spikes, post_spikes, 
                       weight_updates, status);
    
    if (weight_updates.empty()) {
        cout << "PASS: No update outside STDP window\n";
    } else {
        cout << "FAIL: Update generated outside window\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test 5: Multiple Spike Pairs
    //-------------------------------------------------------------------------
    cout << "\nTest 5: Multiple Spike Pairs\n";
    cout << "----------------------------------------\n";
    
    // Clear streams and reset
    while (!weight_updates.empty()) weight_updates.read();
    reset = true;
    snn_learning_engine(enable, reset, config, pre_spikes, post_spikes, 
                       weight_updates, status);
    reset = false;
    
    // Generate multiple pre spikes
    for (int i = 0; i < 5; i++) {
        pre_spike.neuron_id = i;
        pre_spike.timestamp = 500 + i * 10;
        pre_spikes.write(pre_spike);
    }
    
    // Process all pre spikes
    for (int i = 0; i < 5; i++) {
        snn_learning_engine(enable, reset, config, pre_spikes, post_spikes, 
                           weight_updates, status);
    }
    
    // Generate post spike that should pair with all pre spikes
    post_spike.neuron_id = 10;
    post_spike.timestamp = 560;
    post_spikes.write(post_spike);
    
    snn_learning_engine(enable, reset, config, pre_spikes, post_spikes, 
                       weight_updates, status);
    
    // Count updates
    int update_count = 0;
    while (!weight_updates.empty()) {
        weight_update_t update = weight_updates.read();
        update_count++;
        cout << "  Update " << update_count << ": pre=" << update.pre_id 
             << ", post=" << update.post_id << ", delta=" << update.delta << "\n";
    }
    
    if (update_count == 5) {
        cout << "PASS: All " << update_count << " spike pairs generated updates\n";
    } else {
        cout << "FAIL: Expected 5 updates, got " << update_count << "\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test 6: Disable functionality
    //-------------------------------------------------------------------------
    cout << "\nTest 6: Disable Functionality\n";
    cout << "----------------------------------------\n";
    
    enable = false;
    
    // Try to generate spikes when disabled
    pre_spike.neuron_id = 20;
    pre_spike.timestamp = 700;
    pre_spikes.write(pre_spike);
    
    snn_learning_engine(enable, reset, config, pre_spikes, post_spikes, 
                       weight_updates, status);
    
    if ((status & 0x80000000) != 0) {
        cout << "PASS: Disabled flag set in status\n";
    } else {
        cout << "FAIL: Disabled flag not set\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test 7: Performance Test
    //-------------------------------------------------------------------------
    cout << "\nTest 7: Performance Test\n";
    cout << "----------------------------------------\n";
    
    enable = true;
    reset = true;
    snn_learning_engine(enable, reset, config, pre_spikes, post_spikes, 
                       weight_updates, status);
    reset = false;
    
    // Generate burst of activity
    Timer timer;
    timer.start();
    
    for (int t = 0; t < 100; t++) {
        // Random pre spike
        if (t % 3 == 0) {
            pre_spike.neuron_id = rand() % 32;
            pre_spike.timestamp = t * 10;
            pre_spikes.write(pre_spike);
        }
        
        // Random post spike
        if (t % 4 == 0) {
            post_spike.neuron_id = rand() % 32;
            post_spike.timestamp = t * 10 + 5;
            post_spikes.write(post_spike);
        }
        
        // Process
        snn_learning_engine(enable, reset, config, pre_spikes, post_spikes, 
                           weight_updates, status);
    }
    
    timer.stop();
    
    // Count total updates
    int total_updates = 0;
    while (!weight_updates.empty()) {
        weight_updates.read();
        total_updates++;
    }
    
    cout << "Processed 100 time steps in " << timer.getTime() << " cycles\n";
    cout << "Generated " << total_updates << " weight updates\n";
    cout << "Status counter: " << status << "\n";
    
    //-------------------------------------------------------------------------
    // Test Summary
    //-------------------------------------------------------------------------
    cout << "\n==============================================\n";
    cout << "Test Summary\n";
    cout << "==============================================\n";
    cout << "Total Tests: 7\n";
    cout << "Passed: " << (7 - (total_errors > 0 ? 1 : 0)) << "\n";
    cout << "Failed: " << (total_errors > 0 ? 1 : 0) << "\n";
    
    if (total_errors == 0) {
        cout << "\nALL TESTS PASSED!\n";
        return 0;
    } else {
        cout << "\nTEST FAILED with " << total_errors << " errors\n";
        return 1;
    }
}
