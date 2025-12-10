//-----------------------------------------------------------------------------
// Title         : Testbench for Weight Updater
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_weight_updater.cpp
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Tests weight update application and management
//-----------------------------------------------------------------------------

#include <iostream>
#include <iomanip>
#include "../include/weight_updater.h"
#include "test_utils.h"

using namespace std;

// Initialize weight memory
void init_weights(weight_t *weights, int size, weight_t value) {
    for (int i = 0; i < size; i++) {
        weights[i] = value;
    }
}

// Check weight bounds
bool check_weight_bounds(weight_t *weights, int size, weight_t min, weight_t max) {
    for (int i = 0; i < size; i++) {
        if (weights[i] < min || weights[i] > max) {
            return false;
        }
    }
    return true;
}

int main() {
    cout << "==============================================\n";
    cout << "Weight Updater Testbench\n";
    cout << "==============================================\n";
    
    // Allocate weight memory
    weight_t weight_memory[MAX_SYNAPSES];
    
    // Test streams
    hls::stream<weight_update_t> updates_in("updates_in");
    
    // Configuration
    weight_config_t config;
    config.max_weight = 100;
    config.min_weight = -100;
    config.enable_decay = false;
    config.decay_rate = 250; // Minimal decay
    config.enable_normalization = false;
    config.norm_target = 1000;
    
    // Control
    bool enable = true;
    bool reset = false;
    ap_uint<32> updates_applied;
    
    int total_errors = 0;
    
    //-------------------------------------------------------------------------
    // Test 1: Basic Weight Update
    //-------------------------------------------------------------------------
    cout << "\nTest 1: Basic Weight Update\n";
    cout << "----------------------------------------\n";
    
    // Initialize weights
    init_weights(weight_memory, MAX_SYNAPSES, 50);
    
    // Create weight update
    weight_update_t update;
    update.pre_id = 0;
    update.post_id = 1;
    update.delta = 20;
    update.timestamp = 100;
    updates_in.write(update);
    
    // Apply update
    weight_updater(enable, reset, updates_in, weight_memory, config, updates_applied);
    
    // Check result
    int addr = update.pre_id * MAX_NEURONS + update.post_id;
    if (weight_memory[addr] == 70) { // 50 + 20
        cout << "PASS: Weight updated correctly (50 + 20 = 70)\n";
    } else {
        cout << "FAIL: Weight = " << weight_memory[addr] << ", expected 70\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test 2: Weight Bounds Checking
    //-------------------------------------------------------------------------
    cout << "\nTest 2: Weight Bounds Checking\n";
    cout << "----------------------------------------\n";
    
    // Test upper bound
    weight_memory[10] = 95;
    update.pre_id = 0;
    update.post_id = 10;
    update.delta = 20; // Would exceed max
    updates_in.write(update);
    
    weight_updater(enable, reset, updates_in, weight_memory, config, updates_applied);
    
    if (weight_memory[10] == config.max_weight) {
        cout << "PASS: Upper bound enforced (" << weight_memory[10] << ")\n";
    } else {
        cout << "FAIL: Upper bound not enforced\n";
        total_errors++;
    }
    
    // Test lower bound
    weight_memory[20] = -95;
    update.pre_id = 0;
    update.post_id = 20;
    update.delta = -20; // Would exceed min
    updates_in.write(update);
    
    weight_updater(enable, reset, updates_in, weight_memory, config, updates_applied);
    
    if (weight_memory[20] == config.min_weight) {
        cout << "PASS: Lower bound enforced (" << weight_memory[20] << ")\n";
    } else {
        cout << "FAIL: Lower bound not enforced\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test 3: Weight Decay
    //-------------------------------------------------------------------------
    cout << "\nTest 3: Weight Decay\n";
    cout << "----------------------------------------\n";
    
    config.enable_decay = true;
    config.decay_rate = 128; // 50% decay (128/256)
    
    // Positive weight decay
    weight_memory[30] = 80;
    update.pre_id = 0;
    update.post_id = 30;
    update.delta = 0; // No change, just decay
    updates_in.write(update);
    
    weight_updater(enable, reset, updates_in, weight_memory, config, updates_applied);
    
    // Should decay by ~50%
    if (weight_memory[30] < 80 && weight_memory[30] > 30) {
        cout << "PASS: Positive weight decayed to " << weight_memory[30] << "\n";
    } else {
        cout << "FAIL: Incorrect decay\n";
        total_errors++;
    }
    
    // Negative weight decay
    weight_memory[31] = -60;
    update.pre_id = 0;
    update.post_id = 31;
    update.delta = 0;
    updates_in.write(update);
    
    weight_updater(enable, reset, updates_in, weight_memory, config, updates_applied);
    
    if (weight_memory[31] > -60 && weight_memory[31] < 0) {
        cout << "PASS: Negative weight decayed to " << weight_memory[31] << "\n";
    } else {
        cout << "FAIL: Incorrect negative decay\n";
        total_errors++;
    }
    
    config.enable_decay = false; // Disable for next tests
    
    //-------------------------------------------------------------------------
    // Test 4: Multiple Updates
    //-------------------------------------------------------------------------
    cout << "\nTest 4: Multiple Updates\n";
    cout << "----------------------------------------\n";
    
    // Reset counter
    reset = true;
    weight_updater(enable, reset, updates_in, weight_memory, config, updates_applied);
    reset = false;
    
    // Send multiple updates
    for (int i = 0; i < 10; i++) {
        update.pre_id = i;
        update.post_id = i + 1;
        update.delta = i * 5;
        updates_in.write(update);
    }
    
    // Apply all updates
    for (int i = 0; i < 10; i++) {
        weight_updater(enable, reset, updates_in, weight_memory, config, updates_applied);
    }
    
    if (updates_applied == 10) {
        cout << "PASS: All 10 updates applied\n";
    } else {
        cout << "FAIL: Only " << updates_applied << " updates applied\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test 5: Invalid Address Handling
    //-------------------------------------------------------------------------
    cout << "\nTest 5: Invalid Address Handling\n";
    cout << "----------------------------------------\n";
    
    ap_uint<32> prev_count = updates_applied;
    
    // Try to update beyond valid range (NUM_NEURONS=256, so valid IDs are 0-255)
    update.pre_id = 512; // Definitely invalid - well beyond MAX_NEURONS
    update.post_id = 512;
    update.delta = 50;
    updates_in.write(update);
    
    weight_updater(enable, reset, updates_in, weight_memory, config, updates_applied);
    
    if (updates_applied == prev_count) {
        cout << "PASS: Invalid address ignored\n";
    } else {
        cout << "FAIL: Invalid address not handled properly\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test 6: Disable Functionality
    //-------------------------------------------------------------------------
    cout << "\nTest 6: Disable Functionality\n";
    cout << "----------------------------------------\n";
    
    enable = false;
    prev_count = updates_applied;
    
    update.pre_id = 2;
    update.post_id = 3;
    update.delta = 25;
    updates_in.write(update);
    
    weight_updater(enable, reset, updates_in, weight_memory, config, updates_applied);
    
    if (updates_applied == prev_count) {
        cout << "PASS: No updates when disabled\n";
    } else {
        cout << "FAIL: Updates applied when disabled\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test 7: Performance Test
    //-------------------------------------------------------------------------
    cout << "\nTest 7: Performance Test\n";
    cout << "----------------------------------------\n";
    
    enable = true;
    reset = true;
    weight_updater(enable, reset, updates_in, weight_memory, config, updates_applied);
    reset = false;
    
    // Generate burst of updates
    Timer timer;
    timer.start();
    
    for (int i = 0; i < 1000; i++) {
        update.pre_id = rand() % NUM_NEURONS;
        update.post_id = rand() % NUM_NEURONS;
        update.delta = (rand() % 20) - 10;
        updates_in.write(update);
    }
    
    // Apply all updates
    for (int i = 0; i < 1000; i++) {
        weight_updater(enable, reset, updates_in, weight_memory, config, updates_applied);
    }
    
    timer.stop();
    
    cout << "Applied " << updates_applied << " updates in " << timer.getTime() << " cycles\n";
    cout << "Throughput: " << (float)updates_applied / timer.getTime() << " updates/cycle\n";
    
    // Verify all weights are within bounds
    if (check_weight_bounds(weight_memory, MAX_SYNAPSES, config.min_weight, config.max_weight)) {
        cout << "PASS: All weights within bounds after stress test\n";
    } else {
        cout << "FAIL: Some weights out of bounds\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test Summary
    //-------------------------------------------------------------------------
    cout << "\n==============================================\n";
    cout << "Test Summary\n";
    cout << "==============================================\n";
    cout << "Total Tests: 7\n";
    cout << "Errors: " << total_errors << "\n";
    
    if (total_errors == 0) {
        cout << "\nALL TESTS PASSED!\n";
        return 0;
    } else {
        cout << "\nTEST FAILED with " << total_errors << " errors\n";
        return 1;
    }
}
