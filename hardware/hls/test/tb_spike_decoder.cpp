//-----------------------------------------------------------------------------
// Title         : Testbench for Spike Decoder
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_spike_decoder.cpp
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Tests spike train to output decoding
//-----------------------------------------------------------------------------

#include <iostream>
#include <iomanip>
#include "../include/spike_decoder.h"
#include "test_utils.h"

using namespace std;

// Generate spike pattern for specific neuron
void generate_spike_pattern(
    hls::stream<spike_event_t> &spikes,
    int neuron_id,
    int num_spikes,
    int start_time,
    int interval
) {
    for (int i = 0; i < num_spikes; i++) {
        spike_event_t spike;
        spike.neuron_id = neuron_id;
        spike.timestamp = start_time + i * interval;
        spike.weight = 50;
        spikes.write(spike);
    }
}

int main() {
    cout << "==============================================\n";
    cout << "Spike Decoder Testbench\n";
    cout << "==============================================\n";
    
    // Test streams
    hls::stream<spike_event_t> spikes_in("spikes_in");
    hls::stream<output_data_t> data_out("data_out");
    
    // Configuration
    decoder_config_t config;
    config.num_outputs = 10; // 10 output classes
    config.window_size = 100;
    config.rate_alpha = 0.1;
    config.enable_softmax = false;
    config.temperature = 1.0;
    
    // Control
    bool enable = true;
    ap_uint<32> status;
    
    int total_errors = 0;
    
    //-------------------------------------------------------------------------
    // Test 1: Spike Count Decoding
    //-------------------------------------------------------------------------
    cout << "\nTest 1: Spike Count Decoding\n";
    cout << "----------------------------------------\n";
    
    config.decoding_type = SPIKE_COUNT;
    
    // Generate different spike counts for each output neuron
    for (int i = 0; i < config.num_outputs; i++) {
        generate_spike_pattern(spikes_in, i, (i + 1) * 5, 0, 2);
    }
    
    // Process for one window
    for (int t = 0; t < config.window_size; t++) {
        spike_decoder(enable, config, spikes_in, data_out, status);
    }
    
    // Check output
    if (!data_out.empty()) {
        output_data_t output = data_out.read();
        cout << "Winner neuron: " << (int)output.class_id << "\n";
        cout << "Confidence: " << (int)output.confidence << "\n";
        
        if (output.class_id == 9) { // Neuron 9 should have most spikes
            cout << "PASS: Correct winner based on spike count\n";
        } else {
            cout << "FAIL: Wrong winner neuron\n";
            total_errors++;
        }
        
        // Display all counts
        cout << "Spike counts: ";
        for (int i = 0; i < 10; i++) {
            cout << output.values[i] << " ";
        }
        cout << "\n";
    } else {
        cout << "FAIL: No output generated\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test 2: Spike Rate Decoding
    //-------------------------------------------------------------------------
    cout << "\nTest 2: Spike Rate Decoding\n";
    cout << "----------------------------------------\n";
    
    config.decoding_type = SPIKE_RATE;
    
    // Clear previous data
    while (!data_out.empty()) data_out.read();
    
    // Generate bursts with different rates
    for (int window = 0; window < 3; window++) {
        // Neuron 5 gets highest rate
        generate_spike_pattern(spikes_in, 5, 20, window * 100, 1);
        // Others get lower rates
        for (int i = 0; i < config.num_outputs; i++) {
            if (i != 5) {
                generate_spike_pattern(spikes_in, i, 5, window * 100, 5);
            }
        }
        
        // Process window
        for (int t = 0; t < config.window_size; t++) {
            spike_decoder(enable, config, spikes_in, data_out, status);
        }
    }
    
    // Check last output
    if (!data_out.empty()) {
        output_data_t output = data_out.read();
        cout << "Winner neuron (rate): " << (int)output.class_id << "\n";
        
        if (output.class_id == 5) {
            cout << "PASS: Correct winner based on spike rate\n";
        } else {
            cout << "FAIL: Wrong winner for rate decoding\n";
            total_errors++;
        }
    }
    
    //-------------------------------------------------------------------------
    // Test 3: Empty Input Handling
    //-------------------------------------------------------------------------
    cout << "\nTest 3: Empty Input Handling\n";
    cout << "----------------------------------------\n";
    
    // Process window with no spikes
    for (int t = 0; t < config.window_size; t++) {
        spike_decoder(enable, config, spikes_in, data_out, status);
    }
    
    if (!data_out.empty()) {
        output_data_t output = data_out.read();
        bool all_zero = true;
        for (int i = 0; i < MAX_OUTPUT_NEURONS; i++) {
            if (output.values[i] != 0) {
                all_zero = false;
                break;
            }
        }
        
        if (all_zero) {
            cout << "PASS: All outputs zero for empty input\n";
        } else {
            cout << "FAIL: Non-zero outputs for empty input\n";
            total_errors++;
        }
    }
    
    //-------------------------------------------------------------------------
    // Test 4: Window Size Effect
    //-------------------------------------------------------------------------
    cout << "\nTest 4: Window Size Effect\n";
    cout << "----------------------------------------\n";
    
    // Test with different window sizes
    int window_sizes[] = {50, 100, 200};
    
    for (int w = 0; w < 3; w++) {
        config.window_size = window_sizes[w];
        config.decoding_type = SPIKE_COUNT;
        
        // Generate consistent spike pattern
        generate_spike_pattern(spikes_in, 3, 10, 0, 5);
        
        // Process
        for (int t = 0; t < config.window_size; t++) {
            spike_decoder(enable, config, spikes_in, data_out, status);
        }
        
        cout << "Window size " << window_sizes[w] << ": ";
        if (!data_out.empty()) {
            output_data_t output = data_out.read();
            cout << "Neuron " << (int)output.class_id << " wins\n";
        }
    }
    
    config.window_size = 100; // Reset to default
    
    //-------------------------------------------------------------------------
    // Test 5: Disable Functionality
    //-------------------------------------------------------------------------
    cout << "\nTest 5: Disable Functionality\n";
    cout << "----------------------------------------\n";
    
    enable = false;
    
    // Try to process when disabled
    generate_spike_pattern(spikes_in, 0, 10, 0, 1);
    spike_decoder(enable, config, spikes_in, data_out, status);
    
    if ((status & 0x80000000) != 0) {
        cout << "PASS: Disabled flag set\n";
    } else {
        cout << "FAIL: Disabled flag not set\n";
        total_errors++;
    }
    
    enable = true; // Re-enable
    
    //-------------------------------------------------------------------------
    // Test 6: Continuous Operation
    //-------------------------------------------------------------------------
    cout << "\nTest 6: Continuous Operation\n";
    cout << "----------------------------------------\n";
    
    config.decoding_type = SPIKE_COUNT;
    
    // Simulate continuous inference
    int correct_classifications = 0;
    
    for (int trial = 0; trial < 10; trial++) {
        // Generate random "winner" class
        int true_class = rand() % config.num_outputs;
        
        // Generate spikes with true class having most
        for (int i = 0; i < config.num_outputs; i++) {
            int num_spikes = (i == true_class) ? 20 : (rand() % 10);
            generate_spike_pattern(spikes_in, i, num_spikes, trial * 100, 2);
        }
        
        // Process window
        for (int t = 0; t < config.window_size; t++) {
            spike_decoder(enable, config, spikes_in, data_out, status);
        }
        
        // Check classification
        if (!data_out.empty()) {
            output_data_t output = data_out.read();
            if (output.class_id == true_class) {
                correct_classifications++;
            }
        }
    }
    
    cout << "Correct classifications: " << correct_classifications << "/10\n";
    if (correct_classifications >= 8) {
        cout << "PASS: Good classification accuracy\n";
    } else {
        cout << "FAIL: Poor classification accuracy\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test Summary
    //-------------------------------------------------------------------------
    cout << "\n==============================================\n";
    cout << "Test Summary\n";
    cout << "==============================================\n";
    cout << "Total Tests: 6\n";
    cout << "Errors: " << total_errors << "\n";
    
    if (total_errors == 0) {
        cout << "\nALL TESTS PASSED!\n";
        return 0;
    } else {
        cout << "\nTEST FAILED with " << total_errors << " errors\n";
        return 1;
    }
}
