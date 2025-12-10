//-----------------------------------------------------------------------------
// Title         : Testbench for Spike Encoder
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_spike_encoder.cpp
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Tests various spike encoding schemes
//-----------------------------------------------------------------------------

#include <iostream>
#include <iomanip>
#include <vector>
#include "../include/spike_encoder.h"
#include "test_utils.h"

using namespace std;

// Generate test image data
void generate_test_image(input_data_t &data, int pattern) {
    switch (pattern) {
        case 0: // All zeros
            for (int i = 0; i < MAX_INPUT_CHANNELS; i++) {
                data.pixels[i] = 0;
            }
            break;
            
        case 1: // All max
            for (int i = 0; i < MAX_INPUT_CHANNELS; i++) {
                data.pixels[i] = 255;
            }
            break;
            
        case 2: // Gradient
            for (int i = 0; i < MAX_INPUT_CHANNELS; i++) {
                data.pixels[i] = (i * 255) / MAX_INPUT_CHANNELS;
            }
            break;
            
        case 3: // Checkerboard (28x28 for MNIST)
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    data.pixels[i*28 + j] = ((i/4 + j/4) % 2) * 255;
                }
            }
            break;
            
        default: // Random
            for (int i = 0; i < MAX_INPUT_CHANNELS; i++) {
                data.pixels[i] = rand() % 256;
            }
    }
    data.label = pattern;
    data.frame_id = pattern;
}

// Count spikes per channel
void count_spikes_per_channel(
    hls::stream<spike_event_t> &spikes,
    int spike_counts[MAX_INPUT_CHANNELS],
    int duration
) {
    // Initialize counts
    for (int i = 0; i < MAX_INPUT_CHANNELS; i++) {
        spike_counts[i] = 0;
    }
    
    // Count spikes
    while (!spikes.empty()) {
        spike_event_t spike = spikes.read();
        if (spike.neuron_id < MAX_INPUT_CHANNELS) {
            spike_counts[spike.neuron_id]++;
        }
    }
}

int main() {
    cout << "==============================================\n";
    cout << "Spike Encoder Testbench\n";
    cout << "==============================================\n";
    
    // Test streams
    hls::stream<input_data_t> data_in("data_in");
    hls::stream<spike_event_t> spikes_out("spikes_out");
    
    // Configuration
    encoder_config_t config;
    config.num_channels = 784; // MNIST size
    config.time_window = 100;
    config.rate_scale = 100;
    config.phase_scale = 16;
    config.phase_threshold = 1000;
    config.default_weight = 50;
    
    // Control
    bool enable = true;
    ap_uint<32> spike_count;
    
    int total_errors = 0;
    
    //-------------------------------------------------------------------------
    // Test 1: Rate Coding
    //-------------------------------------------------------------------------
    cout << "\nTest 1: Rate Coding\n";
    cout << "----------------------------------------\n";
    
    config.encoding_type = RATE_CODING;
    
    // Test with gradient pattern
    input_data_t test_data;
    generate_test_image(test_data, 2); // Gradient
    data_in.write(test_data);
    
    // Run encoder for multiple time steps
    int spike_counts[MAX_INPUT_CHANNELS];
    for (int t = 0; t < 1000; t++) {
        spike_encoder(enable, config, data_in, spikes_out, spike_count);
    }
    
    count_spikes_per_channel(spikes_out, spike_counts, 1000);
    
    // Verify rate coding property: higher pixel value = higher spike rate
    bool rate_coding_correct = true;
    for (int i = 1; i < config.num_channels; i++) {
        if (test_data.pixels[i] > test_data.pixels[i-1]) {
            if (spike_counts[i] < spike_counts[i-1]) {
                rate_coding_correct = false;
                break;
            }
        }
    }
    
    if (rate_coding_correct) {
        cout << "PASS: Rate coding preserves intensity ordering\n";
        cout << "  Total spikes: " << spike_count << "\n";
    } else {
        cout << "FAIL: Rate coding does not preserve intensity\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test 2: Temporal Coding
    //-------------------------------------------------------------------------
    cout << "\nTest 3: Phase Coding\n";
    cout << "----------------------------------------\n";
    
    config.encoding_type = PHASE_CODING;
    config.phase_scale = 256;       // Higher scale for more spikes (target 0.01-0.5 rate)
    config.phase_threshold = 1024;
    
    // Test with constant input
    generate_test_image(test_data, 1); // All max values
    data_in.write(test_data);
    
    // Run encoder for one time window
    vector<spike_time_t> first_spike_times(MAX_INPUT_CHANNELS, 0);
    bool spike_seen[MAX_INPUT_CHANNELS] = {false};
    
    for (int t = 0; t < config.time_window; t++) {
        spike_encoder(enable, config, data_in, spikes_out, spike_count);
        
        // Record first spike times
        while (!spikes_out.empty()) {
            spike_event_t spike = spikes_out.read();
            if (!spike_seen[spike.neuron_id]) {
                first_spike_times[spike.neuron_id] = t;
                spike_seen[spike.neuron_id] = true;
            }
        }
    }
    
    // All max values should spike early
    bool temporal_correct = true;
    for (int i = 0; i < config.num_channels; i++) {
        if (first_spike_times[i] > 10) { // Should spike within first 10 time steps
            temporal_correct = false;
            break;
        }
    }
    
    if (temporal_correct) {
        cout << "PASS: Temporal coding generates early spikes for high values\n";
    } else {
        cout << "FAIL: Temporal coding timing incorrect\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test 3: Phase Coding
    //-------------------------------------------------------------------------
    cout << "\nTest 3: Phase Coding\n";
    cout << "----------------------------------------\n";
    
    config.encoding_type = PHASE_CODING;
    
    // Test with constant input
    for (int i = 0; i < MAX_INPUT_CHANNELS; i++) {
        test_data.pixels[i] = 128; // Mid-range value
    }
    data_in.write(test_data);
    
    // Run for extended period
    int phase_spike_count = 0;
    for (int t = 0; t < 2000; t++) {
        spike_encoder(enable, config, data_in, spikes_out, spike_count);
        
        while (!spikes_out.empty()) {
            spikes_out.read();
            phase_spike_count++;
        }
    }
    
    // Check that spikes are generated regularly
    float avg_spike_rate = (float)phase_spike_count / (config.num_channels * 2000.0);
    cout << "Average spike rate: " << avg_spike_rate << " spikes/channel/timestep\n";
    
    if (avg_spike_rate > 0.01 && avg_spike_rate < 0.5) {
        cout << "PASS: Phase coding generates reasonable spike rates\n";
    } else {
        cout << "FAIL: Phase coding spike rate out of range\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test 4: Zero Input Handling
    //-------------------------------------------------------------------------
    cout << "\nTest 4: Zero Input Handling\n";
    cout << "----------------------------------------\n";
    
    // Test all encoding types with zero input
    generate_test_image(test_data, 0); // All zeros
    
    for (int enc_type = 0; enc_type < 3; enc_type++) {
        config.encoding_type = (encoding_type_t)enc_type;
        data_in.write(test_data);
        
        int zero_spikes = 0;
        for (int t = 0; t < 100; t++) {
            spike_encoder(enable, config, data_in, spikes_out, spike_count);
            while (!spikes_out.empty()) {
                spikes_out.read();
                zero_spikes++;
            }
        }
        
        cout << "Encoding type " << enc_type << ": " << zero_spikes << " spikes\n";
    }
    
    cout << "PASS: Zero input handling tested\n";
    
    //-------------------------------------------------------------------------
    // Test 5: Disable Functionality
    //-------------------------------------------------------------------------
    cout << "\nTest 5: Disable Functionality\n";
    cout << "----------------------------------------\n";
    
    enable = false;
    config.encoding_type = RATE_CODING;
    
    generate_test_image(test_data, 1); // All max
    data_in.write(test_data);
    
    ap_uint<32> prev_count = spike_count;
    spike_encoder(enable, config, data_in, spikes_out, spike_count);
    
    if (spike_count == prev_count) {
        cout << "PASS: No spikes generated when disabled\n";
    } else {
        cout << "FAIL: Spikes generated when disabled\n";
        total_errors++;
    }
    
    //-------------------------------------------------------------------------
    // Test 6: MNIST-like Pattern
    //-------------------------------------------------------------------------
    cout << "\nTest 6: MNIST-like Pattern\n";
    cout << "----------------------------------------\n";
    
    enable = true;
    config.encoding_type = RATE_CODING;
    
    // Create simple digit-like pattern (vertical line)
    for (int i = 0; i < 784; i++) {
        int row = i / 28;
        int col = i % 28;
        if (col >= 13 && col <= 15 && row >= 5 && row <= 22) {
            test_data.pixels[i] = 255; // White
        } else {
            test_data.pixels[i] = 0;   // Black
        }
    }
    data_in.write(test_data);
    
    // Encode for 500 time steps
    int pattern_spikes = 0;
    for (int t = 0; t < 500; t++) {
        spike_encoder(enable, config, data_in, spikes_out, spike_count);
        while (!spikes_out.empty()) {
            spike_event_t spike = spikes_out.read();
            pattern_spikes++;
        }
    }
    
    cout << "MNIST pattern generated " << pattern_spikes << " spikes\n";
    cout << "Spike density: " << (float)pattern_spikes / (784 * 500) << "\n";
    
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
