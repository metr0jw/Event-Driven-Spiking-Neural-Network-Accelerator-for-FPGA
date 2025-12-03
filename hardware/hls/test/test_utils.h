//-----------------------------------------------------------------------------
// Title         : Test Utilities
// Project       : PYNQ-Z2 SNN Accelerator
// File          : test_utils.h
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Common utilities for HLS testbenches
//-----------------------------------------------------------------------------

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include "../include/snn_types.h"

// Simple timer for performance measurement
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool running;
    
public:
    Timer() : running(false) {}
    
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }
    
    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
        running = false;
    }
    
    double getTime() {
        if (running) {
            stop();
        }
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>
                       (end_time - start_time);
        return duration.count();
    }
};

// Print spike event
void print_spike(const spike_event_t &spike) {
    std::cout << "Spike: neuron=" << spike.neuron_id 
              << ", time=" << spike.timestamp 
              << ", weight=" << spike.weight << std::endl;
}

// Print weight update
void print_update(const weight_update_t &update) {
    std::cout << "Update: pre=" << update.pre_id 
              << ", post=" << update.post_id 
              << ", delta=" << update.delta 
              << ", time=" << update.timestamp << std::endl;
}

// Generate random spike train
void generate_random_spikes(
    hls::stream<spike_event_t> &stream,
    int num_spikes,
    int max_neuron_id,
    int max_time
) {
    for (int i = 0; i < num_spikes; i++) {
        spike_event_t spike;
        spike.neuron_id = rand() % max_neuron_id;
        spike.timestamp = rand() % max_time;
        spike.weight = 50 + (rand() % 50);
        stream.write(spike);
    }
}

// Compare floating point values with tolerance
bool float_equal(float a, float b, float tolerance = 0.001) {
    return std::abs(a - b) < tolerance;
}

// Print test header
void print_test_header(const std::string &test_name) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << test_name << "\n";
    std::cout << "========================================\n";
}

// Print pass/fail result
void print_result(bool pass, const std::string &message = "") {
    if (pass) {
        std::cout << "[PASS] ";
    } else {
        std::cout << "[FAIL] ";
    }
    if (!message.empty()) {
        std::cout << message;
    }
    std::cout << std::endl;
}

// Memory usage estimator
void print_memory_usage() {
    int neuron_mem = MAX_NEURONS * sizeof(membrane_t);
    int synapse_mem = MAX_SYNAPSES * sizeof(weight_t);
    int total_mem = neuron_mem + synapse_mem;
    
    std::cout << "Estimated memory usage:\n";
    std::cout << "  Neurons: " << neuron_mem << " bytes\n";
    std::cout << "  Synapses: " << synapse_mem << " bytes\n";
    std::cout << "  Total: " << total_mem << " bytes (" 
              << total_mem/1024 << " KB)\n";
}

#endif // TEST_UTILS_H
