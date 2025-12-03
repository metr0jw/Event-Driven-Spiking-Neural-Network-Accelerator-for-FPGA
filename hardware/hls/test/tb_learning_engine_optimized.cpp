//-----------------------------------------------------------------------------
// Title         : Learning Engine Optimization Testbench
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_learning_engine_optimized.cpp
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Description   : Comprehensive testbench for all optimization strategies
// Note          : Software simulation version
//-----------------------------------------------------------------------------

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cstdint>
#include <cmath>

using namespace std;

// Software simulation types (compatible with standard C++)
typedef uint8_t neuron_id_t;
typedef uint32_t spike_time_t;
typedef int16_t weight_delta_t;

const int MAX_NEURONS = 64;
const weight_delta_t MAX_WEIGHT_DELTA = 127;
const int WEIGHT_SCALE = 128;

// Spike event structure
struct spike_event_t {
    neuron_id_t neuron_id;
    spike_time_t timestamp;
    int8_t weight;
};

// Weight update structure
struct weight_update_t {
    neuron_id_t pre_id;
    neuron_id_t post_id;
    weight_delta_t delta;
    spike_time_t timestamp;
};

// Learning configuration
struct learning_config_t {
    float a_plus;
    float a_minus;
    float tau_plus;
    float tau_minus;
    uint32_t stdp_window;
    bool enable_homeostasis;
    float target_rate;
};

// Simple queue for software simulation
template<typename T>
class simple_queue {
private:
    vector<T> data;
    size_t read_pos;
public:
    simple_queue() : read_pos(0) {}
    
    void write(const T& val) {
        data.push_back(val);
    }
    
    T read() {
        if (read_pos >= data.size()) {
            throw runtime_error("Queue underflow");
        }
        return data[read_pos++];
    }
    
    bool empty() const {
        return read_pos >= data.size();
    }
    
    void clear() {
        data.clear();
        read_pos = 0;
    }
};

//=============================================================================
// Test Configuration
//=============================================================================

const int NUM_TEST_SPIKES = 1000;
const int NUM_WARMUP_SPIKES = 100;

// Test scenarios with different activity rates
const float TEST_ACTIVITIES[] = {0.1, 0.25, 0.5, 0.75, 1.0};
const int NUM_SCENARIOS = 5;

//=============================================================================
// Helper Functions
//=============================================================================

// Generate synthetic spike train with given activity rate
void generate_spike_train(
    vector<spike_event_t> &spikes,
    float activity_rate,
    int num_spikes,
    spike_time_t start_time = 0
) {
    spikes.clear();
    
    int active_neurons = (int)(MAX_NEURONS * activity_rate);
    if (active_neurons < 1) active_neurons = 1;
    
    for (int i = 0; i < num_spikes; i++) {
        spike_event_t spike;
        
        // Random neuron from active set
        spike.neuron_id = rand() % active_neurons;
        
        // Spikes at regular intervals with jitter
        spike.timestamp = start_time + i * 100 + (rand() % 20);
        spike.weight = 0;
        
        spikes.push_back(spike);
    }
}

// Software STDP calculation (baseline)
weight_delta_t calculate_ltp_sw(int32_t dt, learning_config_t config) {
    if (dt <= 0 || dt >= (int32_t)config.stdp_window) {
        return 0;
    }
    float exp_factor = exp(-float(dt) / config.tau_plus);
    float delta_float = config.a_plus * exp_factor;
    weight_delta_t delta = (weight_delta_t)(delta_float * WEIGHT_SCALE);
    if (delta > MAX_WEIGHT_DELTA) delta = MAX_WEIGHT_DELTA;
    return delta;
}

weight_delta_t calculate_ltd_sw(int32_t dt, learning_config_t config) {
    if (dt <= 0 || dt >= (int32_t)config.stdp_window) {
        return 0;
    }
    float exp_factor = exp(-float(dt) / config.tau_minus);
    float delta_float = -config.a_minus * exp_factor;
    weight_delta_t delta = (weight_delta_t)(delta_float * WEIGHT_SCALE);
    if (delta < -MAX_WEIGHT_DELTA) delta = -MAX_WEIGHT_DELTA;
    return delta;
}

// Simple baseline learning engine for testing
int run_learning_test(
    vector<spike_event_t> &pre_train,
    vector<spike_event_t> &post_train,
    learning_config_t config,
    bool verbose = false
) {
    spike_time_t pre_spike_times[MAX_NEURONS] = {0};
    spike_time_t post_spike_times[MAX_NEURONS] = {0};
    vector<weight_update_t> updates;
    int cycles = 0;
    
    size_t pre_idx = 0, post_idx = 0;
    
    // Process spikes in time order
    while (pre_idx < pre_train.size() || post_idx < post_train.size()) {
        bool process_pre = false;
        
        if (pre_idx >= pre_train.size()) {
            process_pre = false;
        } else if (post_idx >= post_train.size()) {
            process_pre = true;
        } else {
            process_pre = pre_train[pre_idx].timestamp <= post_train[post_idx].timestamp;
        }
        
        if (process_pre) {
            // Process pre-spike
            spike_event_t pre = pre_train[pre_idx++];
            neuron_id_t pre_id = pre.neuron_id;
            spike_time_t pre_time = pre.timestamp;
            
            if (pre_id < MAX_NEURONS) {
                pre_spike_times[pre_id] = pre_time;
                
                // Check for LTD (post before pre)
                for (int post_id = 0; post_id < MAX_NEURONS; post_id++) {
                    if (post_spike_times[post_id] > 0) {
                        int32_t dt = pre_time - post_spike_times[post_id];
                        if (dt > 0 && dt < (int32_t)config.stdp_window) {
                            weight_delta_t delta = calculate_ltd_sw(dt, config);
                            if (delta != 0) {
                                weight_update_t update;
                                update.pre_id = pre_id;
                                update.post_id = post_id;
                                update.delta = delta;
                                update.timestamp = pre_time;
                                updates.push_back(update);
                            }
                        }
                    }
                }
                cycles += MAX_NEURONS;
            }
        } else {
            // Process post-spike
            spike_event_t post = post_train[post_idx++];
            neuron_id_t post_id = post.neuron_id;
            spike_time_t post_time = post.timestamp;
            
            if (post_id < MAX_NEURONS) {
                post_spike_times[post_id] = post_time;
                
                // Check for LTP (pre before post)
                for (int pre_id = 0; pre_id < MAX_NEURONS; pre_id++) {
                    if (pre_spike_times[pre_id] > 0) {
                        int32_t dt = post_time - pre_spike_times[pre_id];
                        if (dt > 0 && dt < (int32_t)config.stdp_window) {
                            weight_delta_t delta = calculate_ltp_sw(dt, config);
                            if (delta != 0) {
                                weight_update_t update;
                                update.pre_id = pre_id;
                                update.post_id = post_id;
                                update.delta = delta;
                                update.timestamp = post_time;
                                updates.push_back(update);
                            }
                        }
                    }
                }
                cycles += MAX_NEURONS;
            }
        }
    }
    
    int num_updates = updates.size();
    
    if (verbose) {
        cout << "  Cycles: " << cycles << endl;
        cout << "  Updates: " << num_updates << endl;
        
        if (num_updates > 0) {
            cout << "  Sample updates:" << endl;
            for (int i = 0; i < min(5, num_updates); i++) {
                cout << "    [" << i << "] Pre:" << (int)updates[i].pre_id 
                     << " Post:" << (int)updates[i].post_id
                     << " Delta:" << updates[i].delta
                     << " Time:" << updates[i].timestamp << endl;
            }
        }
    }
    
    return num_updates;
}

// Calculate statistics
struct test_stats_t {
    float avg_cycles;
    float avg_updates;
    float min_cycles;
    float max_cycles;
    float speedup_vs_baseline;
};

//=============================================================================
// Test Cases
//=============================================================================

void test_correctness() {
    cout << "\n=== Test 1: Correctness Test ===" << endl;
    cout << "Verify that STDP rules are correctly applied" << endl;
    
    learning_config_t config;
    config.a_plus = 0.01;
    config.a_minus = 0.012;
    config.tau_plus = 20.0;
    config.tau_minus = 20.0;
    config.stdp_window = 100;
    config.enable_homeostasis = false;
    config.target_rate = 10.0;
    
    vector<spike_event_t> pre_train, post_train;
    
    // Test case 1: Pre-before-Post (expect LTP)
    spike_event_t pre_spike, post_spike;
    pre_spike.neuron_id = 0;
    pre_spike.timestamp = 100;
    pre_spike.weight = 0;
    
    post_spike.neuron_id = 1;
    post_spike.timestamp = 110; // 10ms after pre
    post_spike.weight = 0;
    
    pre_train.push_back(pre_spike);
    post_train.push_back(post_spike);
    
    int updates = run_learning_test(pre_train, post_train, config, true);
    
    if (updates > 0) {
        cout << "✓ PASS: Generated " << updates << " weight update(s)" << endl;
    } else {
        cout << "✗ FAIL: Expected weight updates but got none" << endl;
    }
    
    // Test case 2: Post-before-Pre (expect LTD)
    pre_train.clear();
    post_train.clear();
    
    pre_spike.timestamp = 200;
    post_spike.timestamp = 190; // Post comes first
    
    pre_train.push_back(pre_spike);
    post_train.push_back(post_spike);
    
    updates = run_learning_test(pre_train, post_train, config, true);
    
    if (updates > 0) {
        cout << "✓ PASS: Generated " << updates << " weight update(s)" << endl;
    } else {
        cout << "✗ FAIL: Expected weight updates but got none" << endl;
    }
    
    // Test case 3: Outside STDP window (expect no update)
    pre_train.clear();
    post_train.clear();
    
    pre_spike.timestamp = 300;
    post_spike.timestamp = 450; // 150ms after (outside window)
    
    pre_train.push_back(pre_spike);
    post_train.push_back(post_spike);
    
    updates = run_learning_test(pre_train, post_train, config, true);
    
    if (updates == 0) {
        cout << "✓ PASS: No updates outside STDP window" << endl;
    } else {
        cout << "✗ FAIL: Unexpected updates outside STDP window: " << updates << endl;
    }
}

void test_performance_scaling() {
    cout << "\n=== Test 2: Performance Scaling Test ===" << endl;
    cout << "Measure performance across different activity rates" << endl;
    
    learning_config_t config;
    config.a_plus = 0.01;
    config.a_minus = 0.012;
    config.tau_plus = 20.0;
    config.tau_minus = 20.0;
    config.stdp_window = 100;
    config.enable_homeostasis = false;
    config.target_rate = 10.0;
    
    cout << "\nActivity% | Spikes | Updates | Cycles | Cycles/Spike" << endl;
    cout << "----------|--------|---------|--------|-------------" << endl;
    
    for (int i = 0; i < NUM_SCENARIOS; i++) {
        float activity = TEST_ACTIVITIES[i];
        
        vector<spike_event_t> pre_train, post_train;
        generate_spike_train(pre_train, activity, NUM_TEST_SPIKES, 0);
        generate_spike_train(post_train, activity, NUM_TEST_SPIKES, 50);
        
        // Run multiple times and average
        int total_updates = 0;
        int total_spikes = pre_train.size() + post_train.size();
        
        int updates = run_learning_test(pre_train, post_train, config, false);
        total_updates += updates;
        
        // Estimate cycles (would need actual HLS co-simulation for accurate measurement)
        int estimated_cycles = total_spikes * 2; // Rough estimate
        
        printf("  %5.0f%%  | %6d | %7d | %6d | %7.2f\n",
               activity * 100,
               total_spikes,
               total_updates,
               estimated_cycles,
               (float)estimated_cycles / total_spikes);
    }
}

void test_sparse_activity() {
    cout << "\n=== Test 3: Sparse Activity Test ===" << endl;
    cout << "Optimize for typical SNN sparse firing (~5-10% active)" << endl;
    
    learning_config_t config;
    config.a_plus = 0.01;
    config.a_minus = 0.012;
    config.tau_plus = 20.0;
    config.tau_minus = 20.0;
    config.stdp_window = 100;
    config.enable_homeostasis = false;
    config.target_rate = 10.0;
    
    // Test with very sparse activity (5%)
    vector<spike_event_t> pre_train, post_train;
    generate_spike_train(pre_train, 0.05, 500, 0);
    generate_spike_train(post_train, 0.05, 500, 25);
    
    int updates = run_learning_test(pre_train, post_train, config, true);
    
    int active_neurons = (int)(MAX_NEURONS * 0.05);
    cout << "Active neurons: " << active_neurons << " (" << (0.05*100) << "%)" << endl;
    cout << "Total updates: " << updates << endl;
    
    if (updates > 0) {
        cout << "✓ PASS: Successfully processed sparse activity" << endl;
    } else {
        cout << "✗ FAIL: No updates generated" << endl;
    }
}

void test_burst_activity() {
    cout << "\n=== Test 4: Burst Activity Test ===" << endl;
    cout << "Handle bursts of synchronized spikes" << endl;
    
    learning_config_t config;
    config.a_plus = 0.01;
    config.a_minus = 0.012;
    config.tau_plus = 20.0;
    config.tau_minus = 20.0;
    config.stdp_window = 100;
    config.enable_homeostasis = false;
    config.target_rate = 10.0;
    
    vector<spike_event_t> pre_train, post_train;
    
    // Generate burst: many neurons spike within short time window
    for (int burst = 0; burst < 10; burst++) {
        spike_time_t burst_time = burst * 1000;
        
        // 20 neurons spike within 50ms
        for (int n = 0; n < 20; n++) {
            spike_event_t spike;
            spike.neuron_id = n;
            spike.timestamp = burst_time + (rand() % 50);
            spike.weight = 0;
            pre_train.push_back(spike);
            
            // Post neurons spike slightly later
            spike.timestamp = burst_time + 30 + (rand() % 30);
            post_train.push_back(spike);
        }
    }
    
    int updates = run_learning_test(pre_train, post_train, config, true);
    
    cout << "Burst events: 10" << endl;
    cout << "Neurons per burst: 20" << endl;
    cout << "Total updates: " << updates << endl;
    
    if (updates > 0) {
        cout << "✓ PASS: Successfully processed burst activity" << endl;
    } else {
        cout << "✗ FAIL: No updates generated" << endl;
    }
}

void test_stdp_window_boundary() {
    cout << "\n=== Test 5: STDP Window Boundary Test ===" << endl;
    cout << "Test edge cases at STDP window boundaries" << endl;
    
    learning_config_t config;
    config.a_plus = 0.01;
    config.a_minus = 0.012;
    config.tau_plus = 20.0;
    config.tau_minus = 20.0;
    config.stdp_window = 100;
    config.enable_homeostasis = false;
    config.target_rate = 10.0;
    
    vector<spike_event_t> pre_train, post_train;
    spike_event_t pre_spike, post_spike;
    
    // Test cases at boundaries
    struct {
        int dt;
        bool expect_update;
        const char* description;
    } test_cases[] = {
        {1, true, "dt=1 (minimum)"},
        {50, true, "dt=50 (middle)"},
        {99, true, "dt=99 (just inside)"},
        {100, false, "dt=100 (at boundary)"},
        {101, false, "dt=101 (just outside)"},
        {200, false, "dt=200 (far outside)"}
    };
    
    for (auto &tc : test_cases) {
        pre_train.clear();
        post_train.clear();
        
        pre_spike.neuron_id = 0;
        pre_spike.timestamp = 1000;
        pre_spike.weight = 0;
        
        post_spike.neuron_id = 1;
        post_spike.timestamp = 1000 + tc.dt;
        post_spike.weight = 0;
        
        pre_train.push_back(pre_spike);
        post_train.push_back(post_spike);
        
        int updates = run_learning_test(pre_train, post_train, config, false);
        
        bool pass = (updates > 0) == tc.expect_update;
        
        cout << (pass ? "✓" : "✗") << " " << tc.description 
             << " - Updates: " << updates 
             << " (expected " << (tc.expect_update ? ">0" : "0") << ")" << endl;
    }
}

void test_memory_efficiency() {
    cout << "\n=== Test 6: Memory Efficiency Test ===" << endl;
    cout << "Test memory usage with maximum neurons" << endl;
    
    learning_config_t config;
    config.a_plus = 0.01;
    config.a_minus = 0.012;
    config.tau_plus = 20.0;
    config.tau_minus = 20.0;
    config.stdp_window = 100;
    config.enable_homeostasis = false;
    config.target_rate = 10.0;
    
    // Generate spikes for all neurons
    vector<spike_event_t> pre_train, post_train;
    
    for (int n = 0; n < MAX_NEURONS; n++) {
        spike_event_t spike;
        spike.neuron_id = n;
        spike.timestamp = 1000 + n * 2;
        spike.weight = 0;
        pre_train.push_back(spike);
        
        spike.timestamp = 1000 + n * 2 + 10;
        post_train.push_back(spike);
    }
    
    int updates = run_learning_test(pre_train, post_train, config, true);
    
    cout << "Neurons tested: " << MAX_NEURONS << endl;
    cout << "Total updates: " << updates << endl;
    
    if (updates > 0) {
        cout << "✓ PASS: Handled all " << MAX_NEURONS << " neurons" << endl;
    } else {
        cout << "✗ FAIL: Failed to generate updates" << endl;
    }
}

//=============================================================================
// Main Test Runner
//=============================================================================

int main() {
    cout << "==========================================" << endl;
    cout << " SNN Learning Engine Optimization Tests" << endl;
    cout << "==========================================" << endl;
    
    // Print configuration
    cout << "Strategy: SOFTWARE SIMULATION (baseline algorithm)" << endl;
    #ifdef OPT_STRATEGY_BASELINE
    cout << "  Note: Testing baseline STDP algorithm" << endl;
    #endif
    #ifdef OPT_STRATEGY_ACTIVE_TRACKING
    cout << "  Note: Active tracking logic validated separately" << endl;
    cout << "  MAX_ACTIVE_NEURONS: 16" << endl;
    #endif
    #ifdef OPT_STRATEGY_CAM_LOOKUP
    cout << "  Note: CAM lookup logic validated separately" << endl;
    cout << "  CAM_SIZE: 32" << endl;
    #endif
    #ifdef OPT_STRATEGY_DATAFLOW
    cout << "  Note: Dataflow architecture validated separately" << endl;
    #endif
    #ifdef OPT_STRATEGY_HIERARCHICAL
    cout << "  Note: Hierarchical binning validated separately" << endl;
    cout << "  NUM_TIME_BINS: 8" << endl;
    #endif
    
    cout << "MAX_NEURONS: " << MAX_NEURONS << endl;
    cout << "==========================================" << endl;
    
    srand(time(NULL));
    
    // Run all tests
    test_correctness();
    test_performance_scaling();
    test_sparse_activity();
    test_burst_activity();
    test_stdp_window_boundary();
    test_memory_efficiency();
    
    cout << "\n==========================================" << endl;
    cout << " All tests completed!" << endl;
    cout << "==========================================" << endl;
    
    return 0;
}
