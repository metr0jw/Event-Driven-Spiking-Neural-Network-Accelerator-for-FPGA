# Learning Engine Optimization Guide

## Current Implementation Analysis

### Performance Characteristics
- **Latency**: 128 cycles for 64 neurons (II=2)
- **Memory Access**: Sequential scan of all neurons
- **Throughput**: Limited by O(N) scan per spike

### Bottleneck Identification
```cpp
// Current approach: Full scan
for (int post_id = 0; post_id < MAX_NEURONS; post_id++) {
    if (post_spike_times[post_id] > 0) {
        // Check STDP pairing
    }
}
```

**Problem**: 
- Scans all 64 neurons even if only 5 are active
- No early termination
- Poor cache locality for sparse activity

## Optimization Strategies

### 1. Active Neuron Tracking (Recommended for Small Networks)

**Concept**: Maintain a list of neurons that have spiked recently.

```cpp
// Add to static variables
static neuron_id_t active_pre_neurons[MAX_ACTIVE_NEURONS];
static neuron_id_t active_post_neurons[MAX_ACTIVE_NEURONS];
static ap_uint<8> num_active_pre = 0;
static ap_uint<8> num_active_post = 0;

#pragma HLS ARRAY_PARTITION variable=active_pre_neurons complete
#pragma HLS ARRAY_PARTITION variable=active_post_neurons complete

// When processing pre-spike
if (pre_id < MAX_NEURONS) {
    pre_spike_times[pre_id] = pre_time;
    
    // Add to active list if not already present
    bool found = false;
    CHECK_ACTIVE: for (int i = 0; i < num_active_pre; i++) {
        #pragma HLS UNROLL
        if (active_pre_neurons[i] == pre_id) {
            found = true;
            break;
        }
    }
    
    if (!found && num_active_pre < MAX_ACTIVE_NEURONS) {
        active_pre_neurons[num_active_pre] = pre_id;
        num_active_pre++;
    }
    
    // Only scan active post neurons
    LTD_LOOP: for (int i = 0; i < num_active_post; i++) {
        #pragma HLS PIPELINE II=1
        neuron_id_t post_id = active_post_neurons[i];
        spike_time_t post_time = post_spike_times[post_id];
        
        if (post_time > 0) {
            ap_int<32> dt = pre_time - post_time;
            
            if (dt > 0 && dt < config.stdp_window) {
                weight_delta_t delta = calculate_ltd(dt, config);
                
                if (delta != 0) {
                    weight_update_t update;
                    update.pre_id = pre_id;
                    update.post_id = post_id;
                    update.delta = delta;
                    update.timestamp = pre_time;
                    
                    weight_updates.write(update);
                    update_counter++;
                }
            }
        }
    }
}
```

**Performance**:
- Latency: `num_active_neurons × II` cycles (e.g., 5 active = 5 cycles)
- Speedup: 12.8× for 5 active out of 64 neurons
- Resource cost: 2 × MAX_ACTIVE_NEURONS registers

**Trade-offs**:
- ✅ Dramatic speedup for sparse activity (<20% active)
- ✅ Deterministic performance
- ❌ Extra bookkeeping overhead
- ❌ Limited by MAX_ACTIVE_NEURONS

### 2. Time-Window Based Pruning

**Concept**: Periodically clear old spike times outside STDP window.

```cpp
// Add to static variables
static ap_uint<32> last_cleanup_time = 0;

// Periodic cleanup (every 1000 cycles)
if (pre_time - last_cleanup_time > config.stdp_window * 2) {
    CLEANUP_LOOP: for (int i = 0; i < MAX_NEURONS; i++) {
        #pragma HLS PIPELINE II=1
        
        // Clear old pre-spike times
        if (pre_spike_times[i] > 0 && 
            pre_time - pre_spike_times[i] > config.stdp_window) {
            pre_spike_times[i] = 0;
        }
        
        // Clear old post-spike times
        if (post_spike_times[i] > 0 && 
            pre_time - post_spike_times[i] > config.stdp_window) {
            post_spike_times[i] = 0;
        }
    }
    last_cleanup_time = pre_time;
}
```

**Benefits**:
- Reduces number of valid entries over time
- Simple to implement
- No additional memory

**Performance**: Amortized cost, cleanup every N spikes

### 3. CAM-Based Lookup (For Large Networks)

**Concept**: Use Content Addressable Memory for O(1) spike time lookup.

```cpp
// Requires Xilinx CAM IP or custom implementation
// Pseudo-code:

struct cam_entry_t {
    neuron_id_t neuron_id;
    spike_time_t spike_time;
    bool valid;
};

static cam_entry_t spike_cam[CAM_SIZE];
#pragma HLS ARRAY_PARTITION variable=spike_cam complete

// Insert
void cam_insert(neuron_id_t id, spike_time_t time) {
    #pragma HLS INLINE
    
    // Parallel search for empty slot or matching ID
    CAM_INSERT: for (int i = 0; i < CAM_SIZE; i++) {
        #pragma HLS UNROLL
        if (!spike_cam[i].valid || spike_cam[i].neuron_id == id) {
            spike_cam[i].neuron_id = id;
            spike_cam[i].spike_time = time;
            spike_cam[i].valid = true;
            break;
        }
    }
}

// Lookup all matching entries in parallel
void cam_lookup(neuron_id_t pre_id, spike_time_t pre_time,
                hls::stream<weight_update_t> &updates) {
    #pragma HLS INLINE
    
    CAM_LOOKUP: for (int i = 0; i < CAM_SIZE; i++) {
        #pragma HLS UNROLL factor=8
        
        if (spike_cam[i].valid) {
            ap_int<32> dt = pre_time - spike_cam[i].spike_time;
            
            if (dt > 0 && dt < config.stdp_window) {
                // Generate update
                weight_update_t update;
                update.pre_id = pre_id;
                update.post_id = spike_cam[i].neuron_id;
                update.delta = calculate_ltd(dt, config);
                update.timestamp = pre_time;
                
                if (update.delta != 0) {
                    updates.write(update);
                }
            }
        }
    }
}
```

**Performance**:
- Latency: O(1) for lookup, fully parallel
- Throughput: 1 cycle per spike (with UNROLL)
- Resource cost: CAM_SIZE × (neuron_id + spike_time + 1 bit)

**Trade-offs**:
- ✅ Constant-time lookup
- ✅ Scales to large networks
- ❌ High resource usage (LUT-based CAM)
- ❌ Limited CAM size requires eviction policy

### 4. Dataflow Parallel Processing

**Concept**: Process pre and post spikes in parallel using dataflow.

```cpp
void snn_learning_engine(
    // ... parameters
) {
    #pragma HLS DATAFLOW
    
    // Separate pre and post spike processing
    process_pre_spikes(pre_spikes, post_spike_times, config, weight_updates);
    process_post_spikes(post_spikes, pre_spike_times, config, weight_updates);
    
    // Merge weight updates
    merge_weight_updates(pre_updates, post_updates, weight_updates);
}

void process_pre_spikes(
    hls::stream<spike_event_t> &pre_spikes,
    spike_time_t post_spike_times[MAX_NEURONS],
    learning_config_t config,
    hls::stream<weight_update_t> &updates
) {
    #pragma HLS INLINE off
    
    if (!pre_spikes.empty()) {
        spike_event_t pre_event = pre_spikes.read();
        // ... LTD processing
    }
}

void process_post_spikes(
    hls::stream<spike_event_t> &post_spikes,
    spike_time_t pre_spike_times[MAX_NEURONS],
    learning_config_t config,
    hls::stream<weight_update_t> &updates
) {
    #pragma HLS INLINE off
    
    if (!post_spikes.empty()) {
        spike_event_t post_event = post_spikes.read();
        // ... LTP processing
    }
}
```

**Benefits**:
- 2× throughput (parallel pre/post processing)
- Better pipeline utilization
- Scales with number of processing units

**Challenges**:
- Requires careful memory partitioning
- More complex control logic
- Potential write conflicts on weight_updates stream

### 5. Hierarchical Binning

**Concept**: Organize neurons into time bins for faster temporal matching.

```cpp
#define NUM_TIME_BINS 8
#define BIN_SIZE (STDP_WINDOW / NUM_TIME_BINS)

struct time_bin_t {
    neuron_id_t neurons[MAX_NEURONS / NUM_TIME_BINS];
    ap_uint<8> count;
};

static time_bin_t spike_bins[NUM_TIME_BINS];
#pragma HLS ARRAY_PARTITION variable=spike_bins complete

// Insert spike into appropriate time bin
int bin_idx = (spike_time / BIN_SIZE) % NUM_TIME_BINS;
spike_bins[bin_idx].neurons[spike_bins[bin_idx].count] = neuron_id;
spike_bins[bin_idx].count++;

// Search only relevant bins
for (int bin = 0; bin < NUM_TIME_BINS; bin++) {
    int time_diff = abs(pre_time - bin * BIN_SIZE);
    
    if (time_diff < STDP_WINDOW) {
        // Search this bin
        for (int i = 0; i < spike_bins[bin].count; i++) {
            // Check STDP pairing
        }
    }
}
```

**Performance**:
- Reduces search space by factor of NUM_TIME_BINS
- Balances memory and computation
- Good for medium-sized networks (64-256 neurons)

## Recommended Implementation Strategy

### For Small Networks (≤64 neurons)
✅ **Active Neuron Tracking + Time-Window Pruning**

```cpp
// Combination approach
#define MAX_ACTIVE_NEURONS 16

// Track active neurons
static neuron_id_t active_post[MAX_ACTIVE_NEURONS];
static ap_uint<8> num_active = 0;

// Periodic cleanup
if (cleanup_needed) {
    prune_old_spikes();
}

// Fast scan of active list
for (int i = 0; i < num_active; i++) {
    check_stdp_pairing(active_post[i]);
}
```

**Expected Performance**:
- 90% activity: 58 cycles (vs 128 baseline) → 2.2× speedup
- 50% activity: 32 cycles → 4× speedup
- 10% activity: 6 cycles → 21× speedup

### For Medium Networks (64-256 neurons)
✅ **Hierarchical Binning + Active Tracking**

Combine time-based binning with active neuron lists for best balance.

### For Large Networks (>256 neurons)
✅ **CAM-Based Lookup + Dataflow**

Use CAM for constant-time lookup and dataflow for parallelism.

## Resource Comparison

| Optimization | LUTs | FFs | BRAM | DSP | Speedup |
|-------------|------|-----|------|-----|---------|
| Baseline | 2K | 3K | 2 | 4 | 1× |
| Active Track | 2.5K | 3.5K | 2 | 4 | 4-20× |
| Time Pruning | 2.2K | 3.2K | 2 | 4 | 1.5-2× |
| CAM-based | 8K | 4K | 0 | 4 | 20-50× |
| Dataflow | 4K | 6K | 4 | 8 | 2× |
| Hierarchical | 3K | 4K | 2.5 | 4 | 3-8× |

## Implementation Priority

### Phase 1: Quick Wins (1 week)
1. ✅ Add spike time caching (done in current code)
2. Implement active neuron counter
3. Add early exit conditions

### Phase 2: Medium Effort (2 weeks)
1. Implement active neuron tracking
2. Add time-window pruning
3. Benchmark on real spike trains

### Phase 3: Advanced (4 weeks)
1. Evaluate CAM IP integration
2. Implement dataflow architecture
3. Full system optimization

## Testing Strategy

```cpp
// Synthetic spike patterns for testing
void generate_test_spikes(
    float activity_rate,     // 0.0 to 1.0
    int num_spikes,
    hls::stream<spike_event_t> &spikes
) {
    for (int i = 0; i < num_spikes; i++) {
        spike_event_t spike;
        
        // Random neuron (Poisson distribution)
        spike.neuron_id = rand() % (int)(MAX_NEURONS * activity_rate);
        spike.timestamp = i * 100; // 100 cycle intervals
        
        spikes.write(spike);
    }
}

// Benchmark harness
void benchmark_learning_engine() {
    float activities[] = {0.1, 0.25, 0.5, 0.75, 1.0};
    
    for (float activity : activities) {
        auto start = get_cycle_count();
        
        generate_test_spikes(activity, 1000, pre_spikes);
        generate_test_spikes(activity, 1000, post_spikes);
        
        // Run learning engine
        for (int i = 0; i < 2000; i++) {
            snn_learning_engine(...);
        }
        
        auto end = get_cycle_count();
        
        printf("Activity %.0f%%: %d cycles\n", 
               activity * 100, end - start);
    }
}
```

## Conclusion

For your current design (64 neurons):
1. **Immediate**: Implement active neuron tracking → 4-20× speedup
2. **Next step**: Add time-window pruning → Additional 1.5× speedup
3. **Future**: Consider CAM-based for scaling beyond 256 neurons

The combination of active tracking + pruning gives the best ROI for SNN applications with sparse activity (typical 5-20% firing rate).
