//-----------------------------------------------------------------------------
// Title         : SNN Utility Functions
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_utils.h
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Common utility functions and macros
//-----------------------------------------------------------------------------

#ifndef SNN_UTILS_H
#define SNN_UTILS_H

#include "snn_types.h"
#include <hls_math.h>

// Utility macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLAMP(x,min,max) (MIN(MAX(x,min),max))
#define ABS(x) ((x) < 0 ? -(x) : (x))

// Bit manipulation
#define SET_BIT(val,bit) ((val) | (1 << (bit)))
#define CLEAR_BIT(val,bit) ((val) & ~(1 << (bit)))
#define TOGGLE_BIT(val,bit) ((val) ^ (1 << (bit)))
#define CHECK_BIT(val,bit) (((val) >> (bit)) & 1)

// Mathematical functions
template<typename T>
T saturating_add(T a, T b) {
    #pragma HLS INLINE
    T result = a + b;
    if (b > 0 && result < a) return MAX_WEIGHT;
    if (b < 0 && result > a) return MIN_WEIGHT;
    return result;
}

// Random number generation
class RandomGenerator {
private:
    ap_uint<32> seed;
    
public:
    RandomGenerator(ap_uint<32> init_seed = 0x12345678) : seed(init_seed) {}
    
    ap_uint<32> rand() {
        #pragma HLS INLINE
        // Linear congruential generator
        seed = seed * 1103515245 + 12345;
        return (seed >> 16) & 0x7FFF;
    }
    
    ap_uint<1> rand_bit(ap_uint<8> probability) {
        #pragma HLS INLINE
        // Returns 1 with probability/256 chance
        return (rand() & 0xFF) < probability;
    }
    
    ap_int<8> rand_weight() {
        #pragma HLS INLINE
        return (rand() & 0xFF) - 128;
    }
};

// Performance monitoring
struct performance_counter_t {
    ap_uint<32> cycles;
    ap_uint<32> operations;
    ap_uint<32> stalls;
    
    void reset() {
        #pragma HLS INLINE
        cycles = 0;
        operations = 0;
        stalls = 0;
    }
    
    void update(bool active, bool stalled) {
        #pragma HLS INLINE
        cycles++;
        if (active) operations++;
        if (stalled) stalls++;
    }
    
    ap_uint<8> get_efficiency() {
        #pragma HLS INLINE
        if (cycles == 0) return 0;
        return (operations * 100) / cycles;
    }
};

// Circular buffer for spike history
template<int SIZE>
class SpikeHistory {
private:
    spike_event_t buffer[SIZE];
    ap_uint<16> write_ptr;
    ap_uint<16> count;
    
public:
    void reset() {
        #pragma HLS INLINE
        write_ptr = 0;
        count = 0;
    }
    
    void add(spike_event_t spike) {
        #pragma HLS INLINE
        buffer[write_ptr] = spike;
        write_ptr = (write_ptr + 1) % SIZE;
        if (count < SIZE) count++;
    }
    
    bool find_recent(neuron_id_t id, spike_time_t &time, spike_time_t window) {
        #pragma HLS INLINE
        for (int i = 0; i < MIN(count, SIZE); i++) {
            #pragma HLS UNROLL factor=4
            int idx = (write_ptr - 1 - i + SIZE) % SIZE;
            if (buffer[idx].neuron_id == id && 
                buffer[idx].timestamp >= time - window) {
                time = buffer[idx].timestamp;
                return true;
            }
        }
        return false;
    }
};

// Debug printing (only enabled in simulation)
#if ENABLE_DEBUG_PRINTS
    #define DEBUG_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...)
#endif

#endif // SNN_UTILS_H
