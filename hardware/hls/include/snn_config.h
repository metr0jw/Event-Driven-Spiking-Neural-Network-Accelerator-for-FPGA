//-----------------------------------------------------------------------------
// Title         : SNN Configuration Parameters
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_config.h
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : System-wide configuration parameters
//-----------------------------------------------------------------------------

#ifndef SNN_CONFIG_H
#define SNN_CONFIG_H

// Hardware constraints (based on PYNQ-Z2 resources)
#define PYNQ_Z2_LUT_COUNT 53200
#define PYNQ_Z2_FF_COUNT 106400
#define PYNQ_Z2_BRAM_KB 630
#define PYNQ_Z2_DSP_COUNT 220

// Default network parameters
#define DEFAULT_LEAK_RATE 10
#define DEFAULT_THRESHOLD 1000
#define DEFAULT_REFRACTORY_PERIOD 20

// STDP default parameters
#define DEFAULT_A_PLUS 0.01
#define DEFAULT_A_MINUS 0.01
#define DEFAULT_TAU_PLUS 20.0
#define DEFAULT_TAU_MINUS 20.0
#define DEFAULT_STDP_WINDOW 100

// Performance optimization settings
#define PIPELINE_DEPTH 8
#define UNROLL_FACTOR 4
#define PARTITION_FACTOR 8

// Memory allocation
#define WEIGHT_MEM_SIZE (MAX_NEURONS * MAX_NEURONS)
#define SPIKE_BUFFER_SIZE 1024
#define INPUT_BUFFER_SIZE 2048
#define OUTPUT_BUFFER_SIZE 1024

// Timing constraints
#define TARGET_CLOCK_PERIOD 10  // 100MHz
#define MAX_LATENCY 1000

// Debug settings
#define ENABLE_DEBUG_PRINTS 0
#define ENABLE_PERFORMANCE_COUNTERS 1

#endif // SNN_CONFIG_H
