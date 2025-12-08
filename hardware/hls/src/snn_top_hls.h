//-----------------------------------------------------------------------------
// Title         : SNN Top-Level HLS Header
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_top_hls.h
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Header for unified HLS top-level with on-chip learning
//-----------------------------------------------------------------------------

#ifndef SNN_TOP_HLS_H
#define SNN_TOP_HLS_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

//=============================================================================
// Configuration
//=============================================================================
const int MAX_NEURONS = 256;
const int MAX_SYNAPSES = 65536;  // 256x256

const int WEIGHT_WIDTH = 8;
const int NEURON_ID_WIDTH = 10;  // Supports up to 1024 encoded channels
const int TIMESTAMP_WIDTH = 32;

const int WEIGHT_SCALE = 128;
const int VERSION_ID = 0x20251205;

//=============================================================================
// Operation Modes (use #define for switch-case compatibility)
//=============================================================================
#define MODE_INFERENCE 0       // Forward inference only
#define MODE_TRAIN_STDP 1      // On-chip STDP learning
#define MODE_CHECKPOINT 2      // Stream weights to DDR/PS

//=============================================================================
// Basic Data Types
//=============================================================================
typedef ap_uint<NEURON_ID_WIDTH> neuron_id_t;
typedef ap_int<WEIGHT_WIDTH> weight_t;
typedef ap_int<16> weight_delta_t;
typedef ap_uint<TIMESTAMP_WIDTH> spike_time_t;

const weight_t MAX_WEIGHT = 127;
const weight_t MIN_WEIGHT = -128;

//=============================================================================
// AXI4-Stream Types
//=============================================================================
// Spike packet (AER over AXIS32):
// [31:18] timestamp(14b), [17:10] weight(8b, two's complement), [9:0] neuron_id
typedef ap_axiu<32, 1, 1, 1> axis_spike_t;

// Weight packet: [31:24] reserved, [23:16] weight, [15:8] post_id, [7:0] pre_id
typedef ap_axiu<32, 1, 1, 1> axis_weight_t;

//=============================================================================
// Learning Parameters Structure
//=============================================================================
struct learning_params_t {
    // STDP parameters
    ap_fixed<16,8> a_plus;          // LTP amplitude (0.0 ~ 1.0)
    ap_fixed<16,8> a_minus;         // LTD amplitude (0.0 ~ 1.0)
    ap_uint<16> tau_plus;           // LTP time constant (timesteps)
    ap_uint<16> tau_minus;          // LTD time constant (timesteps)
    ap_uint<16> stdp_window;        // STDP window size (timesteps)
    ap_fixed<16,8> learning_rate;   // Global learning rate
    
    // R-STDP parameters
    bool rstdp_enable;              // Enable reward-modulated STDP
    ap_fixed<16,8> trace_decay;     // Eligibility trace decay rate
    ap_fixed<16,8> reward_scale;    // Reward signal scaling
};

//=============================================================================
// Weight Update Structure
//=============================================================================
struct weight_update_t {
    neuron_id_t pre_id;
    neuron_id_t post_id;
    weight_delta_t delta;
    spike_time_t timestamp;
};

//=============================================================================
// Encoder Configuration
//=============================================================================
const int MAX_INPUT_CHANNELS = 784;  // 28x28 MNIST default

typedef ap_uint<8> pixel_t;          // Single pixel value (0-255)

struct input_data_t {
    pixel_t pixels[MAX_INPUT_CHANNELS];  // Complete input frame
};

// Encoding types (4-bit allocation for future expansion)
#define ENC_NONE           0  // No encoding - direct spike input
#define ENC_RATE_POISSON   1  // Rate coding (Poisson-like)
#define ENC_LATENCY        2  // Latency coding (intensity to latency)
#define ENC_DELTA_SIGMA    3  // Delta-sigma modulation
// Reserved: 4-15 for future encoding methods

struct encoder_config_t {
    ap_uint<4> encoding_type;        // 4-bit: 0=none, 1=rate, 2=latency, 3=delta-sigma
    bool two_neuron_enable;          // Enable two-neuron encoding (pos/neg split)
    ap_uint<8> baseline;             // Baseline for two-neuron (default: 128 for unsigned)
    ap_uint<16> num_steps;           // Total simulation time steps (for rate/latency normalization)
    ap_uint<16> rate_scale;          // Rate coding: threshold scale
    ap_uint<16> latency_window;      // Latency coding: time window (timesteps)
    ap_uint<16> delta_threshold;     // Delta-sigma: threshold for integration
    ap_uint<16> delta_decay;         // Delta-sigma: decay rate
    ap_uint<16> num_channels;        // Number of input channels (will be doubled if two_neuron_enable)
    weight_t default_weight;         // Default spike weight
};

//=============================================================================
// Main Function Declaration
//=============================================================================
void snn_top_hls(
    // AXI4-Lite Control Interface
    ap_uint<32> ctrl_reg,
    ap_uint<32> config_reg,
    learning_params_t learning_params,
    ap_uint<32> &status_reg,
    ap_uint<32> &spike_count_reg,
    ap_uint<32> &weight_sum_reg,
    ap_uint<32> &version_reg,
    
    // AXI4-Stream Spike Input (from PS)
    hls::stream<axis_spike_t> &s_axis_spikes,
    
    // AXI4-Stream Spike Output (to PS)
    hls::stream<axis_spike_t> &m_axis_spikes,
    
    // AXI4-Stream Weight Read (for debugging)
    hls::stream<axis_weight_t> &m_axis_weights,
    
    // Reward signal input (for R-STDP)
    ap_int<8> reward_signal,
    
    // Verilog Interface - Spike Input (to SNN core)
    ap_uint<1> &spike_in_valid,
    ap_uint<8> &spike_in_neuron_id,
    ap_int<8> &spike_in_weight,
    ap_uint<1> spike_in_ready,
    
    // Verilog Interface - Spike Output (from SNN core)
    ap_uint<1> spike_out_valid,
    ap_uint<8> spike_out_neuron_id,
    ap_int<8> spike_out_weight,
    ap_uint<1> &spike_out_ready,
    
    // Verilog Interface - Control signals
    ap_uint<1> &snn_enable,
    ap_uint<1> &snn_reset,
    ap_uint<16> &threshold_out,
    ap_uint<16> &leak_rate_out,
    
    // Verilog Interface - Status signals
    ap_uint<1> snn_ready,
    ap_uint<1> snn_busy
);

//=============================================================================
// Utility Functions
//=============================================================================
void write_weight(neuron_id_t pre_id, neuron_id_t post_id, weight_t weight);
weight_t read_weight(neuron_id_t pre_id, neuron_id_t post_id);
void load_weights_from_stream(hls::stream<axis_weight_t> &weight_stream, ap_uint<16> num_weights);

#endif // SNN_TOP_HLS_H
