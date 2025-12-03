//-----------------------------------------------------------------------------
// Title         : HLS AXI Wrapper Header
// Project       : PYNQ-Z2 SNN Accelerator
// File          : axi_hls_wrapper.h
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : HLS-based AXI wrapper for Hybrid Verilog/HLS integration
//                 Provides AXI4-Lite control registers and AXI4-Stream
//                 spike interfaces compatible with existing Verilog modules.
//-----------------------------------------------------------------------------

#ifndef AXI_HLS_WRAPPER_H
#define AXI_HLS_WRAPPER_H

#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

//=============================================================================
// Configuration
//=============================================================================
#define MAX_NEURONS         64
#define MAX_AXONS           64
#define WEIGHT_WIDTH        8
#define NEURON_ID_WIDTH     8
#define AXI_DATA_WIDTH      32
#define AXIS_DATA_WIDTH     32

// Memory sizes for weight configuration
#define WEIGHT_MEM_SIZE     (MAX_NEURONS * MAX_AXONS)
#define CONN_MEM_SIZE       (MAX_NEURONS * 8)  // Max 8 connections per neuron

//=============================================================================
// Register Map (Compatible with existing Verilog axi_lite_regs)
//=============================================================================
// Base registers (0x00 - 0x1F)
#define REG_CTRL            0x00    // Control register
#define REG_STATUS          0x04    // Status register (read-only)
#define REG_CONFIG          0x08    // Configuration register
#define REG_SPIKE_COUNT     0x0C    // Spike counter (read-only)
#define REG_LEAK_RATE       0x10    // Neuron leak rate
#define REG_THRESHOLD       0x14    // Neuron threshold
#define REG_REFRAC          0x18    // Refractory period
#define REG_VERSION         0x1C    // Version register (read-only)

// Control register bits
#define CTRL_ENABLE         0x01    // SNN enable
#define CTRL_RESET          0x02    // Soft reset
#define CTRL_CLEAR_COUNTERS 0x04    // Clear spike counters
#define CTRL_LEARNING_EN    0x08    // Enable learning
#define CTRL_IRQ_EN         0x10    // Enable interrupt

// Status register bits
#define STAT_READY          0x01    // System ready
#define STAT_BUSY           0x02    // Processing in progress
#define STAT_ERROR          0x04    // Error flag
#define STAT_OVERFLOW       0x08    // Buffer overflow

// Version info
#define VERSION_ID          0x20241203  // YYYYMMDD format

//=============================================================================
// AXI4-Stream Data Types
//=============================================================================
// Input spike packet format (from PS to PL):
// [31:24] Reserved
// [23:16] Weight (8-bit signed)
// [15:8]  Reserved
// [7:0]   Neuron ID (8-bit)
typedef ap_axiu<AXIS_DATA_WIDTH, 1, 1, 1> axis_spike_t;

// Output spike packet format (from PL to PS):
// [31:16] Timestamp (16-bit)
// [15:8]  Weight (8-bit)
// [7:0]   Neuron ID (8-bit)
typedef ap_axiu<AXIS_DATA_WIDTH, 1, 1, 1> axis_output_t;

//=============================================================================
// Internal Data Structures
//=============================================================================
// Spike event for internal processing
struct spike_in_t {
    ap_uint<NEURON_ID_WIDTH> neuron_id;
    ap_int<WEIGHT_WIDTH> weight;
    bool valid;
};

// Output spike from SNN core
struct spike_out_t {
    ap_uint<NEURON_ID_WIDTH> neuron_id;
    ap_uint<WEIGHT_WIDTH> weight;
    ap_uint<16> timestamp;
    bool valid;
};

// Control/Status registers structure
struct ctrl_regs_t {
    ap_uint<32> ctrl;
    ap_uint<32> config;
    ap_uint<16> leak_rate;
    ap_uint<16> threshold;
    ap_uint<16> refractory_period;
};

struct status_regs_t {
    ap_uint<32> status;
    ap_uint<32> spike_count;
    ap_uint<32> version;
};

//=============================================================================
// Top-level Function Prototype
//=============================================================================
/**
 * @brief HLS AXI Wrapper - Hybrid interface between PS and Verilog SNN core
 * 
 * This module provides:
 * - AXI4-Lite slave for control/status registers
 * - AXI4-Stream slave for input spikes (from PS)
 * - AXI4-Stream master for output spikes (to PS)
 * - Simple signal interface to Verilog SNN modules
 * 
 * @param s_axi_ctrl      AXI4-Lite control registers (bundled)
 * @param s_axis_spikes   AXI4-Stream input spike stream
 * @param m_axis_spikes   AXI4-Stream output spike stream
 * @param spike_in_valid  Spike valid signal to Verilog
 * @param spike_in_data   Spike data to Verilog [neuron_id, weight]
 * @param spike_in_ready  Ready signal from Verilog
 * @param spike_out_valid Output valid from Verilog
 * @param spike_out_data  Output data from Verilog
 * @param spike_out_ready Ready signal to Verilog
 * @param ctrl_signals    Control signals to Verilog
 * @param status_signals  Status signals from Verilog
 */
void axi_hls_wrapper(
    // AXI4-Lite Control Interface (Control registers)
    ap_uint<32> ctrl_reg,
    ap_uint<32> config_reg,
    ap_uint<16> leak_rate,
    ap_uint<16> threshold,
    ap_uint<16> refractory_period,
    ap_uint<32> &status_reg,
    ap_uint<32> &spike_count_out,
    ap_uint<32> &version_reg,
    
    // AXI4-Stream Spike Input (from PS)
    hls::stream<axis_spike_t> &s_axis_spikes,
    
    // AXI4-Stream Spike Output (to PS)
    hls::stream<axis_output_t> &m_axis_spikes,
    
    // Verilog Interface - Spike Input (to SNN core)
    ap_uint<1> &spike_in_valid,
    ap_uint<NEURON_ID_WIDTH> &spike_in_neuron_id,
    ap_int<WEIGHT_WIDTH> &spike_in_weight,
    ap_uint<1> spike_in_ready,
    
    // Verilog Interface - Spike Output (from SNN core)
    ap_uint<1> spike_out_valid,
    ap_uint<NEURON_ID_WIDTH> spike_out_neuron_id,
    ap_int<WEIGHT_WIDTH> spike_out_weight,
    ap_uint<1> &spike_out_ready,
    
    // Verilog Interface - Control signals
    ap_uint<1> &snn_enable,
    ap_uint<1> &snn_reset,
    ap_uint<1> &clear_counters,
    ap_uint<16> &leak_rate_out,
    ap_uint<16> &threshold_out,
    ap_uint<16> &refractory_out,
    
    // Verilog Interface - Status signals
    ap_uint<1> snn_ready,
    ap_uint<1> snn_busy,
    ap_uint<1> snn_error,
    ap_uint<32> snn_spike_count
);

//=============================================================================
// Alternative: Simplified Wrapper for direct AXI passthrough
//=============================================================================
/**
 * @brief Simplified AXI wrapper with minimal logic
 * 
 * Use this version if you want HLS to only handle AXI protocol
 * and pass data directly to Verilog without buffering.
 */
void axi_hls_wrapper_simple(
    // AXI4-Lite registers (all in one bundle)
    ctrl_regs_t ctrl_regs,
    status_regs_t &status_regs,
    
    // AXI4-Stream interfaces
    hls::stream<axis_spike_t> &s_axis_spikes,
    hls::stream<axis_output_t> &m_axis_spikes,
    
    // Direct wire interface to Verilog
    ap_uint<32> &spike_to_verilog,
    ap_uint<1> &spike_valid_to_verilog,
    ap_uint<1> spike_ready_from_verilog,
    ap_uint<32> spike_from_verilog,
    ap_uint<1> spike_valid_from_verilog,
    ap_uint<1> &spike_ready_to_verilog,
    ap_uint<32> &ctrl_to_verilog,
    ap_uint<32> status_from_verilog
);

#endif // AXI_HLS_WRAPPER_H
