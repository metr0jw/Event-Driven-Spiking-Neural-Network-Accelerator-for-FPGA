//-----------------------------------------------------------------------------
// Title         : SNN Energy Monitor (Verilog-2001)
// Project       : PYNQ-Z2 SNN Accelerator  
// File          : snn_energy_monitor.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Description   : Real-time energy estimation for AC-based SNN processing
//                 
//                 ENERGY MODEL PARAMETERS (45nm technology reference):
//                 ====================================================
//                 Operation       | Energy (pJ) | Notes
//                 ----------------|-------------|---------------------------
//                 32-bit FP MAC   | 4.6         | Floating-point multiply-add
//                 32-bit INT MAC  | 3.1         | Integer multiply-add
//                 32-bit INT AC   | 0.9         | Integer add only (our design)
//                 SRAM read (8KB) | 5.0         | Per access
//                 SRAM write      | 5.0         | Per access
//                 Register access | 0.1         | Flip-flop read/write
//                 
//                 SPARSITY IMPACT:
//                 ================
//                 Sparsity | Active Ops | Energy Ratio vs Dense
//                 ---------|------------|----------------------
//                 50%      | 50%        | 0.5x
//                 90%      | 10%        | 0.1x
//                 95%      | 5%         | 0.05x
//                 99%      | 1%         | 0.01x
//                 
//                 Combined with AC (not MAC): 0.01x × (0.9/4.6) ≈ 0.002x
//                 = 500x energy reduction vs dense ANN with FP MAC!
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module snn_energy_monitor #(
    // Energy values in femtojoules (fJ) for integer arithmetic
    // Multiply by 1000 to convert pJ to fJ
    parameter ENERGY_MAC_FP32   = 4600,     // 4.6 pJ in fJ
    parameter ENERGY_MAC_INT32  = 3100,     // 3.1 pJ in fJ
    parameter ENERGY_AC_INT32   = 900,      // 0.9 pJ in fJ
    parameter ENERGY_SRAM_READ  = 5000,     // 5.0 pJ in fJ
    parameter ENERGY_SRAM_WRITE = 5000,     // 5.0 pJ in fJ
    parameter ENERGY_REG_ACCESS = 100       // 0.1 pJ in fJ
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 enable,
    
    //=========================================================================
    // Operation Counters from SNN Layers
    //=========================================================================
    input  wire [31:0]          ac_ops_conv1d,
    input  wire [31:0]          ac_ops_conv2d,
    input  wire [31:0]          ac_ops_fc,
    input  wire [31:0]          ac_ops_synapse,
    
    input  wire [31:0]          mem_reads_conv1d,
    input  wire [31:0]          mem_reads_conv2d,
    input  wire [31:0]          mem_reads_fc,
    input  wire [31:0]          mem_reads_synapse,
    
    input  wire [31:0]          spikes_input,
    input  wire [31:0]          spikes_output,
    
    //=========================================================================
    // Energy Estimation Outputs (in femtojoules)
    //=========================================================================
    output reg  [63:0]          energy_ac_total,        // Total AC operation energy
    output reg  [63:0]          energy_memory_total,    // Total memory access energy
    output reg  [63:0]          energy_total,           // Grand total
    
    // Per-layer breakdown
    output reg  [63:0]          energy_conv1d,
    output reg  [63:0]          energy_conv2d,
    output reg  [63:0]          energy_fc,
    output reg  [63:0]          energy_synapse,
    
    //=========================================================================
    // Comparison with ANN (hypothetical MAC-based processing)
    //=========================================================================
    output reg  [63:0]          energy_ann_equivalent,  // What ANN would consume
    output reg  [31:0]          energy_savings_ratio,   // SNN/ANN ratio (fixed-point Q16.16)
    
    //=========================================================================
    // Statistics
    //=========================================================================
    output reg  [31:0]          total_ac_ops,
    output reg  [31:0]          total_mem_accesses,
    output reg  [31:0]          total_spikes,
    output reg  [31:0]          sparsity_percent       // Q8.8 format (0-100%)
);

    //=========================================================================
    // Internal Registers
    //=========================================================================
    reg [63:0] acc_ac_ops;
    reg [63:0] acc_mem_reads;
    reg [63:0] acc_spikes_in;
    reg [63:0] acc_spikes_out;
    
    // Previous values for delta calculation
    reg [31:0] prev_ac_conv1d, prev_ac_conv2d, prev_ac_fc, prev_ac_synapse;
    reg [31:0] prev_mem_conv1d, prev_mem_conv2d, prev_mem_fc, prev_mem_synapse;
    reg [31:0] prev_spikes_in, prev_spikes_out;
    
    //=========================================================================
    // Energy Calculation
    //=========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all accumulators
            acc_ac_ops <= 64'b0;
            acc_mem_reads <= 64'b0;
            acc_spikes_in <= 64'b0;
            acc_spikes_out <= 64'b0;
            
            prev_ac_conv1d <= 32'b0;
            prev_ac_conv2d <= 32'b0;
            prev_ac_fc <= 32'b0;
            prev_ac_synapse <= 32'b0;
            
            prev_mem_conv1d <= 32'b0;
            prev_mem_conv2d <= 32'b0;
            prev_mem_fc <= 32'b0;
            prev_mem_synapse <= 32'b0;
            
            prev_spikes_in <= 32'b0;
            prev_spikes_out <= 32'b0;
            
            energy_ac_total <= 64'b0;
            energy_memory_total <= 64'b0;
            energy_total <= 64'b0;
            
            energy_conv1d <= 64'b0;
            energy_conv2d <= 64'b0;
            energy_fc <= 64'b0;
            energy_synapse <= 64'b0;
            
            energy_ann_equivalent <= 64'b0;
            energy_savings_ratio <= 32'h00010000;  // 1.0 in Q16.16
            
            total_ac_ops <= 32'b0;
            total_mem_accesses <= 32'b0;
            total_spikes <= 32'b0;
            sparsity_percent <= 32'b0;
            
        end else if (enable) begin
            //=================================================================
            // Calculate deltas (new operations since last cycle)
            //=================================================================
            
            // Update accumulators with new operations
            acc_ac_ops <= acc_ac_ops + 
                         (ac_ops_conv1d - prev_ac_conv1d) +
                         (ac_ops_conv2d - prev_ac_conv2d) +
                         (ac_ops_fc - prev_ac_fc) +
                         (ac_ops_synapse - prev_ac_synapse);
                         
            acc_mem_reads <= acc_mem_reads +
                            (mem_reads_conv1d - prev_mem_conv1d) +
                            (mem_reads_conv2d - prev_mem_conv2d) +
                            (mem_reads_fc - prev_mem_fc) +
                            (mem_reads_synapse - prev_mem_synapse);
                            
            acc_spikes_in <= acc_spikes_in + (spikes_input - prev_spikes_in);
            acc_spikes_out <= acc_spikes_out + (spikes_output - prev_spikes_out);
            
            // Store current values for next delta
            prev_ac_conv1d <= ac_ops_conv1d;
            prev_ac_conv2d <= ac_ops_conv2d;
            prev_ac_fc <= ac_ops_fc;
            prev_ac_synapse <= ac_ops_synapse;
            
            prev_mem_conv1d <= mem_reads_conv1d;
            prev_mem_conv2d <= mem_reads_conv2d;
            prev_mem_fc <= mem_reads_fc;
            prev_mem_synapse <= mem_reads_synapse;
            
            prev_spikes_in <= spikes_input;
            prev_spikes_out <= spikes_output;
            
            //=================================================================
            // Calculate Energy (AC operations)
            // Energy = num_ops × energy_per_op
            //=================================================================
            
            // Per-layer energy (AC ops + memory reads)
            energy_conv1d <= ac_ops_conv1d * ENERGY_AC_INT32 + 
                            mem_reads_conv1d * ENERGY_SRAM_READ;
                            
            energy_conv2d <= ac_ops_conv2d * ENERGY_AC_INT32 + 
                            mem_reads_conv2d * ENERGY_SRAM_READ;
                            
            energy_fc <= ac_ops_fc * ENERGY_AC_INT32 + 
                        mem_reads_fc * ENERGY_SRAM_READ;
                        
            energy_synapse <= ac_ops_synapse * ENERGY_AC_INT32 + 
                             mem_reads_synapse * ENERGY_SRAM_READ;
            
            // Total energies
            energy_ac_total <= acc_ac_ops * ENERGY_AC_INT32;
            energy_memory_total <= acc_mem_reads * ENERGY_SRAM_READ;
            energy_total <= energy_ac_total + energy_memory_total;
            
            //=================================================================
            // ANN Equivalent Energy (if using MAC instead of AC)
            // Assumes same number of operations but with MAC energy
            //=================================================================
            energy_ann_equivalent <= acc_ac_ops * ENERGY_MAC_INT32 + 
                                    acc_mem_reads * ENERGY_SRAM_READ;
            
            //=================================================================
            // Energy Savings Ratio (Q16.16 fixed-point)
            // ratio = SNN_energy / ANN_energy
            //=================================================================
            if (energy_ann_equivalent > 0) begin
                // Shift left by 16 for Q16.16 format, then divide
                energy_savings_ratio <= (energy_total << 16) / energy_ann_equivalent;
            end
            
            //=================================================================
            // Statistics
            //=================================================================
            total_ac_ops <= acc_ac_ops[31:0];
            total_mem_accesses <= acc_mem_reads[31:0];
            total_spikes <= acc_spikes_in[31:0] + acc_spikes_out[31:0];
            
            // Sparsity calculation (simplified)
            // Would need total possible operations for accurate sparsity
        end
    end

endmodule
