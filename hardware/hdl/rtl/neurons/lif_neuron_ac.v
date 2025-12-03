//-----------------------------------------------------------------------------
// Title         : Energy-Efficient AC-based LIF Neuron (Verilog-2001)
// Project       : PYNQ-Z2 SNN Accelerator
// File          : lif_neuron_ac.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Description   : LIF neuron using Accumulate-only (AC) operations
//                 NO multiply operations - only additions/subtractions
//                 
//                 Energy Analysis:
//                 - ANN MAC: ~4.6pJ per operation (45nm technology)
//                 - SNN AC:  ~0.9pJ per operation (45nm technology)
//                 - Energy savings: ~5x per synaptic operation
//                 
//                 Key insight: In SNNs, spikes are binary (0 or 1)
//                 - spike=1: accumulate weight (AC operation)
//                 - spike=0: skip operation entirely (sparse processing)
//                 This eliminates the multiply in MAC = Multiply + Accumulate
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module lif_neuron_ac #(
    parameter NEURON_ID         = 0,
    parameter DATA_WIDTH        = 16,    // Q8.8 fixed-point membrane potential
    parameter WEIGHT_WIDTH      = 8,     // INT8 weights
    parameter THRESHOLD_WIDTH   = 16,    // Q8.8 threshold
    parameter LEAK_SHIFT        = 4,     // Leak = vmem >> LEAK_SHIFT (no multiply!)
    parameter REFRAC_CYCLES     = 5      // Refractory period in cycles
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         enable,
    
    //=========================================================================
    // AC-based Synaptic Input Interface
    // Key: spike_valid indicates binary spike (1=spike occurred)
    // When spike_valid=1, we ADD weight to membrane (AC operation)
    // When spike_valid=0, NO operation needed (energy saved via sparsity)
    //=========================================================================
    input  wire                         spike_valid,      // Binary spike input
    input  wire signed [WEIGHT_WIDTH-1:0] weight,         // Pre-loaded weight
    input  wire                         is_excitatory,    // 1: add, 0: subtract
    
    // Configurable parameters
    input  wire [THRESHOLD_WIDTH-1:0]   threshold,
    input  wire [DATA_WIDTH-1:0]        reset_potential,
    
    // Outputs
    output reg                          spike_out,
    output reg  [DATA_WIDTH-1:0]        membrane_potential,
    output wire                         is_refractory,
    
    // Statistics for energy monitoring
    output reg  [31:0]                  ac_operation_count,  // Number of AC ops
    output reg  [31:0]                  spike_skip_count     // Skipped due to sparsity
);

    //=========================================================================
    // Internal State
    //=========================================================================
    reg signed [DATA_WIDTH-1:0] v_mem;           // Membrane potential (signed)
    reg [3:0]                   refrac_counter;  // Refractory counter
    
    assign is_refractory = (refrac_counter > 0);
    
    //=========================================================================
    // AC-based Synaptic Integration (NO MULTIPLY!)
    // 
    // Traditional ANN: output = sum(input[i] * weight[i])  -- MAC operations
    // Our SNN:         output = sum(weight[i]) where spike[i]=1  -- AC only!
    //
    // Energy benefit: Eliminate ~3.7pJ per operation (multiply cost)
    //=========================================================================
    wire signed [DATA_WIDTH-1:0] weight_extended;
    wire signed [DATA_WIDTH:0]   v_mem_after_spike;
    wire signed [DATA_WIDTH:0]   v_mem_after_leak;
    
    // Sign-extend weight to membrane potential width
    assign weight_extended = {{(DATA_WIDTH-WEIGHT_WIDTH){weight[WEIGHT_WIDTH-1]}}, weight};
    
    // AC operation: Add or subtract weight based on synapse type
    // This is the CORE energy savings - no multiply needed!
    assign v_mem_after_spike = is_excitatory ? 
                               (v_mem + weight_extended) :  // Excitatory: ADD
                               (v_mem - weight_extended);   // Inhibitory: SUBTRACT
    
    //=========================================================================
    // Leak Implementation using SHIFT (NO MULTIPLY!)
    // 
    // Traditional: v_mem = v_mem * decay_factor
    // Our design:  v_mem = v_mem - (v_mem >> LEAK_SHIFT)
    //
    // Approximation: decay ≈ 1 - 2^(-LEAK_SHIFT)
    // LEAK_SHIFT=4: decay ≈ 0.9375
    // LEAK_SHIFT=3: decay ≈ 0.875
    // LEAK_SHIFT=5: decay ≈ 0.96875
    //=========================================================================
    wire signed [DATA_WIDTH-1:0] leak_amount;
    assign leak_amount = v_mem >>> LEAK_SHIFT;  // Arithmetic right shift
    assign v_mem_after_leak = v_mem - leak_amount;
    
    //=========================================================================
    // Main Neuron Dynamics
    //=========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_mem <= 0;
            refrac_counter <= 0;
            spike_out <= 1'b0;
            membrane_potential <= 0;
            ac_operation_count <= 0;
            spike_skip_count <= 0;
            
        end else if (enable) begin
            spike_out <= 1'b0;  // Default: no output spike
            
            if (refrac_counter > 0) begin
                //=============================================================
                // Refractory Period: No integration, just count down
                //=============================================================
                refrac_counter <= refrac_counter - 1;
                v_mem <= reset_potential;
                
            end else begin
                //=============================================================
                // Active State: Process incoming spikes
                //=============================================================
                
                if (spike_valid) begin
                    //=========================================================
                    // ENERGY-EFFICIENT AC OPERATION
                    // Only executed when spike_valid=1
                    // Sparse spikes mean most cycles skip this entirely!
                    //=========================================================
                    ac_operation_count <= ac_operation_count + 1;
                    
                    // Saturating addition/subtraction
                    if (v_mem_after_spike[DATA_WIDTH] != v_mem_after_spike[DATA_WIDTH-1]) begin
                        // Overflow detected - saturate
                        if (v_mem_after_spike[DATA_WIDTH]) begin
                            v_mem <= {1'b1, {(DATA_WIDTH-1){1'b0}}};  // Min value
                        end else begin
                            v_mem <= {1'b0, {(DATA_WIDTH-1){1'b1}}};  // Max value
                        end
                    end else begin
                        v_mem <= v_mem_after_spike[DATA_WIDTH-1:0];
                    end
                    
                end else begin
                    //=========================================================
                    // NO SPIKE - Apply leak only (energy saved!)
                    // This is the sparsity benefit of SNNs
                    //=========================================================
                    spike_skip_count <= spike_skip_count + 1;
                    
                    // Apply leak using shift (no multiply)
                    if (v_mem_after_leak[DATA_WIDTH] != v_mem_after_leak[DATA_WIDTH-1]) begin
                        v_mem <= 0;  // Saturate at zero for leak
                    end else begin
                        v_mem <= v_mem_after_leak[DATA_WIDTH-1:0];
                    end
                end
                
                //=============================================================
                // Spike Generation (Threshold comparison - no multiply)
                //=============================================================
                if ($signed(v_mem) >= $signed(threshold)) begin
                    spike_out <= 1'b1;
                    refrac_counter <= REFRAC_CYCLES;
                    v_mem <= reset_potential;
                end
            end
            
            // Update output
            membrane_potential <= v_mem;
        end
    end

endmodule
