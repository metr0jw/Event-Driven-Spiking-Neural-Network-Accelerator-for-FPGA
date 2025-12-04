//-----------------------------------------------------------------------------
// Title         : Leaky Integrate-and-Fire Neuron Model
// Project       : PYNQ-Z2 SNN Accelerator
// File          : lif_neuron.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : LIF neuron with shift-based exponential leak (no multiplier)
//                 Leak uses configurable shift: tau = 1 - 2^(-leak_shift)
//                 leak_rate[2:0] = shift amount (1-7), leak_rate[7:3] = fine tune
//                 Example: shift=3 -> tau=0.875, shift=4 -> tau=0.9375
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module lif_neuron #(
    parameter NEURON_ID         = 0,
    parameter DATA_WIDTH        = 16,    // Width for membrane potential
    parameter WEIGHT_WIDTH      = 8,     // Width for synaptic weights
    parameter THRESHOLD_WIDTH   = 16,    // Width for threshold value
    parameter LEAK_WIDTH        = 8,     // Width for leak config [7:3]=fine, [2:0]=shift
    parameter REFRAC_WIDTH      = 8      // Width for refractory period
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         enable,
    
    // Synaptic input interface
    input  wire                         syn_valid,
    input  wire [WEIGHT_WIDTH-1:0]      syn_weight,
    input  wire                         syn_excitatory,    // 1: excitatory, 0: inhibitory
    
    // Neuron parameters (configurable)
    input  wire [THRESHOLD_WIDTH-1:0]   threshold,
    input  wire [LEAK_WIDTH-1:0]        leak_rate,         // [2:0]=primary shift, [7:3]=secondary shift (0=disabled)
    input  wire [REFRAC_WIDTH-1:0]      refractory_period,
    input  wire                         reset_potential_en,
    input  wire [DATA_WIDTH-1:0]        reset_potential,
    
    // Spike output
    output reg                          spike_out,
    output reg  [DATA_WIDTH-1:0]        membrane_potential,
    
    // Debug/monitoring outputs
    output wire                         is_refractory,
    output wire [REFRAC_WIDTH-1:0]      refrac_count
);

    // Internal state registers
    reg [DATA_WIDTH-1:0]        v_mem;           // Membrane potential
    reg [REFRAC_WIDTH-1:0]      refrac_counter;  // Refractory counter
    reg                         spike_reg;       // Spike register
    
    // Wire assignments for monitoring
    assign is_refractory = (refrac_counter > 0);
    assign refrac_count = refrac_counter;
    
    // =========================================================================
    // Shift-based Exponential Leak (No Multiplier!)
    // =========================================================================
    // tau = 1 - 2^(-shift1) - 2^(-shift2)  (if shift2 != 0)
    // Example configurations:
    //   leak_rate = 8'b00000_011 (shift1=3, shift2=0) -> tau = 0.875
    //   leak_rate = 8'b00100_011 (shift1=3, shift2=4) -> tau = 0.8125  
    //   leak_rate = 8'b00000_100 (shift1=4, shift2=0) -> tau = 0.9375
    //   leak_rate = 8'b00110_011 (shift1=3, shift2=6) -> tau ≈ 0.859
    //   leak_rate = 8'b00110_100 (shift1=4, shift2=6) -> tau ≈ 0.922
    // =========================================================================
    
    wire [2:0] leak_shift1 = leak_rate[2:0];     // Primary shift (1-7)
    wire [4:0] leak_shift2_cfg = leak_rate[7:3]; // Secondary shift config (0=disabled)
    wire [2:0] leak_shift2 = leak_shift2_cfg[2:0]; // Secondary shift value
    wire       leak_shift2_en = (leak_shift2_cfg != 5'd0); // Enable secondary
    
    // Calculate leak amount using barrel shifter
    wire [DATA_WIDTH-1:0] leak_primary;
    wire [DATA_WIDTH-1:0] leak_secondary;
    wire [DATA_WIDTH-1:0] leak_total;
    
    // Primary leak: v_mem >> shift1
    assign leak_primary = (leak_shift1 == 3'd1) ? (v_mem >> 1) :
                          (leak_shift1 == 3'd2) ? (v_mem >> 2) :
                          (leak_shift1 == 3'd3) ? (v_mem >> 3) :
                          (leak_shift1 == 3'd4) ? (v_mem >> 4) :
                          (leak_shift1 == 3'd5) ? (v_mem >> 5) :
                          (leak_shift1 == 3'd6) ? (v_mem >> 6) :
                          (leak_shift1 == 3'd7) ? (v_mem >> 7) :
                          {DATA_WIDTH{1'b0}};  // shift=0: no leak
    
    // Secondary leak: v_mem >> shift2 (for finer tau control)
    assign leak_secondary = (!leak_shift2_en) ? {DATA_WIDTH{1'b0}} :
                            (leak_shift2 == 3'd1) ? (v_mem >> 1) :
                            (leak_shift2 == 3'd2) ? (v_mem >> 2) :
                            (leak_shift2 == 3'd3) ? (v_mem >> 3) :
                            (leak_shift2 == 3'd4) ? (v_mem >> 4) :
                            (leak_shift2 == 3'd5) ? (v_mem >> 5) :
                            (leak_shift2 == 3'd6) ? (v_mem >> 6) :
                            (leak_shift2 == 3'd7) ? (v_mem >> 7) :
                            {DATA_WIDTH{1'b0}};
    
    // Total leak amount (saturating add)
    wire [DATA_WIDTH:0] leak_sum = {1'b0, leak_primary} + {1'b0, leak_secondary};
    assign leak_total = leak_sum[DATA_WIDTH] ? {DATA_WIDTH{1'b1}} : leak_sum[DATA_WIDTH-1:0];
    
    // Leaked membrane potential: v_mem - leak_total (saturate at 0)
    wire [DATA_WIDTH:0] v_mem_leaked_ext = {1'b0, v_mem} - {1'b0, leak_total};
    wire [DATA_WIDTH-1:0] v_mem_leaked = v_mem_leaked_ext[DATA_WIDTH] ? {DATA_WIDTH{1'b0}} : 
                                          v_mem_leaked_ext[DATA_WIDTH-1:0];
    
    // =========================================================================
    // Synaptic Integration
    // =========================================================================
    wire signed [DATA_WIDTH:0] syn_contribution;
    wire signed [DATA_WIDTH:0] v_mem_with_syn;
    
    // Calculate synaptic contribution (excitatory positive, inhibitory negative)
    assign syn_contribution = syn_excitatory ? 
                             {{(DATA_WIDTH-WEIGHT_WIDTH+1){1'b0}}, syn_weight} : 
                             -{{(DATA_WIDTH-WEIGHT_WIDTH+1){1'b0}}, syn_weight};
    
    // Add synaptic input to membrane potential
    assign v_mem_with_syn = $signed({1'b0, v_mem}) + syn_contribution;
    
    // Saturated membrane potential after synaptic input
    wire [DATA_WIDTH-1:0] v_mem_syn_saturated;
    assign v_mem_syn_saturated = v_mem_with_syn[DATA_WIDTH] ? {DATA_WIDTH{1'b0}} :        // Negative: saturate at 0
                                 (|v_mem_with_syn[DATA_WIDTH:DATA_WIDTH-1]) ? {DATA_WIDTH{1'b1}} : // Overflow: max
                                 v_mem_with_syn[DATA_WIDTH-1:0];
    
    // Select between synaptic update and leak
    wire [DATA_WIDTH-1:0] v_mem_next = syn_valid ? v_mem_syn_saturated : v_mem_leaked;
    
    // Check if spike should be generated
    wire spike_condition;
    assign spike_condition = (v_mem_next >= threshold) && (refrac_counter == 0);
    
    // =========================================================================
    // Main neuron dynamics
    // =========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            v_mem <= {DATA_WIDTH{1'b0}};
            refrac_counter <= {REFRAC_WIDTH{1'b0}};
            spike_reg <= 1'b0;
            membrane_potential <= {DATA_WIDTH{1'b0}};
        end else if (enable) begin
            spike_reg <= 1'b0;  // Default: no spike
            
            if (refrac_counter > 0) begin
                // In refractory period: count down and keep membrane potential at reset
                refrac_counter <= refrac_counter - 1'b1;
                v_mem <= reset_potential_en ? reset_potential : {DATA_WIDTH{1'b0}};
                membrane_potential <= reset_potential_en ? reset_potential : {DATA_WIDTH{1'b0}};
            end else begin
                // Normal operation: update membrane potential
                
                // Check for spike generation
                if (spike_condition) begin
                    spike_reg <= 1'b1;
                    refrac_counter <= refractory_period;
                    v_mem <= reset_potential_en ? reset_potential : {DATA_WIDTH{1'b0}};
                    membrane_potential <= reset_potential_en ? reset_potential : {DATA_WIDTH{1'b0}};
                end else begin
                    // No spike: update membrane with new value
                    v_mem <= v_mem_next;
                    membrane_potential <= v_mem_next;
                end
            end
        end
    end
    
    // Register spike output
    always @(posedge clk) begin
        if (!rst_n) begin
            spike_out <= 1'b0;
        end else begin
            spike_out <= spike_reg & enable;
        end
    end

endmodule
