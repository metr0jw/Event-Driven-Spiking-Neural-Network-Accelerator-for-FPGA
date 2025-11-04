//-----------------------------------------------------------------------------
// Title         : Leaky Integrate-and-Fire Neuron Model
// Project       : PYNQ-Z2 SNN Accelerator
// File          : lif_neuron.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Digital LIF neuron with configurable parameters
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module lif_neuron #(
    parameter NEURON_ID         = 0,
    parameter DATA_WIDTH        = 16,    // Width for membrane potential
    parameter WEIGHT_WIDTH      = 8,     // Width for synaptic weights
    parameter THRESHOLD_WIDTH   = 16,    // Width for threshold value
    parameter LEAK_WIDTH        = 8,     // Width for leak rate
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
    input  wire [LEAK_WIDTH-1:0]        leak_rate,
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
    
    // Membrane potential update logic with saturation arithmetic
    wire signed [DATA_WIDTH:0] v_mem_next;
    wire signed [DATA_WIDTH:0] syn_contribution;
    wire signed [DATA_WIDTH:0] leak_contribution;
    
    // Calculate synaptic contribution (excitatory positive, inhibitory negative)
    assign syn_contribution = syn_excitatory ? 
                             {{(DATA_WIDTH-WEIGHT_WIDTH+1){1'b0}}, syn_weight} : 
                             -{{(DATA_WIDTH-WEIGHT_WIDTH+1){1'b0}}, syn_weight};
    
    // Calculate leak (always negative)
    assign leak_contribution = -{{(DATA_WIDTH-LEAK_WIDTH+1){1'b0}}, leak_rate};
    
    // Next membrane potential calculation
    assign v_mem_next = syn_valid ? 
                       ($signed({1'b0, v_mem}) + syn_contribution) : 
                       ($signed({1'b0, v_mem}) + leak_contribution);
    
    // Main neuron dynamics
    always @(posedge clk) begin
        if (!rst_n) begin
            v_mem <= 16'd0;
            refrac_counter <= 8'd0;
            spike_reg <= 1'b0;
            membrane_potential <= 16'd0;
        end else if (enable) begin
            spike_reg <= 1'b0;  // Default: no spike
            
            if (refrac_counter > 0) begin
                // In refractory period: count down and keep membrane potential at reset
                refrac_counter <= refrac_counter - 1'b1;
                v_mem <= reset_potential_en ? reset_potential : 16'd0;
            end else begin
                // Normal operation: update membrane potential
                
                // Apply saturation to prevent overflow
                if (v_mem_next[DATA_WIDTH]) begin
                    // Negative result - saturate at 0
                    v_mem <= 16'd0;
                end else if (|v_mem_next[DATA_WIDTH:DATA_WIDTH-1]) begin
                    // Positive overflow - saturate at max
                    v_mem <= {DATA_WIDTH{1'b1}};
                end else begin
                    // Normal update
                    v_mem <= v_mem_next[DATA_WIDTH-1:0];
                end
                
                // Check for spike generation
                if (v_mem >= threshold) begin
                    spike_reg <= 1'b1;
                    refrac_counter <= refractory_period;
                    v_mem <= reset_potential_en ? reset_potential : 16'd0;
                end
            end
            
            // Update outputs
            membrane_potential <= v_mem;
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
