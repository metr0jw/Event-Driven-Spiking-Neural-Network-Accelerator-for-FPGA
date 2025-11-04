//-----------------------------------------------------------------------------
// Simple LIF Neuron Testbench (Verilog-2001 Compatible)
// For testing basic simulation functionality
// Title         : SNN LIF Neuron Simple Testbench
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_simple_lif.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Simple testbench for LIF neuron module
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_simple_lif();

    // Parameters
    parameter CLK_PERIOD = 10;  // 100MHz clock
    parameter DATA_WIDTH = 16;
    parameter WEIGHT_WIDTH = 8;
    parameter THRESHOLD_WIDTH = 16;
    parameter LEAK_WIDTH = 8;
    parameter REFRAC_WIDTH = 8;

    // Clock and reset
    reg clk;
    reg rst_n;
    reg enable;
    
    // Inputs to LIF neuron
    reg syn_valid;
    reg [WEIGHT_WIDTH-1:0] syn_weight;
    reg syn_excitatory;
    reg [THRESHOLD_WIDTH-1:0] threshold;
    reg [LEAK_WIDTH-1:0] leak_rate;
    reg [REFRAC_WIDTH-1:0] refractory_period;
    reg reset_potential_en;
    reg [DATA_WIDTH-1:0] reset_potential;
    
    // Outputs from LIF neuron
    wire spike_out;
    wire [DATA_WIDTH-1:0] membrane_potential;
    wire is_refractory;
    wire [REFRAC_WIDTH-1:0] refrac_count;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // LIF Neuron instantiation
    lif_neuron #(
        .NEURON_ID(0),
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .THRESHOLD_WIDTH(THRESHOLD_WIDTH),
        .LEAK_WIDTH(LEAK_WIDTH),
        .REFRAC_WIDTH(REFRAC_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .syn_valid(syn_valid),
        .syn_weight(syn_weight),
        .syn_excitatory(syn_excitatory),
        .threshold(threshold),
        .leak_rate(leak_rate),
        .refractory_period(refractory_period),
        .reset_potential_en(reset_potential_en),
        .reset_potential(reset_potential),
        .spike_out(spike_out),
        .membrane_potential(membrane_potential),
        .is_refractory(is_refractory),
        .refrac_count(refrac_count)
    );
    
    // Test stimulus
    initial begin
        $dumpfile("waves.vcd");
        $dumpvars(0, tb_simple_lif);
        
        // Initialize signals
        rst_n = 0;
        enable = 0;
        syn_valid = 0;
        syn_weight = 0;
        syn_excitatory = 1;
        threshold = 16'h1000;  // Set threshold
        leak_rate = 8'h02;     // Small leak
        refractory_period = 8'h05; // 5 cycle refractory
        reset_potential_en = 0;
        reset_potential = 0;
        
        // Reset sequence
        #(CLK_PERIOD*2);
        rst_n = 1;
        #(CLK_PERIOD*2);
        enable = 1;
        
        $display("Starting LIF neuron simulation...");
        $display("Threshold: %d, Leak rate: %d", threshold, leak_rate);
        
        // Apply some synaptic inputs
        repeat(10) begin
            @(posedge clk);
            syn_valid = 1;
            syn_weight = 8'h20;  // Moderate weight
            syn_excitatory = 1;
            @(posedge clk);
            syn_valid = 0;
            syn_weight = 0;
            
            // Wait a few cycles
            repeat(3) @(posedge clk);
            
            // Monitor membrane potential
            $display("Time: %0t, Membrane: %d, Spike: %b", 
                     $time, membrane_potential, spike_out);
        end
        
        // Wait and finish
        repeat(20) @(posedge clk);
        
        $display("Simulation completed successfully!");
        $finish;
    end
    
    // Monitor for spikes
    always @(posedge clk) begin
        if (spike_out) begin
            $display("*** SPIKE detected at time %0t! Membrane was: %d ***", 
                     $time, membrane_potential);
        end
    end
    
    // Safety timeout
    initial begin
        #100000;  // 100us timeout
        $display("ERROR: Simulation timeout!");
        $finish;
    end

endmodule
