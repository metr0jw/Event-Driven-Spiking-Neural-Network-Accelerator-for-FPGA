//-----------------------------------------------------------------------------
// Title         : Hybrid SNN Accelerator Top Module
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_top_hybrid.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Top module for Hybrid HLS-Verilog SNN accelerator
//                 Uses HLS-generated AXI wrapper IP with Verilog SNN core
//
// Architecture:
//   This module wraps the Verilog SNN core modules and provides
//   wire interfaces that connect to the HLS AXI wrapper.
//   
//   In Vivado Block Design:
//   - ZYNQ PS connects to axi_hls_wrapper via AXI4-Lite/Stream
//   - axi_hls_wrapper connects to this module via simple wires
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module snn_top_hybrid #(
    // Network parameters
    parameter NUM_NEURONS = 64,
    parameter NUM_AXONS = 64,
    parameter WEIGHT_WIDTH = 8,
    parameter NEURON_ID_WIDTH = 8,
    parameter AXON_ID_WIDTH = 6,
    parameter MEMBRANE_WIDTH = 16
)(
    //-------------------------------------------------------------------------
    // Clock and Reset
    //-------------------------------------------------------------------------
    input  wire                         clk,
    input  wire                         rst_n,
    
    //-------------------------------------------------------------------------
    // Control Interface (from HLS wrapper)
    //-------------------------------------------------------------------------
    input  wire                         snn_enable,
    input  wire                         snn_reset,
    input  wire                         clear_counters,
    input  wire [15:0]                  leak_rate,
    input  wire [15:0]                  threshold,
    input  wire [15:0]                  refractory_period,
    
    //-------------------------------------------------------------------------
    // Status Interface (to HLS wrapper)
    //-------------------------------------------------------------------------
    output wire                         snn_ready,
    output wire                         snn_busy,
    output wire                         snn_error,
    output wire [31:0]                  spike_count,
    
    //-------------------------------------------------------------------------
    // Spike Input Interface (from HLS wrapper)
    //-------------------------------------------------------------------------
    input  wire                         spike_in_valid,
    input  wire [NEURON_ID_WIDTH-1:0]   spike_in_neuron_id,
    input  wire signed [WEIGHT_WIDTH-1:0] spike_in_weight,
    output wire                         spike_in_ready,
    
    //-------------------------------------------------------------------------
    // Spike Output Interface (to HLS wrapper)
    //-------------------------------------------------------------------------
    output wire                         spike_out_valid,
    output wire [NEURON_ID_WIDTH-1:0]   spike_out_neuron_id,
    output wire signed [WEIGHT_WIDTH-1:0] spike_out_weight,
    input  wire                         spike_out_ready,
    
    //-------------------------------------------------------------------------
    // Optional: Debug LEDs
    //-------------------------------------------------------------------------
    output wire [3:0]                   debug_led
);

    //=========================================================================
    // Internal Signals
    //=========================================================================
    // Combined reset
    wire sys_rst_n;
    assign sys_rst_n = rst_n & ~snn_reset;
    
    // Internal spike buses
    wire                        synapse_spike_valid;
    wire [NEURON_ID_WIDTH-1:0]  synapse_spike_neuron_id;
    wire [WEIGHT_WIDTH-1:0]     synapse_spike_weight;
    wire                        synapse_spike_exc_inh;
    
    wire                        router_spike_valid;
    wire [NEURON_ID_WIDTH-1:0]  router_spike_dest_id;
    wire [WEIGHT_WIDTH-1:0]     router_spike_weight;
    wire                        router_spike_exc_inh;
    wire                        router_spike_ready;
    
    wire                        neuron_spike_valid;
    wire [NEURON_ID_WIDTH-1:0]  neuron_spike_id;
    
    // Status signals
    wire                        synapse_busy;
    wire                        router_busy;
    wire                        fifo_overflow;
    
    // Spike counters
    reg [31:0]                  input_spike_count;
    reg [31:0]                  output_spike_count;
    
    //=========================================================================
    // Spike Counter Logic
    //=========================================================================
    always @(posedge clk or negedge sys_rst_n) begin
        if (!sys_rst_n || clear_counters) begin
            input_spike_count <= 32'd0;
            output_spike_count <= 32'd0;
        end else if (snn_enable) begin
            if (spike_in_valid && spike_in_ready)
                input_spike_count <= input_spike_count + 1;
            if (spike_out_valid && spike_out_ready)
                output_spike_count <= output_spike_count + 1;
        end
    end
    
    assign spike_count = input_spike_count + output_spike_count;
    
    //=========================================================================
    // Status Generation
    //=========================================================================
    assign snn_ready = snn_enable & ~snn_busy;
    assign snn_busy = synapse_busy | router_busy;
    assign snn_error = fifo_overflow;
    
    // Debug LEDs
    assign debug_led[0] = snn_enable;
    assign debug_led[1] = spike_in_valid;
    assign debug_led[2] = spike_out_valid;
    assign debug_led[3] = snn_busy;
    
    //=========================================================================
    // Synapse Array
    //=========================================================================
    synapse_array #(
        .NUM_AXONS(NUM_AXONS),
        .NUM_NEURONS(NUM_NEURONS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .USE_BRAM(1)
    ) synapse_array_inst (
        .clk(clk),
        .rst_n(sys_rst_n),
        
        // Input spike
        .spike_in_valid(spike_in_valid & snn_enable),
        .spike_in_axon_id(spike_in_neuron_id[AXON_ID_WIDTH-1:0]),
        
        // Output to neurons/router
        .spike_out_valid(synapse_spike_valid),
        .spike_out_neuron_id(synapse_spike_neuron_id),
        .spike_out_weight(synapse_spike_weight),
        .spike_out_exc_inh(synapse_spike_exc_inh),
        
        // Weight configuration (directly from top for simplicity)
        // In production, route through config interface
        .weight_we(1'b0),
        .weight_addr_axon({AXON_ID_WIDTH{1'b0}}),
        .weight_addr_neuron({$clog2(NUM_NEURONS){1'b0}}),
        .weight_data({1'b0, {WEIGHT_WIDTH{1'b0}}}),
        
        // Status
        .busy(synapse_busy)
    );
    
    // Input ready when synapse array can accept
    assign spike_in_ready = ~synapse_busy;
    
    //=========================================================================
    // Spike Router
    //=========================================================================
    spike_router #(
        .NUM_NEURONS(NUM_NEURONS),
        .NEURON_ID_WIDTH(NEURON_ID_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .MAX_FANOUT(8),
        .FIFO_DEPTH(16)
    ) spike_router_inst (
        .clk(clk),
        .rst_n(sys_rst_n),
        
        // Input from synapse array
        .spike_in_valid(synapse_spike_valid),
        .spike_in_src_id(synapse_spike_neuron_id),
        .spike_in_weight(synapse_spike_weight),
        .spike_in_exc_inh(synapse_spike_exc_inh),
        .spike_in_ready(), // Not used in this config
        
        // Output to neuron array
        .spike_out_valid(router_spike_valid),
        .spike_out_dest_id(router_spike_dest_id),
        .spike_out_weight(router_spike_weight),
        .spike_out_exc_inh(router_spike_exc_inh),
        .spike_out_ready(router_spike_ready),
        
        // Configuration (static for now)
        .config_we(1'b0),
        .config_src_id({NEURON_ID_WIDTH{1'b0}}),
        .config_conn_idx(3'd0),
        .config_data(24'd0),
        .config_conn_count(4'd0),
        
        // Status
        .busy(router_busy),
        .fifo_overflow(fifo_overflow),
        .spike_counter()
    );
    
    //=========================================================================
    // LIF Neuron Array
    //=========================================================================
    // Generate neuron instances
    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin : gen_neurons
            wire neuron_spike;
            wire neuron_input_valid;
            wire [WEIGHT_WIDTH-1:0] neuron_input_weight;
            wire neuron_exc_inh;
            
            // Route input to this neuron
            assign neuron_input_valid = router_spike_valid && 
                                       (router_spike_dest_id == i);
            assign neuron_input_weight = router_spike_weight;
            assign neuron_exc_inh = router_spike_exc_inh;
            
            lif_neuron #(
                .MEMBRANE_WIDTH(MEMBRANE_WIDTH),
                .WEIGHT_WIDTH(WEIGHT_WIDTH)
            ) neuron_inst (
                .clk(clk),
                .rst_n(sys_rst_n),
                .enable(snn_enable),
                
                // Input
                .spike_in_valid(neuron_input_valid),
                .spike_in_weight(neuron_input_weight),
                .spike_in_exc_inh(neuron_exc_inh),
                
                // Parameters
                .threshold(threshold[MEMBRANE_WIDTH-1:0]),
                .leak_rate(leak_rate[WEIGHT_WIDTH-1:0]),
                .refractory_period(refractory_period[7:0]),
                
                // Output
                .spike_out(neuron_spike),
                
                // Debug
                .membrane_potential(),
                .refractory_counter()
            );
            
            // Collect neuron spikes
            // Note: In a real implementation, use proper arbitration
        end
    endgenerate
    
    // Simple output spike generation (first neuron that spikes)
    // In production, implement proper output arbitration
    reg [NEURON_ID_WIDTH-1:0] output_neuron_id;
    reg output_spike_reg;
    reg output_valid_reg;
    
    integer j;
    always @(posedge clk or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            output_spike_reg <= 1'b0;
            output_valid_reg <= 1'b0;
            output_neuron_id <= {NEURON_ID_WIDTH{1'b0}};
        end else begin
            output_spike_reg <= 1'b0;
            output_valid_reg <= 1'b0;
            
            // Check each neuron for spike
            for (j = 0; j < NUM_NEURONS; j = j + 1) begin
                if (gen_neurons[j].neuron_inst.spike_out && !output_spike_reg) begin
                    output_spike_reg <= 1'b1;
                    output_valid_reg <= 1'b1;
                    output_neuron_id <= j[NEURON_ID_WIDTH-1:0];
                end
            end
        end
    end
    
    assign spike_out_valid = output_valid_reg & spike_out_ready;
    assign spike_out_neuron_id = output_neuron_id;
    assign spike_out_weight = 8'sd100;  // Default output weight
    
    // Router ready based on output ready
    assign router_spike_ready = 1'b1;  // Always accept for now

endmodule
