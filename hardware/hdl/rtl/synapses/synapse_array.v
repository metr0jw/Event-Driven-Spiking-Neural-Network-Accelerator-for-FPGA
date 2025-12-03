//-----------------------------------------------------------------------------
// Title         : Synapse Array with Weight Storage
// Project       : PYNQ-Z2 SNN Accelerator
// File          : synapse_array.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Configurable synapse array with BRAM weight storage
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module synapse_array #(
    parameter NUM_AXONS         = 64,      // Number of input axons
    parameter NUM_NEURONS       = 64,      // Number of output neurons  
    parameter WEIGHT_WIDTH      = 8,       // Bits per weight
    parameter AXON_ID_WIDTH     = $clog2(NUM_AXONS),
    parameter NEURON_ID_WIDTH   = $clog2(NUM_NEURONS),
    parameter USE_BRAM          = 1        // Use BRAM (1) or distributed RAM (0)
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input spike from axons
    input  wire                         spike_in_valid,
    input  wire [AXON_ID_WIDTH-1:0]    spike_in_axon_id,
    
    // Output spikes to neurons
    output reg                          spike_out_valid,
    output reg  [NEURON_ID_WIDTH-1:0]  spike_out_neuron_id,
    output reg  [WEIGHT_WIDTH-1:0]     spike_out_weight,
    output reg                          spike_out_exc_inh,    // Sign bit
    
    // Weight configuration interface
    input  wire                         weight_we,
    input  wire [AXON_ID_WIDTH-1:0]    weight_addr_axon,
    input  wire [NEURON_ID_WIDTH-1:0]  weight_addr_neuron,
    input  wire [WEIGHT_WIDTH:0]        weight_data,           // Includes sign
    
    // Control
    input  wire                         enable
);

    // State machine for sequential neuron processing
    localparam IDLE     = 2'd0;
    localparam FETCH    = 2'd1;
    localparam DELIVER  = 2'd2;
    
    reg [1:0] state;
    reg [NEURON_ID_WIDTH-1:0] neuron_counter;
    reg [AXON_ID_WIDTH-1:0] current_axon;
    reg spike_pending;
    
    // Weight memory interface
    wire [WEIGHT_WIDTH:0] weight_out;
    wire weight_valid;
    
    // Address calculation for weight memory
    wire [$clog2(NUM_AXONS * NUM_NEURONS)-1:0] read_addr;
    wire [$clog2(NUM_AXONS * NUM_NEURONS)-1:0] write_addr;
    
    assign read_addr = (current_axon * NUM_NEURONS) + neuron_counter;
    assign write_addr = (weight_addr_axon * NUM_NEURONS) + weight_addr_neuron;
    
    // Instantiate weight memory
    weight_memory #(
        .NUM_WEIGHTS(NUM_AXONS * NUM_NEURONS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH + 1),  // +1 for sign bit
        .USE_BRAM(USE_BRAM)
    ) weight_mem_inst (
        .clk(clk),
        .rst_n(rst_n),
        
        // Read port
        .read_en(state == FETCH),
        .read_addr(read_addr),
        .read_data(weight_out),
        
        // Write port
        .write_en(weight_we),
        .write_addr(write_addr),
        .write_data(weight_data),
        
        .read_valid(weight_valid)
    );
    
    // Input spike capture
    always @(posedge clk) begin
        if (!rst_n) begin
            spike_pending <= 1'b0;
            current_axon <= 0;
        end else if (spike_in_valid && state == IDLE) begin
            spike_pending <= 1'b1;
            current_axon <= spike_in_axon_id;
        end else if (state == DELIVER && neuron_counter == NUM_NEURONS - 1) begin
            spike_pending <= 1'b0;
        end
    end
    
    // State machine
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            neuron_counter <= 0;
        end else if (enable) begin
            case (state)
                IDLE: begin
                    if (spike_pending) begin
                        state <= FETCH;
                        neuron_counter <= 0;
                    end
                end
                
                FETCH: begin
                    state <= DELIVER;
                end
                
                DELIVER: begin
                    if (neuron_counter == NUM_NEURONS - 1) begin
                        state <= IDLE;
                        neuron_counter <= 0;
                    end else begin
                        neuron_counter <= neuron_counter + 1'b1;
                        state <= FETCH;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end
    
    // Output generation
    always @(posedge clk) begin
        if (!rst_n) begin
            spike_out_valid <= 1'b0;
            spike_out_neuron_id <= 0;
            spike_out_weight <= 0;
            spike_out_exc_inh <= 1'b1;
        end else begin
            spike_out_valid <= 1'b0;
            
            if (state == DELIVER && weight_valid) begin
                // Only output if weight is non-zero
                if (|weight_out[WEIGHT_WIDTH-1:0]) begin
                    spike_out_valid <= 1'b1;
                    spike_out_neuron_id <= neuron_counter;
                    spike_out_weight <= weight_out[WEIGHT_WIDTH-1:0];
                    spike_out_exc_inh <= weight_out[WEIGHT_WIDTH];  // Sign bit
                end
            end
        end
    end

endmodule
