//-----------------------------------------------------------------------------
// Title         : Energy-Efficient AC-based Synapse Array (Verilog-2001)
// Project       : PYNQ-Z2 SNN Accelerator
// File          : synapse_array_ac.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Description   : Synapse array using Accumulate-only (AC) operations
//                 Exploits spike sparsity for maximum energy efficiency
//                 
//                 Energy Model (45nm technology):
//                 - MAC operation: ~4.6pJ
//                 - AC operation:  ~0.9pJ
//                 - Memory access: ~5pJ per BRAM read
//                 
//                 Sparsity Optimization:
//                 - Only access weights when spike occurs
//                 - Skip all processing for zero spikes
//                 - Typical SNN sparsity: 90-99% (huge energy savings!)
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module synapse_array_ac #(
    parameter NUM_PRE_NEURONS   = 64,      // Pre-synaptic neurons (inputs)
    parameter NUM_POST_NEURONS  = 64,      // Post-synaptic neurons (outputs)
    parameter WEIGHT_WIDTH      = 8,       // INT8 weights
    parameter VMEM_WIDTH        = 16,      // Q8.8 membrane potential
    parameter USE_BRAM          = 1,       // 1: BRAM, 0: Distributed RAM
    
    // Calculated parameters (Verilog-2001 compatible)
    parameter PRE_ID_WIDTH  = 6,  // log2(64) = 6
    parameter POST_ID_WIDTH = 6   // log2(64) = 6
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         enable,
    
    //=========================================================================
    // Sparse Spike Input Interface
    // Only valid spikes trigger processing - zero spikes are FREE!
    //=========================================================================
    input  wire                         spike_in_valid,
    input  wire [PRE_ID_WIDTH-1:0]      spike_in_pre_id,
    
    //=========================================================================
    // AC Output Interface
    // Delivers weight to post-synaptic neuron for accumulation
    //=========================================================================
    output reg                          ac_out_valid,
    output reg  [POST_ID_WIDTH-1:0]     ac_out_post_id,
    output reg  signed [WEIGHT_WIDTH-1:0] ac_out_weight,
    output reg                          ac_out_excitatory,
    
    //=========================================================================
    // Weight Configuration Interface
    //=========================================================================
    input  wire                         weight_we,
    input  wire [PRE_ID_WIDTH-1:0]      weight_pre_id,
    input  wire [POST_ID_WIDTH-1:0]     weight_post_id,
    input  wire signed [WEIGHT_WIDTH-1:0] weight_data,
    input  wire                         weight_sign,      // 1: excitatory, 0: inhibitory
    
    //=========================================================================
    // Energy Monitoring
    //=========================================================================
    output reg  [31:0]                  total_spike_count,
    output reg  [31:0]                  ac_operation_count,
    output reg  [31:0]                  memory_access_count,
    output wire                         busy
);

    //=========================================================================
    // State Machine
    //=========================================================================
    localparam IDLE     = 3'd0;
    localparam CAPTURE  = 3'd1;
    localparam FETCH    = 3'd2;
    localparam WAIT_MEM = 3'd3;
    localparam DELIVER  = 3'd4;
    localparam NEXT     = 3'd5;
    
    reg [2:0] state;
    reg [POST_ID_WIDTH-1:0] post_counter;
    reg [PRE_ID_WIDTH-1:0]  current_pre_id;
    
    assign busy = (state != IDLE);
    
    //=========================================================================
    // Weight Memory
    // Only accessed when spike occurs - sparsity saves memory bandwidth!
    //=========================================================================
    localparam NUM_WEIGHTS = NUM_PRE_NEURONS * NUM_POST_NEURONS;
    localparam ADDR_WIDTH = PRE_ID_WIDTH + POST_ID_WIDTH;
    
    // Weight storage with sign bit
    reg signed [WEIGHT_WIDTH-1:0] weight_mem [0:NUM_WEIGHTS-1];
    reg sign_mem [0:NUM_WEIGHTS-1];  // Excitatory/Inhibitory
    
    // Memory addressing
    wire [ADDR_WIDTH-1:0] read_addr;
    wire [ADDR_WIDTH-1:0] write_addr;
    reg signed [WEIGHT_WIDTH-1:0] weight_read_data;
    reg sign_read_data;
    reg mem_read_valid;
    
    assign read_addr = (current_pre_id * NUM_POST_NEURONS) + post_counter;
    assign write_addr = (weight_pre_id * NUM_POST_NEURONS) + weight_post_id;
    
    //=========================================================================
    // Weight Memory Write
    //=========================================================================
    integer i;
    initial begin
        for (i = 0; i < NUM_WEIGHTS; i = i + 1) begin
            weight_mem[i] = 0;
            sign_mem[i] = 1'b1;  // Default excitatory
        end
    end
    
    always @(posedge clk) begin
        if (weight_we) begin
            weight_mem[write_addr] <= weight_data;
            sign_mem[write_addr] <= weight_sign;
        end
    end
    
    //=========================================================================
    // Weight Memory Read (only when processing spike!)
    //=========================================================================
    always @(posedge clk) begin
        if (state == FETCH) begin
            weight_read_data <= weight_mem[read_addr];
            sign_read_data <= sign_mem[read_addr];
            mem_read_valid <= 1'b1;
        end else begin
            mem_read_valid <= 1'b0;
        end
    end
    
    //=========================================================================
    // Spike Processing State Machine
    // 
    // ENERGY OPTIMIZATION:
    // - IDLE state: No spike = No processing = Zero energy
    // - Only transitions to CAPTURE when valid spike arrives
    // - Exploits SNN sparsity (90-99% of cycles are idle)
    //=========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            post_counter <= 0;
            current_pre_id <= 0;
            total_spike_count <= 0;
            ac_operation_count <= 0;
            memory_access_count <= 0;
            
        end else if (enable) begin
            case (state)
                //=============================================================
                // IDLE: Wait for spike (zero energy consumption)
                //=============================================================
                IDLE: begin
                    if (spike_in_valid) begin
                        state <= CAPTURE;
                        current_pre_id <= spike_in_pre_id;
                        total_spike_count <= total_spike_count + 1;
                    end
                end
                
                //=============================================================
                // CAPTURE: Store spike info and start processing
                //=============================================================
                CAPTURE: begin
                    post_counter <= 0;
                    state <= FETCH;
                end
                
                //=============================================================
                // FETCH: Read weight from memory (one memory access)
                //=============================================================
                FETCH: begin
                    memory_access_count <= memory_access_count + 1;
                    state <= WAIT_MEM;
                end
                
                //=============================================================
                // WAIT_MEM: Wait for memory read latency
                //=============================================================
                WAIT_MEM: begin
                    state <= DELIVER;
                end
                
                //=============================================================
                // DELIVER: Output weight for AC operation
                // This is where the energy savings happen!
                // Traditional: output = spike * weight (MAC)
                // Our design:  output = weight (if spike=1) (AC only!)
                //=============================================================
                DELIVER: begin
                    // Only output non-zero weights (additional sparsity!)
                    if (weight_read_data != 0) begin
                        ac_operation_count <= ac_operation_count + 1;
                    end
                    state <= NEXT;
                end
                
                //=============================================================
                // NEXT: Move to next post-synaptic neuron
                //=============================================================
                NEXT: begin
                    if (post_counter == NUM_POST_NEURONS - 1) begin
                        state <= IDLE;
                        post_counter <= 0;
                    end else begin
                        post_counter <= post_counter + 1;
                        state <= FETCH;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end
    
    //=========================================================================
    // AC Output Generation
    // 
    // OUTPUT FORMAT:
    // - ac_out_valid: Indicates valid AC operation
    // - ac_out_weight: The weight value to accumulate
    // - ac_out_excitatory: Add (1) or subtract (0)
    //
    // RECEIVING NEURON OPERATION:
    // if (ac_out_valid)
    //     if (ac_out_excitatory)
    //         membrane += ac_out_weight;  // AC operation (ADD)
    //     else
    //         membrane -= ac_out_weight;  // AC operation (SUB)
    //=========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ac_out_valid <= 1'b0;
            ac_out_post_id <= 0;
            ac_out_weight <= 0;
            ac_out_excitatory <= 1'b1;
            
        end else begin
            ac_out_valid <= 1'b0;  // Default: no output
            
            if (state == DELIVER && mem_read_valid) begin
                // Skip zero weights (weight sparsity optimization)
                if (weight_read_data != 0) begin
                    ac_out_valid <= 1'b1;
                    ac_out_post_id <= post_counter;
                    ac_out_weight <= weight_read_data;
                    ac_out_excitatory <= sign_read_data;
                end
            end
        end
    end

endmodule
