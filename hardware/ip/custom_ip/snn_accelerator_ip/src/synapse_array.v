//-----------------------------------------------------------------------------
// Title         : Synapse Array with Parallel BRAM Banking
// Project       : PYNQ-Z2 SNN Accelerator
// File          : synapse_array.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : High-performance synapse array with:
//                 - Multiple parallel read ports using BRAM banking
//                 - Pipelined weight lookups
//                 - Support for larger network sizes (256x256)
//                 - DSP-assisted weight scaling (optional)
// Note          : Iverilog/Verilator compatible version
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module synapse_array #(
    parameter NUM_AXONS         = 256,     // Number of input axons (scaled up)
    parameter NUM_NEURONS       = 256,     // Number of output neurons (scaled up)
    parameter WEIGHT_WIDTH      = 8,       // Bits per weight
    parameter NUM_READ_PORTS    = 8,       // Parallel read ports (BRAM banks)
    parameter AXON_ID_WIDTH     = $clog2(NUM_AXONS),
    parameter NEURON_ID_WIDTH   = $clog2(NUM_NEURONS),
    parameter USE_BRAM          = 1,       // Use BRAM for storage
    parameter USE_DSP           = 1        // Use DSP for weight scaling
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input spike from axons
    input  wire                         spike_in_valid,
    input  wire [AXON_ID_WIDTH-1:0]     spike_in_axon_id,
    output wire                         spike_in_ready,
    
    // Output spikes to neurons - pipelined output
    output reg                          spike_out_valid,
    output reg  [NEURON_ID_WIDTH-1:0]   spike_out_neuron_id,
    output reg  [WEIGHT_WIDTH-1:0]      spike_out_weight,
    output reg                          spike_out_exc_inh,
    
    // Weight configuration interface (AXI-mapped)
    input  wire                         weight_we,
    input  wire [AXON_ID_WIDTH-1:0]     weight_addr_axon,
    input  wire [NEURON_ID_WIDTH-1:0]   weight_addr_neuron,
    input  wire [WEIGHT_WIDTH:0]        weight_data,  // +1 for sign bit
    
    // Batch write interface for faster initialization
    input  wire                         batch_we,
    input  wire [AXON_ID_WIDTH-1:0]     batch_axon_id,
    input  wire [NUM_READ_PORTS*(WEIGHT_WIDTH+1)-1:0] batch_weights,
    input  wire [NEURON_ID_WIDTH-1:0]   batch_start_neuron,
    
    // Control
    input  wire                         enable,
    
    // Status
    output wire                         busy,
    output reg  [31:0]                  spike_count
);

    //=========================================================================
    // Local Parameters
    //=========================================================================
    localparam NEURONS_PER_CYCLE = NUM_READ_PORTS;
    localparam NUM_CYCLES = (NUM_NEURONS + NEURONS_PER_CYCLE - 1) / NEURONS_PER_CYCLE;
    localparam CYCLE_COUNTER_WIDTH = $clog2(NUM_CYCLES + 1);
    
    // Memory organization: one bank per read port for parallel access
    localparam NEURONS_PER_BANK = (NUM_NEURONS + NUM_READ_PORTS - 1) / NUM_READ_PORTS;
    localparam BANK_ADDR_WIDTH = $clog2(NEURONS_PER_BANK);
    
    // Total memory depth per bank
    localparam BANK_DEPTH = NUM_AXONS * NEURONS_PER_BANK;
    localparam BANK_ADDR_TOTAL_WIDTH = $clog2(BANK_DEPTH);

    //=========================================================================
    // State Machine
    //=========================================================================
    localparam [2:0] ST_IDLE     = 3'd0;
    localparam [2:0] ST_FETCH    = 3'd1;
    localparam [2:0] ST_WAIT     = 3'd2;
    localparam [2:0] ST_DELIVER  = 3'd3;
    localparam [2:0] ST_NEXT     = 3'd4;
    
    reg [2:0] state;
    reg [CYCLE_COUNTER_WIDTH-1:0] cycle_counter;
    reg [AXON_ID_WIDTH-1:0] current_axon;
    reg spike_pending;

    //=========================================================================
    // Weight Memory - Flat Arrays for Iverilog Compatibility
    //=========================================================================
    // Weight storage: NUM_READ_PORTS banks, each BANK_DEPTH deep
    (* ram_style = "block" *)
    reg [WEIGHT_WIDTH:0] weight_mem_0 [0:BANK_DEPTH-1];
    (* ram_style = "block" *)
    reg [WEIGHT_WIDTH:0] weight_mem_1 [0:BANK_DEPTH-1];
    (* ram_style = "block" *)
    reg [WEIGHT_WIDTH:0] weight_mem_2 [0:BANK_DEPTH-1];
    (* ram_style = "block" *)
    reg [WEIGHT_WIDTH:0] weight_mem_3 [0:BANK_DEPTH-1];
    (* ram_style = "block" *)
    reg [WEIGHT_WIDTH:0] weight_mem_4 [0:BANK_DEPTH-1];
    (* ram_style = "block" *)
    reg [WEIGHT_WIDTH:0] weight_mem_5 [0:BANK_DEPTH-1];
    (* ram_style = "block" *)
    reg [WEIGHT_WIDTH:0] weight_mem_6 [0:BANK_DEPTH-1];
    (* ram_style = "block" *)
    reg [WEIGHT_WIDTH:0] weight_mem_7 [0:BANK_DEPTH-1];
    
    // Read data registers for each bank
    reg [WEIGHT_WIDTH:0] read_data_0, read_data_1, read_data_2, read_data_3;
    reg [WEIGHT_WIDTH:0] read_data_4, read_data_5, read_data_6, read_data_7;
    reg read_valid_0, read_valid_1, read_valid_2, read_valid_3;
    reg read_valid_4, read_valid_5, read_valid_6, read_valid_7;
    
    // Address calculation wires
    wire [BANK_ADDR_TOTAL_WIDTH-1:0] read_addr [0:NUM_READ_PORTS-1];
    wire [NEURON_ID_WIDTH-1:0] target_neuron [0:NUM_READ_PORTS-1];
    wire [BANK_ADDR_WIDTH-1:0] neuron_in_bank [0:NUM_READ_PORTS-1];
    
    // Generate address calculations
    genvar b;
    generate
        for (b = 0; b < NUM_READ_PORTS; b = b + 1) begin : addr_calc
            assign target_neuron[b] = cycle_counter * NUM_READ_PORTS + b;
            assign neuron_in_bank[b] = target_neuron[b] / NUM_READ_PORTS;
            assign read_addr[b] = current_axon * NEURONS_PER_BANK + neuron_in_bank[b];
        end
    endgenerate
    
    // Bank 0 Read/Write
    always @(posedge clk) begin
        if (!rst_n) begin
            read_data_0 <= 0;
            read_valid_0 <= 0;
        end else if (state == ST_FETCH) begin
            if (target_neuron[0] < NUM_NEURONS) begin
                read_data_0 <= weight_mem_0[read_addr[0]];
                read_valid_0 <= 1'b1;
            end else begin
                read_data_0 <= 0;
                read_valid_0 <= 1'b0;
            end
        end else if (state == ST_IDLE) begin
            read_valid_0 <= 1'b0;
        end
        
        // Write logic for bank 0
        if (weight_we && (weight_addr_neuron % NUM_READ_PORTS == 0)) begin
            weight_mem_0[weight_addr_axon * NEURONS_PER_BANK + weight_addr_neuron / NUM_READ_PORTS] <= weight_data;
        end else if (batch_we && ((batch_start_neuron + 0) % NUM_READ_PORTS == 0) && (batch_start_neuron + 0 < NUM_NEURONS)) begin
            weight_mem_0[batch_axon_id * NEURONS_PER_BANK + (batch_start_neuron + 0) / NUM_READ_PORTS] <= batch_weights[0*(WEIGHT_WIDTH+1) +: (WEIGHT_WIDTH+1)];
        end
    end
    
    // Bank 1 Read/Write
    always @(posedge clk) begin
        if (!rst_n) begin
            read_data_1 <= 0;
            read_valid_1 <= 0;
        end else if (state == ST_FETCH) begin
            if (target_neuron[1] < NUM_NEURONS) begin
                read_data_1 <= weight_mem_1[read_addr[1]];
                read_valid_1 <= 1'b1;
            end else begin
                read_data_1 <= 0;
                read_valid_1 <= 1'b0;
            end
        end else if (state == ST_IDLE) begin
            read_valid_1 <= 1'b0;
        end
        
        if (weight_we && (weight_addr_neuron % NUM_READ_PORTS == 1)) begin
            weight_mem_1[weight_addr_axon * NEURONS_PER_BANK + weight_addr_neuron / NUM_READ_PORTS] <= weight_data;
        end else if (batch_we && ((batch_start_neuron + 1) % NUM_READ_PORTS == 1) && (batch_start_neuron + 1 < NUM_NEURONS)) begin
            weight_mem_1[batch_axon_id * NEURONS_PER_BANK + (batch_start_neuron + 1) / NUM_READ_PORTS] <= batch_weights[1*(WEIGHT_WIDTH+1) +: (WEIGHT_WIDTH+1)];
        end
    end
    
    // Bank 2 Read/Write
    always @(posedge clk) begin
        if (!rst_n) begin
            read_data_2 <= 0;
            read_valid_2 <= 0;
        end else if (state == ST_FETCH) begin
            if (target_neuron[2] < NUM_NEURONS) begin
                read_data_2 <= weight_mem_2[read_addr[2]];
                read_valid_2 <= 1'b1;
            end else begin
                read_data_2 <= 0;
                read_valid_2 <= 1'b0;
            end
        end else if (state == ST_IDLE) begin
            read_valid_2 <= 1'b0;
        end
        
        if (weight_we && (weight_addr_neuron % NUM_READ_PORTS == 2)) begin
            weight_mem_2[weight_addr_axon * NEURONS_PER_BANK + weight_addr_neuron / NUM_READ_PORTS] <= weight_data;
        end else if (batch_we && ((batch_start_neuron + 2) % NUM_READ_PORTS == 2) && (batch_start_neuron + 2 < NUM_NEURONS)) begin
            weight_mem_2[batch_axon_id * NEURONS_PER_BANK + (batch_start_neuron + 2) / NUM_READ_PORTS] <= batch_weights[2*(WEIGHT_WIDTH+1) +: (WEIGHT_WIDTH+1)];
        end
    end
    
    // Bank 3 Read/Write
    always @(posedge clk) begin
        if (!rst_n) begin
            read_data_3 <= 0;
            read_valid_3 <= 0;
        end else if (state == ST_FETCH) begin
            if (target_neuron[3] < NUM_NEURONS) begin
                read_data_3 <= weight_mem_3[read_addr[3]];
                read_valid_3 <= 1'b1;
            end else begin
                read_data_3 <= 0;
                read_valid_3 <= 1'b0;
            end
        end else if (state == ST_IDLE) begin
            read_valid_3 <= 1'b0;
        end
        
        if (weight_we && (weight_addr_neuron % NUM_READ_PORTS == 3)) begin
            weight_mem_3[weight_addr_axon * NEURONS_PER_BANK + weight_addr_neuron / NUM_READ_PORTS] <= weight_data;
        end else if (batch_we && ((batch_start_neuron + 3) % NUM_READ_PORTS == 3) && (batch_start_neuron + 3 < NUM_NEURONS)) begin
            weight_mem_3[batch_axon_id * NEURONS_PER_BANK + (batch_start_neuron + 3) / NUM_READ_PORTS] <= batch_weights[3*(WEIGHT_WIDTH+1) +: (WEIGHT_WIDTH+1)];
        end
    end
    
    // Bank 4 Read/Write
    always @(posedge clk) begin
        if (!rst_n) begin
            read_data_4 <= 0;
            read_valid_4 <= 0;
        end else if (state == ST_FETCH) begin
            if (target_neuron[4] < NUM_NEURONS) begin
                read_data_4 <= weight_mem_4[read_addr[4]];
                read_valid_4 <= 1'b1;
            end else begin
                read_data_4 <= 0;
                read_valid_4 <= 1'b0;
            end
        end else if (state == ST_IDLE) begin
            read_valid_4 <= 1'b0;
        end
        
        if (weight_we && (weight_addr_neuron % NUM_READ_PORTS == 4)) begin
            weight_mem_4[weight_addr_axon * NEURONS_PER_BANK + weight_addr_neuron / NUM_READ_PORTS] <= weight_data;
        end else if (batch_we && ((batch_start_neuron + 4) % NUM_READ_PORTS == 4) && (batch_start_neuron + 4 < NUM_NEURONS)) begin
            weight_mem_4[batch_axon_id * NEURONS_PER_BANK + (batch_start_neuron + 4) / NUM_READ_PORTS] <= batch_weights[4*(WEIGHT_WIDTH+1) +: (WEIGHT_WIDTH+1)];
        end
    end
    
    // Bank 5 Read/Write
    always @(posedge clk) begin
        if (!rst_n) begin
            read_data_5 <= 0;
            read_valid_5 <= 0;
        end else if (state == ST_FETCH) begin
            if (target_neuron[5] < NUM_NEURONS) begin
                read_data_5 <= weight_mem_5[read_addr[5]];
                read_valid_5 <= 1'b1;
            end else begin
                read_data_5 <= 0;
                read_valid_5 <= 1'b0;
            end
        end else if (state == ST_IDLE) begin
            read_valid_5 <= 1'b0;
        end
        
        if (weight_we && (weight_addr_neuron % NUM_READ_PORTS == 5)) begin
            weight_mem_5[weight_addr_axon * NEURONS_PER_BANK + weight_addr_neuron / NUM_READ_PORTS] <= weight_data;
        end else if (batch_we && ((batch_start_neuron + 5) % NUM_READ_PORTS == 5) && (batch_start_neuron + 5 < NUM_NEURONS)) begin
            weight_mem_5[batch_axon_id * NEURONS_PER_BANK + (batch_start_neuron + 5) / NUM_READ_PORTS] <= batch_weights[5*(WEIGHT_WIDTH+1) +: (WEIGHT_WIDTH+1)];
        end
    end
    
    // Bank 6 Read/Write
    always @(posedge clk) begin
        if (!rst_n) begin
            read_data_6 <= 0;
            read_valid_6 <= 0;
        end else if (state == ST_FETCH) begin
            if (target_neuron[6] < NUM_NEURONS) begin
                read_data_6 <= weight_mem_6[read_addr[6]];
                read_valid_6 <= 1'b1;
            end else begin
                read_data_6 <= 0;
                read_valid_6 <= 1'b0;
            end
        end else if (state == ST_IDLE) begin
            read_valid_6 <= 1'b0;
        end
        
        if (weight_we && (weight_addr_neuron % NUM_READ_PORTS == 6)) begin
            weight_mem_6[weight_addr_axon * NEURONS_PER_BANK + weight_addr_neuron / NUM_READ_PORTS] <= weight_data;
        end else if (batch_we && ((batch_start_neuron + 6) % NUM_READ_PORTS == 6) && (batch_start_neuron + 6 < NUM_NEURONS)) begin
            weight_mem_6[batch_axon_id * NEURONS_PER_BANK + (batch_start_neuron + 6) / NUM_READ_PORTS] <= batch_weights[6*(WEIGHT_WIDTH+1) +: (WEIGHT_WIDTH+1)];
        end
    end
    
    // Bank 7 Read/Write
    always @(posedge clk) begin
        if (!rst_n) begin
            read_data_7 <= 0;
            read_valid_7 <= 0;
        end else if (state == ST_FETCH) begin
            if (target_neuron[7] < NUM_NEURONS) begin
                read_data_7 <= weight_mem_7[read_addr[7]];
                read_valid_7 <= 1'b1;
            end else begin
                read_data_7 <= 0;
                read_valid_7 <= 1'b0;
            end
        end else if (state == ST_IDLE) begin
            read_valid_7 <= 1'b0;
        end
        
        if (weight_we && (weight_addr_neuron % NUM_READ_PORTS == 7)) begin
            weight_mem_7[weight_addr_axon * NEURONS_PER_BANK + weight_addr_neuron / NUM_READ_PORTS] <= weight_data;
        end else if (batch_we && ((batch_start_neuron + 7) % NUM_READ_PORTS == 7) && (batch_start_neuron + 7 < NUM_NEURONS)) begin
            weight_mem_7[batch_axon_id * NEURONS_PER_BANK + (batch_start_neuron + 7) / NUM_READ_PORTS] <= batch_weights[7*(WEIGHT_WIDTH+1) +: (WEIGHT_WIDTH+1)];
        end
    end

    //=========================================================================
    // Input Spike Capture
    //=========================================================================
    assign spike_in_ready = (state == ST_IDLE) && !spike_pending;
    assign busy = (state != ST_IDLE) || spike_pending;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            spike_pending <= 1'b0;
            current_axon <= 0;
        end else if (spike_in_valid && spike_in_ready) begin
            spike_pending <= 1'b1;
            current_axon <= spike_in_axon_id;
        end else if (state == ST_NEXT && cycle_counter >= NUM_CYCLES - 1) begin
            spike_pending <= 1'b0;
        end
    end

    //=========================================================================
    // State Machine
    //=========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            cycle_counter <= 0;
        end else if (enable) begin
            case (state)
                ST_IDLE: begin
                    cycle_counter <= 0;
                    if (spike_pending) begin
                        state <= ST_FETCH;
                    end
                end
                
                ST_FETCH: begin
                    // Initiate parallel reads from all banks
                    state <= ST_WAIT;
                end
                
                ST_WAIT: begin
                    // Wait for BRAM read latency (1 cycle)
                    state <= ST_DELIVER;
                end
                
                ST_DELIVER: begin
                    // Output weights from all banks sequentially
                    state <= ST_NEXT;
                end
                
                ST_NEXT: begin
                    if (cycle_counter >= NUM_CYCLES - 1) begin
                        state <= ST_IDLE;
                        cycle_counter <= 0;
                    end else begin
                        cycle_counter <= cycle_counter + 1;
                        state <= ST_FETCH;
                    end
                end
                
                default: state <= ST_IDLE;
            endcase
        end
    end

    //=========================================================================
    // Output Generation (Round-robin from banks)
    //=========================================================================
    reg [3:0] output_bank_sel;
    reg [NUM_READ_PORTS-1:0] bank_valid_latched;
    reg [WEIGHT_WIDTH:0] bank_data_latched [0:NUM_READ_PORTS-1];
    reg [NEURON_ID_WIDTH-1:0] bank_neuron_latched [0:NUM_READ_PORTS-1];
    reg output_pending;
    reg [CYCLE_COUNTER_WIDTH-1:0] output_cycle;
    
    // Current bank read data and valid (combinational mux)
    wire [WEIGHT_WIDTH:0] current_bank_data;
    wire current_bank_valid;
    
    assign current_bank_data = (output_bank_sel == 0) ? bank_data_latched[0] :
                               (output_bank_sel == 1) ? bank_data_latched[1] :
                               (output_bank_sel == 2) ? bank_data_latched[2] :
                               (output_bank_sel == 3) ? bank_data_latched[3] :
                               (output_bank_sel == 4) ? bank_data_latched[4] :
                               (output_bank_sel == 5) ? bank_data_latched[5] :
                               (output_bank_sel == 6) ? bank_data_latched[6] :
                                                        bank_data_latched[7];
    
    assign current_bank_valid = bank_valid_latched[output_bank_sel];
    
    // Latch bank outputs
    always @(posedge clk) begin
        if (!rst_n) begin
            bank_valid_latched <= 0;
            output_pending <= 0;
            output_cycle <= 0;
            bank_data_latched[0] <= 0;
            bank_data_latched[1] <= 0;
            bank_data_latched[2] <= 0;
            bank_data_latched[3] <= 0;
            bank_data_latched[4] <= 0;
            bank_data_latched[5] <= 0;
            bank_data_latched[6] <= 0;
            bank_data_latched[7] <= 0;
            bank_neuron_latched[0] <= 0;
            bank_neuron_latched[1] <= 0;
            bank_neuron_latched[2] <= 0;
            bank_neuron_latched[3] <= 0;
            bank_neuron_latched[4] <= 0;
            bank_neuron_latched[5] <= 0;
            bank_neuron_latched[6] <= 0;
            bank_neuron_latched[7] <= 0;
        end else if (state == ST_WAIT) begin
            // Prepare to latch in DELIVER
            output_cycle <= cycle_counter;
        end else if (state == ST_DELIVER) begin
            // Latch all bank outputs
            bank_valid_latched[0] <= read_valid_0 && |read_data_0[WEIGHT_WIDTH-1:0];
            bank_valid_latched[1] <= read_valid_1 && |read_data_1[WEIGHT_WIDTH-1:0];
            bank_valid_latched[2] <= read_valid_2 && |read_data_2[WEIGHT_WIDTH-1:0];
            bank_valid_latched[3] <= read_valid_3 && |read_data_3[WEIGHT_WIDTH-1:0];
            bank_valid_latched[4] <= read_valid_4 && |read_data_4[WEIGHT_WIDTH-1:0];
            bank_valid_latched[5] <= read_valid_5 && |read_data_5[WEIGHT_WIDTH-1:0];
            bank_valid_latched[6] <= read_valid_6 && |read_data_6[WEIGHT_WIDTH-1:0];
            bank_valid_latched[7] <= read_valid_7 && |read_data_7[WEIGHT_WIDTH-1:0];
            
            bank_data_latched[0] <= read_data_0;
            bank_data_latched[1] <= read_data_1;
            bank_data_latched[2] <= read_data_2;
            bank_data_latched[3] <= read_data_3;
            bank_data_latched[4] <= read_data_4;
            bank_data_latched[5] <= read_data_5;
            bank_data_latched[6] <= read_data_6;
            bank_data_latched[7] <= read_data_7;
            
            bank_neuron_latched[0] <= output_cycle * NUM_READ_PORTS + 0;
            bank_neuron_latched[1] <= output_cycle * NUM_READ_PORTS + 1;
            bank_neuron_latched[2] <= output_cycle * NUM_READ_PORTS + 2;
            bank_neuron_latched[3] <= output_cycle * NUM_READ_PORTS + 3;
            bank_neuron_latched[4] <= output_cycle * NUM_READ_PORTS + 4;
            bank_neuron_latched[5] <= output_cycle * NUM_READ_PORTS + 5;
            bank_neuron_latched[6] <= output_cycle * NUM_READ_PORTS + 6;
            bank_neuron_latched[7] <= output_cycle * NUM_READ_PORTS + 7;
            
            output_pending <= 1'b1;
            output_bank_sel <= 0;
        end else if (output_pending) begin
            // Cycle through valid banks
            if (output_bank_sel >= NUM_READ_PORTS - 1) begin
                output_pending <= 1'b0;
            end else begin
                output_bank_sel <= output_bank_sel + 1;
            end
        end
    end
    
    // Current neuron from mux
    wire [NEURON_ID_WIDTH-1:0] current_neuron;
    assign current_neuron = (output_bank_sel == 0) ? bank_neuron_latched[0] :
                            (output_bank_sel == 1) ? bank_neuron_latched[1] :
                            (output_bank_sel == 2) ? bank_neuron_latched[2] :
                            (output_bank_sel == 3) ? bank_neuron_latched[3] :
                            (output_bank_sel == 4) ? bank_neuron_latched[4] :
                            (output_bank_sel == 5) ? bank_neuron_latched[5] :
                            (output_bank_sel == 6) ? bank_neuron_latched[6] :
                                                     bank_neuron_latched[7];
    
    // Generate output spikes
    always @(posedge clk) begin
        if (!rst_n) begin
            spike_out_valid <= 1'b0;
            spike_out_neuron_id <= 0;
            spike_out_weight <= 0;
            spike_out_exc_inh <= 1'b1;
            spike_count <= 0;
        end else begin
            spike_out_valid <= 1'b0;
            
            if (output_pending && current_bank_valid) begin
                spike_out_valid <= 1'b1;
                spike_out_neuron_id <= current_neuron;
                spike_out_weight <= current_bank_data[WEIGHT_WIDTH-1:0];
                spike_out_exc_inh <= current_bank_data[WEIGHT_WIDTH];
                spike_count <= spike_count + 1;
            end
        end
    end

    //=========================================================================
    // Memory Initialization
    //=========================================================================
    integer init_i;
    initial begin
        for (init_i = 0; init_i < BANK_DEPTH; init_i = init_i + 1) begin
            weight_mem_0[init_i] = 0;
            weight_mem_1[init_i] = 0;
            weight_mem_2[init_i] = 0;
            weight_mem_3[init_i] = 0;
            weight_mem_4[init_i] = 0;
            weight_mem_5[init_i] = 0;
            weight_mem_6[init_i] = 0;
            weight_mem_7[init_i] = 0;
        end
    end

endmodule
