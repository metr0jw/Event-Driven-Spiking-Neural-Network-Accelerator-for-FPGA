//-----------------------------------------------------------------------------
// Title         : Weight Memory with Configurable Storage (Verilog-2001)
// Project       : PYNQ-Z2 SNN Accelerator
// File          : weight_memory_verilog.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Parameterized weight storage using BRAM or distributed RAM
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module weight_memory #(
    parameter NUM_WEIGHTS   = 4096,    // Total number of weights
    parameter WEIGHT_WIDTH  = 9,       // Bits per weight (8 + sign)
    parameter ADDR_WIDTH    = 12,      // log2(4096) = 12 bits for address
    parameter USE_BRAM      = 1,       // 1: BRAM, 0: Distributed RAM
    parameter INIT_FILE     = ""       // Memory initialization file
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Read port
    input  wire                     read_en,
    input  wire [ADDR_WIDTH-1:0]    read_addr,
    output reg  [WEIGHT_WIDTH-1:0]  read_data,
    output reg                      read_valid,
    
    // Write port
    input  wire                     write_en,
    input  wire [ADDR_WIDTH-1:0]    write_addr,
    input  wire [WEIGHT_WIDTH-1:0]  write_data
);

    // Weight storage
    generate
        if (USE_BRAM) begin : bram_storage
            // Use Xilinx Block RAM primitive for PYNQ-Z2
            (* ram_style = "block" *)
            reg [WEIGHT_WIDTH-1:0] bram_array [0:NUM_WEIGHTS-1];
            
            // Variable for initialization loop
            integer i;
            
            // Initialize memory if file provided
            initial begin
                if (INIT_FILE != "") begin
                    $readmemh(INIT_FILE, bram_array);
                end else begin
                    for (i = 0; i < NUM_WEIGHTS; i = i + 1) begin
                        bram_array[i] = 0;
                    end
                end
            end
            
            // Read port - 1 cycle latency
            always @(posedge clk) begin
                if (!rst_n) begin
                    read_data <= 0;
                    read_valid <= 1'b0;
                end else begin
                    if (read_en) begin
                        read_data <= bram_array[read_addr];
                        read_valid <= 1'b1;
                    end else begin
                        read_valid <= 1'b0;
                    end
                end
            end
            
            // Write port
            always @(posedge clk) begin
                if (write_en) begin
                    bram_array[write_addr] <= write_data;
                end
            end
            
        end else begin : distributed_storage
            // Use distributed RAM for smaller arrays
            (* ram_style = "distributed" *)
            reg [WEIGHT_WIDTH-1:0] dist_ram_array [0:NUM_WEIGHTS-1];
            
            // Variable for initialization loop  
            integer j;
            
            // Initialize memory
            initial begin
                if (INIT_FILE != "") begin
                    $readmemh(INIT_FILE, dist_ram_array);
                end else begin
                    for (j = 0; j < NUM_WEIGHTS; j = j + 1) begin
                        dist_ram_array[j] = 0;
                    end
                end
            end
            
            // Read port - combinational for distributed RAM
            always @(*) begin
                if (read_en) begin
                    read_data = dist_ram_array[read_addr];
                    read_valid = 1'b1;
                end else begin
                    read_data = 0;
                    read_valid = 1'b0;
                end
            end
            
            // Write port
            always @(posedge clk) begin
                if (write_en) begin
                    dist_ram_array[write_addr] <= write_data;
                end
            end
            
        end
    endgenerate

endmodule
