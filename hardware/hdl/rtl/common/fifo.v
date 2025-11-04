//-----------------------------------------------------------------------------
// Title         : Synchronous FIFO
// Project       : PYNQ-Z2 SNN Accelerator
// File          : fifo.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Parameterizable synchronous FIFO with status flags
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module fifo #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH = 16,
    parameter ALMOST_FULL_THRESHOLD = DEPTH - 2,
    parameter ALMOST_EMPTY_THRESHOLD = 2
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Write interface
    input  wire                     wr_en,
    input  wire [DATA_WIDTH-1:0]    wr_data,
    output wire                     full,
    output wire                     almost_full,
    
    // Read interface
    input  wire                     rd_en,
    output reg  [DATA_WIDTH-1:0]    rd_data,
    output wire                     empty,
    output wire                     almost_empty,
    
    // Status
    output wire [$clog2(DEPTH):0]   count,
    output reg                      overflow,
    output reg                      underflow
);

    // Local parameters
    localparam ADDR_WIDTH = $clog2(DEPTH);
    
    // Memory array
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    
    // Pointers
    reg [ADDR_WIDTH:0] wr_ptr;
    reg [ADDR_WIDTH:0] rd_ptr;
    
    // Internal signals
    wire [ADDR_WIDTH:0] wr_ptr_next;
    wire [ADDR_WIDTH:0] rd_ptr_next;
    wire wr_en_qualified;
    wire rd_en_qualified;
    
    // Pointer logic
    assign wr_ptr_next = wr_ptr + 1'b1;
    assign rd_ptr_next = rd_ptr + 1'b1;
    
    // Status flags
    assign full = (wr_ptr[ADDR_WIDTH] != rd_ptr[ADDR_WIDTH]) && 
                  (wr_ptr[ADDR_WIDTH-1:0] == rd_ptr[ADDR_WIDTH-1:0]);
    assign empty = (wr_ptr == rd_ptr);
    
    // Count calculation
    wire [ADDR_WIDTH:0] count_raw;
    assign count_raw = wr_ptr - rd_ptr;
    assign count = count_raw;
    
    // Almost full/empty flags
    assign almost_full = (count >= ALMOST_FULL_THRESHOLD);
    assign almost_empty = (count <= ALMOST_EMPTY_THRESHOLD);
    
    // Qualified write/read enables
    assign wr_en_qualified = wr_en && !full;
    assign rd_en_qualified = rd_en && !empty;
    
    // Write logic
    always @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr <= 0;
        end else begin
            if (wr_en_qualified) begin
                mem[wr_ptr[ADDR_WIDTH-1:0]] <= wr_data;
                wr_ptr <= wr_ptr_next;
            end
        end
    end
    
    // Read logic
    always @(posedge clk) begin
        if (!rst_n) begin
            rd_ptr <= 0;
            rd_data <= 0;
        end else begin
            if (rd_en_qualified) begin
                rd_data <= mem[rd_ptr[ADDR_WIDTH-1:0]];
                rd_ptr <= rd_ptr_next;
            end
        end
    end
    
    // Overflow/Underflow detection
    always @(posedge clk) begin
        if (!rst_n) begin
            overflow <= 1'b0;
            underflow <= 1'b0;
        end else begin
            if (wr_en && full)
                overflow <= 1'b1;
            if (rd_en && empty)
                underflow <= 1'b1;
        end
    end

endmodule
