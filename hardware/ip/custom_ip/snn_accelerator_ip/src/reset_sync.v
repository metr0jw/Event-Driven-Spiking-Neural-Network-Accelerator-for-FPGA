//-----------------------------------------------------------------------------
// Title         : Reset Synchronizer
// Project       : PYNQ-Z2 SNN Accelerator
// File          : reset_sync.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Asynchronous reset synchronizer
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module reset_sync (
    input  wire clk,
    input  wire async_rst_n,
    output reg  sync_rst_n
);

    // 2-stage synchronizer for reset
    reg rst_meta;
    
    always @(posedge clk or negedge async_rst_n) begin
        if (!async_rst_n) begin
            rst_meta <= 1'b0;
            sync_rst_n <= 1'b0;
        end else begin
            rst_meta <= 1'b1;
            sync_rst_n <= rst_meta;
        end
    end

endmodule

// Enhanced version with configurable stages
module reset_sync_n #(
    parameter STAGES = 2,
    parameter INIT_VALUE = 0
)(
    input  wire clk,
    input  wire async_rst_n,
    output wire sync_rst_n
);

    reg [STAGES-1:0] sync_chain;
    
    always @(posedge clk or negedge async_rst_n) begin
        if (!async_rst_n) begin
            sync_chain <= {STAGES{1'b0}};
        end else begin
            sync_chain <= {sync_chain[STAGES-2:0], 1'b1};
        end
    end
    
    assign sync_rst_n = sync_chain[STAGES-1];

endmodule

// Power-on reset generator
module por_gen #(
    parameter DELAY_CYCLES = 16
)(
    input  wire clk,
    output reg  por_n
);

    reg [$clog2(DELAY_CYCLES)-1:0] por_counter;
    
    initial begin
        por_counter = 0;
        por_n = 1'b0;
    end
    
    always @(posedge clk) begin
        if (por_counter < DELAY_CYCLES-1) begin
            por_counter <= por_counter + 1'b1;
            por_n <= 1'b0;
        end else begin
            por_n <= 1'b1;
        end
    end

endmodule
