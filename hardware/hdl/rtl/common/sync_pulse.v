//-----------------------------------------------------------------------------
// Title         : Pulse Synchronizer
// Project       : PYNQ-Z2 SNN Accelerator
// File          : sync_pulse.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : CDC pulse synchronizer with handshaking
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module sync_pulse (
    // Source clock domain
    input  wire src_clk,
    input  wire src_rst_n,
    input  wire src_pulse,
    output wire src_busy,
    
    // Destination clock domain
    input  wire dst_clk,
    input  wire dst_rst_n,
    output reg  dst_pulse
);

    // Toggle synchronizer for pulse
    reg src_toggle;
    reg [2:0] dst_sync;
    reg dst_toggle_prev;
    
    // Source domain: convert pulse to toggle
    always @(posedge src_clk) begin
        if (!src_rst_n) begin
            src_toggle <= 1'b0;
        end else if (src_pulse) begin
            src_toggle <= ~src_toggle;
        end
    end
    
    // Destination domain: synchronize and detect edge
    always @(posedge dst_clk) begin
        if (!dst_rst_n) begin
            dst_sync <= 3'b000;
            dst_toggle_prev <= 1'b0;
            dst_pulse <= 1'b0;
        end else begin
            // 3-stage synchronizer
            dst_sync <= {dst_sync[1:0], src_toggle};
            dst_toggle_prev <= dst_sync[2];
            
            // Edge detection
            dst_pulse <= dst_sync[2] ^ dst_toggle_prev;
        end
    end
    
    // Busy signal (optional)
    assign src_busy = 1'b0; // Simple version doesn't track busy

endmodule

// Enhanced version with acknowledge
module sync_pulse_ack (
    // Source clock domain
    input  wire src_clk,
    input  wire src_rst_n,
    input  wire src_pulse,
    output wire src_busy,
    output reg  src_done,
    
    // Destination clock domain
    input  wire dst_clk,
    input  wire dst_rst_n,
    output reg  dst_pulse
);

    // Handshake signals
    reg src_req;
    reg [2:0] src_ack_sync;
    reg dst_ack;
    reg [2:0] dst_req_sync;
    
    // Source domain
    always @(posedge src_clk) begin
        if (!src_rst_n) begin
            src_req <= 1'b0;
            src_done <= 1'b0;
            src_ack_sync <= 3'b000;
        end else begin
            // Synchronize acknowledge
            src_ack_sync <= {src_ack_sync[1:0], dst_ack};
            
            // Handshake logic
            src_done <= 1'b0;
            if (src_pulse && !src_req && !src_ack_sync[2]) begin
                src_req <= 1'b1;
            end else if (src_req && src_ack_sync[2]) begin
                src_req <= 1'b0;
                src_done <= 1'b1;
            end
        end
    end
    
    // Destination domain
    always @(posedge dst_clk) begin
        if (!dst_rst_n) begin
            dst_req_sync <= 3'b000;
            dst_ack <= 1'b0;
            dst_pulse <= 1'b0;
        end else begin
            // Synchronize request
            dst_req_sync <= {dst_req_sync[1:0], src_req};
            
            // Generate pulse and acknowledge
            dst_pulse <= 1'b0;
            if (dst_req_sync[2] && !dst_ack) begin
                dst_pulse <= 1'b1;
                dst_ack <= 1'b1;
            end else if (!dst_req_sync[2] && dst_ack) begin
                dst_ack <= 1'b0;
            end
        end
    end
    
    assign src_busy = src_req || src_ack_sync[2];

endmodule
