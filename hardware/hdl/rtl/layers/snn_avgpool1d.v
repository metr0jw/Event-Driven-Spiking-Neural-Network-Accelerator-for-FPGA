//-----------------------------------------------------------------------------
// Title         : SNN 1D Average Pooling Layer (Verilog-2001 Compatible)
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_avgpool1d.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : 1D average pooling for temporal/sequential data in SNNs
//                 Compatible with snn_conv1d output
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module snn_avgpool1d #(
    parameter INPUT_LENGTH = 128,
    parameter INPUT_CHANNELS = 32,
    parameter POOL_SIZE = 2,
    parameter STRIDE = 2,
    parameter OUTPUT_LENGTH = INPUT_LENGTH / STRIDE,
    
    // Precision parameters
    parameter VMEM_WIDTH = 16,          // Q8.8 membrane potential
    parameter ADDR_WIDTH = 16,
    parameter TIMESTAMP_WIDTH = 16
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    // Input spike interface (AXI-Stream)
    input wire s_axis_input_tvalid,
    input wire [31:0] s_axis_input_tdata,
    input wire s_axis_input_tlast,
    output reg s_axis_input_tready,
    
    // Output spike interface (AXI-Stream)
    output reg m_axis_output_tvalid,
    output reg [31:0] m_axis_output_tdata,
    output reg m_axis_output_tlast,
    input wire m_axis_output_tready,
    
    // Configuration
    input wire config_valid,
    input wire [31:0] config_data,
    
    // Status
    output reg busy,
    output reg layer_done
);

    // Accumulator memory for Verilog-2001 compatibility
    reg [VMEM_WIDTH-1:0] accumulator_mem [0:INPUT_CHANNELS*OUTPUT_LENGTH-1];
    reg [7:0] spike_count_mem [0:INPUT_CHANNELS*OUTPUT_LENGTH-1];
    
    // Input spike buffer
    reg input_spike_buffer [0:INPUT_LENGTH-1];
    reg [7:0] input_channel_id;
    reg [15:0] input_position;
    
    // Internal signals
    reg [2:0] pool_state;
    reg [15:0] current_ch;
    reg [15:0] current_pos;
    reg [31:0] pool_sum;
    reg [7:0] pool_count;
    integer k; // Moved outside always block for Verilog-2001
    
    // State machine parameters
    localparam IDLE = 3'b000;
    localparam PROCESS_SPIKES = 3'b001;
    localparam POOLING = 3'b010;
    localparam OUTPUT_SPIKES = 3'b011;
    localparam DONE = 3'b100;
    
    // Helper function to calculate flattened index
    function [31:0] pool_index;
        input [15:0] ch;
        input [15:0] pos;
        begin
            pool_index = ch * OUTPUT_LENGTH + pos;
        end
    endfunction
    
    // Input spike parsing
    always @(*) begin
        if (s_axis_input_tvalid && s_axis_input_tready) begin
            input_channel_id = s_axis_input_tdata[15:8];
            input_position = s_axis_input_tdata[31:16];
        end else begin
            input_channel_id = 8'b0;
            input_position = 16'b0;
        end
    end
    
    // Main state machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pool_state <= IDLE;
            current_ch <= 16'b0;
            current_pos <= 16'b0;
            pool_sum <= 32'b0;
            pool_count <= 8'b0;
            
            s_axis_input_tready <= 1'b0;
            m_axis_output_tvalid <= 1'b0;
            m_axis_output_tdata <= 32'b0;
            m_axis_output_tlast <= 1'b0;
            
            busy <= 1'b0;
            layer_done <= 1'b0;
            
        end else if (enable) begin
            case (pool_state)
                IDLE: begin
                    if (config_valid) begin
                        pool_state <= PROCESS_SPIKES;
                        s_axis_input_tready <= 1'b1;
                        busy <= 1'b1;
                    end else begin
                        s_axis_input_tready <= 1'b1;
                        busy <= 1'b0;
                    end
                end
                
                PROCESS_SPIKES: begin
                    if (s_axis_input_tvalid && s_axis_input_tready) begin
                        // Store input spike
                        if (input_position < INPUT_LENGTH) begin
                            input_spike_buffer[input_position] <= 1'b1;
                        end
                        
                        if (s_axis_input_tlast) begin
                            pool_state <= POOLING;
                            s_axis_input_tready <= 1'b0;
                            current_ch <= 16'b0;
                            current_pos <= 16'b0;
                        end
                    end
                end
                
                POOLING: begin
                    // Perform 1D average pooling
                    pool_sum <= 32'b0;
                    pool_count <= 8'b0;
                    
                    // Count spikes in pool window
                    for (k = 0; k < POOL_SIZE; k = k + 1) begin
                        if ((current_pos * STRIDE + k) < INPUT_LENGTH) begin
                            if (input_spike_buffer[current_pos * STRIDE + k]) begin
                                pool_count <= pool_count + 1;
                            end
                        end
                    end
                    
                    // Store average
                    accumulator_mem[pool_index(current_ch, current_pos)] <= pool_count;
                    spike_count_mem[pool_index(current_ch, current_pos)] <= pool_count;
                    
                    // Move to next position
                    if (current_pos < OUTPUT_LENGTH - 1) begin
                        current_pos <= current_pos + 1;
                    end else begin
                        current_pos <= 16'b0;
                        if (current_ch < INPUT_CHANNELS - 1) begin
                            current_ch <= current_ch + 1;
                        end else begin
                            pool_state <= OUTPUT_SPIKES;
                            current_ch <= 16'b0;
                            current_pos <= 16'b0;
                        end
                    end
                end
                
                OUTPUT_SPIKES: begin
                    if (m_axis_output_tready) begin
                        // Check if we should output a spike
                        if (spike_count_mem[pool_index(current_ch, current_pos)] > 0) begin
                            m_axis_output_tvalid <= 1'b1;
                            m_axis_output_tdata <= {current_pos, current_ch[7:0], 7'b0, 1'b1};
                        end else begin
                            m_axis_output_tvalid <= 1'b0;
                        end
                        
                        // Move to next position
                        if (current_pos < OUTPUT_LENGTH - 1) begin
                            current_pos <= current_pos + 1;
                        end else begin
                            current_pos <= 16'b0;
                            if (current_ch < INPUT_CHANNELS - 1) begin
                                current_ch <= current_ch + 1;
                            end else begin
                                m_axis_output_tlast <= 1'b1;
                                pool_state <= DONE;
                            end
                        end
                    end
                end
                
                DONE: begin
                    m_axis_output_tvalid <= 1'b0;
                    m_axis_output_tlast <= 1'b0;
                    busy <= 1'b0;
                    layer_done <= 1'b1;
                    pool_state <= IDLE;
                end
                
                default: begin
                    pool_state <= IDLE;
                end
            endcase
        end
    end
    
    // Initialize memories
    integer i;
    initial begin
        for (i = 0; i < INPUT_CHANNELS * OUTPUT_LENGTH; i = i + 1) begin
            accumulator_mem[i] = 16'b0;
            spike_count_mem[i] = 8'b0;
        end
        
        for (i = 0; i < INPUT_LENGTH; i = i + 1) begin
            input_spike_buffer[i] = 1'b0;
        end
    end

endmodule
