//-----------------------------------------------------------------------------
// Title         : SNN Average Pooling Layer
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_avgpool2d.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : PyTorch-compatible SNN 2D average pooling layer
//                 Area-efficient implementation without surrogate gradients
//-----------------------------------------------------------------------------

module snn_avgpool2d #(
    parameter INPUT_WIDTH    = 28,     // Input feature map width
    parameter INPUT_HEIGHT   = 28,     // Input feature map height
    parameter INPUT_CHANNELS = 32,     // Number of input channels
    parameter POOL_SIZE      = 2,      // Pooling window size (2x2, 3x3, etc.)
    parameter STRIDE         = 2,      // Pooling stride
    parameter VMEM_WIDTH     = 16,     // Membrane potential precision
    parameter POOL_THRESHOLD = 16'h2000 // Pooling threshold (0.125 in Q8.8)
)(
    input wire clk,
    input wire reset,
    input wire enable,
    
    // Input spike interface (AXI-Stream)
    input wire [31:0] s_axis_input_tdata,    // {channel[7:0], y[7:0], x[7:0], valid[7:0]}
    input wire s_axis_input_tvalid,
    output reg s_axis_input_tready,
    input wire s_axis_input_tlast,
    
    // Output spike interface (AXI-Stream)
    output reg [31:0] m_axis_output_tdata,   // {channel[7:0], y[7:0], x[7:0], valid[7:0]}
    output reg m_axis_output_tvalid,
    input wire m_axis_output_tready,
    output reg m_axis_output_tlast,
    
    // Configuration
    input wire [15:0] threshold_config,
    input wire [7:0] decay_factor,           // Membrane leak factor
    input wire [7:0] pooling_weight,         // Weight for each spike in pool
    
    // Status
    output reg [31:0] input_spike_count,
    output reg [31:0] output_spike_count,
    output reg computation_done
);

    // Calculated parameters
    localparam OUTPUT_WIDTH = (INPUT_WIDTH - POOL_SIZE) / STRIDE + 1;
    localparam OUTPUT_HEIGHT = (INPUT_HEIGHT - POOL_SIZE) / STRIDE + 1;
    
    // Internal signals
    reg [7:0] input_x, input_y, input_ch;
    reg input_spike_valid;
    
    // Pooling accumulator memory (for each output position)
    reg signed [VMEM_WIDTH-1:0] pool_accumulator [0:INPUT_CHANNELS-1][0:OUTPUT_HEIGHT-1][0:OUTPUT_WIDTH-1];
    
    // Spike counting for average calculation
    reg [7:0] spike_count [0:INPUT_CHANNELS-1][0:OUTPUT_HEIGHT-1][0:OUTPUT_WIDTH-1];
    
    // Timing control for pooling windows
    reg [15:0] time_window_counter;
    reg [15:0] pooling_window_size;  // Number of time steps for pooling window
    
    // Output spike generation
    reg [7:0] output_x, output_y, output_ch;
    reg output_spike_pending;
    reg [2:0] output_state;
    
    // Performance counters
    reg [31:0] local_input_count;
    reg [31:0] local_output_count;
    
    // Loop variables for reset
    integer i, j, k;
    integer ch, y, x;  // Loop variables for decay
    
    // Pool coordinate calculation registers
    reg [7:0] pool_x, pool_y;
    
    // Output calculation registers (moved outside always block for Verilog-2001)
    reg signed [VMEM_WIDTH-1:0] avg_value;
    reg [7:0] current_spike_count;
    
    // Input spike parsing
    always @(*) begin
        input_x = s_axis_input_tdata[7:0];
        input_y = s_axis_input_tdata[15:8];
        input_ch = s_axis_input_tdata[23:16];
        input_spike_valid = s_axis_input_tdata[31:24] != 0;
    end
    
    // Calculate which pooling window contains the input coordinate
    function [7:0] calc_pool_x;
        input [7:0] input_x;
        begin
            calc_pool_x = input_x / STRIDE;
        end
    endfunction
    
    function [7:0] calc_pool_y;
        input [7:0] input_y;
        begin
            calc_pool_y = input_y / STRIDE;
        end
    endfunction
    
    // Check if input coordinate contributes to pooling window
    function is_in_pool_window;
        input [7:0] in_x, in_y, pool_x, pool_y;
        reg [7:0] pool_start_x, pool_start_y;
        reg [7:0] pool_end_x, pool_end_y;
        begin
            pool_start_x = pool_x * STRIDE;
            pool_start_y = pool_y * STRIDE;
            pool_end_x = pool_start_x + POOL_SIZE - 1;
            pool_end_y = pool_start_y + POOL_SIZE - 1;
            
            is_in_pool_window = (in_x >= pool_start_x) && (in_x <= pool_end_x) &&
                               (in_y >= pool_start_y) && (in_y <= pool_end_y);
        end
    endfunction
    
    // Main processing logic
    always @(posedge clk) begin
        if (reset) begin
            // Reset all state
            s_axis_input_tready <= 1'b1;
            m_axis_output_tvalid <= 1'b0;
            m_axis_output_tlast <= 1'b0;
            computation_done <= 1'b0;
            output_spike_pending <= 1'b0;
            output_state <= 3'b000;
            local_input_count <= 32'b0;
            local_output_count <= 32'b0;
            time_window_counter <= 16'b0;
            pooling_window_size <= 16'd100; // Default: 100 time steps per pooling window
            
            // Reset accumulators and spike counts
            for (i = 0; i < INPUT_CHANNELS; i = i + 1) begin
                for (j = 0; j < OUTPUT_HEIGHT; j = j + 1) begin
                    for (k = 0; k < OUTPUT_WIDTH; k = k + 1) begin
                        pool_accumulator[i][j][k] <= 0;
                        spike_count[i][j][k] <= 0;
                    end
                end
            end
            
        end else if (enable) begin
            
            // Time window management
            time_window_counter <= time_window_counter + 1;
            
            // Input spike processing
            if (s_axis_input_tvalid && s_axis_input_tready && input_spike_valid) begin
                local_input_count <= local_input_count + 1;
                
                // Calculate which pooling windows this spike contributes to
                pool_x = calc_pool_x(input_x);
                pool_y = calc_pool_y(input_y);
                
                // Check bounds
                if (pool_x < OUTPUT_WIDTH && pool_y < OUTPUT_HEIGHT &&
                    is_in_pool_window(input_x, input_y, pool_x, pool_y)) begin
                    
                    // Add spike to accumulator
                    pool_accumulator[input_ch][pool_y][pool_x] <= 
                        pool_accumulator[input_ch][pool_y][pool_x] + pooling_weight;
                    
                    // Increment spike count for average calculation
                    spike_count[input_ch][pool_y][pool_x] <= 
                        spike_count[input_ch][pool_y][pool_x] + 1;
                end
            end
            
            // Pooling window completion and output generation
            if (time_window_counter >= pooling_window_size) begin
                time_window_counter <= 16'b0;
                
                if (!output_spike_pending) begin
                    output_state <= 3'b001; // Start output generation
                    output_ch <= 8'b0;
                    output_y <= 8'b0;
                    output_x <= 8'b0;
                end
            end
            
            // Output spike generation state machine
            case (output_state)
                3'b001: begin // Check current position for spike
                    current_spike_count = spike_count[output_ch][output_y][output_x];
                    
                    if (current_spike_count > 0) begin
                        // Calculate average: accumulator / spike_count
                        avg_value = pool_accumulator[output_ch][output_y][output_x] / current_spike_count;
                        
                        if (avg_value >= threshold_config) begin
                            // Generate output spike
                            output_spike_pending <= 1'b1;
                            local_output_count <= local_output_count + 1;
                        end
                    end
                    
                    // Reset accumulator for next window
                    pool_accumulator[output_ch][output_y][output_x] <= 0;
                    spike_count[output_ch][output_y][output_x] <= 0;
                    
                    output_state <= 3'b010;
                end
                
                3'b010: begin // Move to next position
                    if (output_x < OUTPUT_WIDTH - 1) begin
                        output_x <= output_x + 1;
                        output_state <= 3'b001;
                    end else if (output_y < OUTPUT_HEIGHT - 1) begin
                        output_x <= 8'b0;
                        output_y <= output_y + 1;
                        output_state <= 3'b001;
                    end else if (output_ch < INPUT_CHANNELS - 1) begin
                        output_x <= 8'b0;
                        output_y <= 8'b0;
                        output_ch <= output_ch + 1;
                        output_state <= 3'b001;
                    end else begin
                        // Finished all positions
                        output_state <= 3'b000;
                    end
                end
            endcase
            
            // Output spike transmission
            if (output_spike_pending && m_axis_output_tready) begin
                m_axis_output_tdata <= {8'h01, output_ch, output_y, output_x};
                m_axis_output_tvalid <= 1'b1;
                m_axis_output_tlast <= 1'b0;
                output_spike_pending <= 1'b0;
            end else if (!output_spike_pending) begin
                m_axis_output_tvalid <= 1'b0;
            end
            
            // Apply decay to all accumulators (membrane leak)
            if (time_window_counter[3:0] == 4'b0000) begin // Every 16 cycles
                for (ch = 0; ch < INPUT_CHANNELS; ch = ch + 1) begin
                    for (y = 0; y < OUTPUT_HEIGHT; y = y + 1) begin
                        for (x = 0; x < OUTPUT_WIDTH; x = x + 1) begin
                            pool_accumulator[ch][y][x] <= 
                                (pool_accumulator[ch][y][x] * decay_factor) >> 8;
                        end
                    end
                end
            end
        end
    end
    
    // Output assignments
    always @(*) begin
        input_spike_count = local_input_count;
        output_spike_count = local_output_count;
        computation_done = (output_state == 3'b000) && !output_spike_pending;
    end

endmodule
