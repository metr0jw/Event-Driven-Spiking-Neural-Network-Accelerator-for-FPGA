//-----------------------------------------------------------------------------
// Title         : SNN Max Pooling Layer
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_maxpool2d.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : PyTorch-compatible SNN 2D max pooling layer
//                 Spike-based implementation using temporal max detection
//-----------------------------------------------------------------------------

module snn_maxpool2d #(
    parameter INPUT_WIDTH    = 28,     // Input feature map width
    parameter INPUT_HEIGHT   = 28,     // Input feature map height
    parameter INPUT_CHANNELS = 32,     // Number of input channels
    parameter POOL_SIZE      = 2,      // Pooling window size (2x2, 3x3, etc.)
    parameter STRIDE         = 2,      // Pooling stride
    parameter TIME_WIDTH     = 16      // Timestamp precision
)(
    input wire clk,
    input wire reset,
    input wire enable,
    
    // Input spike interface (AXI-Stream)
    input wire [47:0] s_axis_input_tdata,    // {timestamp[15:0], channel[7:0], y[7:0], x[7:0], valid[7:0]}
    input wire s_axis_input_tvalid,
    output reg s_axis_input_tready,
    input wire s_axis_input_tlast,
    
    // Output spike interface (AXI-Stream)
    output reg [47:0] m_axis_output_tdata,   // {timestamp[15:0], channel[7:0], y[7:0], x[7:0], valid[7:0]}
    output reg m_axis_output_tvalid,
    input wire m_axis_output_tready,
    output reg m_axis_output_tlast,
    
    // Configuration
    input wire [15:0] pooling_window_time,   // Time window for pooling (in cycles)
    input wire winner_take_all_enable,       // Enable winner-take-all mode
    
    // Status
    output reg [31:0] input_spike_count,
    output reg [31:0] output_spike_count,
    output reg computation_done
);

    // Calculated parameters
    localparam OUTPUT_WIDTH = (INPUT_WIDTH - POOL_SIZE) / STRIDE + 1;
    localparam OUTPUT_HEIGHT = (INPUT_HEIGHT - POOL_SIZE) / STRIDE + 1;
    
    // Internal signals
    reg [15:0] input_timestamp;
    reg [7:0] input_x, input_y, input_ch;
    reg input_spike_valid;
    
    // Spike timing memory for max pooling (earliest spike wins)
    reg [TIME_WIDTH-1:0] earliest_spike_time [0:INPUT_CHANNELS-1][0:OUTPUT_HEIGHT-1][0:OUTPUT_WIDTH-1];
    reg [7:0] earliest_spike_x [0:INPUT_CHANNELS-1][0:OUTPUT_HEIGHT-1][0:OUTPUT_WIDTH-1];
    reg [7:0] earliest_spike_y [0:INPUT_CHANNELS-1][0:OUTPUT_HEIGHT-1][0:OUTPUT_WIDTH-1];
    reg spike_present [0:INPUT_CHANNELS-1][0:OUTPUT_HEIGHT-1][0:OUTPUT_WIDTH-1];
    
    // Spike strength memory (for max pooling based on spike frequency)
    reg [7:0] spike_strength [0:INPUT_CHANNELS-1][0:OUTPUT_HEIGHT-1][0:OUTPUT_WIDTH-1];
    
    // Timing control
    reg [TIME_WIDTH-1:0] current_time;
    reg [TIME_WIDTH-1:0] window_start_time;
    reg pooling_window_active;
    
    // Output generation
    reg [7:0] output_x, output_y, output_ch;
    reg output_spike_pending;
    reg [2:0] output_state;
    reg [TIME_WIDTH-1:0] output_timestamp;
    
    // Performance counters
    reg [31:0] local_input_count;
    reg [31:0] local_output_count;
    
    // Loop variables for reset
    integer i, j, k;
    
    // Pool coordinate calculation registers
    reg [7:0] pool_x, pool_y;
    
    // Input spike parsing
    always @(*) begin
        input_timestamp = s_axis_input_tdata[47:32];
        input_ch = s_axis_input_tdata[31:24];
        input_y = s_axis_input_tdata[23:16];
        input_x = s_axis_input_tdata[15:8];
        input_spike_valid = s_axis_input_tdata[7:0] != 0;
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
            current_time <= 16'b0;
            window_start_time <= 16'b0;
            pooling_window_active <= 1'b0;
            
            // Reset spike memories
            for (i = 0; i < INPUT_CHANNELS; i = i + 1) begin
                for (j = 0; j < OUTPUT_HEIGHT; j = j + 1) begin
                    for (k = 0; k < OUTPUT_WIDTH; k = k + 1) begin
                        earliest_spike_time[i][j][k] <= {TIME_WIDTH{1'b1}}; // Max value
                        earliest_spike_x[i][j][k] <= 8'b0;
                        earliest_spike_y[i][j][k] <= 8'b0;
                        spike_present[i][j][k] <= 1'b0;
                        spike_strength[i][j][k] <= 8'b0;
                    end
                end
            end
            
        end else if (enable) begin
            
            // Update current time
            current_time <= current_time + 1;
            
            // Manage pooling windows
            if (!pooling_window_active) begin
                pooling_window_active <= 1'b1;
                window_start_time <= current_time;
            end else if (current_time - window_start_time >= pooling_window_time) begin
                // Window completed, generate outputs
                if (!output_spike_pending) begin
                    output_state <= 3'b001;
                    output_ch <= 8'b0;
                    output_y <= 8'b0;
                    output_x <= 8'b0;
                    pooling_window_active <= 1'b0;
                end
            end
            
            // Input spike processing
            if (s_axis_input_tvalid && s_axis_input_tready && input_spike_valid) begin
                local_input_count <= local_input_count + 1;
                
                // Calculate which pooling windows this spike contributes to
                pool_x = calc_pool_x(input_x);
                pool_y = calc_pool_y(input_y);
                
                // Check bounds and window membership
                if (pool_x < OUTPUT_WIDTH && pool_y < OUTPUT_HEIGHT &&
                    is_in_pool_window(input_x, input_y, pool_x, pool_y)) begin
                    
                    if (winner_take_all_enable) begin
                        // Winner-take-all: earliest spike wins
                        if (!spike_present[input_ch][pool_y][pool_x] || 
                            input_timestamp < earliest_spike_time[input_ch][pool_y][pool_x]) begin
                            
                            earliest_spike_time[input_ch][pool_y][pool_x] <= input_timestamp;
                            earliest_spike_x[input_ch][pool_y][pool_x] <= input_x;
                            earliest_spike_y[input_ch][pool_y][pool_x] <= input_y;
                            spike_present[input_ch][pool_y][pool_x] <= 1'b1;
                        end
                    end else begin
                        // Frequency-based max pooling: count spikes
                        spike_strength[input_ch][pool_y][pool_x] <= 
                            spike_strength[input_ch][pool_y][pool_x] + 1;
                        spike_present[input_ch][pool_y][pool_x] <= 1'b1;
                        
                        // Update earliest timestamp for output timing
                        if (!spike_present[input_ch][pool_y][pool_x] || 
                            input_timestamp < earliest_spike_time[input_ch][pool_y][pool_x]) begin
                            earliest_spike_time[input_ch][pool_y][pool_x] <= input_timestamp;
                        end
                    end
                end
            end
            
            // Output spike generation state machine
            case (output_state)
                3'b001: begin // Check current position for output spike
                    if (spike_present[output_ch][output_y][output_x]) begin
                        if (winner_take_all_enable || 
                            spike_strength[output_ch][output_y][output_x] > 0) begin
                            
                            // Generate output spike
                            output_spike_pending <= 1'b1;
                            output_timestamp <= earliest_spike_time[output_ch][output_y][output_x];
                            local_output_count <= local_output_count + 1;
                        end
                    end
                    
                    // Reset for next window
                    spike_present[output_ch][output_y][output_x] <= 1'b0;
                    spike_strength[output_ch][output_y][output_x] <= 8'b0;
                    earliest_spike_time[output_ch][output_y][output_x] <= {TIME_WIDTH{1'b1}};
                    
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
                        // Finished all positions, start new window
                        output_state <= 3'b000;
                        pooling_window_active <= 1'b1;
                        window_start_time <= current_time;
                    end
                end
            endcase
            
            // Output spike transmission
            if (output_spike_pending && m_axis_output_tready) begin
                m_axis_output_tdata <= {output_timestamp, output_ch, output_y, output_x, 8'h01};
                m_axis_output_tvalid <= 1'b1;
                m_axis_output_tlast <= 1'b0;
                output_spike_pending <= 1'b0;
            end else if (!output_spike_pending) begin
                m_axis_output_tvalid <= 1'b0;
            end
        end
    end
    
    // Output assignments
    always @(*) begin
        input_spike_count = local_input_count;
        output_spike_count = local_output_count;
        computation_done = (output_state == 3'b000) && !output_spike_pending && !pooling_window_active;
    end

endmodule
