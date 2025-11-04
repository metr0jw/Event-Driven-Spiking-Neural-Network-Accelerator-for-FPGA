//-----------------------------------------------------------------------------
// Title         : SNN Pooling Layer
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_pooling.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : PyTorch-compatible pooling layer for SNN (Max/Avg/Adaptive)
//-----------------------------------------------------------------------------

module snn_pooling #(
    // Layer parameters
    parameter INPUT_WIDTH = 28,
    parameter INPUT_HEIGHT = 28,
    parameter INPUT_CHANNELS = 32,
    parameter KERNEL_SIZE = 2,
    parameter STRIDE = 2,
    parameter PADDING = 0,
    
    // Precision parameters
    parameter DATA_WIDTH = 8,           // INT8 for activations
    parameter GRADIENT_WIDTH = 16,      // FP16 for surrogate gradients
    
    // Pooling type
    parameter POOL_TYPE = "MAX",         // "MAX", "AVG", "ADAPTIVE_MAX", "ADAPTIVE_AVG"
    
    // Calculate log2 ceiling manually for Verilog-2001 compatibility
    parameter KERNEL_BITS = (KERNEL_SIZE <= 2) ? 2 :
                           (KERNEL_SIZE <= 4) ? 3 :
                           (KERNEL_SIZE <= 8) ? 4 :
                           (KERNEL_SIZE <= 16) ? 5 : 6,
    parameter SUM_BITS = KERNEL_BITS + KERNEL_BITS  // For KERNEL_SIZE*KERNEL_SIZE
) (
    input wire clk,
    input wire reset,
    input wire enable,
    
    // Control signals
    input wire training_mode,
    input wire [1:0] pool_mode,         // 0: forward, 1: backward
    
    // Configuration (for adaptive pooling)
    input wire [15:0] target_width,
    input wire [15:0] target_height,
    
    // Input feature maps (AXI-Stream)
    input wire [DATA_WIDTH-1:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    input wire s_axis_input_tlast,
    input wire [7:0] s_axis_input_tuser, // Channel info
    
    // Output feature maps (AXI-Stream)
    output wire [DATA_WIDTH-1:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready,
    output wire m_axis_output_tlast,
    output wire [7:0] m_axis_output_tuser,
    
    // Gradient interface for backpropagation
    input wire [GRADIENT_WIDTH-1:0] s_axis_grad_input_tdata,
    input wire s_axis_grad_input_tvalid,
    output wire s_axis_grad_input_tready,
    
    output wire [GRADIENT_WIDTH-1:0] m_axis_grad_output_tdata,
    output wire m_axis_grad_output_tvalid,
    input wire m_axis_grad_output_tready,
    
    // Performance counters
    output wire [31:0] pool_ops_count,
    output wire [31:0] spike_count
);

    // Calculate output dimensions
    function integer calc_output_dim;
        input integer input_dim;
        input integer kernel_size;
        input integer stride;
        input integer padding;
        begin
            calc_output_dim = (input_dim + 2*padding - kernel_size) / stride + 1;
        end
    endfunction
    
    // Adaptive pooling stride calculation
    function integer calc_adaptive_stride;
        input integer input_dim;
        input integer target_dim;
        begin
            calc_adaptive_stride = input_dim / target_dim;
        end
    endfunction
    
    localparam OUTPUT_WIDTH = (POOL_TYPE == "ADAPTIVE_MAX" || POOL_TYPE == "ADAPTIVE_AVG") ? 
                              16 : calc_output_dim(INPUT_WIDTH, KERNEL_SIZE, STRIDE, PADDING);
    localparam OUTPUT_HEIGHT = (POOL_TYPE == "ADAPTIVE_MAX" || POOL_TYPE == "ADAPTIVE_AVG") ? 
                               16 : calc_output_dim(INPUT_HEIGHT, KERNEL_SIZE, STRIDE, PADDING);
    
    // Internal signals
    reg [2:0] state;
    localparam IDLE = 3'b000,
               LOAD_INPUT = 3'b001,
               POOLING = 3'b010,
               OUTPUT = 3'b011,
               GRADIENT_CALC = 3'b100;
    
    // Counters and coordinates
    reg [15:0] input_x, input_y, input_ch;
    reg [15:0] output_x, output_y, output_ch;
    reg [15:0] kernel_x, kernel_y;
    reg [31:0] operation_count;
    reg [31:0] spike_counter;
    
    // Dynamic stride for adaptive pooling
    wire [15:0] effective_stride_x, effective_stride_y;
    wire [15:0] effective_kernel_x, effective_kernel_y;
    
    assign effective_stride_x = (POOL_TYPE == "ADAPTIVE_MAX" || POOL_TYPE == "ADAPTIVE_AVG") ?
                               calc_adaptive_stride(INPUT_WIDTH, target_width) : STRIDE;
    assign effective_stride_y = (POOL_TYPE == "ADAPTIVE_MAX" || POOL_TYPE == "ADAPTIVE_AVG") ?
                               calc_adaptive_stride(INPUT_HEIGHT, target_height) : STRIDE;
    assign effective_kernel_x = (POOL_TYPE == "ADAPTIVE_MAX" || POOL_TYPE == "ADAPTIVE_AVG") ?
                               effective_stride_x : KERNEL_SIZE;
    assign effective_kernel_y = (POOL_TYPE == "ADAPTIVE_MAX" || POOL_TYPE == "ADAPTIVE_AVG") ?
                               effective_stride_y : KERNEL_SIZE;
    
    // Input buffer for pooling window - flattened for Verilog-2001
    reg [DATA_WIDTH-1:0] pool_window_mem [0:KERNEL_SIZE*KERNEL_SIZE-1];
    reg [KERNEL_SIZE-1:0] window_valid_mem [0:KERNEL_SIZE-1];
    
    // Line buffers for input streaming - flattened for Verilog-2001  
    reg [DATA_WIDTH-1:0] line_buffer_mem [0:KERNEL_SIZE*INPUT_WIDTH-1];
    reg line_buffer_valid [0:KERNEL_SIZE-1];
    
    // Max pooling logic
    wire [DATA_WIDTH-1:0] max_pool_result;
    wire [3:0] max_position; // For gradient backprop tracking
    
    // Average pooling logic
    wire [DATA_WIDTH+SUM_BITS-1:0] avg_pool_sum;
    wire [DATA_WIDTH-1:0] avg_pool_result;
    
    // Gradient tracking for max pooling - flattened for Verilog-2001
    reg [3:0] max_indices_mem [0:OUTPUT_HEIGHT*OUTPUT_WIDTH*INPUT_CHANNELS-1];
    reg max_indices_valid;
    
    // Max pooling implementation
    max_pool_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .KERNEL_SIZE(KERNEL_SIZE)
    ) max_pool_inst (
        .clk(clk),
        .reset(reset),
        .pool_window(pool_window),
        .window_valid(window_valid),
        .max_result(max_pool_result),
        .max_position(max_position)
    );
    
    // Average pooling implementation
    avg_pool_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .KERNEL_SIZE(KERNEL_SIZE)
    ) avg_pool_inst (
        .clk(clk),
        .reset(reset),
        .pool_window(pool_window),
        .window_valid(window_valid),
        .avg_sum(avg_pool_sum),
        .avg_result(avg_pool_result)
    );
    
    // Output selection based on pooling type
    reg [DATA_WIDTH-1:0] pool_result;
    always @(*) begin
        case (POOL_TYPE)
            "MAX", "ADAPTIVE_MAX": pool_result = max_pool_result;
            "AVG", "ADAPTIVE_AVG": pool_result = avg_pool_result;
            default: pool_result = max_pool_result;
        endcase
    end
    
    // Main state machine
    always @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            input_x <= 0;
            input_y <= 0;
            input_ch <= 0;
            output_x <= 0;
            output_y <= 0;
            output_ch <= 0;
            kernel_x <= 0;
            kernel_y <= 0;
            operation_count <= 0;
            spike_counter <= 0;
            max_indices_valid <= 1'b0;
        end else if (enable) begin
            case (state)
                IDLE: begin
                    if (s_axis_input_tvalid && pool_mode == 2'b00) begin // Forward pass
                        state <= LOAD_INPUT;
                        input_x <= 0;
                        input_y <= 0;
                        input_ch <= 0;
                    end else if (training_mode && pool_mode == 2'b01) begin // Backward pass
                        state <= GRADIENT_CALC;
                    end
                end
                
                LOAD_INPUT: begin
                    if (s_axis_input_tvalid && s_axis_input_tready) begin
                        // Store input in line buffer
                        line_buffer[input_y % KERNEL_SIZE][input_x] <= s_axis_input_tdata;
                        line_buffer_valid[input_y % KERNEL_SIZE] <= 1'b1;
                        
                        // Count spikes
                        if (s_axis_input_tdata != 0) begin
                            spike_counter <= spike_counter + 1;
                        end
                        
                        // Fill pooling window when ready
                        if (input_x >= kernel_x && input_y >= kernel_y &&
                            input_x < kernel_x + effective_kernel_x &&
                            input_y < kernel_y + effective_kernel_y) begin
                            pool_window[input_y - kernel_y][input_x - kernel_x] <= s_axis_input_tdata;
                            window_valid[input_y - kernel_y][input_x - kernel_x] <= 1'b1;
                        end
                        
                        // Update coordinates
                        if (input_x == INPUT_WIDTH - 1) begin
                            input_x <= 0;
                            if (input_y == INPUT_HEIGHT - 1) begin
                                input_y <= 0;
                                if (input_ch == INPUT_CHANNELS - 1) begin
                                    input_ch <= 0;
                                    state <= POOLING;
                                    output_x <= 0;
                                    output_y <= 0;
                                    output_ch <= 0;
                                end else begin
                                    input_ch <= input_ch + 1;
                                end
                            end else begin
                                input_y <= input_y + 1;
                            end
                        end else begin
                            input_x <= input_x + 1;
                        end
                    end
                end
                
                POOLING: begin
                    // Perform pooling for current output position
                    if (m_axis_output_tready) begin
                        // Store max indices for gradient computation
                        if (POOL_TYPE == "MAX" || POOL_TYPE == "ADAPTIVE_MAX") begin
                            max_indices[output_y][output_x][output_ch] <= max_position;
                            max_indices_valid <= 1'b1;
                        end
                        
                        // Move to next output position
                        if (output_x == OUTPUT_WIDTH - 1) begin
                            output_x <= 0;
                            if (output_y == OUTPUT_HEIGHT - 1) begin
                                output_y <= 0;
                                if (output_ch == INPUT_CHANNELS - 1) begin
                                    output_ch <= 0;
                                    state <= OUTPUT;
                                end else begin
                                    output_ch <= output_ch + 1;
                                end
                            end else begin
                                output_y <= output_y + 1;
                            end
                        end else begin
                            output_x <= output_x + 1;
                        end
                        
                        operation_count <= operation_count + 1;
                    end
                end
                
                OUTPUT: begin
                    if (m_axis_output_tvalid && m_axis_output_tready) begin
                        if (m_axis_output_tlast) begin
                            state <= IDLE;
                        end
                    end
                end
                
                GRADIENT_CALC: begin
                    // Handle gradient computation for backpropagation
                    if (s_axis_grad_input_tvalid) begin
                        // Distribute gradients based on pooling type
                        state <= OUTPUT;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end
    
    // AXI-Stream interfaces
    assign s_axis_input_tready = (state == LOAD_INPUT) && enable;
    assign m_axis_output_tdata = pool_result;
    assign m_axis_output_tvalid = (state == POOLING) || (state == OUTPUT);
    assign m_axis_output_tlast = (output_x == OUTPUT_WIDTH-1) && 
                                (output_y == OUTPUT_HEIGHT-1) && 
                                (output_ch == INPUT_CHANNELS-1);
    assign m_axis_output_tuser = output_ch[7:0];
    
    // Gradient interfaces
    assign s_axis_grad_input_tready = (state == GRADIENT_CALC);
    
    // Gradient computation for backpropagation
    gradient_backprop #(
        .DATA_WIDTH(DATA_WIDTH),
        .GRADIENT_WIDTH(GRADIENT_WIDTH),
        .KERNEL_SIZE(KERNEL_SIZE),
        .INPUT_WIDTH(INPUT_WIDTH),
        .INPUT_HEIGHT(INPUT_HEIGHT),
        .OUTPUT_WIDTH(OUTPUT_WIDTH),
        .OUTPUT_HEIGHT(OUTPUT_HEIGHT),
        .POOL_TYPE(POOL_TYPE)
    ) grad_backprop_inst (
        .clk(clk),
        .reset(reset),
        .enable(training_mode && (state == GRADIENT_CALC)),
        .pool_type(POOL_TYPE),
        .max_indices(max_indices),
        .grad_input(s_axis_grad_input_tdata),
        .grad_input_valid(s_axis_grad_input_tvalid),
        .grad_output(m_axis_grad_output_tdata),
        .grad_output_valid(m_axis_grad_output_tvalid)
    );
    
    // Performance counters
    assign pool_ops_count = operation_count;
    assign spike_count = spike_counter;
    
endmodule

// Max pooling engine
module max_pool_engine #(
    parameter DATA_WIDTH = 8,
    parameter KERNEL_SIZE = 2
) (
    input wire clk,
    input wire reset,
    input wire [DATA_WIDTH-1:0] pool_window [0:KERNEL_SIZE-1][0:KERNEL_SIZE-1],
    input wire [KERNEL_SIZE-1:0] window_valid [0:KERNEL_SIZE-1],
    output reg [DATA_WIDTH-1:0] max_result,
    output reg [3:0] max_position
);
    
    integer i, j;
    reg [DATA_WIDTH-1:0] current_max;
    reg [3:0] current_pos;
    
    always @(posedge clk) begin
        if (reset) begin
            max_result <= 0;
            max_position <= 0;
        end else begin
            current_max = 0;
            current_pos = 0;
            
            // Find maximum value and its position
            for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
                for (j = 0; j < KERNEL_SIZE; j = j + 1) begin
                    if (window_valid[i][j] && pool_window[i][j] > current_max) begin
                        current_max = pool_window[i][j];
                        current_pos = i * KERNEL_SIZE + j;
                    end
                end
            end
            
            max_result <= current_max;
            max_position <= current_pos;
        end
    end
    
endmodule

// Average pooling engine
module avg_pool_engine #(
    parameter DATA_WIDTH = 8,
    parameter KERNEL_SIZE = 2
) (
    input wire clk,
    input wire reset,
    input wire [DATA_WIDTH-1:0] pool_window [0:KERNEL_SIZE-1][0:KERNEL_SIZE-1],
    input wire [KERNEL_SIZE-1:0] window_valid [0:KERNEL_SIZE-1],
    output reg [DATA_WIDTH+$clog2(KERNEL_SIZE*KERNEL_SIZE)-1:0] avg_sum,
    output reg [DATA_WIDTH-1:0] avg_result
);
    
    integer i, j;
    reg [DATA_WIDTH+$clog2(KERNEL_SIZE*KERNEL_SIZE)-1:0] sum;
    reg [$clog2(KERNEL_SIZE*KERNEL_SIZE):0] valid_count;
    
    always @(posedge clk) begin
        if (reset) begin
            avg_sum <= 0;
            avg_result <= 0;
        end else begin
            sum = 0;
            valid_count = 0;
            
            // Sum all valid values
            for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
                for (j = 0; j < KERNEL_SIZE; j = j + 1) begin
                    if (window_valid[i][j]) begin
                        sum = sum + pool_window[i][j];
                        valid_count = valid_count + 1;
                    end
                end
            end
            
            avg_sum <= sum;
            // Divide by number of valid elements
            avg_result <= (valid_count > 0) ? sum / valid_count : 0;
        end
    end
    
endmodule

// Gradient backpropagation for pooling layers
module gradient_backprop #(
    parameter DATA_WIDTH = 8,
    parameter GRADIENT_WIDTH = 16,
    parameter KERNEL_SIZE = 2,
    parameter INPUT_WIDTH = 28,
    parameter INPUT_HEIGHT = 28,
    parameter OUTPUT_WIDTH = 14,
    parameter OUTPUT_HEIGHT = 14,
    parameter POOL_TYPE = "MAX"
) (
    input wire clk,
    input wire reset,
    input wire enable,
    input wire [8*8-1:0] pool_type, // String parameter
    input wire [3:0] max_indices [0:OUTPUT_HEIGHT-1][0:OUTPUT_WIDTH-1],
    input wire [GRADIENT_WIDTH-1:0] grad_input,
    input wire grad_input_valid,
    output reg [GRADIENT_WIDTH-1:0] grad_output,
    output reg grad_output_valid
);
    
    reg [15:0] grad_x, grad_y;
    reg [GRADIENT_WIDTH-1:0] gradient_buffer [0:INPUT_HEIGHT-1][0:INPUT_WIDTH-1];
    
    always @(posedge clk) begin
        if (reset) begin
            grad_output <= 0;
            grad_output_valid <= 1'b0;
            grad_x <= 0;
            grad_y <= 0;
        end else if (enable && grad_input_valid) begin
            if (POOL_TYPE == "MAX") begin
                // For max pooling, gradient goes only to the max element
                integer max_idx = max_indices[grad_y][grad_x];
                integer max_row = max_idx / KERNEL_SIZE;
                integer max_col = max_idx % KERNEL_SIZE;
                integer input_row = grad_y * KERNEL_SIZE + max_row;
                integer input_col = grad_x * KERNEL_SIZE + max_col;
                
                gradient_buffer[input_row][input_col] <= grad_input;
            end else begin // Average pooling
                // For average pooling, gradient is distributed equally
                integer i, j;
                wire [GRADIENT_WIDTH-1:0] distributed_grad = grad_input / (KERNEL_SIZE * KERNEL_SIZE);
                
                for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
                    for (j = 0; j < KERNEL_SIZE; j = j + 1) begin
                        integer input_row = grad_y * KERNEL_SIZE + i;
                        integer input_col = grad_x * KERNEL_SIZE + j;
                        
                        if (input_row < INPUT_HEIGHT && input_col < INPUT_WIDTH) begin
                            gradient_buffer[input_row][input_col] <= distributed_grad;
                        end
                    end
                end
            end
            
            grad_output <= gradient_buffer[grad_y][grad_x];
            grad_output_valid <= 1'b1;
            
            // Update coordinates
            if (grad_x == OUTPUT_WIDTH - 1) begin
                grad_x <= 0;
                if (grad_y == OUTPUT_HEIGHT - 1) begin
                    grad_y <= 0;
                end else begin
                    grad_y <= grad_y + 1;
                end
            end else begin
                grad_x <= grad_x + 1;
            end
        end else begin
            grad_output_valid <= 1'b0;
        end
    end
    
endmodule
