//-----------------------------------------------------------------------------
// Title         : SNN Convolutional Layer
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_conv2d.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Description   : PyTorch-compatible 2D convolution for SNN with mixed precision
//-----------------------------------------------------------------------------

module snn_conv2d #(
    // Layer parameters
    parameter INPUT_WIDTH = 28,
    parameter INPUT_HEIGHT = 28,
    parameter INPUT_CHANNELS = 1,
    parameter OUTPUT_CHANNELS = 32,
    parameter KERNEL_SIZE = 3,
    parameter STRIDE = 1,
    parameter PADDING = 1,
    
    // Precision parameters
    parameter WEIGHT_WIDTH = 8,         // INT8 weights for forward pass
    parameter ACTIVATION_WIDTH = 8,     // INT8 activations for forward pass
    parameter GRADIENT_WIDTH = 16,      // FP16 for surrogate gradients
    parameter BIAS_WIDTH = 16,
    
    // Buffer parameters
    parameter MAX_BATCH_SIZE = 32,
    parameter PIPELINE_DEPTH = 4
) (
    input wire clk,
    input wire reset,
    input wire enable,
    
    // Control signals
    input wire training_mode,           // 0: inference (INT8), 1: training (FP16)
    input wire [1:0] conv_mode,         // 0: forward, 1: backward_data, 2: backward_weight
    
    // Input feature maps (AXI-Stream)
    input wire [ACTIVATION_WIDTH-1:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    input wire s_axis_input_tlast,
    input wire [7:0] s_axis_input_tuser, // Channel info
    
    // Output feature maps (AXI-Stream)
    output wire [ACTIVATION_WIDTH-1:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready,
    output wire m_axis_output_tlast,
    output wire [7:0] m_axis_output_tuser,
    
    // Weight interface (AXI-Lite)
    input wire [31:0] s_axi_weight_awaddr,
    input wire [2:0] s_axi_weight_awprot,
    input wire s_axi_weight_awvalid,
    output wire s_axi_weight_awready,
    input wire [31:0] s_axi_weight_wdata,
    input wire [3:0] s_axi_weight_wstrb,
    input wire s_axi_weight_wvalid,
    output wire s_axi_weight_wready,
    output wire [1:0] s_axi_weight_bresp,
    output wire s_axi_weight_bvalid,
    input wire s_axi_weight_bready,
    
    // Gradient output (for surrogate gradient)
    output wire [GRADIENT_WIDTH-1:0] m_axis_gradient_tdata,
    output wire m_axis_gradient_tvalid,
    input wire m_axis_gradient_tready,
    
    // Performance counters
    output wire [31:0] conv_ops_count,
    output wire [31:0] spike_count
);

    // Calculate output dimensions
    localparam OUTPUT_WIDTH = (INPUT_WIDTH + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;
    localparam OUTPUT_HEIGHT = (INPUT_HEIGHT + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;
    
    // Weight memory parameters
    localparam WEIGHT_MEMORY_SIZE = OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE;
    localparam WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_MEMORY_SIZE);
    
    // Internal signals
    wire [ACTIVATION_WIDTH-1:0] input_buffer_data;
    wire input_buffer_valid;
    wire input_buffer_ready;
    
    wire [WEIGHT_WIDTH-1:0] weight_data;
    wire weight_valid;
    
    wire [ACTIVATION_WIDTH-1:0] conv_result;
    wire conv_result_valid;
    wire conv_result_ready;
    
    // State machine
    reg [2:0] state;
    localparam IDLE = 3'b000,
               LOAD_INPUT = 3'b001,
               CONVOLUTION = 3'b010,
               OUTPUT = 3'b011,
               GRADIENT_CALC = 3'b100;
    
    // Counters
    reg [15:0] input_x, input_y, input_ch;
    reg [15:0] output_x, output_y, output_ch;
    reg [15:0] kernel_x, kernel_y;
    reg [31:0] operation_count;
    reg [31:0] spike_counter;
    
    // Input line buffer for sliding window
    reg [ACTIVATION_WIDTH-1:0] line_buffer [0:INPUT_CHANNELS-1][0:KERNEL_SIZE-1][0:INPUT_WIDTH-1];
    reg [KERNEL_SIZE-1:0] line_buffer_valid [0:INPUT_CHANNELS-1];
    
    // Weight memory (dual-port for concurrent read/write)
    reg [WEIGHT_WIDTH-1:0] weight_memory [0:WEIGHT_MEMORY_SIZE-1];
    reg [WEIGHT_ADDR_WIDTH-1:0] weight_addr_read, weight_addr_write;
    reg weight_write_enable;
    
    // Bias memory
    reg signed [BIAS_WIDTH-1:0] bias_memory [0:OUTPUT_CHANNELS-1];
    
    // Convolution engine with parallel multipliers
    wire [ACTIVATION_WIDTH+WEIGHT_WIDTH-1:0] mult_results [0:KERNEL_SIZE*KERNEL_SIZE-1];
    wire [ACTIVATION_WIDTH+WEIGHT_WIDTH+$clog2(KERNEL_SIZE*KERNEL_SIZE)-1:0] accumulator;
    
    // Generate parallel multipliers for each kernel position
    genvar i, j;
    generate
        for (i = 0; i < KERNEL_SIZE; i = i + 1) begin : gen_mult_i
            for (j = 0; j < KERNEL_SIZE; j = j + 1) begin : gen_mult_j
                localparam mult_idx = i * KERNEL_SIZE + j;
                
                // INT8 multiplier for forward pass
                wire [ACTIVATION_WIDTH-1:0] input_pixel;
                wire [WEIGHT_WIDTH-1:0] weight_kernel;
                
                assign input_pixel = (state == CONVOLUTION && 
                                    kernel_x == j && kernel_y == i) ? 
                                    input_buffer_data : 8'b0;
                assign weight_kernel = weight_memory[weight_addr_read + mult_idx];
                
                // Signed multiplication
                assign mult_results[mult_idx] = $signed(input_pixel) * $signed(weight_kernel);
            end
        end
    endgenerate
    
    // Accumulation tree for parallel addition
    wire [ACTIVATION_WIDTH+WEIGHT_WIDTH+$clog2(KERNEL_SIZE*KERNEL_SIZE)-1:0] sum_tree [0:KERNEL_SIZE*KERNEL_SIZE-1];
    
    // First level - pair-wise addition
    generate
        for (i = 0; i < KERNEL_SIZE*KERNEL_SIZE/2; i = i + 1) begin : gen_sum_level1
            assign sum_tree[i] = mult_results[2*i] + mult_results[2*i+1];
        end
    endgenerate
    
    // Continue until single result
    assign accumulator = sum_tree[0] + sum_tree[1] + sum_tree[2] + sum_tree[3] + 
                        sum_tree[4] + sum_tree[5] + sum_tree[6] + sum_tree[7] + 
                        sum_tree[8]; // For 3x3 kernel
    
    // Add bias and apply activation (ReLU for compatibility)
    wire signed [ACTIVATION_WIDTH+WEIGHT_WIDTH+$clog2(KERNEL_SIZE*KERNEL_SIZE)-1:0] biased_result;
    assign biased_result = accumulator + bias_memory[output_ch];
    
    // ReLU activation and quantization back to INT8
    assign conv_result = (biased_result > 0) ? 
                        ((biased_result > 127) ? 8'd127 : biased_result[ACTIVATION_WIDTH-1:0]) : 
                        8'd0;
    
    // Surrogate gradient calculation (FP16)
    reg [GRADIENT_WIDTH-1:0] surrogate_gradient;
    wire [GRADIENT_WIDTH-1:0] gradient_input;
    
    // Fast sigmoid surrogate gradient: grad = 1 / (1 + |x|)
    // Implemented using lookup table for efficiency
    surrogate_gradient_lut surrogate_lut (
        .clk(clk),
        .input_val(biased_result[15:0]),  // Convert to 16-bit for gradient calc
        .gradient(gradient_input)
    );
    
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
            weight_addr_read <= 0;
            surrogate_gradient <= 0;
        end else if (enable) begin
            case (state)
                IDLE: begin
                    if (s_axis_input_tvalid && conv_mode == 2'b00) begin // Forward pass
                        state <= LOAD_INPUT;
                        input_x <= 0;
                        input_y <= 0;
                        input_ch <= 0;
                    end
                end
                
                LOAD_INPUT: begin
                    if (s_axis_input_tvalid && s_axis_input_tready) begin
                        // Store input in line buffer
                        line_buffer[input_ch][input_y % KERNEL_SIZE][input_x] <= s_axis_input_tdata;
                        line_buffer_valid[input_ch][input_y % KERNEL_SIZE] <= 1'b1;
                        
                        // Count spikes (non-zero values)
                        if (s_axis_input_tdata != 0) begin
                            spike_counter <= spike_counter + 1;
                        end
                        
                        // Update coordinates
                        if (input_x == INPUT_WIDTH - 1) begin
                            input_x <= 0;
                            if (input_ch == INPUT_CHANNELS - 1) begin
                                input_ch <= 0;
                                if (input_y == INPUT_HEIGHT - 1) begin
                                    input_y <= 0;
                                    state <= CONVOLUTION;
                                    output_x <= 0;
                                    output_y <= 0;
                                    output_ch <= 0;
                                end else begin
                                    input_y <= input_y + 1;
                                end
                            end else begin
                                input_ch <= input_ch + 1;
                            end
                        end else begin
                            input_x <= input_x + 1;
                        end
                    end
                end
                
                CONVOLUTION: begin
                    // Perform convolution for current output position
                    if (conv_result_ready) begin
                        // Calculate weight address
                        weight_addr_read <= (output_ch * INPUT_CHANNELS + input_ch) * 
                                          KERNEL_SIZE * KERNEL_SIZE + 
                                          kernel_y * KERNEL_SIZE + kernel_x;
                        
                        // Update kernel position
                        if (kernel_x == KERNEL_SIZE - 1) begin
                            kernel_x <= 0;
                            if (kernel_y == KERNEL_SIZE - 1) begin
                                kernel_y <= 0;
                                if (input_ch == INPUT_CHANNELS - 1) begin
                                    input_ch <= 0;
                                    // Move to next output position
                                    if (output_x == OUTPUT_WIDTH - 1) begin
                                        output_x <= 0;
                                        if (output_y == OUTPUT_HEIGHT - 1) begin
                                            output_y <= 0;
                                            if (output_ch == OUTPUT_CHANNELS - 1) begin
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
                                end else begin
                                    input_ch <= input_ch + 1;
                                end
                            end else begin
                                kernel_y <= kernel_y + 1;
                            end
                        end else begin
                            kernel_x <= kernel_x + 1;
                        end
                        
                        // Calculate surrogate gradient if in training mode
                        if (training_mode) begin
                            surrogate_gradient <= gradient_input;
                        end
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
                    if (training_mode && conv_mode == 2'b01) begin // Backward data
                        // Implement gradient computation logic
                        state <= OUTPUT;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end
    
    // AXI-Stream input interface
    assign s_axis_input_tready = (state == LOAD_INPUT) && enable;
    assign input_buffer_data = s_axis_input_tdata;
    assign input_buffer_valid = s_axis_input_tvalid && s_axis_input_tready;
    
    // AXI-Stream output interface
    assign m_axis_output_tdata = conv_result;
    assign m_axis_output_tvalid = (state == OUTPUT) && conv_result_valid;
    assign conv_result_ready = m_axis_output_tready;
    assign conv_result_valid = (state == CONVOLUTION) || (state == OUTPUT);
    assign m_axis_output_tlast = (output_x == OUTPUT_WIDTH-1) && 
                                (output_y == OUTPUT_HEIGHT-1) && 
                                (output_ch == OUTPUT_CHANNELS-1);
    assign m_axis_output_tuser = output_ch[7:0];
    
    // Gradient output interface
    assign m_axis_gradient_tdata = surrogate_gradient;
    assign m_axis_gradient_tvalid = training_mode && (state == CONVOLUTION);
    
    // Weight loading interface (AXI-Lite)
    axi_lite_weight_interface #(
        .WEIGHT_MEMORY_SIZE(WEIGHT_MEMORY_SIZE),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .BIAS_CHANNELS(OUTPUT_CHANNELS),
        .BIAS_WIDTH(BIAS_WIDTH)
    ) weight_if (
        .clk(clk),
        .reset(reset),
        .s_axi_awaddr(s_axi_weight_awaddr),
        .s_axi_awprot(s_axi_weight_awprot),
        .s_axi_awvalid(s_axi_weight_awvalid),
        .s_axi_awready(s_axi_weight_awready),
        .s_axi_wdata(s_axi_weight_wdata),
        .s_axi_wstrb(s_axi_weight_wstrb),
        .s_axi_wvalid(s_axi_weight_wvalid),
        .s_axi_wready(s_axi_weight_wready),
        .s_axi_bresp(s_axi_weight_bresp),
        .s_axi_bvalid(s_axi_weight_bvalid),
        .s_axi_bready(s_axi_weight_bready),
        .weight_memory(weight_memory),
        .bias_memory(bias_memory),
        .weight_write_enable(weight_write_enable),
        .weight_addr_write(weight_addr_write)
    );
    
    // Performance counters
    assign conv_ops_count = operation_count;
    assign spike_count = spike_counter;
    
endmodule

// Surrogate gradient lookup table for fast approximation
module surrogate_gradient_lut (
    input wire clk,
    input wire [15:0] input_val,
    output reg [15:0] gradient
);
    
    // Fast sigmoid approximation: 1 / (1 + |x|)
    // Using 8-bit LUT for efficiency
    wire [7:0] lut_addr;
    wire [15:0] lut_data;
    
    assign lut_addr = input_val[15] ? (~input_val[14:7] + 1) : input_val[14:7]; // abs(x) >> 7
    
    // Pre-computed gradient values in FP16 format
    // gradient[i] = 1.0 / (1.0 + i/128.0) converted to FP16
    reg [15:0] gradient_lut [0:255];
    
    initial begin
        // Initialize with pre-computed FP16 values
        gradient_lut[0] = 16'h3C00;   // 1.0 in FP16
        gradient_lut[1] = 16'h3BFE;   // ~0.996
        gradient_lut[2] = 16'h3BFB;   // ~0.992
        gradient_lut[3] = 16'h3BF8;   // ~0.988
        // ... continue for all 256 values
        // This would be generated by a script in practice
        
        // Fill remaining with decreasing values
        integer i;
        for (i = 4; i < 256; i = i + 1) begin
            gradient_lut[i] = 16'h3C00 - (i << 4); // Approximation
        end
    end
    
    assign lut_data = gradient_lut[lut_addr];
    
    always @(posedge clk) begin
        gradient <= lut_data;
    end
    
endmodule

// AXI-Lite interface for weight loading
module axi_lite_weight_interface #(
    parameter WEIGHT_MEMORY_SIZE = 1024,
    parameter WEIGHT_WIDTH = 8,
    parameter BIAS_CHANNELS = 32,
    parameter BIAS_WIDTH = 16
) (
    input wire clk,
    input wire reset,
    
    // AXI-Lite interface
    input wire [31:0] s_axi_awaddr,
    input wire [2:0] s_axi_awprot,
    input wire s_axi_awvalid,
    output reg s_axi_awready,
    input wire [31:0] s_axi_wdata,
    input wire [3:0] s_axi_wstrb,
    input wire s_axi_wvalid,
    output reg s_axi_wready,
    output reg [1:0] s_axi_bresp,
    output reg s_axi_bvalid,
    input wire s_axi_bready,
    
    // Memory interfaces
    output reg [WEIGHT_WIDTH-1:0] weight_memory [0:WEIGHT_MEMORY_SIZE-1],
    output reg signed [BIAS_WIDTH-1:0] bias_memory [0:BIAS_CHANNELS-1],
    output reg weight_write_enable,
    output reg [$clog2(WEIGHT_MEMORY_SIZE)-1:0] weight_addr_write
);
    
    // AXI-Lite state machine
    reg [1:0] axi_state;
    localparam AXI_IDLE = 2'b00,
               AXI_WRITE = 2'b01,
               AXI_RESP = 2'b10;
    
    reg [31:0] write_addr;
    
    always @(posedge clk) begin
        if (reset) begin
            axi_state <= AXI_IDLE;
            s_axi_awready <= 1'b0;
            s_axi_wready <= 1'b0;
            s_axi_bvalid <= 1'b0;
            s_axi_bresp <= 2'b00;
            weight_write_enable <= 1'b0;
        end else begin
            case (axi_state)
                AXI_IDLE: begin
                    if (s_axi_awvalid && s_axi_wvalid) begin
                        s_axi_awready <= 1'b1;
                        s_axi_wready <= 1'b1;
                        write_addr <= s_axi_awaddr;
                        axi_state <= AXI_WRITE;
                    end
                end
                
                AXI_WRITE: begin
                    s_axi_awready <= 1'b0;
                    s_axi_wready <= 1'b0;
                    
                    // Decode address and write to appropriate memory
                    if (write_addr[31:16] == 16'h0000) begin
                        // Weight memory
                        weight_addr_write <= write_addr[$clog2(WEIGHT_MEMORY_SIZE)-1:0];
                        weight_memory[write_addr[$clog2(WEIGHT_MEMORY_SIZE)-1:0]] <= s_axi_wdata[WEIGHT_WIDTH-1:0];
                        weight_write_enable <= 1'b1;
                    end else if (write_addr[31:16] == 16'h0001) begin
                        // Bias memory
                        bias_memory[write_addr[$clog2(BIAS_CHANNELS)-1:0]] <= s_axi_wdata[BIAS_WIDTH-1:0];
                    end
                    
                    s_axi_bvalid <= 1'b1;
                    s_axi_bresp <= 2'b00; // OKAY
                    axi_state <= AXI_RESP;
                end
                
                AXI_RESP: begin
                    weight_write_enable <= 1'b0;
                    if (s_axi_bready) begin
                        s_axi_bvalid <= 1'b0;
                        axi_state <= AXI_IDLE;
                    end
                end
            endcase
        end
    end
    
endmodule
