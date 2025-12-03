//-----------------------------------------------------------------------------
// Title         : SNN Pooling Layer (Forward Only)
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_pooling.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Description   : Forward-only pooling layer. Surrogate gradient and backprop
//                 paths have been removed to support pure STDP-based learning.
//-----------------------------------------------------------------------------

module snn_pooling #(
    // Layer parameters
    parameter INPUT_WIDTH    = 28,
    parameter INPUT_HEIGHT   = 28,
    parameter INPUT_CHANNELS = 32,
    parameter KERNEL_SIZE    = 2,
    parameter STRIDE         = 2,
    parameter PADDING        = 0,

    // Precision parameters
    parameter DATA_WIDTH     = 8,

    // Pooling type
    parameter POOL_TYPE      = "MAX"   // "MAX" or "AVG"
) (
    input  wire clk,
    input  wire reset,
    input  wire enable,

    // Optional adaptive pooling configuration (ignored in this forward-only core)
    input  wire [15:0] target_width,
    input  wire [15:0] target_height,

    // Input feature maps (AXI-Stream style interface)
    input  wire [DATA_WIDTH-1:0] s_axis_input_tdata,
    input  wire                  s_axis_input_tvalid,
    output wire                  s_axis_input_tready,
    input  wire                  s_axis_input_tlast,
    input  wire [7:0]            s_axis_input_tuser,

    // Output feature maps (AXI-Stream style interface)
    output reg  [DATA_WIDTH-1:0] m_axis_output_tdata,
    output reg                   m_axis_output_tvalid,
    input  wire                  m_axis_output_tready,
    output reg                   m_axis_output_tlast,
    output reg  [7:0]            m_axis_output_tuser,

    // Performance counters
    output reg [31:0] pool_ops_count,
    output reg [31:0] spike_count
);

    localparam OUTPUT_WIDTH  = (INPUT_WIDTH  + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;
    localparam OUTPUT_HEIGHT = (INPUT_HEIGHT + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;
    localparam integer SUM_WIDTH = DATA_WIDTH + 8; // Enough headroom for typical kernel sizes

    // Internal storage for a single input frame
    reg [DATA_WIDTH-1:0] input_buffer [0:INPUT_CHANNELS-1][0:INPUT_HEIGHT-1][0:INPUT_WIDTH-1];

    // FSM states
    localparam ST_IDLE    = 2'b00;
    localparam ST_LOAD    = 2'b01;
    localparam ST_COMPUTE = 2'b10;
    localparam ST_OUTPUT  = 2'b11;

    reg [1:0] state;

    // Input traversal counters
    reg [15:0] in_x;
    reg [15:0] in_y;
    reg [15:0] in_ch;

    // Output traversal counters
    reg [15:0] out_x;
    reg [15:0] out_y;
    reg [15:0] out_ch;

    // Helper wires
    wire input_last_position  = (in_x  == INPUT_WIDTH-1)  &&
                                (in_y  == INPUT_HEIGHT-1) &&
                                (in_ch == INPUT_CHANNELS-1);

    wire output_last_position = (out_x  == OUTPUT_WIDTH-1)  &&
                                (out_y  == OUTPUT_HEIGHT-1) &&
                                (out_ch == INPUT_CHANNELS-1);

    wire store_enable = enable && s_axis_input_tvalid && s_axis_input_tready;

    // Ready when waiting for or loading a frame
    assign s_axis_input_tready = enable && ((state == ST_IDLE) || (state == ST_LOAD));

    // Consume adaptive pooling inputs to avoid unused warnings (not implemented here)
    wire unused_targets;
    assign unused_targets = |target_width | |target_height;

    // ------------------------------------------------------------------
    // Sequential logic
    // ------------------------------------------------------------------
    integer ch_init;
    integer y_init;
    integer x_init;
    integer ky;
    integer kx;
    integer src_y;
    integer src_x;
    integer sample_count;
    reg [DATA_WIDTH-1:0] sample;
    reg [DATA_WIDTH-1:0] local_max;
    reg [SUM_WIDTH-1:0]  local_sum;

    always @(posedge clk) begin
        if (reset) begin
            state <= ST_IDLE;

            in_x <= 0;
            in_y <= 0;
            in_ch <= 0;
            out_x <= 0;
            out_y <= 0;
            out_ch <= 0;

            m_axis_output_tvalid <= 1'b0;
            m_axis_output_tdata  <= {DATA_WIDTH{1'b0}};
            m_axis_output_tlast  <= 1'b0;
            m_axis_output_tuser  <= 8'd0;

            pool_ops_count <= 32'd0;
            spike_count    <= 32'd0;

            // Clear buffer
            for (ch_init = 0; ch_init < INPUT_CHANNELS; ch_init = ch_init + 1) begin
                for (y_init = 0; y_init < INPUT_HEIGHT; y_init = y_init + 1) begin
                    for (x_init = 0; x_init < INPUT_WIDTH; x_init = x_init + 1) begin
                        input_buffer[ch_init][y_init][x_init] <= {DATA_WIDTH{1'b0}};
                    end
                end
            end

        end else if (enable) begin
            case (state)
                ST_IDLE: begin
                    m_axis_output_tvalid <= 1'b0;
                    m_axis_output_tlast  <= 1'b0;

                    if (store_enable) begin
                        input_buffer[in_ch][in_y][in_x] <= s_axis_input_tdata;
                        if (s_axis_input_tdata != {DATA_WIDTH{1'b0}}) begin
                            spike_count <= spike_count + 1;
                        end

                        if (input_last_position) begin
                            in_x <= 0;
                            in_y <= 0;
                            in_ch <= 0;
                            out_x <= 0;
                            out_y <= 0;
                            out_ch <= 0;
                            state <= ST_COMPUTE;
                        end else begin
                            if (in_x == INPUT_WIDTH-1) begin
                                in_x <= 0;
                                if (in_ch == INPUT_CHANNELS-1) begin
                                    in_ch <= 0;
                                    if (in_y == INPUT_HEIGHT-1) begin
                                        in_y <= 0;
                                    end else begin
                                        in_y <= in_y + 1;
                                    end
                                end else begin
                                    in_ch <= in_ch + 1;
                                end
                            end else begin
                                in_x <= in_x + 1;
                            end
                            state <= ST_LOAD;
                        end
                    end
                end

                ST_LOAD: begin
                    if (store_enable) begin
                        input_buffer[in_ch][in_y][in_x] <= s_axis_input_tdata;
                        if (s_axis_input_tdata != {DATA_WIDTH{1'b0}}) begin
                            spike_count <= spike_count + 1;
                        end

                        if (input_last_position) begin
                            in_x <= 0;
                            in_y <= 0;
                            in_ch <= 0;
                            out_x <= 0;
                            out_y <= 0;
                            out_ch <= 0;
                            state <= ST_COMPUTE;
                        end else begin
                            if (in_x == INPUT_WIDTH-1) begin
                                in_x <= 0;
                                if (in_ch == INPUT_CHANNELS-1) begin
                                    in_ch <= 0;
                                    if (in_y == INPUT_HEIGHT-1) begin
                                        in_y <= 0;
                                    end else begin
                                        in_y <= in_y + 1;
                                    end
                                end else begin
                                    in_ch <= in_ch + 1;
                                end
                            end else begin
                                in_x <= in_x + 1;
                            end
                        end
                    end
                end

                ST_COMPUTE: begin
                    local_max = {DATA_WIDTH{1'b0}};
                    local_sum = {SUM_WIDTH{1'b0}};
                    sample_count = 0;

                    for (ky = 0; ky < KERNEL_SIZE; ky = ky + 1) begin
                        src_y = out_y * STRIDE + ky - PADDING;
                        for (kx = 0; kx < KERNEL_SIZE; kx = kx + 1) begin
                            src_x = out_x * STRIDE + kx - PADDING;
                            if (src_x >= 0 && src_x < INPUT_WIDTH &&
                                src_y >= 0 && src_y < INPUT_HEIGHT) begin
                                sample = input_buffer[out_ch][src_y][src_x];
                                if ((sample_count == 0) || (sample > local_max)) begin
                                    local_max = sample;
                                end
                                local_sum = local_sum + sample;
                                sample_count = sample_count + 1;
                            end
                        end
                    end

                    if (POOL_TYPE == "AVG" || POOL_TYPE == "ADAPTIVE_AVG") begin
                        if (sample_count > 0) begin
                            m_axis_output_tdata <= local_sum / sample_count;
                        end else begin
                            m_axis_output_tdata <= {DATA_WIDTH{1'b0}};
                        end
                    end else begin
                        m_axis_output_tdata <= local_max;
                    end

                    m_axis_output_tvalid <= 1'b1;
                    m_axis_output_tuser  <= out_ch[7:0];
                    m_axis_output_tlast  <= output_last_position;
                    state <= ST_OUTPUT;
                end

                ST_OUTPUT: begin
                    if (m_axis_output_tvalid && m_axis_output_tready) begin
                        pool_ops_count <= pool_ops_count + 1;
                        m_axis_output_tvalid <= 1'b0;

                        if (output_last_position) begin
                            out_x <= 0;
                            out_y <= 0;
                            out_ch <= 0;
                            state <= ST_IDLE;
                        end else begin
                            if (out_x == OUTPUT_WIDTH-1) begin
                                out_x <= 0;
                                if (out_y == OUTPUT_HEIGHT-1) begin
                                    out_y <= 0;
                                    out_ch <= out_ch + 1;
                                end else begin
                                    out_y <= out_y + 1;
                                end
                            end else begin
                                out_x <= out_x + 1;
                            end
                            state <= ST_COMPUTE;
                        end
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
