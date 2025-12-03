//-----------------------------------------------------------------------------
// Title         : SNN 1D Convolution Layer (Verilog-2001 Compatible)
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_conv1d.v
// Author        : Jiwoon Lee (@metr0jw)
// Organisation  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Lightweight event-driven 1D convolution block used by the
//                  Python accelerator simulation as well as RTL test benches.
//                  The implementation focuses on functional correctness so the
//                  test benches can validate integration without requiring the
//                  full, optimised pipeline.
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module snn_conv1d #(
    parameter INPUT_LENGTH    = 128,
    parameter INPUT_CHANNELS  = 16,
    parameter OUTPUT_CHANNELS = 32,
    parameter KERNEL_SIZE     = 3,
    parameter STRIDE          = 1,
    parameter PADDING         = 1,
    parameter WEIGHT_WIDTH    = 8,
    parameter VMEM_WIDTH      = 16,
    parameter THRESHOLD       = 16'h0100,
    parameter DECAY_FACTOR    = 8'hF0
) (
    input  wire                     clk,
    input  wire                     reset,
    input  wire                     enable,

    // AXI-Stream input spikes: {channel[15:0], position[15:0]}
    input  wire [31:0]              s_axis_input_tdata,
    input  wire                     s_axis_input_tvalid,
    output reg                      s_axis_input_tready,
    input  wire                     s_axis_input_tlast,

    // AXI-Stream output spikes: {channel[15:0], position[15:0]}
    output reg  [31:0]              m_axis_output_tdata,
    output reg                      m_axis_output_tvalid,
    input  wire                     m_axis_output_tready,
    output reg                      m_axis_output_tlast,

    // External weight memory interface (simple read port)
    input  wire [WEIGHT_WIDTH-1:0]  weight_data,
    output reg  [15:0]              weight_addr,
    output reg                      weight_read_en,

    // Runtime configuration
    input  wire [15:0]              threshold_config,
    input  wire [7:0]               decay_config,
    input  wire                     learning_enable,

    // Status / counters
    output reg  [31:0]              input_spike_count,
    output reg  [31:0]              output_spike_count,
    output reg                      computation_done,
    output reg  [31:0]              cycle_count
);

    // ---------------------------------------------------------------------
    // Derived constants
    // ---------------------------------------------------------------------
    localparam OUTPUT_LENGTH = (INPUT_LENGTH + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;
    localparam TOTAL_WEIGHTS = OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE;
    localparam TOTAL_OUTPUTS = OUTPUT_CHANNELS * OUTPUT_LENGTH;
    localparam OUT_FIFO_DEPTH = 8;

    localparam integer VMEM_MAX_INT = (1 << (VMEM_WIDTH-1)) - 1;
    localparam integer VMEM_MIN_INT = - (1 << (VMEM_WIDTH-1));

    // ---------------------------------------------------------------------
    // Internal storage
    // ---------------------------------------------------------------------
    reg signed [WEIGHT_WIDTH-1:0] weight_mem [0:TOTAL_WEIGHTS-1];
    reg signed [VMEM_WIDTH-1:0]   membrane_mem [0:TOTAL_OUTPUTS-1];

    // Output FIFO for spikes (simple ring buffer)
    reg [31:0] out_fifo_data [0:OUT_FIFO_DEPTH-1];
    reg [2:0]  out_fifo_wr_ptr;
    reg [2:0]  out_fifo_rd_ptr;
    reg [3:0]  out_fifo_count;

    wire out_fifo_empty = (out_fifo_count == 0);
    wire out_fifo_full  = (out_fifo_count == OUT_FIFO_DEPTH);

    // ---------------------------------------------------------------------
    // Control state
    // ---------------------------------------------------------------------
    localparam ST_LOAD  = 2'b00;
    localparam ST_READY = 2'b01;
    localparam ST_PROC  = 2'b10;

    reg [1:0] state;
    reg [15:0] load_index;
    reg [15:0] load_prev_addr;
    reg        load_prev_valid;

    reg [15:0] latched_channel;
    reg [15:0] latched_position;
    reg        latched_last;

    reg [15:0] process_out_channel;
    reg [15:0] process_kernel_idx;

    reg last_packet_pending;
    reg processing_active;

    integer idx_int;

    // ---------------------------------------------------------------------
    // Helper tasks / functions
    // ---------------------------------------------------------------------
    function integer weight_index;
        input integer oc;
        input integer ic;
        input integer k;
        begin
            weight_index = ((oc * INPUT_CHANNELS) + ic) * KERNEL_SIZE + k;
        end
    endfunction

    function integer membrane_index;
        input integer oc;
        input integer pos;
        begin
            membrane_index = oc * OUTPUT_LENGTH + pos;
        end
    endfunction

    // ---------------------------------------------------------------------
    // Reset logic for memories and counters
    // ---------------------------------------------------------------------
    integer init_i;
    always @(posedge clk) begin
        if (reset) begin
            state <= ST_LOAD;
            load_index <= 0;
            weight_addr <= 0;
            weight_read_en <= 1'b0;

            s_axis_input_tready <= 1'b0;
            m_axis_output_tvalid <= 1'b0;
            m_axis_output_tdata <= 32'b0;
            m_axis_output_tlast <= 1'b0;

            input_spike_count <= 32'b0;
            output_spike_count <= 32'b0;
            cycle_count <= 32'b0;
            computation_done <= 1'b0;

            latched_channel <= 0;
            latched_position <= 0;
            latched_last <= 1'b0;
            process_out_channel <= 0;
            process_kernel_idx <= 0;
            last_packet_pending <= 1'b0;
            processing_active <= 1'b0;

            load_prev_addr <= 16'd0;
            load_prev_valid <= 1'b0;

            out_fifo_wr_ptr <= 3'b000;
            out_fifo_rd_ptr <= 3'b000;
            out_fifo_count  <= 4'd0;

            for (init_i = 0; init_i < TOTAL_OUTPUTS; init_i = init_i + 1) begin
                membrane_mem[init_i] <= {VMEM_WIDTH{1'b0}};
            end
        end else if (enable) begin
            cycle_count <= cycle_count + 1;

            // Default assignments
            m_axis_output_tlast <= 1'b0;
            computation_done <= 1'b0;

            case (state)
                // ---------------------------------------------------------
                // Weight loading phase
                // ---------------------------------------------------------
                ST_LOAD: begin
                    computation_done <= 1'b0;

                    if (load_prev_valid) begin
                        weight_mem[load_prev_addr] <= weight_data;
                        load_prev_valid <= 1'b0;
                    end

                    if (load_index < TOTAL_WEIGHTS) begin
                        weight_read_en <= 1'b1;
                        weight_addr    <= load_index;
                        load_prev_addr <= load_index;
                        load_prev_valid <= 1'b1;
                        load_index <= load_index + 1;
                    end else begin
                        weight_read_en <= 1'b0;
                        if (!load_prev_valid) begin
                            state <= ST_READY;
                            s_axis_input_tready <= 1'b1;
                        end
                    end
                end

                // ---------------------------------------------------------
                // Ready to accept an input spike
                // ---------------------------------------------------------
                ST_READY: begin
                    if (s_axis_input_tvalid && s_axis_input_tready) begin
                        s_axis_input_tready <= 1'b0;
                        latched_channel  <= s_axis_input_tdata[31:16];
                        latched_position <= s_axis_input_tdata[15:0];
                        latched_last     <= s_axis_input_tlast;
                        process_out_channel <= 0;
                        process_kernel_idx  <= 0;
                        processing_active   <= 1'b1;
                        input_spike_count   <= input_spike_count + 1;
                        if (s_axis_input_tlast)
                            last_packet_pending <= 1'b1;
                        state <= ST_PROC;
                    end else if (!processing_active && last_packet_pending && out_fifo_empty) begin
                        computation_done <= 1'b1;
                        last_packet_pending <= 1'b0;
                    end
                end

                // ---------------------------------------------------------
                // Iterate through output channels and kernel positions
                // ---------------------------------------------------------
                ST_PROC: begin
                    integer numerator;
                    integer out_position_int;
                    integer mem_idx;
                    integer w_idx;
                    integer current_mem;
                    integer leak_term;
                    integer new_mem;
                    integer threshold_int;
                    integer in_channel_idx;

                    numerator = latched_position + PADDING - process_kernel_idx;
                    if (numerator >= 0 && (numerator % STRIDE) == 0) begin
                        out_position_int = numerator / STRIDE;
                        if (out_position_int >= 0 && out_position_int < OUTPUT_LENGTH) begin
                            mem_idx = membrane_index(process_out_channel, out_position_int);
                            in_channel_idx = latched_channel % INPUT_CHANNELS;
                            w_idx   = weight_index(process_out_channel, in_channel_idx, process_kernel_idx);

                            current_mem = membrane_mem[mem_idx];
                            leak_term   = (current_mem * decay_config) >>> 8;
                            new_mem     = current_mem - leak_term + weight_mem[w_idx];

                            if (new_mem > VMEM_MAX_INT) new_mem = VMEM_MAX_INT;
                            if (new_mem < VMEM_MIN_INT) new_mem = VMEM_MIN_INT;

                            threshold_int = threshold_config;
                            if (new_mem >= threshold_int) begin
                                if (!out_fifo_full) begin
                                    out_fifo_data[out_fifo_wr_ptr] <= {process_out_channel[15:0], out_position_int[15:0]};
                                    out_fifo_wr_ptr <= out_fifo_wr_ptr + 1'b1;
                                    out_fifo_count  <= out_fifo_count + 1'b1;
                                    output_spike_count <= output_spike_count + 1;
                                end
                                new_mem = new_mem - threshold_int;
                                if (new_mem < VMEM_MIN_INT) new_mem = VMEM_MIN_INT;
                            end

                            membrane_mem[mem_idx] <= $signed(new_mem[VMEM_WIDTH-1:0]);
                        end
                    end

                    if (process_kernel_idx == KERNEL_SIZE-1) begin
                        process_kernel_idx <= 0;
                        if (process_out_channel == OUTPUT_CHANNELS-1) begin
                            process_out_channel <= 0;
                            processing_active <= 1'b0;
                            state <= ST_READY;
                            s_axis_input_tready <= 1'b1;
                            if (latched_last)
                                latched_last <= 1'b0;
                        end else begin
                            process_out_channel <= process_out_channel + 1;
                        end
                    end else begin
                        process_kernel_idx <= process_kernel_idx + 1;
                    end
                end

                default: state <= ST_LOAD;
            endcase

            // -----------------------------------------------------------------
            // Output FIFO management (AXI-Stream master side)
            // -----------------------------------------------------------------
            if (!out_fifo_empty) begin
                m_axis_output_tdata  <= out_fifo_data[out_fifo_rd_ptr];
                m_axis_output_tvalid <= 1'b1;
                    if (m_axis_output_tready) begin
                    out_fifo_rd_ptr <= out_fifo_rd_ptr + 1'b1;
                    out_fifo_count  <= out_fifo_count - 1'b1;
                    if (last_packet_pending && (out_fifo_count == 1) && !processing_active && state == ST_READY) begin
                        m_axis_output_tlast <= 1'b1;
                        last_packet_pending <= 1'b0;
                            computation_done <= 1'b1;
                    end
                end
            end else begin
                m_axis_output_tvalid <= 1'b0;
            end
        end
    end

    // Prevent unused warning for learning flag (future extension)
    always @(*) begin
        if (!learning_enable) begin
            // No-op – learning hardware not yet implemented
        end
    end

endmodule
