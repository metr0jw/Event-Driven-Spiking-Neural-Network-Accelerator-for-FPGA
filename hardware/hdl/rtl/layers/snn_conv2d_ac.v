//-----------------------------------------------------------------------------
// Title         : Energy-Efficient AC-based SNN 2D Convolution (Verilog-2001)
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_conv2d_ac.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Description   : 2D Convolution using Accumulate-only (AC) operations
//                 Optimized for image processing with maximum energy efficiency
//                 
//                 ENERGY ANALYSIS:
//                 ================
//                 ANN Conv2D (per output pixel):
//                   ops = C_in × K × K MAC operations
//                   energy = C_in × K × K × 4.6pJ ≈ 44pJ (for 3×3, C=1)
//                   
//                 SNN Conv2D (per input spike):
//                   ops = C_out × K × K AC operations (only if spike=1)
//                   energy = C_out × K × K × 0.9pJ × sparsity
//                   
//                 With 95% sparsity:
//                   effective_energy ≈ C_out × K × K × 0.9pJ × 0.05
//                   savings ≈ 100x vs ANN!
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module snn_conv2d_ac #(
    // Input dimensions
    parameter INPUT_HEIGHT      = 28,
    parameter INPUT_WIDTH       = 28,
    parameter INPUT_CHANNELS    = 1,
    
    // Convolution parameters
    parameter OUTPUT_CHANNELS   = 32,
    parameter KERNEL_SIZE       = 3,
    parameter STRIDE            = 1,
    parameter PADDING           = 1,
    
    // Calculated output dimensions
    parameter OUTPUT_HEIGHT     = (INPUT_HEIGHT + 2*PADDING - KERNEL_SIZE) / STRIDE + 1,
    parameter OUTPUT_WIDTH      = (INPUT_WIDTH + 2*PADDING - KERNEL_SIZE) / STRIDE + 1,
    
    // Precision parameters  
    parameter WEIGHT_WIDTH      = 8,        // INT8 weights
    parameter VMEM_WIDTH        = 16,       // Q8.8 membrane potential
    parameter THRESHOLD         = 16'h0100, // 1.0 in Q8.8 format
    parameter LEAK_SHIFT        = 4,        // Decay ≈ 0.9375
    
    // Address widths
    parameter CH_WIDTH          = 5,        // log2(32) = 5
    parameter ROW_WIDTH         = 5,        // log2(28) = 5
    parameter COL_WIDTH         = 5         // log2(28) = 5
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         enable,
    
    //=========================================================================
    // Sparse Spike Input (AXI-Stream)
    // Format: {channel[7:0], row[7:0], col[7:0], timestamp[7:0]}
    //=========================================================================
    input  wire                         s_axis_spike_tvalid,
    input  wire [31:0]                  s_axis_spike_tdata,
    input  wire                         s_axis_spike_tlast,
    output reg                          s_axis_spike_tready,
    
    //=========================================================================
    // Sparse Spike Output (AXI-Stream)
    //=========================================================================
    output reg                          m_axis_spike_tvalid,
    output reg  [31:0]                  m_axis_spike_tdata,
    output reg                          m_axis_spike_tlast,
    input  wire                         m_axis_spike_tready,
    
    //=========================================================================
    // Weight Memory Interface
    //=========================================================================
    output reg                          weight_rd_en,
    output reg  [15:0]                  weight_addr,
    input  wire signed [WEIGHT_WIDTH-1:0] weight_data,
    input  wire                         weight_valid,
    
    //=========================================================================
    // Configuration
    //=========================================================================
    input  wire [VMEM_WIDTH-1:0]        config_threshold,
    input  wire                         config_valid,
    
    //=========================================================================
    // Energy Monitoring
    //=========================================================================
    output reg  [31:0]                  input_spike_count,
    output reg  [31:0]                  output_spike_count,
    output reg  [31:0]                  ac_operation_count,
    output reg  [31:0]                  memory_access_count,
    output reg  [31:0]                  cycle_count,
    output wire                         busy
);

    //=========================================================================
    // State Machine
    //=========================================================================
    localparam IDLE           = 4'd0;
    localparam PARSE_SPIKE    = 4'd1;
    localparam INIT_CONV      = 4'd2;
    localparam FETCH_WEIGHT   = 4'd3;
    localparam WAIT_WEIGHT    = 4'd4;
    localparam AC_ACCUMULATE  = 4'd5;
    localparam NEXT_KERNEL_X  = 4'd6;
    localparam NEXT_KERNEL_Y  = 4'd7;
    localparam NEXT_OUT_CH    = 4'd8;
    localparam CHECK_OUTPUTS  = 4'd9;
    localparam OUTPUT_SPIKE   = 4'd10;
    localparam APPLY_LEAK     = 4'd11;
    
    reg [3:0] state;
    assign busy = (state != IDLE);
    
    //=========================================================================
    // Input Spike Registers
    //=========================================================================
    reg [CH_WIDTH-1:0]  in_ch;
    reg [ROW_WIDTH-1:0] in_row;
    reg [COL_WIDTH-1:0] in_col;
    reg [7:0]           in_timestamp;
    
    //=========================================================================
    // Convolution Counters
    //=========================================================================
    reg [CH_WIDTH-1:0]  out_ch;
    reg [2:0]           kernel_row;
    reg [2:0]           kernel_col;
    reg [ROW_WIDTH-1:0] out_row;
    reg [COL_WIDTH-1:0] out_col;
    
    //=========================================================================
    // Membrane Potential Memory
    // Flattened: index = out_ch * (OUTPUT_HEIGHT * OUTPUT_WIDTH) + 
    //                    out_row * OUTPUT_WIDTH + out_col
    //=========================================================================
    localparam VMEM_SIZE = OUTPUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH;
    reg signed [VMEM_WIDTH-1:0] vmem_memory [0:VMEM_SIZE-1];
    
    // Memory index calculation
    function [15:0] vmem_idx;
        input [CH_WIDTH-1:0]  ch;
        input [ROW_WIDTH-1:0] row;
        input [COL_WIDTH-1:0] col;
        begin
            vmem_idx = ch * (OUTPUT_HEIGHT * OUTPUT_WIDTH) + row * OUTPUT_WIDTH + col;
        end
    endfunction
    
    //=========================================================================
    // Weight Address Calculation
    // Layout: [out_ch][in_ch][kernel_row][kernel_col]
    //=========================================================================
    function [15:0] weight_idx;
        input [CH_WIDTH-1:0] o_ch;
        input [CH_WIDTH-1:0] i_ch;
        input [2:0] k_row;
        input [2:0] k_col;
        begin
            weight_idx = o_ch * (INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE) +
                        i_ch * (KERNEL_SIZE * KERNEL_SIZE) +
                        k_row * KERNEL_SIZE + k_col;
        end
    endfunction
    
    //=========================================================================
    // Current Threshold
    //=========================================================================
    reg [VMEM_WIDTH-1:0] threshold_reg;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            threshold_reg <= THRESHOLD;
        else if (config_valid)
            threshold_reg <= config_threshold;
    end
    
    //=========================================================================
    // Memory Initialization
    //=========================================================================
    integer init_i;
    initial begin
        for (init_i = 0; init_i < VMEM_SIZE; init_i = init_i + 1) begin
            vmem_memory[init_i] = 0;
        end
    end
    
    //=========================================================================
    // Processing Variables
    //=========================================================================
    reg signed [VMEM_WIDTH-1:0] current_vmem;
    reg signed [VMEM_WIDTH:0]   vmem_new;
    reg [15:0]                  current_vmem_idx;
    reg [ROW_WIDTH-1:0]         target_out_row;
    reg [COL_WIDTH-1:0]         target_out_col;
    reg                         valid_output_pos;
    
    // Output scan registers
    reg [CH_WIDTH-1:0]  scan_ch;
    reg [ROW_WIDTH-1:0] scan_row;
    reg [COL_WIDTH-1:0] scan_col;
    
    //=========================================================================
    // Main State Machine
    //=========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            s_axis_spike_tready <= 1'b1;
            m_axis_spike_tvalid <= 1'b0;
            m_axis_spike_tdata <= 32'b0;
            m_axis_spike_tlast <= 1'b0;
            
            weight_rd_en <= 1'b0;
            weight_addr <= 16'b0;
            
            input_spike_count <= 32'b0;
            output_spike_count <= 32'b0;
            ac_operation_count <= 32'b0;
            memory_access_count <= 32'b0;
            cycle_count <= 32'b0;
            
            out_ch <= 0;
            kernel_row <= 0;
            kernel_col <= 0;
            
        end else if (enable) begin
            cycle_count <= cycle_count + 1;
            m_axis_spike_tvalid <= 1'b0;
            weight_rd_en <= 1'b0;
            
            case (state)
                //=============================================================
                // IDLE: Wait for spike (zero energy consumption)
                //=============================================================
                IDLE: begin
                    s_axis_spike_tready <= 1'b1;
                    if (s_axis_spike_tvalid) begin
                        state <= PARSE_SPIKE;
                    end
                end
                
                //=============================================================
                // PARSE_SPIKE: Extract spike coordinates
                //=============================================================
                PARSE_SPIKE: begin
                    s_axis_spike_tready <= 1'b0;
                    in_ch <= s_axis_spike_tdata[31:24];
                    in_row <= s_axis_spike_tdata[23:16];
                    in_col <= s_axis_spike_tdata[15:8];
                    in_timestamp <= s_axis_spike_tdata[7:0];
                    input_spike_count <= input_spike_count + 1;
                    state <= INIT_CONV;
                end
                
                //=============================================================
                // INIT_CONV: Initialize convolution for this spike
                //=============================================================
                INIT_CONV: begin
                    out_ch <= 0;
                    kernel_row <= 0;
                    kernel_col <= 0;
                    state <= FETCH_WEIGHT;
                end
                
                //=============================================================
                // FETCH_WEIGHT: Request weight from memory
                //=============================================================
                FETCH_WEIGHT: begin
                    weight_addr <= weight_idx(out_ch, in_ch, kernel_row, kernel_col);
                    weight_rd_en <= 1'b1;
                    memory_access_count <= memory_access_count + 1;
                    state <= WAIT_WEIGHT;
                end
                
                //=============================================================
                // WAIT_WEIGHT: Wait for memory
                //=============================================================
                WAIT_WEIGHT: begin
                    if (weight_valid) begin
                        state <= AC_ACCUMULATE;
                    end
                end
                
                //=============================================================
                // AC_ACCUMULATE: THE CORE AC OPERATION!
                // 
                // Calculate affected output position and accumulate weight
                // NO MULTIPLY - just addition!
                //
                // Output position: 
                //   out_row = (in_row + PADDING - kernel_row) / STRIDE
                //   out_col = (in_col + PADDING - kernel_col) / STRIDE
                //=============================================================
                AC_ACCUMULATE: begin
                    // Calculate target output position
                    target_out_row <= (in_row + PADDING - kernel_row);
                    target_out_col <= (in_col + PADDING - kernel_col);
                    
                    // Check if output position is valid
                    valid_output_pos <= (in_row + PADDING >= kernel_row) &&
                                       (in_col + PADDING >= kernel_col) &&
                                       ((in_row + PADDING - kernel_row) < OUTPUT_HEIGHT) &&
                                       ((in_col + PADDING - kernel_col) < OUTPUT_WIDTH);
                    
                    if (valid_output_pos) begin
                        // Get current vmem index
                        current_vmem_idx <= vmem_idx(out_ch, target_out_row, target_out_col);
                        current_vmem <= vmem_memory[vmem_idx(out_ch, target_out_row, target_out_col)];
                        
                        // AC OPERATION: vmem += weight (NO MULTIPLY!)
                        vmem_new <= vmem_memory[vmem_idx(out_ch, target_out_row, target_out_col)] +
                                   {{(VMEM_WIDTH-WEIGHT_WIDTH){weight_data[WEIGHT_WIDTH-1]}}, weight_data};
                        
                        // Store with saturation
                        if (vmem_new[VMEM_WIDTH] != vmem_new[VMEM_WIDTH-1]) begin
                            vmem_memory[current_vmem_idx] <= vmem_new[VMEM_WIDTH] ?
                                {1'b1, {(VMEM_WIDTH-1){1'b0}}} : {1'b0, {(VMEM_WIDTH-1){1'b1}}};
                        end else begin
                            vmem_memory[current_vmem_idx] <= vmem_new[VMEM_WIDTH-1:0];
                        end
                        
                        ac_operation_count <= ac_operation_count + 1;
                    end
                    
                    state <= NEXT_KERNEL_X;
                end
                
                //=============================================================
                // NEXT_KERNEL_X: Move to next kernel column
                //=============================================================
                NEXT_KERNEL_X: begin
                    if (kernel_col < KERNEL_SIZE - 1) begin
                        kernel_col <= kernel_col + 1;
                        state <= FETCH_WEIGHT;
                    end else begin
                        kernel_col <= 0;
                        state <= NEXT_KERNEL_Y;
                    end
                end
                
                //=============================================================
                // NEXT_KERNEL_Y: Move to next kernel row
                //=============================================================
                NEXT_KERNEL_Y: begin
                    if (kernel_row < KERNEL_SIZE - 1) begin
                        kernel_row <= kernel_row + 1;
                        state <= FETCH_WEIGHT;
                    end else begin
                        kernel_row <= 0;
                        state <= NEXT_OUT_CH;
                    end
                end
                
                //=============================================================
                // NEXT_OUT_CH: Move to next output channel
                //=============================================================
                NEXT_OUT_CH: begin
                    if (out_ch < OUTPUT_CHANNELS - 1) begin
                        out_ch <= out_ch + 1;
                        state <= FETCH_WEIGHT;
                    end else begin
                        // Done with all channels for this input spike
                        state <= CHECK_OUTPUTS;
                        scan_ch <= 0;
                        scan_row <= 0;
                        scan_col <= 0;
                    end
                end
                
                //=============================================================
                // CHECK_OUTPUTS: Scan for neurons that should spike
                //=============================================================
                CHECK_OUTPUTS: begin
                    if (vmem_memory[vmem_idx(scan_ch, scan_row, scan_col)] >= 
                        $signed(threshold_reg)) begin
                        state <= OUTPUT_SPIKE;
                    end else begin
                        // Move to next position
                        if (scan_col < OUTPUT_WIDTH - 1) begin
                            scan_col <= scan_col + 1;
                        end else if (scan_row < OUTPUT_HEIGHT - 1) begin
                            scan_col <= 0;
                            scan_row <= scan_row + 1;
                        end else if (scan_ch < OUTPUT_CHANNELS - 1) begin
                            scan_col <= 0;
                            scan_row <= 0;
                            scan_ch <= scan_ch + 1;
                        end else begin
                            // Done scanning
                            state <= IDLE;
                            s_axis_spike_tready <= 1'b1;
                        end
                    end
                end
                
                //=============================================================
                // OUTPUT_SPIKE: Generate sparse output spike
                //=============================================================
                OUTPUT_SPIKE: begin
                    if (m_axis_spike_tready) begin
                        m_axis_spike_tvalid <= 1'b1;
                        m_axis_spike_tdata <= {scan_ch, scan_row, scan_col, in_timestamp};
                        output_spike_count <= output_spike_count + 1;
                        
                        // Reset membrane after spike
                        vmem_memory[vmem_idx(scan_ch, scan_row, scan_col)] <= 0;
                        
                        // Continue scanning
                        if (scan_col < OUTPUT_WIDTH - 1) begin
                            scan_col <= scan_col + 1;
                        end else if (scan_row < OUTPUT_HEIGHT - 1) begin
                            scan_col <= 0;
                            scan_row <= scan_row + 1;
                        end else if (scan_ch < OUTPUT_CHANNELS - 1) begin
                            scan_col <= 0;
                            scan_row <= 0;
                            scan_ch <= scan_ch + 1;
                        end else begin
                            state <= IDLE;
                            s_axis_spike_tready <= 1'b1;
                        end
                        state <= CHECK_OUTPUTS;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule
