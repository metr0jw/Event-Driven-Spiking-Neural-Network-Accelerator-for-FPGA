//-----------------------------------------------------------------------------
// Title         : Energy-Efficient AC-based SNN 1D Convolution (Verilog-2001)
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_conv1d.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Description   : 1D Convolution using Accumulate-only (AC) operations
//                 
//                 KEY INSIGHT - Why AC instead of MAC:
//                 =====================================
//                 In traditional ANN
//                   output[j] = Σ(input[i] * weight[i,j])  -- MAC operations
//                   
//                 In SNNs with binary spikes:
//                   output[j] = Σ(weight[i,j]) where spike[i]=1  -- AC only!
//                   
//                 Since spike ∈ {0,1}:
//                   spike * weight = weight (if spike=1)
//                   spike * weight = 0      (if spike=0, skip entirely!)
//                 
//                 ENERGY COMPARISON (45nm technology):
//                 - 32-bit FP MAC: ~4.6 pJ
//                 - 32-bit INT MAC: ~3.1 pJ  
//                 - 32-bit INT AC:  ~0.9 pJ
//                 - Energy ratio:  ~3-5x savings per operation
//                 
//                 SPARSITY BONUS:
//                 - Typical SNN sparsity: 90-99%
//                 - Only 1-10% of operations actually execute
//                 - Combined savings: 30-500x vs dense ANN
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module snn_conv1d #(
    parameter INPUT_LENGTH      = 128,
    parameter INPUT_CHANNELS    = 16,
    parameter OUTPUT_CHANNELS   = 32,
    parameter KERNEL_SIZE       = 3,
    parameter STRIDE            = 1,
    parameter PADDING           = 1,
    
    // Calculated output length
    parameter OUTPUT_LENGTH     = (INPUT_LENGTH + 2*PADDING - KERNEL_SIZE) / STRIDE + 1,
    
    // Precision parameters
    parameter WEIGHT_WIDTH      = 8,        // INT8 weights
    parameter VMEM_WIDTH        = 16,       // Q8.8 membrane potential
    parameter THRESHOLD         = 16'h0100, // 1.0 in Q8.8
    parameter LEAK_SHIFT        = 4,        // Leak = vmem >> 4 (≈0.9375 decay)
    
    // Address widths (Verilog-2001 compatible)
    parameter CH_ADDR_WIDTH     = 5,        // log2(32) = 5
    parameter POS_ADDR_WIDTH    = 7         // log2(128) = 7
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         enable,
    
    //=========================================================================
    // Sparse Spike Input Interface (AXI-Stream)
    // Only non-zero spikes are transmitted - zero spikes don't exist!
    //=========================================================================
    input  wire                         s_axis_spike_tvalid,
    input  wire [31:0]                  s_axis_spike_tdata,   // {ch[15:8], pos[7:0], timestamp[15:0]}
    input  wire                         s_axis_spike_tlast,
    output reg                          s_axis_spike_tready,
    
    //=========================================================================
    // Sparse Spike Output Interface (AXI-Stream)
    // Only output neurons that spike - exploits output sparsity!
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
    // Energy Monitoring Statistics
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
    localparam RECEIVE_SPIKE  = 4'd1;
    localparam CALC_POSITIONS = 4'd2;
    localparam FETCH_WEIGHT   = 4'd3;
    localparam WAIT_WEIGHT    = 4'd4;
    localparam ACCUMULATE     = 4'd5;
    localparam NEXT_KERNEL    = 4'd6;
    localparam NEXT_CHANNEL   = 4'd7;
    localparam CHECK_SPIKE    = 4'd8;
    localparam OUTPUT_SPIKE   = 4'd9;
    localparam APPLY_LEAK     = 4'd10;
    localparam DONE           = 4'd11;
    
    reg [3:0] state;
    assign busy = (state != IDLE);
    
    //=========================================================================
    // Input Spike Parsing
    //=========================================================================
    reg [CH_ADDR_WIDTH-1:0]  input_ch;
    reg [POS_ADDR_WIDTH-1:0] input_pos;
    reg [15:0]               input_timestamp;
    
    //=========================================================================
    // Convolution Counters
    //=========================================================================
    reg [CH_ADDR_WIDTH-1:0]  out_ch_counter;
    reg [2:0]                kernel_counter;
    reg [POS_ADDR_WIDTH-1:0] out_pos;
    
    //=========================================================================
    // Membrane Potential Memory (Flattened for Verilog-2001)
    // Index = out_ch * OUTPUT_LENGTH + out_pos
    //=========================================================================
    reg signed [VMEM_WIDTH-1:0] membrane_mem [0:OUTPUT_CHANNELS*OUTPUT_LENGTH-1];
    
    // Helper function for memory indexing
    function [15:0] mem_idx;
        input [CH_ADDR_WIDTH-1:0] ch;
        input [POS_ADDR_WIDTH-1:0] pos;
        begin
            mem_idx = ch * OUTPUT_LENGTH + pos;
        end
    endfunction
    
    //=========================================================================
    // Weight Address Calculation
    // weight_addr = out_ch * (INPUT_CHANNELS * KERNEL_SIZE) + 
    //               input_ch * KERNEL_SIZE + kernel_pos
    //=========================================================================
    function [15:0] calc_weight_addr;
        input [CH_ADDR_WIDTH-1:0] o_ch;
        input [CH_ADDR_WIDTH-1:0] i_ch;
        input [2:0] k_pos;
        begin
            calc_weight_addr = o_ch * (INPUT_CHANNELS * KERNEL_SIZE) + 
                              i_ch * KERNEL_SIZE + k_pos;
        end
    endfunction
    
    //=========================================================================
    // Current Threshold (configurable or default)
    //=========================================================================
    reg [VMEM_WIDTH-1:0] current_threshold;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            current_threshold <= THRESHOLD;
        else if (config_valid)
            current_threshold <= config_threshold;
    end
    
    //=========================================================================
    // Memory Initialization
    //=========================================================================
    integer i;
    initial begin
        for (i = 0; i < OUTPUT_CHANNELS * OUTPUT_LENGTH; i = i + 1) begin
            membrane_mem[i] = 0;
        end
    end
    
    //=========================================================================
    // Main Processing State Machine
    //=========================================================================
    reg signed [VMEM_WIDTH-1:0] current_vmem;
    reg signed [VMEM_WIDTH:0]   vmem_after_ac;
    reg [15:0]                  current_mem_idx;
    reg [POS_ADDR_WIDTH-1:0]    spike_output_pos;
    reg [CH_ADDR_WIDTH-1:0]     spike_output_ch;
    
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
            
            out_ch_counter <= 0;
            kernel_counter <= 0;
            out_pos <= 0;
            
        end else if (enable) begin
            cycle_count <= cycle_count + 1;
            
            // Default outputs
            m_axis_spike_tvalid <= 1'b0;
            weight_rd_en <= 1'b0;
            
            case (state)
                //=============================================================
                // IDLE: Wait for input spike (zero energy when no spikes)
                //=============================================================
                IDLE: begin
                    s_axis_spike_tready <= 1'b1;
                    if (s_axis_spike_tvalid) begin
                        state <= RECEIVE_SPIKE;
                    end
                end
                
                //=============================================================
                // RECEIVE_SPIKE: Parse incoming spike data
                //=============================================================
                RECEIVE_SPIKE: begin
                    s_axis_spike_tready <= 1'b0;
                    input_ch <= s_axis_spike_tdata[23:16];
                    input_pos <= s_axis_spike_tdata[7:0];
                    input_timestamp <= s_axis_spike_tdata[15:0];
                    input_spike_count <= input_spike_count + 1;
                    
                    out_ch_counter <= 0;
                    state <= CALC_POSITIONS;
                end
                
                //=============================================================
                // CALC_POSITIONS: Calculate which output positions are affected
                //=============================================================
                CALC_POSITIONS: begin
                    kernel_counter <= 0;
                    // Calculate output position from input position
                    // out_pos = (input_pos + PADDING - kernel_idx) / STRIDE
                    out_pos <= (input_pos + PADDING) / STRIDE;
                    state <= FETCH_WEIGHT;
                end
                
                //=============================================================
                // FETCH_WEIGHT: Request weight from memory
                //=============================================================
                FETCH_WEIGHT: begin
                    weight_addr <= calc_weight_addr(out_ch_counter, input_ch, kernel_counter);
                    weight_rd_en <= 1'b1;
                    memory_access_count <= memory_access_count + 1;
                    state <= WAIT_WEIGHT;
                end
                
                //=============================================================
                // WAIT_WEIGHT: Wait for memory read
                //=============================================================
                WAIT_WEIGHT: begin
                    weight_rd_en <= 1'b0;
                    if (weight_valid) begin
                        state <= ACCUMULATE;
                    end
                end
                
                //=============================================================
                // ACCUMULATE: THE AC OPERATION!
                // 
                // This is where energy savings happen:
                // - Traditional MAC: vmem += spike * weight (multiply + add)
                // - Our AC:          vmem += weight        (add only!)
                //
                // Since spike=1 (we only process valid spikes), 
                // spike * weight = weight, so multiply is eliminated!
                //=============================================================
                ACCUMULATE: begin
                    // Calculate output position for this kernel element
                    if (out_pos >= kernel_counter && out_pos - kernel_counter < OUTPUT_LENGTH) begin
                        current_mem_idx <= mem_idx(out_ch_counter, out_pos - kernel_counter);
                        current_vmem <= membrane_mem[mem_idx(out_ch_counter, out_pos - kernel_counter)];
                        
                        // AC OPERATION: Simply add weight (no multiply!)
                        vmem_after_ac <= membrane_mem[mem_idx(out_ch_counter, out_pos - kernel_counter)] 
                                        + {{(VMEM_WIDTH-WEIGHT_WIDTH){weight_data[WEIGHT_WIDTH-1]}}, weight_data};
                        
                        // Update membrane with saturation
                        if (vmem_after_ac[VMEM_WIDTH] != vmem_after_ac[VMEM_WIDTH-1]) begin
                            // Overflow - saturate
                            membrane_mem[current_mem_idx] <= vmem_after_ac[VMEM_WIDTH] ? 
                                {1'b1, {(VMEM_WIDTH-1){1'b0}}} : {1'b0, {(VMEM_WIDTH-1){1'b1}}};
                        end else begin
                            membrane_mem[current_mem_idx] <= vmem_after_ac[VMEM_WIDTH-1:0];
                        end
                        
                        ac_operation_count <= ac_operation_count + 1;
                    end
                    
                    state <= NEXT_KERNEL;
                end
                
                //=============================================================
                // NEXT_KERNEL: Move to next kernel position
                //=============================================================
                NEXT_KERNEL: begin
                    if (kernel_counter < KERNEL_SIZE - 1) begin
                        kernel_counter <= kernel_counter + 1;
                        state <= FETCH_WEIGHT;
                    end else begin
                        state <= NEXT_CHANNEL;
                    end
                end
                
                //=============================================================
                // NEXT_CHANNEL: Move to next output channel
                //=============================================================
                NEXT_CHANNEL: begin
                    if (out_ch_counter < OUTPUT_CHANNELS - 1) begin
                        out_ch_counter <= out_ch_counter + 1;
                        kernel_counter <= 0;
                        state <= FETCH_WEIGHT;
                    end else begin
                        // Done processing this input spike
                        state <= CHECK_SPIKE;
                        spike_output_ch <= 0;
                        spike_output_pos <= 0;
                    end
                end
                
                //=============================================================
                // CHECK_SPIKE: Check if any neuron should fire
                //=============================================================
                CHECK_SPIKE: begin
                    if (membrane_mem[mem_idx(spike_output_ch, spike_output_pos)] >= 
                        $signed(current_threshold)) begin
                        state <= OUTPUT_SPIKE;
                    end else begin
                        // Move to next position
                        if (spike_output_pos < OUTPUT_LENGTH - 1) begin
                            spike_output_pos <= spike_output_pos + 1;
                        end else if (spike_output_ch < OUTPUT_CHANNELS - 1) begin
                            spike_output_pos <= 0;
                            spike_output_ch <= spike_output_ch + 1;
                        end else begin
                            // Done checking all neurons
                            state <= IDLE;
                            s_axis_spike_tready <= 1'b1;
                        end
                    end
                end
                
                //=============================================================
                // OUTPUT_SPIKE: Generate output spike (sparse output!)
                //=============================================================
                OUTPUT_SPIKE: begin
                    if (m_axis_spike_tready) begin
                        m_axis_spike_tvalid <= 1'b1;
                        m_axis_spike_tdata <= {8'b0, spike_output_ch, spike_output_pos, input_timestamp};
                        output_spike_count <= output_spike_count + 1;
                        
                        // Reset membrane potential after spike
                        membrane_mem[mem_idx(spike_output_ch, spike_output_pos)] <= 0;
                        
                        // Continue checking
                        if (spike_output_pos < OUTPUT_LENGTH - 1) begin
                            spike_output_pos <= spike_output_pos + 1;
                        end else if (spike_output_ch < OUTPUT_CHANNELS - 1) begin
                            spike_output_pos <= 0;
                            spike_output_ch <= spike_output_ch + 1;
                        end else begin
                            state <= IDLE;
                            s_axis_spike_tready <= 1'b1;
                        end
                        state <= CHECK_SPIKE;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule
