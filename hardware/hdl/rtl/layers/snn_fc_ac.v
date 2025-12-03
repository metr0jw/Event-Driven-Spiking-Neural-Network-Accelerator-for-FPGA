//-----------------------------------------------------------------------------
// Title         : Energy-Efficient AC-based SNN Fully Connected Layer
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_fc_ac.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Description   : Fully Connected layer using Accumulate-only operations
//                 
//                 FC Layer Energy Analysis:
//                 =========================
//                 ANN FC: N_in × N_out MAC operations per inference
//                 SNN FC: N_active × N_out AC operations (N_active << N_in)
//                 
//                 Example (1000 input, 100 output, 5% sparsity):
//                 - ANN: 1000 × 100 × 4.6pJ = 460,000 pJ
//                 - SNN: 50 × 100 × 0.9pJ = 4,500 pJ
//                 - Savings: ~100x energy reduction!
//                 
//                 This layer is particularly efficient because:
//                 1. FC layers have dense weight matrices
//                 2. SNN sparsity dramatically reduces operations
//                 3. Only AC (add) operations, no multiply
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module snn_fc_ac #(
    parameter NUM_INPUTS        = 256,      // Input neurons
    parameter NUM_OUTPUTS       = 10,       // Output neurons (e.g., 10 classes)
    parameter WEIGHT_WIDTH      = 8,        // INT8 weights
    parameter VMEM_WIDTH        = 16,       // Q8.8 membrane potential
    parameter THRESHOLD         = 16'h0100, // 1.0 in Q8.8
    parameter LEAK_SHIFT        = 4,        // Decay using shift
    
    // Address widths
    parameter IN_ADDR_WIDTH     = 8,        // log2(256) = 8
    parameter OUT_ADDR_WIDTH    = 4         // log2(16) >= log2(10)
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         enable,
    
    //=========================================================================
    // Sparse Spike Input
    //=========================================================================
    input  wire                         spike_in_valid,
    input  wire [IN_ADDR_WIDTH-1:0]     spike_in_id,
    input  wire [7:0]                   spike_in_timestamp,
    output reg                          spike_in_ready,
    
    //=========================================================================
    // Sparse Spike Output
    //=========================================================================
    output reg                          spike_out_valid,
    output reg  [OUT_ADDR_WIDTH-1:0]    spike_out_id,
    output reg  [7:0]                   spike_out_timestamp,
    input  wire                         spike_out_ready,
    
    //=========================================================================
    // Weight Memory Interface
    //=========================================================================
    output reg                          weight_rd_en,
    output reg  [IN_ADDR_WIDTH+OUT_ADDR_WIDTH-1:0] weight_addr,
    input  wire signed [WEIGHT_WIDTH-1:0] weight_data,
    input  wire                         weight_valid,
    
    //=========================================================================
    // Classification Output (for inference)
    // Returns membrane potentials of all output neurons
    //=========================================================================
    input  wire                         read_output_en,
    input  wire [OUT_ADDR_WIDTH-1:0]    read_output_id,
    output reg  signed [VMEM_WIDTH-1:0] read_output_vmem,
    
    //=========================================================================
    // Control
    //=========================================================================
    input  wire                         clear_state,        // Clear all membrane potentials
    input  wire                         inference_done,     // Signal end of inference
    
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
    output wire                         busy
);

    //=========================================================================
    // State Machine
    //=========================================================================
    localparam IDLE         = 3'd0;
    localparam CAPTURE      = 3'd1;
    localparam FETCH_WEIGHT = 3'd2;
    localparam WAIT_WEIGHT  = 3'd3;
    localparam AC_UPDATE    = 3'd4;
    localparam NEXT_OUTPUT  = 3'd5;
    localparam CHECK_SPIKE  = 3'd6;
    localparam OUTPUT_SPIKE = 3'd7;
    
    reg [2:0] state;
    assign busy = (state != IDLE);
    
    //=========================================================================
    // Membrane Potential Memory
    //=========================================================================
    reg signed [VMEM_WIDTH-1:0] vmem_mem [0:NUM_OUTPUTS-1];
    
    //=========================================================================
    // Processing Registers
    //=========================================================================
    reg [IN_ADDR_WIDTH-1:0]  current_input_id;
    reg [7:0]                current_timestamp;
    reg [OUT_ADDR_WIDTH-1:0] out_counter;
    reg signed [VMEM_WIDTH-1:0] current_vmem;
    reg signed [VMEM_WIDTH:0]   vmem_new;
    
    //=========================================================================
    // Threshold Register
    //=========================================================================
    reg [VMEM_WIDTH-1:0] threshold_reg;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            threshold_reg <= THRESHOLD;
        else if (config_valid)
            threshold_reg <= config_threshold;
    end
    
    //=========================================================================
    // Memory Initialization and Clear
    //=========================================================================
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < NUM_OUTPUTS; i = i + 1) begin
                vmem_mem[i] <= 0;
            end
        end else if (clear_state) begin
            for (i = 0; i < NUM_OUTPUTS; i = i + 1) begin
                vmem_mem[i] <= 0;
            end
        end
    end
    
    //=========================================================================
    // Output Memory Read (for classification result)
    //=========================================================================
    always @(posedge clk) begin
        if (read_output_en) begin
            read_output_vmem <= vmem_mem[read_output_id];
        end
    end
    
    //=========================================================================
    // Main Processing State Machine
    //=========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            spike_in_ready <= 1'b1;
            spike_out_valid <= 1'b0;
            spike_out_id <= 0;
            spike_out_timestamp <= 0;
            
            weight_rd_en <= 1'b0;
            weight_addr <= 0;
            
            input_spike_count <= 0;
            output_spike_count <= 0;
            ac_operation_count <= 0;
            
            out_counter <= 0;
            
        end else if (enable) begin
            spike_out_valid <= 1'b0;
            weight_rd_en <= 1'b0;
            
            case (state)
                //=============================================================
                // IDLE: Wait for input spike
                //=============================================================
                IDLE: begin
                    spike_in_ready <= 1'b1;
                    if (spike_in_valid) begin
                        state <= CAPTURE;
                    end
                end
                
                //=============================================================
                // CAPTURE: Store input spike info
                //=============================================================
                CAPTURE: begin
                    spike_in_ready <= 1'b0;
                    current_input_id <= spike_in_id;
                    current_timestamp <= spike_in_timestamp;
                    input_spike_count <= input_spike_count + 1;
                    out_counter <= 0;
                    state <= FETCH_WEIGHT;
                end
                
                //=============================================================
                // FETCH_WEIGHT: Request weight for current input→output pair
                // Weight layout: input_id * NUM_OUTPUTS + output_id
                //=============================================================
                FETCH_WEIGHT: begin
                    weight_addr <= current_input_id * NUM_OUTPUTS + out_counter;
                    weight_rd_en <= 1'b1;
                    state <= WAIT_WEIGHT;
                end
                
                //=============================================================
                // WAIT_WEIGHT: Wait for memory
                //=============================================================
                WAIT_WEIGHT: begin
                    if (weight_valid) begin
                        state <= AC_UPDATE;
                    end
                end
                
                //=============================================================
                // AC_UPDATE: Accumulate weight into membrane potential
                // 
                // THE CORE AC OPERATION:
                // vmem[out_id] += weight[in_id][out_id]
                // 
                // No multiply needed because spike is binary (1)!
                //=============================================================
                AC_UPDATE: begin
                    current_vmem <= vmem_mem[out_counter];
                    
                    // AC operation: add weight
                    vmem_new <= vmem_mem[out_counter] + 
                               {{(VMEM_WIDTH-WEIGHT_WIDTH){weight_data[WEIGHT_WIDTH-1]}}, weight_data};
                    
                    // Store with saturation
                    if (vmem_new[VMEM_WIDTH] != vmem_new[VMEM_WIDTH-1]) begin
                        vmem_mem[out_counter] <= vmem_new[VMEM_WIDTH] ?
                            {1'b1, {(VMEM_WIDTH-1){1'b0}}} : {1'b0, {(VMEM_WIDTH-1){1'b1}}};
                    end else begin
                        vmem_mem[out_counter] <= vmem_new[VMEM_WIDTH-1:0];
                    end
                    
                    ac_operation_count <= ac_operation_count + 1;
                    state <= NEXT_OUTPUT;
                end
                
                //=============================================================
                // NEXT_OUTPUT: Move to next output neuron
                //=============================================================
                NEXT_OUTPUT: begin
                    if (out_counter < NUM_OUTPUTS - 1) begin
                        out_counter <= out_counter + 1;
                        state <= FETCH_WEIGHT;
                    end else begin
                        // Done with all outputs for this input spike
                        out_counter <= 0;
                        state <= CHECK_SPIKE;
                    end
                end
                
                //=============================================================
                // CHECK_SPIKE: Check if any output neuron should fire
                //=============================================================
                CHECK_SPIKE: begin
                    if (vmem_mem[out_counter] >= $signed(threshold_reg)) begin
                        state <= OUTPUT_SPIKE;
                    end else begin
                        if (out_counter < NUM_OUTPUTS - 1) begin
                            out_counter <= out_counter + 1;
                        end else begin
                            state <= IDLE;
                            spike_in_ready <= 1'b1;
                        end
                    end
                end
                
                //=============================================================
                // OUTPUT_SPIKE: Generate output spike
                //=============================================================
                OUTPUT_SPIKE: begin
                    if (spike_out_ready) begin
                        spike_out_valid <= 1'b1;
                        spike_out_id <= out_counter;
                        spike_out_timestamp <= current_timestamp;
                        output_spike_count <= output_spike_count + 1;
                        
                        // Reset membrane after spike
                        vmem_mem[out_counter] <= 0;
                        
                        // Continue checking
                        if (out_counter < NUM_OUTPUTS - 1) begin
                            out_counter <= out_counter + 1;
                            state <= CHECK_SPIKE;
                        end else begin
                            state <= IDLE;
                            spike_in_ready <= 1'b1;
                        end
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule
