//-----------------------------------------------------------------------------
// Title         : SNN Layer Manager
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_layer_manager.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : PyTorch-compatible SNN layer management and routing
//                 Supports Conv2d, AvgPool2d, MaxPool2d, and Linear layers
//-----------------------------------------------------------------------------

module snn_layer_manager #(
    parameter MAX_LAYERS     = 16,     // Maximum number of layers
    parameter DATA_WIDTH     = 48,     // Spike data width
    parameter CONFIG_WIDTH   = 32,     // Configuration width
    parameter WEIGHT_WIDTH   = 8,      // Weight precision
    parameter VMEM_WIDTH     = 16      // Membrane potential precision
)(
    input wire clk,
    input wire reset,
    input wire enable,
    
    // Input interface (from previous layer or input encoder)
    input wire [DATA_WIDTH-1:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    input wire s_axis_input_tlast,
    
    // Output interface (to next layer or output decoder)
    output wire [DATA_WIDTH-1:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready,
    output wire m_axis_output_tlast,
    
    // Configuration interface
    input wire [7:0] config_layer_id,
    input wire [7:0] config_layer_type,    // 0: Conv2d, 1: AvgPool2d, 2: MaxPool2d, 3: Linear
    input wire [CONFIG_WIDTH-1:0] config_data,
    input wire config_write,
    
    // Weight loading interface
    input wire [7:0] weight_layer_id,
    input wire [15:0] weight_addr,
    input wire [WEIGHT_WIDTH-1:0] weight_data,
    input wire weight_write,
    
    // Layer execution control
    input wire [7:0] execute_layer_id,
    input wire execute_start,
    output wire execute_done,
    
    // Status and debug
    output wire [31:0] total_input_spikes,
    output wire [31:0] total_output_spikes,
    output wire [MAX_LAYERS-1:0] layer_active_status,
    output wire [7:0] current_layer_id
);

    // Layer type definitions
    localparam LAYER_CONV1D    = 4'h0;
    localparam LAYER_CONV2D    = 4'h1;
    localparam LAYER_AVGPOOL2D = 4'h2;
    localparam LAYER_MAXPOOL2D = 4'h3;
    localparam LAYER_DENSE     = 4'h4;
    localparam LAYER_INACTIVE  = 4'hF;
    
    // Layer configuration storage
    reg [7:0] layer_types [0:MAX_LAYERS-1];
    reg [CONFIG_WIDTH-1:0] layer_configs [0:MAX_LAYERS-1][0:15]; // 16 config words per layer
    reg layer_enabled [0:MAX_LAYERS-1];
    
    // Current execution state
    reg [7:0] active_layer;
    reg execution_active;
    reg [2:0] exec_state;
    
    // Inter-layer connection signals
    wire [DATA_WIDTH-1:0] layer_input_tdata [0:MAX_LAYERS-1];
    wire layer_input_tvalid [0:MAX_LAYERS-1];
    wire layer_input_tready [0:MAX_LAYERS-1];
    wire layer_input_tlast [0:MAX_LAYERS-1];
    
    wire [DATA_WIDTH-1:0] layer_output_tdata [0:MAX_LAYERS-1];
    wire layer_output_tvalid [0:MAX_LAYERS-1];
    wire layer_output_tready [0:MAX_LAYERS-1];
    wire layer_output_tlast [0:MAX_LAYERS-1];
    
    // Layer-specific control signals
    wire [31:0] layer_input_spike_count [0:MAX_LAYERS-1];
    wire [31:0] layer_output_spike_count [0:MAX_LAYERS-1];
    wire layer_computation_done [0:MAX_LAYERS-1];
    
    // Aggregated statistics
    reg [31:0] total_input_count;
    reg [31:0] total_output_count;
    
    // Loop variables (Verilog-2001 compatible)
    integer cfg_i, cfg_j;
    
    // Configuration management
    always @(posedge clk) begin
        if (reset) begin
            for (cfg_i = 0; cfg_i < MAX_LAYERS; cfg_i = cfg_i + 1) begin
                layer_types[cfg_i] <= LAYER_INACTIVE;
                layer_enabled[cfg_i] <= 1'b0;
                for (cfg_j = 0; cfg_j < 16; cfg_j = cfg_j + 1) begin
                    layer_configs[cfg_i][cfg_j] <= 32'b0;
                end
            end
            active_layer <= 8'hFF;
            execution_active <= 1'b0;
            exec_state <= 3'b000;
            total_input_count <= 32'b0;
            total_output_count <= 32'b0;
        end else begin
            // Configuration updates
            if (config_write && config_layer_id < MAX_LAYERS) begin
                if (config_data[31:24] == 8'hFF) begin
                    // Layer type configuration
                    layer_types[config_layer_id] <= config_data[7:0];
                    layer_enabled[config_layer_id] <= 1'b1;
                end else begin
                    // Layer parameter configuration
                    layer_configs[config_layer_id][config_data[31:28]] <= config_data;
                end
            end
            
            // Execution control
            if (execute_start && !execution_active) begin
                active_layer <= execute_layer_id;
                execution_active <= 1'b1;
                exec_state <= 3'b001;
            end else if (execution_active && layer_computation_done[active_layer]) begin
                execution_active <= 1'b0;
                exec_state <= 3'b000;
            end
            
            // Update statistics
            total_input_count <= total_input_count + (s_axis_input_tvalid && s_axis_input_tready ? 1 : 0);
            total_output_count <= total_output_count + (m_axis_output_tvalid && m_axis_output_tready ? 1 : 0);
        end
    end
    
    //=========================================================================
    // Layer Pipeline Architecture
    //=========================================================================
    // Design approach: Instead of dynamic layer type selection at runtime,
    // we use a fixed pipeline with enable signals controlled by layer_enabled.
    // Each slot can be configured with parameters, but the layer type is
    // determined by the pipeline position (compile-time).
    //
    // For runtime layer type selection, use a crossbar switch architecture
    // with pre-instantiated layer processors.
    //=========================================================================
    
    // Internal pipeline signals
    wire [DATA_WIDTH-1:0] pipe_data [0:MAX_LAYERS];
    wire pipe_valid [0:MAX_LAYERS];
    wire pipe_ready [0:MAX_LAYERS];
    wire pipe_last [0:MAX_LAYERS];
    
    // Connect input to pipeline start
    assign pipe_data[0] = s_axis_input_tdata;
    assign pipe_valid[0] = s_axis_input_tvalid;
    assign pipe_last[0] = s_axis_input_tlast;
    assign s_axis_input_tready = pipe_ready[0];
    
    // Connect pipeline end to output
    assign m_axis_output_tdata = pipe_data[MAX_LAYERS];
    assign m_axis_output_tvalid = pipe_valid[MAX_LAYERS];
    assign m_axis_output_tlast = pipe_last[MAX_LAYERS];
    assign pipe_ready[MAX_LAYERS] = m_axis_output_tready;
    
    //=========================================================================
    // Layer Pipeline Generation
    //=========================================================================
    // Each layer slot acts as a configurable processing element
    // When layer_enabled[i] is 0, the slot passes data through
    // When layer_enabled[i] is 1, the slot processes data based on config
    //=========================================================================
    
    genvar i;
    generate
        for (i = 0; i < MAX_LAYERS; i = i + 1) begin : layer_pipeline
            
            // Per-layer processing registers
            reg [DATA_WIDTH-1:0] proc_data_out;
            reg proc_valid_out;
            reg proc_last_out;
            reg proc_ready_out;
            
            // Layer processing state machine
            reg [2:0] layer_state;
            localparam L_IDLE    = 3'd0;
            localparam L_RECEIVE = 3'd1;
            localparam L_PROCESS = 3'd2;
            localparam L_OUTPUT  = 3'd3;
            localparam L_DONE    = 3'd4;
            
            // Processing buffer
            reg [DATA_WIDTH-1:0] data_buffer;
            reg data_buffered;
            reg last_buffered;
            
            // Computation done flag for this layer
            reg layer_done;
            assign layer_computation_done[i] = layer_done;
            
            always @(posedge clk) begin
                if (reset) begin
                    layer_state <= L_IDLE;
                    proc_data_out <= {DATA_WIDTH{1'b0}};
                    proc_valid_out <= 1'b0;
                    proc_last_out <= 1'b0;
                    proc_ready_out <= 1'b1;
                    data_buffer <= {DATA_WIDTH{1'b0}};
                    data_buffered <= 1'b0;
                    last_buffered <= 1'b0;
                    layer_done <= 1'b0;
                end else if (!enable) begin
                    // Pass through when disabled
                    proc_data_out <= pipe_data[i];
                    proc_valid_out <= pipe_valid[i];
                    proc_last_out <= pipe_last[i];
                    proc_ready_out <= pipe_ready[i+1];
                    layer_done <= 1'b1;
                end else if (!layer_enabled[i]) begin
                    // Pass through when layer not configured
                    proc_data_out <= pipe_data[i];
                    proc_valid_out <= pipe_valid[i];
                    proc_last_out <= pipe_last[i];
                    proc_ready_out <= pipe_ready[i+1];
                    layer_done <= 1'b1;
                end else begin
                    // Layer is enabled - process based on layer type
                    layer_done <= 1'b0;
                    
                    case (layer_state)
                        L_IDLE: begin
                            proc_ready_out <= 1'b1;
                            proc_valid_out <= 1'b0;
                            if (pipe_valid[i] && proc_ready_out) begin
                                data_buffer <= pipe_data[i];
                                last_buffered <= pipe_last[i];
                                data_buffered <= 1'b1;
                                layer_state <= L_PROCESS;
                            end
                        end
                        
                        L_PROCESS: begin
                            proc_ready_out <= 1'b0;
                            // Simple processing based on layer type
                            // This can be extended with actual computation
                            case (layer_types[i][3:0])
                                LAYER_CONV1D, LAYER_CONV2D: begin
                                    // Placeholder: pass through (actual conv would be here)
                                    proc_data_out <= data_buffer;
                                end
                                LAYER_AVGPOOL2D: begin
                                    // Placeholder: simple scaling (actual pooling would be here)
                                    proc_data_out <= data_buffer;
                                end
                                LAYER_MAXPOOL2D: begin
                                    // Placeholder: pass through (actual max pooling would be here)
                                    proc_data_out <= data_buffer;
                                end
                                LAYER_DENSE: begin
                                    // Placeholder: pass through (actual dense would be here)
                                    proc_data_out <= data_buffer;
                                end
                                default: begin
                                    proc_data_out <= data_buffer;
                                end
                            endcase
                            proc_last_out <= last_buffered;
                            layer_state <= L_OUTPUT;
                        end
                        
                        L_OUTPUT: begin
                            proc_valid_out <= 1'b1;
                            if (pipe_ready[i+1]) begin
                                proc_valid_out <= 1'b0;
                                data_buffered <= 1'b0;
                                if (last_buffered) begin
                                    layer_state <= L_DONE;
                                end else begin
                                    layer_state <= L_IDLE;
                                end
                            end
                        end
                        
                        L_DONE: begin
                            layer_done <= 1'b1;
                            proc_ready_out <= 1'b0;
                            proc_valid_out <= 1'b0;
                            // Wait for new execution command
                            if (!execution_active) begin
                                layer_state <= L_IDLE;
                            end
                        end
                        
                        default: layer_state <= L_IDLE;
                    endcase
                end
            end
            
            // Connect to pipeline
            assign pipe_data[i+1] = proc_data_out;
            assign pipe_valid[i+1] = proc_valid_out;
            assign pipe_last[i+1] = proc_last_out;
            assign pipe_ready[i] = proc_ready_out;
            
            // Statistics tracking
            reg [31:0] input_count;
            reg [31:0] output_count;
            
            always @(posedge clk) begin
                if (reset) begin
                    input_count <= 32'b0;
                    output_count <= 32'b0;
                end else begin
                    if (pipe_valid[i] && pipe_ready[i])
                        input_count <= input_count + 1;
                    if (pipe_valid[i+1] && pipe_ready[i+1])
                        output_count <= output_count + 1;
                end
            end
            
            assign layer_input_spike_count[i] = input_count;
            assign layer_output_spike_count[i] = output_count;
            
            // Unused inter-layer signals - tie off
            assign layer_input_tdata[i] = pipe_data[i];
            assign layer_input_tvalid[i] = pipe_valid[i];
            assign layer_input_tlast[i] = pipe_last[i];
            assign layer_input_tready[i] = pipe_ready[i];
            assign layer_output_tdata[i] = pipe_data[i+1];
            assign layer_output_tvalid[i] = pipe_valid[i+1];
            assign layer_output_tlast[i] = pipe_last[i+1];
            assign layer_output_tready[i] = pipe_ready[i+1];
            
        end
    endgenerate
    
    // Status outputs
    assign execute_done = !execution_active;
    assign current_layer_id = active_layer;
    assign total_input_spikes = total_input_count;
    assign total_output_spikes = total_output_count;
    
    // Layer active status - generate layer_active_status from layer_enabled array
    genvar k;
    generate
        for (k = 0; k < MAX_LAYERS; k = k + 1) begin : gen_layer_status
            assign layer_active_status[k] = layer_enabled[k];
        end
    endgenerate

endmodule
