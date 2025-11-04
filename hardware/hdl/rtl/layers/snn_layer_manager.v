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
    
    // Configuration management
    always @(posedge clk) begin
        if (reset) begin
            integer i, j;
            for (i = 0; i < MAX_LAYERS; i = i + 1) begin
                layer_types[i] <= LAYER_INACTIVE;
                layer_enabled[i] <= 1'b0;
                for (j = 0; j < 16; j = j + 1) begin
                    layer_configs[i][j] <= 32'b0;
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
    
    // Layer instantiation with generate blocks
    genvar i;
    generate
        for (i = 0; i < MAX_LAYERS; i = i + 1) begin : layer_gen
            
            // Layer instantiation based on type
            case (layer_config[i*CONFIG_WIDTH +: 4])  // Layer type in lower 4 bits
                LAYER_CONV1D: begin
                    // 1D Convolution layer instantiation would go here
                    // For now, pass through as placeholder
                    assign layer_outputs[i] = layer_inputs[i];
                end
                
                LAYER_CONV2D: begin
                    // 2D Convolution layer instantiation would go here
                    // For now, pass through as placeholder
                    assign layer_outputs[i] = layer_inputs[i];
                end
                
                LAYER_AVGPOOL2D: begin
                    // Average pooling layer
                    snn_avgpool2d #(
                        .INPUT_WIDTH(28),
                        .INPUT_HEIGHT(28),
                        .INPUT_CHANNELS(32),
                        .POOL_SIZE(2),
                        .STRIDE(2),
                        .VMEM_WIDTH(VMEM_WIDTH),
                        .POOL_THRESHOLD(16'h2000)
                    ) avgpool_layer (
                        .clk(clk),
                        .reset(reset),
                        .enable(enable && (layer_types[i] == LAYER_AVGPOOL2D) && 
                                (active_layer == i) && execution_active),
                        
                        .s_axis_input_tdata(layer_input_tdata[i][31:0]),
                        .s_axis_input_tvalid(layer_input_tvalid[i] && (layer_types[i] == LAYER_AVGPOOL2D)),
                        .s_axis_input_tready(layer_input_tready[i]),
                        .s_axis_input_tlast(layer_input_tlast[i]),
                        
                        .m_axis_output_tdata(layer_output_tdata[i][31:0]),
                        .m_axis_output_tvalid(layer_output_tvalid[i]),
                        .m_axis_output_tready(layer_output_tready[i]),
                        .m_axis_output_tlast(layer_output_tlast[i]),
                        
                        .threshold_config(layer_configs[i][0][15:0]),
                        .decay_factor(layer_configs[i][1][7:0]),
                        .pooling_weight(layer_configs[i][2][7:0]),
                        
                        .input_spike_count(layer_input_spike_count[i]),
                        .output_spike_count(layer_output_spike_count[i]),
                        .computation_done(layer_computation_done[i])
                    );
                end
                
                // Max pooling layer
                LAYER_MAXPOOL2D: begin
                    snn_maxpool2d #(
                        .INPUT_WIDTH(28),
                        .INPUT_HEIGHT(28),
                        .INPUT_CHANNELS(32),
                        .POOL_SIZE(2),
                        .STRIDE(2),
                        .TIME_WIDTH(16)
                    ) maxpool_layer (
                        .clk(clk),
                        .reset(reset),
                        .enable(enable && (layer_types[i] == LAYER_MAXPOOL2D) && 
                                (active_layer == i) && execution_active),
                        
                        .s_axis_input_tdata(layer_input_tdata[i]),
                        .s_axis_input_tvalid(layer_input_tvalid[i] && (layer_types[i] == LAYER_MAXPOOL2D)),
                        .s_axis_input_tready(layer_input_tready[i]),
                        .s_axis_input_tlast(layer_input_tlast[i]),
                        
                        .m_axis_output_tdata(layer_output_tdata[i]),
                        .m_axis_output_tvalid(layer_output_tvalid[i]),
                        .m_axis_output_tready(layer_output_tready[i]),
                        .m_axis_output_tlast(layer_output_tlast[i]),
                        
                        .pooling_window_time(layer_configs[i][0][15:0]),
                        .winner_take_all_enable(layer_configs[i][1][0]),
                        
                        .input_spike_count(layer_input_spike_count[i]),
                        .output_spike_count(layer_output_spike_count[i]),
                        .computation_done(layer_computation_done[i])
                    );
                end
                
                // Set unused outputs to zero for inactive layers
                assign layer_output_tdata[i] = (layer_types[i] != LAYER_INACTIVE && 
                                               active_layer == i && execution_active) ? 
                                              layer_output_tdata[i] : {DATA_WIDTH{1'b0}};
                assign layer_output_tvalid[i] = (layer_types[i] != LAYER_INACTIVE && 
                                                active_layer == i && execution_active) ? 
                                               layer_output_tvalid[i] : 1'b0;
            endcase
        end
    endgenerate
    
    // Input routing - connect to active layer
    assign s_axis_input_tready = (active_layer < MAX_LAYERS) ? 
                                layer_input_tready[active_layer] : 1'b1;
    
    generate
        for (i = 0; i < MAX_LAYERS; i = i + 1) begin : input_routing
            assign layer_input_tdata[i] = (active_layer == i) ? s_axis_input_tdata : {DATA_WIDTH{1'b0}};
            assign layer_input_tvalid[i] = (active_layer == i) ? s_axis_input_tvalid : 1'b0;
            assign layer_input_tlast[i] = (active_layer == i) ? s_axis_input_tlast : 1'b0;
        end
    endgenerate
    
    // Output routing - connect from active layer
    assign m_axis_output_tdata = (active_layer < MAX_LAYERS) ? 
                                layer_output_tdata[active_layer] : {DATA_WIDTH{1'b0}};
    assign m_axis_output_tvalid = (active_layer < MAX_LAYERS) ? 
                                 layer_output_tvalid[active_layer] : 1'b0;
    assign m_axis_output_tlast = (active_layer < MAX_LAYERS) ? 
                                layer_output_tlast[active_layer] : 1'b0;
    
    generate
        for (i = 0; i < MAX_LAYERS; i = i + 1) begin : output_routing
            assign layer_output_tready[i] = (active_layer == i) ? m_axis_output_tready : 1'b0;
        end
    endgenerate
    
    // Status outputs
    assign execute_done = !execution_active;
    assign current_layer_id = active_layer;
    assign total_input_spikes = total_input_count;
    assign total_output_spikes = total_output_count;
    
    generate
        for (i = 0; i < MAX_LAYERS; i = 
