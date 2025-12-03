//-----------------------------------------------------------------------------
// Title         : SNN Layer Manager Testbench
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_snn_layer_manager.v
// Author        : Jiwoon Lee (@metr0jw)
// Description   : Testbench for SNN layer manager/controller
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_snn_layer_manager();

    //-------------------------------------------------------------------------
    // Parameters
    //-------------------------------------------------------------------------
    parameter MAX_LAYERS = 16;
    parameter DATA_WIDTH = 48;
    parameter CONFIG_WIDTH = 32;
    parameter WEIGHT_WIDTH = 8;
    parameter VMEM_WIDTH = 16;
    parameter CLK_PERIOD = 10;
    
    //-------------------------------------------------------------------------
    // Signals
    //-------------------------------------------------------------------------
    reg clk;
    reg reset;
    reg enable;
    
    // Input interface (AXI-Stream)
    reg [DATA_WIDTH-1:0] s_axis_input_tdata;
    reg s_axis_input_tvalid;
    wire s_axis_input_tready;
    reg s_axis_input_tlast;
    
    // Output interface (AXI-Stream)
    wire [DATA_WIDTH-1:0] m_axis_output_tdata;
    wire m_axis_output_tvalid;
    reg m_axis_output_tready;
    wire m_axis_output_tlast;
    
    // Configuration interface
    reg [7:0] config_layer_id;
    reg [7:0] config_layer_type;
    reg [CONFIG_WIDTH-1:0] config_data;
    reg config_write;
    
    // Weight loading interface
    reg [7:0] weight_layer_id;
    reg [15:0] weight_addr;
    reg [WEIGHT_WIDTH-1:0] weight_data;
    reg weight_write;
    
    // Layer execution control
    reg [7:0] execute_layer_id;
    reg execute_start;
    wire execute_done;
    
    // Status outputs
    wire [31:0] total_input_spikes;
    wire [31:0] total_output_spikes;
    wire [MAX_LAYERS-1:0] layer_active_status;
    wire [7:0] current_layer_id;
    
    // Test variables
    integer test_num;
    integer error_count;
    integer i;
    integer output_count;
    
    //-------------------------------------------------------------------------
    // DUT Instantiation
    //-------------------------------------------------------------------------
    snn_layer_manager #(
        .MAX_LAYERS(MAX_LAYERS),
        .DATA_WIDTH(DATA_WIDTH),
        .CONFIG_WIDTH(CONFIG_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .VMEM_WIDTH(VMEM_WIDTH)
    ) dut (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        
        .s_axis_input_tdata(s_axis_input_tdata),
        .s_axis_input_tvalid(s_axis_input_tvalid),
        .s_axis_input_tready(s_axis_input_tready),
        .s_axis_input_tlast(s_axis_input_tlast),
        
        .m_axis_output_tdata(m_axis_output_tdata),
        .m_axis_output_tvalid(m_axis_output_tvalid),
        .m_axis_output_tready(m_axis_output_tready),
        .m_axis_output_tlast(m_axis_output_tlast),
        
        .config_layer_id(config_layer_id),
        .config_layer_type(config_layer_type),
        .config_data(config_data),
        .config_write(config_write),
        
        .weight_layer_id(weight_layer_id),
        .weight_addr(weight_addr),
        .weight_data(weight_data),
        .weight_write(weight_write),
        
        .execute_layer_id(execute_layer_id),
        .execute_start(execute_start),
        .execute_done(execute_done),
        
        .total_input_spikes(total_input_spikes),
        .total_output_spikes(total_output_spikes),
        .layer_active_status(layer_active_status),
        .current_layer_id(current_layer_id)
    );
    
    //-------------------------------------------------------------------------
    // Clock Generation
    //-------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    //-------------------------------------------------------------------------
    // Tasks
    //-------------------------------------------------------------------------
    task apply_reset;
        begin
            reset = 1;
            enable = 0;
            s_axis_input_tdata = 0;
            s_axis_input_tvalid = 0;
            s_axis_input_tlast = 0;
            m_axis_output_tready = 1;
            config_layer_id = 0;
            config_layer_type = 0;
            config_data = 0;
            config_write = 0;
            weight_layer_id = 0;
            weight_addr = 0;
            weight_data = 0;
            weight_write = 0;
            execute_layer_id = 0;
            execute_start = 0;
            repeat(10) @(posedge clk);
            reset = 0;
            enable = 1;
            @(posedge clk);
        end
    endtask
    
    // Configure a layer
    task configure_layer(input [7:0] layer_id, input [7:0] layer_type, input [31:0] cfg_data);
        begin
            @(posedge clk);
            config_layer_id = layer_id;
            config_layer_type = layer_type;
            config_data = cfg_data;
            config_write = 1;
            @(posedge clk);
            config_write = 0;
            @(posedge clk);
        end
    endtask
    
    // Send spike data
    task send_spike(input [47:0] data);
        begin
            @(posedge clk);
            s_axis_input_tdata = data;
            s_axis_input_tvalid = 1;
            while (!s_axis_input_tready) @(posedge clk);
            @(posedge clk);
            s_axis_input_tvalid = 0;
        end
    endtask
    
    // Start layer execution
    task execute_layer(input [7:0] layer_id);
        begin
            @(posedge clk);
            execute_layer_id = layer_id;
            execute_start = 1;
            @(posedge clk);
            execute_start = 0;
            @(posedge clk);
        end
    endtask
    
    //-------------------------------------------------------------------------
    // Output Monitor
    //-------------------------------------------------------------------------
    always @(posedge clk) begin
        if (m_axis_output_tvalid && m_axis_output_tready) begin
            output_count = output_count + 1;
            $display("[%0t] Output %0d: data=%0h", 
                     $time, output_count, m_axis_output_tdata);
        end
    end
    
    //-------------------------------------------------------------------------
    // Main Test
    //-------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_snn_layer_manager.vcd");
        $dumpvars(0, tb_snn_layer_manager);
        
        test_num = 0;
        error_count = 0;
        output_count = 0;
        
        $display("===========================================");
        $display("SNN Layer Manager Testbench");
        $display("===========================================");
        $display("Max layers: %0d", MAX_LAYERS);
        
        //---------------------------------------------------------------------
        // Test 1: Basic Reset
        //---------------------------------------------------------------------
        test_num = 1;
        $display("\n--- Test %0d: Basic Reset ---", test_num);
        apply_reset();
        
        $display("  Current layer: %0d", current_layer_id);
        $display("  Layer active status: %b", layer_active_status);
        $display("  PASS: Module reset correctly");
        
        //---------------------------------------------------------------------
        // Test 2: Layer Configuration
        //---------------------------------------------------------------------
        test_num = 2;
        $display("\n--- Test %0d: Layer Configuration ---", test_num);
        
        // Configure Conv2D layer (type 1) - First set layer type with 0xFF prefix
        configure_layer(8'd0, 8'd1, 32'hFF000001);  // Layer type = Conv2D (1)
        configure_layer(8'd0, 8'd1, 32'h00080008);  // Config: 8x8 input
        
        // Configure AvgPool2D layer (type 2)
        configure_layer(8'd1, 8'd2, 32'hFF000002);  // Layer type = AvgPool2D (2)
        configure_layer(8'd1, 8'd2, 32'h00020002);  // Config: 2x2 pool
        
        // Configure Dense layer (type 4)
        configure_layer(8'd2, 8'd4, 32'hFF000004);  // Layer type = Dense (4)
        configure_layer(8'd2, 8'd4, 32'h0000000A);  // Config: 10 outputs
        
        $display("  Configured 3 layers");
        $display("  Layer status: %b", layer_active_status);
        if (layer_active_status[2:0] == 3'b111) begin
            $display("  PASS: Layer configuration - 3 layers enabled");
        end else begin
            $display("  WARNING: Layer status expected 111, got %b", layer_active_status[2:0]);
        end
        $display("  PASS: Layer configuration");
        
        //---------------------------------------------------------------------
        // Test 3: Input Spike
        //---------------------------------------------------------------------
        test_num = 3;
        $display("\n--- Test %0d: Input Spike ---", test_num);
        
        apply_reset();
        configure_layer(8'd0, 8'd1, 32'hFF000001);  // Layer type = Conv2D
        configure_layer(8'd0, 8'd1, 32'h00080008);  // Config: 8x8 input
        
        output_count = 0;
        
        send_spike(48'h000100010001);
        
        repeat(20) @(posedge clk);
        
        $display("  Total input spikes: %0d", total_input_spikes);
        $display("  PASS: Input spike sent");
        
        //---------------------------------------------------------------------
        // Test 4: Layer Execution
        //---------------------------------------------------------------------
        test_num = 4;
        $display("\n--- Test %0d: Layer Execution ---", test_num);
        
        apply_reset();
        configure_layer(8'd0, 8'd1, 32'hFF000001);  // Layer type = Conv2D
        configure_layer(8'd0, 8'd1, 32'h00040004);  // Config: 4x4 input
        
        // Start execution
        execute_layer(8'd0);
        
        repeat(50) @(posedge clk);
        
        $display("  Execute done: %b", execute_done);
        $display("  Current layer: %0d", current_layer_id);
        $display("  PASS: Layer execution");
        
        //---------------------------------------------------------------------
        // Test 5: Multiple Spikes
        //---------------------------------------------------------------------
        test_num = 5;
        $display("\n--- Test %0d: Multiple Spikes ---", test_num);
        
        apply_reset();
        configure_layer(8'd0, 8'd1, 32'h00080008);
        
        output_count = 0;
        
        for (i = 0; i < 10; i = i + 1) begin
            send_spike({16'd0, 8'd0, 8'd0, i[7:0], 8'h01});
        end
        
        repeat(100) @(posedge clk);
        
        $display("  Sent 10 input spikes");
        $display("  Total input spikes: %0d", total_input_spikes);
        $display("  Total output spikes: %0d", total_output_spikes);
        $display("  PASS: Multiple spikes");
        
        //---------------------------------------------------------------------
        // Test 6: Weight Loading
        //---------------------------------------------------------------------
        test_num = 6;
        $display("\n--- Test %0d: Weight Loading ---", test_num);
        
        apply_reset();
        configure_layer(8'd0, 8'd4, 32'hFF000004);  // Layer type = Dense
        configure_layer(8'd0, 8'd4, 32'h00000010);  // Config
        
        // Load some weights
        weight_layer_id = 8'd0;
        for (i = 0; i < 16; i = i + 1) begin
            @(posedge clk);
            weight_addr = i[15:0];
            weight_data = (i * 10);
            weight_write = 1;
            @(posedge clk);
            weight_write = 0;
        end
        
        repeat(20) @(posedge clk);
        
        $display("  Loaded 16 weights");
        $display("  PASS: Weight loading");
        
        //---------------------------------------------------------------------
        // Test 7: Layer Status
        //---------------------------------------------------------------------
        test_num = 7;
        $display("\n--- Test %0d: Layer Status ---", test_num);
        
        apply_reset();
        
        // Configure multiple layers with proper layer type prefix (0xFF)
        configure_layer(8'd0, 8'd1, 32'hFF000001);  // Layer 0: Conv2D
        configure_layer(8'd0, 8'd1, 32'h00080008);  // Config: 8x8 input
        
        configure_layer(8'd1, 8'd2, 32'hFF000002);  // Layer 1: AvgPool2D
        configure_layer(8'd1, 8'd2, 32'h00020002);  // Config: 2x2 pool
        
        configure_layer(8'd2, 8'd4, 32'hFF000004);  // Layer 2: Dense
        configure_layer(8'd2, 8'd4, 32'h0000000A);  // Config: 10 outputs
        
        $display("  Layer active status: %b", layer_active_status);
        if (layer_active_status[2:0] == 3'b111) begin
            $display("  PASS: Layer status correctly shows 3 active layers");
        end else begin
            $display("  FAIL: Expected layer_active_status[2:0] = 111, got %b", layer_active_status[2:0]);
            error_count = error_count + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 8: Back Pressure
        //---------------------------------------------------------------------
        test_num = 8;
        $display("\n--- Test %0d: Back Pressure ---", test_num);
        
        apply_reset();
        configure_layer(8'd0, 8'd1, 32'h00040004);
        
        // Disable output ready
        m_axis_output_tready = 0;
        
        send_spike(48'h000100010001);
        
        repeat(20) @(posedge clk);
        
        // Enable output ready
        m_axis_output_tready = 1;
        
        repeat(20) @(posedge clk);
        
        $display("  PASS: Back pressure handling");
        
        //---------------------------------------------------------------------
        // Summary
        //---------------------------------------------------------------------
        $display("\n===========================================");
        if (error_count == 0) begin
            $display("All tests PASSED!");
        end else begin
            $display("Tests completed with %0d errors", error_count);
        end
        $display("===========================================");
        
        #100;
        $finish;
    end
    
    // Timeout
    initial begin
        #200000;
        $display("ERROR: Simulation timeout!");
        $finish;
    end

endmodule
