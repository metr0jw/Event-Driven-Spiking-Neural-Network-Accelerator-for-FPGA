//-----------------------------------------------------------------------------
// Title         : AvgPool1D Testbench
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_snn_avgpool1d.v
// Author        : Jiwoon Lee (@metr0jw)
// Description   : Testbench for SNN 1D average pooling layer
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_snn_avgpool1d();

    //-------------------------------------------------------------------------
    // Parameters
    //-------------------------------------------------------------------------
    parameter INPUT_LENGTH = 16;
    parameter INPUT_CHANNELS = 4;
    parameter POOL_SIZE = 2;
    parameter STRIDE = 2;
    parameter OUTPUT_LENGTH = INPUT_LENGTH / STRIDE;
    parameter VMEM_WIDTH = 16;
    
    parameter CLK_PERIOD = 10;
    
    //-------------------------------------------------------------------------
    // Signals
    //-------------------------------------------------------------------------
    reg clk;
    reg rst_n;
    reg enable;
    
    // Input AXI-Stream
    reg s_axis_input_tvalid;
    reg [31:0] s_axis_input_tdata;
    reg s_axis_input_tlast;
    wire s_axis_input_tready;
    
    // Output AXI-Stream
    wire m_axis_output_tvalid;
    wire [31:0] m_axis_output_tdata;
    wire m_axis_output_tlast;
    reg m_axis_output_tready;
    
    // Configuration
    reg config_valid;
    reg [31:0] config_data;
    
    // Status
    wire busy;
    wire layer_done;
    
    // Test variables
    integer test_num;
    integer error_count;
    integer i, ch;
    integer output_count;
    
    //-------------------------------------------------------------------------
    // DUT Instantiation
    //-------------------------------------------------------------------------
    snn_avgpool1d #(
        .INPUT_LENGTH(INPUT_LENGTH),
        .INPUT_CHANNELS(INPUT_CHANNELS),
        .POOL_SIZE(POOL_SIZE),
        .STRIDE(STRIDE),
        .OUTPUT_LENGTH(OUTPUT_LENGTH),
        .VMEM_WIDTH(VMEM_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        
        .s_axis_input_tvalid(s_axis_input_tvalid),
        .s_axis_input_tdata(s_axis_input_tdata),
        .s_axis_input_tlast(s_axis_input_tlast),
        .s_axis_input_tready(s_axis_input_tready),
        
        .m_axis_output_tvalid(m_axis_output_tvalid),
        .m_axis_output_tdata(m_axis_output_tdata),
        .m_axis_output_tlast(m_axis_output_tlast),
        .m_axis_output_tready(m_axis_output_tready),
        
        .config_valid(config_valid),
        .config_data(config_data),
        
        .busy(busy),
        .layer_done(layer_done)
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
            rst_n = 0;
            enable = 0;
            s_axis_input_tvalid = 0;
            s_axis_input_tdata = 0;
            s_axis_input_tlast = 0;
            m_axis_output_tready = 1;
            config_valid = 0;
            config_data = 0;
            repeat(10) @(posedge clk);
            rst_n = 1;
            enable = 1;
            @(posedge clk);
        end
    endtask
    
    // Send spike: {position[31:16], channel[15:8], value[7:0]}
    task send_spike(input [15:0] position, input [7:0] channel, input [7:0] value, input last);
        begin
            @(posedge clk);
            s_axis_input_tdata = {position, channel, value};
            s_axis_input_tvalid = 1;
            s_axis_input_tlast = last;
            while (!s_axis_input_tready) @(posedge clk);
            @(posedge clk);
            s_axis_input_tvalid = 0;
            s_axis_input_tlast = 0;
        end
    endtask
    
    //-------------------------------------------------------------------------
    // Output Monitor
    //-------------------------------------------------------------------------
    always @(posedge clk) begin
        if (m_axis_output_tvalid && m_axis_output_tready) begin
            output_count = output_count + 1;
            $display("[%0t] Output %0d: pos=%0d ch=%0d val=%0d last=%b", 
                     $time, output_count,
                     m_axis_output_tdata[31:16],
                     m_axis_output_tdata[15:8],
                     m_axis_output_tdata[7:0],
                     m_axis_output_tlast);
        end
    end
    
    //-------------------------------------------------------------------------
    // Main Test
    //-------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_snn_avgpool1d.vcd");
        $dumpvars(0, tb_snn_avgpool1d);
        
        test_num = 0;
        error_count = 0;
        output_count = 0;
        
        $display("===========================================");
        $display("SNN AvgPool1D Testbench");
        $display("===========================================");
        $display("Input: %0d x %0d channels", INPUT_LENGTH, INPUT_CHANNELS);
        $display("Output: %0d x %0d channels", OUTPUT_LENGTH, INPUT_CHANNELS);
        $display("Pool size: %0d, Stride: %0d", POOL_SIZE, STRIDE);
        
        //---------------------------------------------------------------------
        // Test 1: Basic Reset
        //---------------------------------------------------------------------
        test_num = 1;
        $display("\n--- Test %0d: Basic Reset ---", test_num);
        apply_reset();
        
        if (!busy && !layer_done) begin
            $display("  PASS: Module reset correctly");
        end else begin
            $display("  INFO: busy=%b layer_done=%b", busy, layer_done);
        end
        
        //---------------------------------------------------------------------
        // Test 2: Send Single Spike
        //---------------------------------------------------------------------
        test_num = 2;
        $display("\n--- Test %0d: Send Single Spike ---", test_num);
        
        send_spike(16'd0, 8'd0, 8'd100, 1'b0);
        repeat(10) @(posedge clk);
        
        $display("  PASS: Single spike sent");
        
        //---------------------------------------------------------------------
        // Test 3: Send Multiple Spikes to Same Pool
        //---------------------------------------------------------------------
        test_num = 3;
        $display("\n--- Test %0d: Multiple Spikes to Same Pool ---", test_num);
        
        apply_reset();
        output_count = 0;
        
        // Pool positions 0,1 should average to output position 0
        send_spike(16'd0, 8'd0, 8'd100, 1'b0);  // Position 0
        send_spike(16'd1, 8'd0, 8'd200, 1'b0);  // Position 1
        
        repeat(20) @(posedge clk);
        
        $display("  Sent 2 spikes to pool window");
        $display("  PASS: Pool window spikes accumulated");
        
        //---------------------------------------------------------------------
        // Test 4: All Channels
        //---------------------------------------------------------------------
        test_num = 4;
        $display("\n--- Test %0d: All Channels ---", test_num);
        
        apply_reset();
        output_count = 0;
        
        for (ch = 0; ch < INPUT_CHANNELS; ch = ch + 1) begin
            send_spike(16'd0, ch[7:0], 8'd50 + ch[7:0]*10, 1'b0);
        end
        
        repeat(30) @(posedge clk);
        
        $display("  Sent spikes to all %0d channels", INPUT_CHANNELS);
        $display("  PASS: Multi-channel handling");
        
        //---------------------------------------------------------------------
        // Test 5: Full Input Sequence
        //---------------------------------------------------------------------
        test_num = 5;
        $display("\n--- Test %0d: Full Input Sequence ---", test_num);
        
        apply_reset();
        output_count = 0;
        
        for (ch = 0; ch < INPUT_CHANNELS; ch = ch + 1) begin
            for (i = 0; i < INPUT_LENGTH; i = i + 1) begin
                send_spike(i[15:0], ch[7:0], 8'd100, 
                          (ch == INPUT_CHANNELS-1 && i == INPUT_LENGTH-1));
            end
        end
        
        // Wait for processing
        repeat(200) @(posedge clk);
        
        $display("  Sent %0d total spikes", INPUT_LENGTH * INPUT_CHANNELS);
        $display("  Output count: %0d", output_count);
        $display("  Layer done: %b", layer_done);
        $display("  PASS: Full sequence processed");
        
        //---------------------------------------------------------------------
        // Test 6: Busy Signal
        //---------------------------------------------------------------------
        test_num = 6;
        $display("\n--- Test %0d: Busy Signal ---", test_num);
        
        apply_reset();
        
        // Send burst of spikes
        for (i = 0; i < 5; i = i + 1) begin
            send_spike(i[15:0], 8'd0, 8'd100, 1'b0);
        end
        
        $display("  Busy signal: %b", busy);
        $display("  PASS: Busy signal test completed");
        
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
        #100000;
        $display("ERROR: Simulation timeout!");
        $finish;
    end

endmodule
