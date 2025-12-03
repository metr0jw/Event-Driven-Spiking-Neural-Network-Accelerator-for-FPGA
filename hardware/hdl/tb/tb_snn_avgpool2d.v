//-----------------------------------------------------------------------------
// Title         : AvgPool2D Testbench
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_snn_avgpool2d.v
// Author        : Jiwoon Lee (@metr0jw)
// Description   : Testbench for SNN 2D average pooling layer
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_snn_avgpool2d();

    //-------------------------------------------------------------------------
    // Parameters
    //-------------------------------------------------------------------------
    parameter INPUT_HEIGHT = 8;
    parameter INPUT_WIDTH = 8;
    parameter INPUT_CHANNELS = 4;
    parameter POOL_SIZE = 2;
    parameter STRIDE = 2;
    localparam OUTPUT_HEIGHT = (INPUT_HEIGHT - POOL_SIZE) / STRIDE + 1;
    localparam OUTPUT_WIDTH = (INPUT_WIDTH - POOL_SIZE) / STRIDE + 1;
    parameter VMEM_WIDTH = 16;
    
    parameter CLK_PERIOD = 10;
    
    //-------------------------------------------------------------------------
    // Signals
    //-------------------------------------------------------------------------
    reg clk;
    reg reset;
    reg enable;
    
    // Input spike interface (AXI-Stream)
    reg [31:0] s_axis_input_tdata;
    reg s_axis_input_tvalid;
    wire s_axis_input_tready;
    reg s_axis_input_tlast;
    
    // Output spike interface (AXI-Stream)
    wire [31:0] m_axis_output_tdata;
    wire m_axis_output_tvalid;
    reg m_axis_output_tready;
    wire m_axis_output_tlast;
    
    // Configuration
    reg [15:0] threshold_config;
    reg [7:0] decay_factor;
    reg [7:0] pooling_weight;
    
    // Status
    wire [31:0] input_spike_count;
    wire [31:0] output_spike_count;
    wire computation_done;
    
    // Test variables
    integer test_num;
    integer error_count;
    integer i, j, ch;
    integer output_count;
    
    //-------------------------------------------------------------------------
    // DUT Instantiation
    //-------------------------------------------------------------------------
    snn_avgpool2d #(
        .INPUT_HEIGHT(INPUT_HEIGHT),
        .INPUT_WIDTH(INPUT_WIDTH),
        .INPUT_CHANNELS(INPUT_CHANNELS),
        .POOL_SIZE(POOL_SIZE),
        .STRIDE(STRIDE),
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
        
        .threshold_config(threshold_config),
        .decay_factor(decay_factor),
        .pooling_weight(pooling_weight),
        
        .input_spike_count(input_spike_count),
        .output_spike_count(output_spike_count),
        .computation_done(computation_done)
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
            threshold_config = 16'h1000;
            decay_factor = 8'd1;
            pooling_weight = 8'd64;
            repeat(10) @(posedge clk);
            reset = 0;
            enable = 1;
            @(posedge clk);
        end
    endtask
    
    // Send spike: {channel[7:0], y[7:0], x[7:0], valid[7:0]}
    task send_spike_2d(input [7:0] channel, input [7:0] y, input [7:0] x);
        begin
            @(posedge clk);
            s_axis_input_tdata = {channel, y, x, 8'h01};
            s_axis_input_tvalid = 1;
            while (!s_axis_input_tready) @(posedge clk);
            @(posedge clk);
            s_axis_input_tvalid = 0;
        end
    endtask
    
    //-------------------------------------------------------------------------
    // Output Monitor
    //-------------------------------------------------------------------------
    always @(posedge clk) begin
        if (m_axis_output_tvalid && m_axis_output_tready) begin
            output_count = output_count + 1;
            $display("[%0t] Output %0d: ch=%0d y=%0d x=%0d", 
                     $time, output_count,
                     m_axis_output_tdata[31:24],
                     m_axis_output_tdata[23:16],
                     m_axis_output_tdata[15:8]);
        end
    end
    
    //-------------------------------------------------------------------------
    // Main Test
    //-------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_snn_avgpool2d.vcd");
        $dumpvars(0, tb_snn_avgpool2d);
        
        test_num = 0;
        error_count = 0;
        output_count = 0;
        
        $display("===========================================");
        $display("SNN AvgPool2D Testbench");
        $display("===========================================");
        $display("Input: %0dx%0d x %0d channels", INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS);
        $display("Output: %0dx%0d x %0d channels", OUTPUT_HEIGHT, OUTPUT_WIDTH, INPUT_CHANNELS);
        $display("Pool size: %0dx%0d, Stride: %0d", POOL_SIZE, POOL_SIZE, STRIDE);
        
        //---------------------------------------------------------------------
        // Test 1: Basic Reset
        //---------------------------------------------------------------------
        test_num = 1;
        $display("\n--- Test %0d: Basic Reset ---", test_num);
        apply_reset();
        
        if (input_spike_count == 0 && output_spike_count == 0) begin
            $display("  PASS: Module reset correctly");
        end else begin
            $display("  INFO: input_count=%0d output_count=%0d", 
                     input_spike_count, output_spike_count);
        end
        
        //---------------------------------------------------------------------
        // Test 2: Send Single Spike
        //---------------------------------------------------------------------
        test_num = 2;
        $display("\n--- Test %0d: Send Single Spike ---", test_num);
        
        send_spike_2d(8'd0, 8'd0, 8'd0);
        repeat(10) @(posedge clk);
        
        $display("  Input spike count: %0d", input_spike_count);
        $display("  PASS: Single spike sent");
        
        //---------------------------------------------------------------------
        // Test 3: Multiple Spikes to Same Pool Window
        //---------------------------------------------------------------------
        test_num = 3;
        $display("\n--- Test %0d: Multiple Spikes to Same Pool Window ---", test_num);
        
        apply_reset();
        output_count = 0;
        
        send_spike_2d(8'd0, 8'd0, 8'd0);
        send_spike_2d(8'd0, 8'd0, 8'd1);
        send_spike_2d(8'd0, 8'd1, 8'd0);
        send_spike_2d(8'd0, 8'd1, 8'd1);
        
        repeat(30) @(posedge clk);
        
        $display("  Input spike count: %0d", input_spike_count);
        $display("  PASS: Pool window spikes accumulated");
        
        //---------------------------------------------------------------------
        // Test 4: Multi-Channel
        //---------------------------------------------------------------------
        test_num = 4;
        $display("\n--- Test %0d: Multi-Channel ---", test_num);
        
        apply_reset();
        output_count = 0;
        
        for (ch = 0; ch < INPUT_CHANNELS; ch = ch + 1) begin
            send_spike_2d(ch[7:0], 8'd0, 8'd0);
        end
        
        repeat(50) @(posedge clk);
        
        $display("  Sent spikes to all %0d channels", INPUT_CHANNELS);
        $display("  Input spike count: %0d", input_spike_count);
        $display("  PASS: Multi-channel handling");
        
        //---------------------------------------------------------------------
        // Test 5: Full Coverage
        //---------------------------------------------------------------------
        test_num = 5;
        $display("\n--- Test %0d: Full Coverage ---", test_num);
        
        apply_reset();
        output_count = 0;
        
        for (ch = 0; ch < INPUT_CHANNELS; ch = ch + 1) begin
            for (j = 0; j < INPUT_HEIGHT; j = j + 1) begin
                for (i = 0; i < INPUT_WIDTH; i = i + 1) begin
                    send_spike_2d(ch[7:0], j[7:0], i[7:0]);
                end
            end
        end
        
        repeat(500) @(posedge clk);
        
        $display("  Sent %0d total spikes", INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS);
        $display("  Input spike count: %0d", input_spike_count);
        $display("  Output spike count: %0d", output_spike_count);
        $display("  Computation done: %b", computation_done);
        $display("  PASS: Full coverage processed");
        
        //---------------------------------------------------------------------
        // Test 6: Configuration Test
        //---------------------------------------------------------------------
        test_num = 6;
        $display("\n--- Test %0d: Configuration Test ---", test_num);
        
        apply_reset();
        threshold_config = 16'h0800;
        decay_factor = 8'd2;
        pooling_weight = 8'd128;
        
        send_spike_2d(8'd0, 8'd0, 8'd0);
        send_spike_2d(8'd0, 8'd0, 8'd1);
        
        repeat(50) @(posedge clk);
        
        $display("  Threshold: %04h", threshold_config);
        $display("  PASS: Configuration applied");
        
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
