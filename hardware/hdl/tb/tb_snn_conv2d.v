//-----------------------------------------------------------------------------
// Title         : SNN 2D Convolution Testbench (AC Version)
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_snn_conv2d.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Description   : Testbench for AC-based SNN 2D convolution layer
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_snn_conv2d;

    //=========================================================================
    // Parameters
    //=========================================================================
    parameter INPUT_HEIGHT    = 8;
    parameter INPUT_WIDTH     = 8;
    parameter INPUT_CHANNELS  = 1;
    parameter OUTPUT_CHANNELS = 4;
    parameter KERNEL_SIZE     = 3;
    parameter STRIDE          = 1;
    parameter PADDING         = 1;
    parameter OUTPUT_HEIGHT   = (INPUT_HEIGHT + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;
    parameter OUTPUT_WIDTH    = (INPUT_WIDTH + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;
    
    parameter WEIGHT_WIDTH    = 8;
    parameter VMEM_WIDTH      = 16;
    parameter THRESHOLD       = 16'h0080;  // Lower threshold for testing
    parameter LEAK_SHIFT      = 4;
    
    parameter CLK_PERIOD      = 10;
    
    //=========================================================================
    // Signals
    //=========================================================================
    reg clk;
    reg rst_n;
    reg enable;
    
    // Input spike interface
    reg s_axis_spike_tvalid;
    reg [31:0] s_axis_spike_tdata;
    reg s_axis_spike_tlast;
    wire s_axis_spike_tready;
    
    // Output spike interface
    wire m_axis_spike_tvalid;
    wire [31:0] m_axis_spike_tdata;
    wire m_axis_spike_tlast;
    reg m_axis_spike_tready;
    
    // Weight interface
    wire weight_rd_en;
    wire [15:0] weight_addr;
    reg signed [WEIGHT_WIDTH-1:0] weight_data;
    reg weight_valid;
    
    // Configuration
    reg [VMEM_WIDTH-1:0] config_threshold;
    reg config_valid;
    
    // Statistics
    wire [31:0] input_spike_count;
    wire [31:0] output_spike_count;
    wire [31:0] ac_operation_count;
    wire [31:0] memory_access_count;
    wire [31:0] cycle_count;
    wire busy;
    
    // Weight memory model
    reg signed [WEIGHT_WIDTH-1:0] weight_memory [0:65535];
    
    // Test variables
    integer i, j, k;
    integer test_num;
    integer error_count;
    integer output_count;
    
    //=========================================================================
    // DUT Instantiation
    //=========================================================================
    snn_conv2d #(
        .INPUT_HEIGHT(INPUT_HEIGHT),
        .INPUT_WIDTH(INPUT_WIDTH),
        .INPUT_CHANNELS(INPUT_CHANNELS),
        .OUTPUT_CHANNELS(OUTPUT_CHANNELS),
        .KERNEL_SIZE(KERNEL_SIZE),
        .STRIDE(STRIDE),
        .PADDING(PADDING),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .VMEM_WIDTH(VMEM_WIDTH),
        .THRESHOLD(THRESHOLD),
        .LEAK_SHIFT(LEAK_SHIFT)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        
        .s_axis_spike_tvalid(s_axis_spike_tvalid),
        .s_axis_spike_tdata(s_axis_spike_tdata),
        .s_axis_spike_tlast(s_axis_spike_tlast),
        .s_axis_spike_tready(s_axis_spike_tready),
        
        .m_axis_spike_tvalid(m_axis_spike_tvalid),
        .m_axis_spike_tdata(m_axis_spike_tdata),
        .m_axis_spike_tlast(m_axis_spike_tlast),
        .m_axis_spike_tready(m_axis_spike_tready),
        
        .weight_rd_en(weight_rd_en),
        .weight_addr(weight_addr),
        .weight_data(weight_data),
        .weight_valid(weight_valid),
        
        .config_threshold(config_threshold),
        .config_valid(config_valid),
        
        .input_spike_count(input_spike_count),
        .output_spike_count(output_spike_count),
        .ac_operation_count(ac_operation_count),
        .memory_access_count(memory_access_count),
        .cycle_count(cycle_count),
        .busy(busy)
    );
    
    //=========================================================================
    // Clock Generation
    //=========================================================================
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    //=========================================================================
    // Weight Memory Model
    //=========================================================================
    always @(posedge clk) begin
        if (weight_rd_en) begin
            weight_data <= weight_memory[weight_addr];
            weight_valid <= 1'b1;
        end else begin
            weight_valid <= 1'b0;
        end
    end
    
    //=========================================================================
    // Tasks
    //=========================================================================
    task apply_reset;
        begin
            rst_n = 0;
            enable = 0;
            s_axis_spike_tvalid = 0;
            s_axis_spike_tdata = 0;
            s_axis_spike_tlast = 0;
            m_axis_spike_tready = 1;
            config_threshold = THRESHOLD;
            config_valid = 0;
            repeat(10) @(posedge clk);
            rst_n = 1;
            enable = 1;
            @(posedge clk);
        end
    endtask
    
    task init_weights;
        integer idx;
        begin
            // Initialize weights with small positive values
            for (idx = 0; idx < 65536; idx = idx + 1) begin
                weight_memory[idx] = (idx % 8) + 1;  // Values 1-8
            end
        end
    endtask
    
    // Send spike: {channel[7:0], row[7:0], col[7:0], timestamp[7:0]}
    task send_spike(input [7:0] channel, input [7:0] row, input [7:0] col, 
                    input [7:0] timestamp, input last);
        begin
            @(posedge clk);
            s_axis_spike_tdata = {channel, row, col, timestamp};
            s_axis_spike_tvalid = 1;
            s_axis_spike_tlast = last;
            while (!s_axis_spike_tready) @(posedge clk);
            @(posedge clk);
            s_axis_spike_tvalid = 0;
            s_axis_spike_tlast = 0;
        end
    endtask
    
    task wait_idle;
        begin
            while (busy) @(posedge clk);
            repeat(5) @(posedge clk);
        end
    endtask
    
    //=========================================================================
    // Output Monitor
    //=========================================================================
    always @(posedge clk) begin
        if (m_axis_spike_tvalid && m_axis_spike_tready) begin
            output_count = output_count + 1;
            $display("[%0t] Output spike %0d: ch=%0d, row=%0d, col=%0d, ts=%0d", 
                     $time, output_count,
                     m_axis_spike_tdata[31:24],
                     m_axis_spike_tdata[23:16],
                     m_axis_spike_tdata[15:8],
                     m_axis_spike_tdata[7:0]);
        end
    end
    
    //=========================================================================
    // Main Test
    //=========================================================================
    initial begin
        $dumpfile("tb_snn_conv2d.vcd");
        $dumpvars(0, tb_snn_conv2d);
        
        test_num = 0;
        error_count = 0;
        output_count = 0;
        
        $display("===========================================");
        $display("SNN Conv2D (AC) Testbench");
        $display("===========================================");
        $display("Input: %0dx%0dx%0d", INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS);
        $display("Output: %0dx%0dx%0d", OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_CHANNELS);
        $display("Kernel: %0dx%0d, Stride: %0d, Padding: %0d", 
                 KERNEL_SIZE, KERNEL_SIZE, STRIDE, PADDING);
        $display("Threshold: 0x%04h", THRESHOLD);
        
        // Initialize weights
        init_weights();
        
        //---------------------------------------------------------------------
        // Test 1: Basic Reset
        //---------------------------------------------------------------------
        test_num = 1;
        $display("\n--- Test %0d: Basic Reset ---", test_num);
        apply_reset();
        
        if (!busy && input_spike_count == 0) begin
            $display("  PASS: Reset successful");
        end else begin
            $display("  FAIL: Reset state incorrect");
            error_count = error_count + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 2: Single Spike at Center
        //---------------------------------------------------------------------
        test_num = 2;
        $display("\n--- Test %0d: Single Spike at Center ---", test_num);
        apply_reset();
        output_count = 0;
        
        // Send spike at (row=4, col=4, channel=0)
        send_spike(8'd0, 8'd4, 8'd4, 8'd100, 1'b1);
        wait_idle();
        
        $display("  Input spikes: %0d", input_spike_count);
        $display("  AC operations: %0d", ac_operation_count);
        $display("  Memory accesses: %0d", memory_access_count);
        $display("  Expected AC ops: %0d (K×K×C_out = %0d×%0d×%0d)", 
                 KERNEL_SIZE*KERNEL_SIZE*OUTPUT_CHANNELS,
                 KERNEL_SIZE, KERNEL_SIZE, OUTPUT_CHANNELS);
        
        if (input_spike_count == 1) begin
            $display("  PASS: Single spike processed");
        end else begin
            $display("  FAIL: Spike count mismatch");
            error_count = error_count + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 3: Corner Spike (Tests Padding)
        //---------------------------------------------------------------------
        test_num = 3;
        $display("\n--- Test %0d: Corner Spike (Padding Test) ---", test_num);
        apply_reset();
        output_count = 0;
        
        // Send spike at corner (0,0)
        send_spike(8'd0, 8'd0, 8'd0, 8'd50, 1'b1);
        wait_idle();
        
        $display("  Input spikes: %0d", input_spike_count);
        $display("  AC operations: %0d", ac_operation_count);
        $display("  Note: Corner spikes affect fewer output neurons");
        
        if (input_spike_count == 1) begin
            $display("  PASS: Corner spike processed");
        end else begin
            $display("  FAIL: Spike count mismatch");
            error_count = error_count + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 4: Multiple Spikes (Sparse Pattern)
        //---------------------------------------------------------------------
        test_num = 4;
        $display("\n--- Test %0d: Sparse Spike Pattern ---", test_num);
        apply_reset();
        output_count = 0;
        
        // Send 5 spikes at different positions
        send_spike(8'd0, 8'd1, 8'd1, 8'd10, 1'b0);
        send_spike(8'd0, 8'd3, 8'd3, 8'd20, 1'b0);
        send_spike(8'd0, 8'd5, 8'd5, 8'd30, 1'b0);
        send_spike(8'd0, 8'd2, 8'd6, 8'd40, 1'b0);
        send_spike(8'd0, 8'd6, 8'd2, 8'd50, 1'b1);
        
        wait_idle();
        
        $display("  Input spikes: %0d", input_spike_count);
        $display("  Output spikes: %0d", output_spike_count);
        $display("  AC operations: %0d", ac_operation_count);
        $display("  Ops per input spike: %0d", 
                 input_spike_count > 0 ? ac_operation_count / input_spike_count : 0);
        
        if (input_spike_count == 5) begin
            $display("  PASS: Multiple spikes processed");
        end else begin
            $display("  FAIL: Spike count mismatch");
            error_count = error_count + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 5: Dense Spike Pattern (Stress Test)
        //---------------------------------------------------------------------
        test_num = 5;
        $display("\n--- Test %0d: Dense Spike Pattern ---", test_num);
        apply_reset();
        output_count = 0;
        
        // Send spikes in a grid pattern
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                send_spike(8'd0, (i*2), (j*2), (i*4+j), (i == 3 && j == 3) ? 1 : 0);
            end
        end
        
        wait_idle();
        
        $display("  Input spikes: %0d", input_spike_count);
        $display("  Output spikes: %0d", output_spike_count);
        $display("  AC operations: %0d", ac_operation_count);
        $display("  Cycles: %0d", cycle_count);
        
        if (input_spike_count == 16) begin
            $display("  PASS: Dense pattern processed");
        end else begin
            $display("  FAIL: Spike count mismatch (expected 16, got %0d)", input_spike_count);
            error_count = error_count + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 6: Energy Efficiency Analysis
        //---------------------------------------------------------------------
        test_num = 6;
        $display("\n--- Test %0d: Energy Efficiency Analysis ---", test_num);
        apply_reset();
        output_count = 0;
        
        // Send 10 random spikes
        for (i = 0; i < 10; i = i + 1) begin
            send_spike(8'd0, (i*3) % INPUT_HEIGHT, (i*5) % INPUT_WIDTH, 
                      i[7:0]*10, (i == 9));
        end
        
        wait_idle();
        
        $display("  Input spikes: %0d", input_spike_count);
        $display("  Output spikes: %0d", output_spike_count);
        $display("  AC operations: %0d", ac_operation_count);
        $display("  Memory accesses: %0d", memory_access_count);
        $display("  Total cycles: %0d", cycle_count);
        
        // Calculate energy savings estimate
        $display("\n  Energy Analysis:");
        $display("  - Total neurons: %0d", OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNELS);
        $display("  - Updated neurons: ~%0d (sparse updates)", 
                 input_spike_count * KERNEL_SIZE * KERNEL_SIZE);
        $display("  - Sparsity benefit: Only updating affected receptive fields");
        $display("  - AC vs MAC: ~3-5x energy per operation");
        
        $display("  PASS: Energy analysis complete");
        
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
    
    // Timeout watchdog
    initial begin
        #2000000;
        $display("ERROR: Simulation timeout!");
        $finish;
    end

endmodule
