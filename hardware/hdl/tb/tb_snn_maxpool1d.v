//-----------------------------------------------------------------------------
// Title         : SNN 1D Max Pooling Testbench
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_snn_maxpool1d.v
// Author        : Jiwoon Lee (@metr0jw)
// Description   : Testbench for SNN 1D max pooling layer
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_snn_maxpool1d;

    // Parameters
    parameter INPUT_LENGTH = 16;
    parameter INPUT_CHANNELS = 4;
    parameter POOL_SIZE = 2;
    parameter STRIDE = 2;
    parameter OUTPUT_LENGTH = INPUT_LENGTH / STRIDE;
    parameter VMEM_WIDTH = 16;
    parameter TIMESTAMP_WIDTH = 16;
    
    // Clock and reset
    reg clk;
    reg rst_n;
    reg enable;
    
    // Input interface
    reg s_axis_input_tvalid;
    reg [31:0] s_axis_input_tdata;
    reg s_axis_input_tlast;
    wire s_axis_input_tready;
    
    // Output interface
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
    integer i;
    integer error_count;
    integer test_num;
    integer output_spike_count;
    reg [255:0] test_name;
    
    // DUT instantiation
    snn_maxpool1d #(
        .INPUT_LENGTH(INPUT_LENGTH),
        .INPUT_CHANNELS(INPUT_CHANNELS),
        .POOL_SIZE(POOL_SIZE),
        .STRIDE(STRIDE),
        .OUTPUT_LENGTH(OUTPUT_LENGTH),
        .VMEM_WIDTH(VMEM_WIDTH),
        .TIMESTAMP_WIDTH(TIMESTAMP_WIDTH)
    ) DUT (
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
    
    // Clock generation
    always #5 clk = ~clk;
    
    // Task: Initialize test
    task init_test;
        input [255:0] name;
        begin
            test_name = name;
            test_num = test_num + 1;
            $display("\n========================================");
            $display("Test %0d: %0s", test_num, test_name);
            $display("========================================");
        end
    endtask
    
    // Task: Apply reset
    task apply_reset;
        begin
            rst_n = 1'b0;
            repeat(5) @(posedge clk);
            rst_n = 1'b1;
            @(posedge clk);
        end
    endtask
    
    // Task: Send spike
    task send_spike;
        input [7:0] channel;
        input [15:0] position;
        input [15:0] timestamp;
        input is_last;
        begin
            wait(s_axis_input_tready);
            @(posedge clk);
            s_axis_input_tdata = {position, channel, timestamp[7:0], 8'h01};
            s_axis_input_tvalid = 1'b1;
            s_axis_input_tlast = is_last;
            @(posedge clk);
            s_axis_input_tvalid = 1'b0;
            s_axis_input_tlast = 1'b0;
        end
    endtask
    
    // Main test
    initial begin
        // Initialize
        clk = 0;
        rst_n = 0;
        enable = 0;
        s_axis_input_tvalid = 0;
        s_axis_input_tdata = 0;
        s_axis_input_tlast = 0;
        m_axis_output_tready = 1;
        config_valid = 0;
        config_data = 0;
        error_count = 0;
        test_num = 0;
        output_spike_count = 0;
        
        $display("==============================================");
        $display("  SNN 1D Max Pooling Testbench");
        $display("==============================================");
        $display("  INPUT_LENGTH:    %0d", INPUT_LENGTH);
        $display("  INPUT_CHANNELS:  %0d", INPUT_CHANNELS);
        $display("  POOL_SIZE:       %0d", POOL_SIZE);
        $display("  STRIDE:          %0d", STRIDE);
        $display("  OUTPUT_LENGTH:   %0d", OUTPUT_LENGTH);
        $display("==============================================");
        
        // Reset
        apply_reset();
        #100;
        enable = 1;
        
        //---------------------------------------------------------------------
        // Test 1: Basic Pooling Operation
        //---------------------------------------------------------------------
        init_test("Basic Pooling Operation");
        
        // Start pooling
        @(posedge clk);
        config_valid = 1;
        config_data = 32'h0001;
        @(posedge clk);
        config_valid = 0;
        
        // Wait for ready
        wait(s_axis_input_tready);
        
        // Send spikes at different positions
        // Channel 0: spikes at positions 0, 2, 4
        send_spike(8'd0, 16'd0, 16'd100, 1'b0);
        send_spike(8'd0, 16'd2, 16'd150, 1'b0);
        send_spike(8'd0, 16'd4, 16'd200, 1'b0);
        
        // Channel 1: spikes at positions 1, 3, 5
        send_spike(8'd1, 16'd1, 16'd120, 1'b0);
        send_spike(8'd1, 16'd3, 16'd180, 1'b0);
        send_spike(8'd1, 16'd5, 16'd220, 1'b1);  // Last spike
        
        // Wait for processing
        wait(layer_done || $time > 50000);
        
        if (layer_done) begin
            $display("  PASS: Basic pooling completed");
        end else begin
            $display("  ERROR: Timeout waiting for layer_done");
            error_count = error_count + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 2: Dense Spike Pattern
        //---------------------------------------------------------------------
        init_test("Dense Spike Pattern");
        apply_reset();
        enable = 1;
        
        @(posedge clk);
        config_valid = 1;
        config_data = 32'h0001;
        @(posedge clk);
        config_valid = 0;
        
        wait(s_axis_input_tready);
        
        // Send spikes to all positions for channel 0
        for (i = 0; i < INPUT_LENGTH - 1; i = i + 1) begin
            send_spike(8'd0, i[15:0], (i * 10 + 100), 1'b0);
        end
        send_spike(8'd0, (INPUT_LENGTH - 1), 16'd255, 1'b1);
        
        wait(layer_done || $time > 100000);
        
        if (layer_done) begin
            $display("  PASS: Dense spike pattern processed");
        end else begin
            $display("  ERROR: Timeout processing dense pattern");
            error_count = error_count + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 3: Multi-channel Processing
        //---------------------------------------------------------------------
        init_test("Multi-channel Processing");
        apply_reset();
        enable = 1;
        
        @(posedge clk);
        config_valid = 1;
        config_data = 32'h0001;
        @(posedge clk);
        config_valid = 0;
        
        wait(s_axis_input_tready);
        
        // Send spikes to multiple channels
        for (i = 0; i < INPUT_CHANNELS; i = i + 1) begin
            send_spike(i[7:0], 16'd0, (i * 20 + 50), 1'b0);
            send_spike(i[7:0], 16'd1, (i * 20 + 60), 1'b0);
        end
        // Mark last
        send_spike(8'd0, 16'd2, 16'd100, 1'b1);
        
        wait(layer_done || $time > 100000);
        
        if (layer_done) begin
            $display("  PASS: Multi-channel processing completed");
        end else begin
            $display("  ERROR: Timeout in multi-channel processing");
            error_count = error_count + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 4: Output Validation
        //---------------------------------------------------------------------
        init_test("Output Spike Monitoring");
        apply_reset();
        enable = 1;
        output_spike_count = 0;
        
        @(posedge clk);
        config_valid = 1;
        config_data = 32'h0001;
        @(posedge clk);
        config_valid = 0;
        
        wait(s_axis_input_tready);
        
        // Simple pattern: spikes at positions 0, 1 (same pool window)
        send_spike(8'd0, 16'd0, 16'd100, 1'b0);
        send_spike(8'd0, 16'd1, 16'd200, 1'b1);  // Higher timestamp wins
        
        // Monitor outputs
        fork
            begin
                wait(layer_done || $time > 100000);
            end
            begin
                while (!layer_done && $time < 100000) begin
                    @(posedge clk);
                    if (m_axis_output_tvalid && m_axis_output_tready) begin
                        output_spike_count = output_spike_count + 1;
                        $display("  Output spike %0d: data=0x%08x", output_spike_count, m_axis_output_tdata);
                    end
                end
            end
        join
        
        $display("  Total output spikes: %0d", output_spike_count);
        if (layer_done) begin
            $display("  PASS: Output monitoring completed");
        end else begin
            $display("  ERROR: Timeout");
            error_count = error_count + 1;
        end
        
        //---------------------------------------------------------------------
        // Test Summary
        //---------------------------------------------------------------------
        $display("\n========================================");
        $display("Test Summary");
        $display("========================================");
        $display("Total Tests Run: %0d", test_num);
        $display("Total Errors: %0d", error_count);
        
        if (error_count == 0) begin
            $display("All tests PASSED!");
        end else begin
            $display("Tests FAILED with %0d errors!", error_count);
        end
        $display("========================================\n");
        
        $finish;
    end
    
    // Timeout protection
    initial begin
        #500000;
        $display("ERROR: Global timeout reached");
        $finish;
    end

endmodule
