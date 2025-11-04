//-----------------------------------------------------------------------------
// Title         : SNN 1D Convolution Testbench
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_snn_conv1d.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Testbench for SNN 1D convolution layer
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_snn_conv1d;

    // Parameters
    parameter INPUT_LENGTH   = 100;
    parameter INPUT_CHANNELS = 4;
    parameter OUTPUT_CHANNELS = 8;
    parameter KERNEL_SIZE    = 3;
    parameter STRIDE         = 1;
    parameter PADDING        = 1;
    parameter WEIGHT_WIDTH   = 8;
    parameter VMEM_WIDTH     = 16;
    parameter THRESHOLD      = 16'h4000;  // 0.25 in Q8.8
    parameter DECAY_FACTOR   = 8'hE6;     // 0.9 * 256
    
    // Clock and reset
    reg clk;
    reg reset;
    reg enable;
    
    // Input spike interface
    reg [31:0] s_axis_input_tdata;
    reg s_axis_input_tvalid;
    wire s_axis_input_tready;
    reg s_axis_input_tlast;
    
    // Output spike interface
    wire [31:0] m_axis_output_tdata;
    wire m_axis_output_tvalid;
    reg m_axis_output_tready;
    wire m_axis_output_tlast;
    
    // Weight memory interface
    reg [WEIGHT_WIDTH-1:0] weight_data;
    wire [15:0] weight_addr;
    wire weight_read_en;
    
    // Configuration
    reg [15:0] threshold_config;
    reg [7:0] decay_config;
    reg learning_enable;
    
    // Status
    wire [31:0] input_spike_count;
    wire [31:0] output_spike_count;
    wire computation_done;
    wire [31:0] cycle_count;
    
    // Weight memory model
    reg signed [WEIGHT_WIDTH-1:0] weight_memory [0:8191];  // 8K weights
    
    // Test data
    reg [15:0] test_input_positions [0:99];
    reg [15:0] test_input_channels [0:99];
    integer test_spike_count;
    
    // Instantiate DUT
    snn_conv1d #(
        .INPUT_LENGTH(INPUT_LENGTH),
        .INPUT_CHANNELS(INPUT_CHANNELS),
        .OUTPUT_CHANNELS(OUTPUT_CHANNELS),
        .KERNEL_SIZE(KERNEL_SIZE),
        .STRIDE(STRIDE),
        .PADDING(PADDING),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .VMEM_WIDTH(VMEM_WIDTH),
        .THRESHOLD(THRESHOLD),
        .DECAY_FACTOR(DECAY_FACTOR)
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
        .weight_data(weight_data),
        .weight_addr(weight_addr),
        .weight_read_en(weight_read_en),
        .threshold_config(threshold_config),
        .decay_config(decay_config),
        .learning_enable(learning_enable),
        .input_spike_count(input_spike_count),
        .output_spike_count(output_spike_count),
        .computation_done(computation_done),
        .cycle_count(cycle_count)
    );
    
    // Clock generation
    always #5 clk = ~clk;
    
    // Weight memory interface
    always @(posedge clk) begin
        if (weight_read_en) begin
            weight_data <= weight_memory[weight_addr];
        end
    end
    
    // Test stimulus
    initial begin
        // Initialize signals
        clk = 0;
        reset = 1;
        enable = 0;
        s_axis_input_tdata = 32'b0;
        s_axis_input_tvalid = 0;
        s_axis_input_tlast = 0;
        m_axis_output_tready = 1;
        threshold_config = THRESHOLD;
        decay_config = DECAY_FACTOR;
        learning_enable = 0;
        test_spike_count = 0;
        
        // Initialize weight memory with test pattern
        integer i;
        for (i = 0; i < 8192; i = i + 1) begin
            weight_memory[i] = $signed($random % 127);  // Random weights -127 to 126
        end
        
        // Initialize test input data
        for (i = 0; i < 100; i = i + 1) begin
            test_input_positions[i] = $urandom % INPUT_LENGTH;
            test_input_channels[i] = $urandom % INPUT_CHANNELS;
        end
        
        $display("=== SNN 1D Convolution Testbench ===");
        $display("Input Length: %d", INPUT_LENGTH);
        $display("Input Channels: %d", INPUT_CHANNELS);
        $display("Output Channels: %d", OUTPUT_CHANNELS);
        $display("Kernel Size: %d", KERNEL_SIZE);
        $display("Stride: %d", STRIDE);
        $display("Padding: %d", PADDING);
        $display("=====================================");
        
        // Reset sequence
        #100;
        reset = 0;
        #20;
        enable = 1;
        
        // Test 1: Basic spike processing
        $display("\nTest 1: Basic spike processing");
        fork
            // Input spike generation
            begin
                for (i = 0; i < 50; i = i + 1) begin
                    @(posedge clk);
                    if (s_axis_input_tready) begin
                        s_axis_input_tdata = {test_input_channels[i], test_input_positions[i]};
                        s_axis_input_tvalid = 1;
                        test_spike_count = test_spike_count + 1;
                        @(posedge clk);
                        s_axis_input_tvalid = 0;
                        
                        // Add some delay between spikes
                        repeat($urandom % 5 + 1) @(posedge clk);
                    end
                end
            end
            
            // Output spike monitoring
            begin
                integer output_count = 0;
                while (output_count < 10 && $time < 100000) begin
                    @(posedge clk);
                    if (m_axis_output_tvalid && m_axis_output_tready) begin
                        $display("Output spike: channel=%d, position=%d at time %t", 
                                m_axis_output_tdata[31:16], m_axis_output_tdata[15:0], $time);
                        output_count = output_count + 1;
                    end
                end
            end
        join
        
        // Wait for computation to complete
        wait(computation_done);
        $display("Computation completed at time %t", $time);
        $display("Input spikes: %d", input_spike_count);
        $display("Output spikes: %d", output_spike_count);
        $display("Cycles: %d", cycle_count);
        
        // Test 2: Performance stress test
        $display("\nTest 2: Performance stress test");
        reset = 1;
        #20;
        reset = 0;
        #20;
        
        // Generate continuous spike stream
        fork
            begin
                for (i = 0; i < 200; i = i + 1) begin
                    @(posedge clk);
                    if (s_axis_input_tready) begin
                        s_axis_input_tdata = {$urandom % INPUT_CHANNELS, $urandom % INPUT_LENGTH};
                        s_axis_input_tvalid = 1;
                        @(posedge clk);
                        s_axis_input_tvalid = 0;
                    end
                end
            end
            
            begin
                integer timeout = 0;
                while (!computation_done && timeout < 50000) begin
                    @(posedge clk);
                    timeout = timeout + 1;
                end
                if (timeout >= 50000) begin
                    $display("ERROR: Timeout waiting for computation completion");
                end
            end
        join
        
        $display("Stress test completed");
        $display("Final input spikes: %d", input_spike_count);
        $display("Final output spikes: %d", output_spike_count);
        $display("Final cycles: %d", cycle_count);
        
        // Test 3: Threshold sensitivity
        $display("\nTest 3: Threshold sensitivity");
        
        // Test different thresholds
        reg [15:0] thresholds [0:3];
        thresholds[0] = 16'h1000;  // Low threshold
        thresholds[1] = 16'h4000;  // Medium threshold
        thresholds[2] = 16'h8000;  // High threshold
        thresholds[3] = 16'hC000;  // Very high threshold
        
        for (i = 0; i < 4; i = i + 1) begin
            reset = 1;
            threshold_config = thresholds[i];
            #20;
            reset = 0;
            #20;
            
            // Send fixed pattern
            repeat(20) begin
                @(posedge clk);
                if (s_axis_input_tready) begin
                    s_axis_input_tdata = {16'd0, 16'd50};  // Channel 0, position 50
                    s_axis_input_tvalid = 1;
                    @(posedge clk);
                    s_axis_input_tvalid = 0;
                    repeat(5) @(posedge clk);
                end
            end
            
            // Wait for processing
            wait(computation_done);
            $display("Threshold=0x%h: Output spikes=%d", thresholds[i], output_spike_count);
        end
        
        // Test 4: Different kernel sizes
        $display("\nTest 4: Architecture validation");
        reset = 1;
        threshold_config = THRESHOLD;
        #20;
        reset = 0;
        #20;
        
        // Test pattern that should activate multiple output positions
        for (i = 0; i < INPUT_LENGTH; i = i + 2) begin
            @(posedge clk);
            if (s_axis_input_tready) begin
                s_axis_input_tdata = {16'd0, i[15:0]};  // Channel 0, sequential positions
                s_axis_input_tvalid = 1;
                @(posedge clk);
                s_axis_input_tvalid = 0;
                @(posedge clk);
            end
        end
        
        wait(computation_done);
        $display("Sequential pattern test completed");
        $display("Output spikes for sequential pattern: %d", output_spike_count);
        
        // Performance analysis
        real throughput = input_spike_count * 1000.0 / cycle_count;  // spikes per 1000 cycles
        $display("\nPerformance Analysis:");
        $display("Throughput: %.2f spikes per 1000 cycles", throughput);
        
        if (output_spike_count > 0) begin
            $display("✅ SNN 1D Convolution working correctly");
        end else begin
            $display("❌ No output spikes generated - check threshold and weights");
        end
        
        $display("\nTestbench completed successfully!");
        $finish;
    end
    
    // Timeout protection
    initial begin
        #200000;
        $display("ERROR: Global timeout reached");
        $finish;
    end
    
    // Monitor key signals
    initial begin
        $monitor("Time=%t, Enable=%b, InputReady=%b, OutputValid=%b, Done=%b", 
                 $time, enable, s_axis_input_tready, m_axis_output_tvalid, computation_done);
    end

endmodule
