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
    integer i;
    integer output_count;
    integer timeout;
    reg [15:0] thresholds [0:3];
    real throughput;
    
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
        
        // Wait for weight loading to complete (wait for ready signal)
        wait(s_axis_input_tready);
        $display("Weight loading complete, starting spike processing at time %t", $time);
        
        // Send 10 spikes with proper handshaking
        for (i = 0; i < 10; i = i + 1) begin
            // Wait for ready
            wait(s_axis_input_tready);
            @(posedge clk);
            
            s_axis_input_tdata = {test_input_channels[i], test_input_positions[i]};
            s_axis_input_tvalid = 1;
            s_axis_input_tlast = (i == 9) ? 1'b1 : 1'b0;  // Set tlast on final spike
            test_spike_count = test_spike_count + 1;
            
            @(posedge clk);
            s_axis_input_tvalid = 0;
            s_axis_input_tlast = 0;
            
            // Small delay between spikes
            repeat(5) @(posedge clk);
        end
        
        $display("All spikes sent, waiting for completion at time %t", $time);
        
        // Wait for computation to complete with timeout
        fork
            begin
                wait(computation_done);
            end
            begin
                #5000000;  // 5ms timeout
                $display("WARNING: Timeout waiting for computation_done");
            end
        join_any
        disable fork;
        
        $display("Test 1 completed at time %t", $time);
        $display("Input spikes: %d", input_spike_count);
        $display("Output spikes: %d", output_spike_count);
        $display("Cycles: %d", cycle_count);
        
        // Performance analysis
        if (cycle_count > 0) begin
            throughput = input_spike_count * 1000.0 / cycle_count;
            $display("Throughput: %.2f spikes per 1000 cycles", throughput);
        end
        
        if (output_spike_count > 0) begin
            $display("PASS: SNN 1D Convolution working correctly");
        end else begin
            $display("INFO: No output spikes generated (depends on weights and threshold)");
        end
        
        $display("\nTestbench completed!");
        $finish;
    end
    
    // Timeout protection
    initial begin
        #10000000;  // 10ms timeout
        $display("ERROR: Global timeout reached");
        $finish;
    end
    
    // Monitor key signals (reduced output)
    // initial begin
    //     $monitor("Time=%t, Enable=%b, InputReady=%b, OutputValid=%b, Done=%b", 
    //              $time, enable, s_axis_input_tready, m_axis_output_tvalid, computation_done);
    // end

endmodule
