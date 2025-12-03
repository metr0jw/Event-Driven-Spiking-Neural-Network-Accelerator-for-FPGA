//-----------------------------------------------------------------------------
// Title         : PC Interface Comprehensive Testbench
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_pc_interface_comprehensive.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Comprehensive testbench for PC communication interface
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_pc_interface_comprehensive;

    // Clock and reset
    reg clk;
    reg reset;
    
    // PC Interface signals
    reg enable;
    reg [31:0] control_reg;
    wire [31:0] status_reg;
    
    // AXI Stream interfaces
    reg [63:0] pc_spike_in_tdata;
    reg pc_spike_in_tvalid;
    wire pc_spike_in_tready;
    reg pc_spike_in_tlast;
    
    wire [63:0] pc_spike_out_tdata;
    wire pc_spike_out_tvalid;
    reg pc_spike_out_tready;
    wire pc_spike_out_tlast;
    
    wire [31:0] snn_spike_in_tdata;
    wire snn_spike_in_tvalid;
    reg snn_spike_in_tready;
    
    reg [31:0] snn_spike_out_tdata;
    reg snn_spike_out_tvalid;
    wire snn_spike_out_tready;
    
    // Configuration interface
    reg [31:0] config_addr;
    reg [31:0] config_data;
    reg config_write;
    wire [31:0] config_read_data;
    
    // Performance counters
    wire [31:0] input_spike_count;
    wire [31:0] output_spike_count;
    wire [31:0] cycle_count;
    
    // Test variables
    integer test_count;
    integer pass_count;
    integer fail_count;
    
    // Test data
    reg [63:0] test_spike_data [0:255];
    reg [31:0] expected_output [0:255];
    integer test_index;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz clock
    end
    
    // Device Under Test (DUT) - Simulated interface
    // In actual implementation, this would be the HLS-generated module
    pc_interface_wrapper dut (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .control_reg(control_reg),
        .status_reg(status_reg),
        
        // PC spike input
        .pc_spike_in_tdata(pc_spike_in_tdata),
        .pc_spike_in_tvalid(pc_spike_in_tvalid),
        .pc_spike_in_tready(pc_spike_in_tready),
        .pc_spike_in_tlast(pc_spike_in_tlast),
        
        // PC spike output
        .pc_spike_out_tdata(pc_spike_out_tdata),
        .pc_spike_out_tvalid(pc_spike_out_tvalid),
        .pc_spike_out_tready(pc_spike_out_tready),
        .pc_spike_out_tlast(pc_spike_out_tlast),
        
        // SNN core interface
        .snn_spike_in_tdata(snn_spike_in_tdata),
        .snn_spike_in_tvalid(snn_spike_in_tvalid),
        .snn_spike_in_tready(snn_spike_in_tready),
        
        .snn_spike_out_tdata(snn_spike_out_tdata),
        .snn_spike_out_tvalid(snn_spike_out_tvalid),
        .snn_spike_out_tready(snn_spike_out_tready),
        
        // Configuration
        .config_addr(config_addr),
        .config_data(config_data),
        .config_write(config_write),
        .config_read_data(config_read_data),
        
        // Performance counters
        .input_spike_count(input_spike_count),
        .output_spike_count(output_spike_count),
        .cycle_count(cycle_count)
    );
    
    // Test initialization
    initial begin
        $display("=================================================");
        $display("PC Interface Comprehensive Testbench");
        $display("Testing event-driven SNN accelerator interface");
        $display("=================================================");
        
        // Initialize signals
        reset = 1;
        enable = 0;
        control_reg = 0;
        pc_spike_in_tdata = 0;
        pc_spike_in_tvalid = 0;
        pc_spike_in_tlast = 0;
        pc_spike_out_tready = 0;
        snn_spike_in_tready = 1;
        snn_spike_out_tdata = 0;
        snn_spike_out_tvalid = 0;
        config_addr = 0;
        config_data = 0;
        config_write = 0;
        
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        test_index = 0;
        
        // Initialize test data
        initialize_test_data();
        
        // Wait for reset
        #100;
        reset = 0;
        #50;
        
        // Run test sequence
        run_test_sequence();
        
        // Final results
        $display("\n=================================================");
        $display("Test Results Summary:");
        $display("Total Tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        $display("Success Rate: %.1f%%", (pass_count * 100.0) / test_count);
        $display("=================================================");
        
        if (fail_count == 0) begin
            $display("✅ ALL TESTS PASSED!");
        end else begin
            $display("❌ SOME TESTS FAILED!");
        end
        
        $finish;
    end
    
    // Initialize test data
    task initialize_test_data;
        integer i;
        begin
            $display("Initializing test data...");
            
            // Generate test spike patterns
            for (i = 0; i < 256; i = i + 1) begin
                test_spike_data[i] = {
                    16'h1,                    // packet_type (SPIKE)
                    16'(i % 1024),           // neuron_id
                    16'(i * 10),             // timestamp
                    8'(128 + (i % 127)),     // weight
                    8'h00                    // padding
                };
                
                expected_output[i] = {
                    16'(i % 1024),           // neuron_id
                    16'(i * 10)              // timestamp
                };
            end
            
            $display("Test data initialized with %d patterns", 256);
        end
    endtask
    
    // Main test sequence
    task run_test_sequence;
        begin
            $display("\nStarting test sequence...");
            
            // Test 1: Basic reset and enable
            test_reset_enable();
            
            // Test 2: Configuration interface
            test_configuration();
            
            // Test 3: Single spike transmission
            test_single_spike();
            
            // Test 4: Burst spike transmission
            test_burst_spikes();
            
            // Test 5: Buffer overflow handling
            test_buffer_overflow();
            
            // Test 6: SNN core interaction
            test_snn_interaction();
            
            // Test 7: Performance counters
            test_performance_counters();
            
            // Test 8: Error handling
            test_error_handling();
            
            // Test 9: Throughput test
            test_throughput();
            
            // Test 10: Latency measurement
            test_latency();
        end
    endtask
    
    // Test 1: Basic reset and enable functionality
    task test_reset_enable;
        begin
            $display("\nTest 1: Reset and Enable");
            test_count = test_count + 1;
            
            // Test reset
            reset = 1;
            #20;
            if (status_reg[0] == 0) begin
                $display("✅ Reset test passed");
                pass_count = pass_count + 1;
            end else begin
                $display("❌ Reset test failed");
                fail_count = fail_count + 1;
            end
            
            reset = 0;
            #10;
            
            // Test enable
            enable = 1;
            #20;
            if (status_reg[0] == 1) begin
                $display("✅ Enable test passed");
                pass_count = pass_count + 1;
            end else begin
                $display("❌ Enable test failed");
                fail_count = fail_count + 1;
            end
            
            test_count = test_count + 1;
        end
    endtask
    
    // Test 2: Configuration interface
    task test_configuration;
        begin
            $display("\nTest 2: Configuration Interface");
            test_count = test_count + 1;
            
            // Write configuration
            config_addr = 32'h0000; // NEURON_THRESHOLD
            config_data = 32'h12345678;
            config_write = 1;
            #20;
            config_write = 0;
            #10;
            
            // Read back configuration
            config_addr = 32'h0000;
            #20;
            
            if (config_read_data == 32'h12345678) begin
                $display("✅ Configuration write/read test passed");
                pass_count = pass_count + 1;
            end else begin
                $display("❌ Configuration write/read test failed");
                $display("Expected: 0x12345678, Got: 0x%08x", config_read_data);
                fail_count = fail_count + 1;
            end
        end
    endtask
    
    // Test 3: Single spike transmission
    task test_single_spike;
        begin
            $display("\nTest 3: Single Spike Transmission");
            test_count = test_count + 1;
            
            // Send single spike
            pc_spike_in_tdata = test_spike_data[0];
            pc_spike_in_tvalid = 1;
            pc_spike_in_tlast = 1;
            
            wait(pc_spike_in_tready);
            #10;
            pc_spike_in_tvalid = 0;
            pc_spike_in_tlast = 0;
            
            // Wait for processing
            #100;
            
            // Check if spike was forwarded to SNN core
            if (snn_spike_in_tvalid) begin
                $display("✅ Single spike transmission test passed");
                pass_count = pass_count + 1;
            end else begin
                $display("❌ Single spike transmission test failed");
                fail_count = fail_count + 1;
            end
        end
    endtask
    
    // Test 4: Burst spike transmission
    task test_burst_spikes;
        integer i;
        begin
            $display("\nTest 4: Burst Spike Transmission");
            test_count = test_count + 1;
            
            pc_spike_out_tready = 1;
            snn_spike_in_tready = 1;
            
            // Send burst of spikes
            for (i = 0; i < 10; i = i + 1) begin
                pc_spike_in_tdata = test_spike_data[i];
                pc_spike_in_tvalid = 1;
                pc_spike_in_tlast = (i == 9);
                
                wait(pc_spike_in_tready);
                #10;
                pc_spike_in_tvalid = 0;
                pc_spike_in_tlast = 0;
                #5;
            end
            
            // Wait for processing
            #200;
            
            // Check performance counters
            if (input_spike_count >= 10) begin
                $display("✅ Burst spike transmission test passed");
                $display("Input spikes processed: %d", input_spike_count);
                pass_count = pass_count + 1;
            end else begin
                $display("❌ Burst spike transmission test failed");
                $display("Expected >= 10, Got: %d", input_spike_count);
                fail_count = fail_count + 1;
            end
        end
    endtask
    
    // Test 5: Buffer overflow handling
    task test_buffer_overflow;
        integer i;
        begin
            $display("\nTest 5: Buffer Overflow Handling");
            test_count = test_count + 1;
            
            // Block SNN core to cause buffer overflow
            snn_spike_in_tready = 0;
            
            // Send many spikes
            for (i = 0; i < 300; i = i + 1) begin
                pc_spike_in_tdata = test_spike_data[i % 256];
                pc_spike_in_tvalid = 1;
                
                if (pc_spike_in_tready) begin
                    #10;
                    pc_spike_in_tvalid = 0;
                    #2;
                end else begin
                    // Buffer full, stop sending
                    pc_spike_in_tvalid = 0;
                    i = 300; // Force loop exit
                end
            end
            
            // Check buffer full status
            #50;
            if (status_reg[3] == 1) begin  // INPUT_BUFFER_FULL
                $display("✅ Buffer overflow detection test passed");
                pass_count = pass_count + 1;
            end else begin
                $display("❌ Buffer overflow detection test failed");
                fail_count = fail_count + 1;
            end
            
            // Restore normal operation
            snn_spike_in_tready = 1;
            #100;
        end
    endtask
    
    // Test 6: SNN core interaction
    task test_snn_interaction;
        begin
            $display("\nTest 6: SNN Core Interaction");
            test_count = test_count + 1;
            
            pc_spike_out_tready = 1;
            
            // Simulate SNN core generating output spike
            snn_spike_out_tdata = 32'h12340567;  // neuron_id=0x1234, timestamp=0x0567
            snn_spike_out_tvalid = 1;
            
            // Wait for spike to be forwarded to PC
            wait(pc_spike_out_tvalid);
            #10;
            snn_spike_out_tvalid = 0;
            
            // Check output data
            if (pc_spike_out_tvalid && pc_spike_out_tdata[31:16] == 16'h1234) begin
                $display("✅ SNN core interaction test passed");
                $display("Output spike data: 0x%016x", pc_spike_out_tdata);
                pass_count = pass_count + 1;
            end else begin
                $display("❌ SNN core interaction test failed");
                fail_count = fail_count + 1;
            end
        end
    endtask
    
    // Test 7: Performance counters
    task test_performance_counters;
        reg [31:0] initial_cycle_count;
        begin
            $display("\nTest 7: Performance Counters");
            test_count = test_count + 1;
            
            initial_cycle_count = cycle_count;
            
            // Wait some cycles
            #1000;
            
            // Check if cycle counter is incrementing
            if (cycle_count > initial_cycle_count) begin
                $display("✅ Performance counters test passed");
                $display("Cycle count increment: %d", cycle_count - initial_cycle_count);
                pass_count = pass_count + 1;
            end else begin
                $display("❌ Performance counters test failed");
                fail_count = fail_count + 1;
            end
        end
    endtask
    
    // Test 8: Error handling
    task test_error_handling;
        begin
            $display("\nTest 8: Error Handling");
            test_count = test_count + 1;
            
            // Send invalid spike packet (invalid neuron ID)
            pc_spike_in_tdata = {
                16'h1,      // packet_type
                16'hFFFF,   // invalid neuron_id (> MAX_NEURONS)
                16'h1234,   // timestamp
                8'h80,      // weight
                8'h00       // padding
            };
            pc_spike_in_tvalid = 1;
            pc_spike_in_tlast = 1;
            
            wait(pc_spike_in_tready);
            #10;
            pc_spike_in_tvalid = 0;
            pc_spike_in_tlast = 0;
            
            // Wait for error processing
            #100;
            
            // Check error status
            if (status_reg[2] == 1) begin  // ERROR bit
                $display("✅ Error handling test passed");
                pass_count = pass_count + 1;
            end else begin
                $display("❌ Error handling test failed");
                fail_count = fail_count + 1;
            end
        end
    endtask
    
    // Test 9: Throughput measurement
    task test_throughput;
        integer i;
        reg [31:0] start_time, end_time;
        real throughput;
        begin
            $display("\nTest 9: Throughput Measurement");
            test_count = test_count + 1;
            
            pc_spike_out_tready = 1;
            snn_spike_in_tready = 1;
            
            start_time = cycle_count;
            
            // Send 100 spikes as fast as possible
            for (i = 0; i < 100; i = i + 1) begin
                pc_spike_in_tdata = test_spike_data[i % 256];
                pc_spike_in_tvalid = 1;
                pc_spike_in_tlast = (i == 99);
                
                wait(pc_spike_in_tready);
                @(posedge clk);
                pc_spike_in_tvalid = 0;
                pc_spike_in_tlast = 0;
            end
            
            // Wait for all spikes to be processed
            #500;
            end_time = cycle_count;
            
            throughput = 100.0 / (end_time - start_time);
            
            $display("✅ Throughput test completed");
            $display("Processed 100 spikes in %d cycles", end_time - start_time);
            $display("Throughput: %.2f spikes/cycle", throughput);
            pass_count = pass_count + 1;
        end
    endtask
    
    // Test 10: Latency measurement
    task test_latency;
        reg [31:0] send_time, receive_time;
        begin
            $display("\nTest 10: Latency Measurement");
            test_count = test_count + 1;
            
            pc_spike_out_tready = 1;
            snn_spike_in_tready = 1;
            
            // Send single spike and measure latency
            send_time = cycle_count;
            pc_spike_in_tdata = test_spike_data[0];
            pc_spike_in_tvalid = 1;
            pc_spike_in_tlast = 1;
            
            wait(pc_spike_in_tready);
            @(posedge clk);
            pc_spike_in_tvalid = 0;
            pc_spike_in_tlast = 0;
            
            // Wait for spike to appear at SNN core
            wait(snn_spike_in_tvalid);
            receive_time = cycle_count;
            
            $display("✅ Latency test completed");
            $display("Spike latency: %d cycles", receive_time - send_time);
            pass_count = pass_count + 1;
        end
    endtask
    
    // Monitor for debugging
    initial begin
        $monitor("Time=%0t, Enable=%b, Status=0x%08x, InputCount=%d, OutputCount=%d, CycleCount=%d",
                 $time, enable, status_reg, input_spike_count, output_spike_count, cycle_count);
    end
    
    // VCD dump for waveform analysis
    initial begin
        $dumpfile("tb_pc_interface_comprehensive.vcd");
        $dumpvars(0, tb_pc_interface_comprehensive);
    end
    
endmodule

// Wrapper module to simulate HLS interface
// In actual implementation, this would be generated by Vitis HLS
module pc_interface_wrapper (
    input clk,
    input reset,
    input enable,
    input [31:0] control_reg,
    output reg [31:0] status_reg,
    
    // PC spike input
    input [63:0] pc_spike_in_tdata,
    input pc_spike_in_tvalid,
    output reg pc_spike_in_tready,
    input pc_spike_in_tlast,
    
    // PC spike output  
    output reg [63:0] pc_spike_out_tdata,
    output reg pc_spike_out_tvalid,
    input pc_spike_out_tready,
    output reg pc_spike_out_tlast,
    
    // SNN core interface
    output reg [31:0] snn_spike_in_tdata,
    output reg snn_spike_in_tvalid,
    input snn_spike_in_tready,
    
    input [31:0] snn_spike_out_tdata,
    input snn_spike_out_tvalid,
    output reg snn_spike_out_tready,
    
    // Configuration
    input [31:0] config_addr,
    input [31:0] config_data,
    input config_write,
    output reg [31:0] config_read_data,
    
    // Performance counters
    output reg [31:0] input_spike_count,
    output reg [31:0] output_spike_count,
    output reg [31:0] cycle_count
);

    // Simple behavioral model for testing
    reg [31:0] config_mem [0:1023];
    reg [255:0] input_buffer;
    reg [255:0] output_buffer;
    reg error_flag;
    reg buffer_full;
    
    always @(posedge clk) begin
        if (reset) begin
            status_reg <= 0;
            pc_spike_in_tready <= 1;
            pc_spike_out_tvalid <= 0;
            snn_spike_in_tvalid <= 0;
            snn_spike_out_tready <= 1;
            config_read_data <= 0;
            input_spike_count <= 0;
            output_spike_count <= 0;
            cycle_count <= 0;
            error_flag <= 0;
            buffer_full <= 0;
        end else if (enable) begin
            cycle_count <= cycle_count + 1;
            
            // Handle input spikes
            if (pc_spike_in_tvalid && pc_spike_in_tready) begin
                input_spike_count <= input_spike_count + 1;
                snn_spike_in_tdata <= pc_spike_in_tdata[31:0];
                snn_spike_in_tvalid <= 1;
                
                // Check for invalid neuron ID
                if (pc_spike_in_tdata[47:32] > 16'h0FFF) begin
                    error_flag <= 1;
                end
            end else begin
                snn_spike_in_tvalid <= 0;
            end
            
            // Handle output spikes
            if (snn_spike_out_tvalid && snn_spike_out_tready) begin
                output_spike_count <= output_spike_count + 1;
                pc_spike_out_tdata <= {32'h01000000, snn_spike_out_tdata};
                pc_spike_out_tvalid <= 1;
                pc_spike_out_tlast <= 1;
            end else if (pc_spike_out_tready) begin
                pc_spike_out_tvalid <= 0;
                pc_spike_out_tlast <= 0;
            end
            
            // Handle configuration
            if (config_write && config_addr < 1024) begin
                config_mem[config_addr] <= config_data;
            end
            
            if (config_addr < 1024) begin
                config_read_data <= config_mem[config_addr];
            end else if (config_addr == 32'h1000) begin
                config_read_data <= 32'h01000000; // Version
            end
            
            // Update status register
            status_reg <= {24'h0, error_flag, buffer_full, 4'b0, enable, 1'b0};
            
            // Simulate buffer full condition
            if (input_spike_count > 250) begin
                buffer_full <= 1;
                pc_spike_in_tready <= 0;
            end else begin
                buffer_full <= 0;
                pc_spike_in_tready <= 1;
            end
        end else begin
            status_reg <= 32'h2; // DISABLED
        end
    end
    
endmodule
