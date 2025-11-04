//-----------------------------------------------------------------------------
// Title         : Testbench for Spike Router
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_spike_router.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Comprehensive testbench for spike router module
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_spike_router();

    // Parameters
    localparam NUM_NEURONS      = 64;
    localparam MAX_FANOUT       = 32;
    localparam WEIGHT_WIDTH     = 8;
    // Calculate log2 ceiling manually for Verilog-2001 compatibility
    localparam NEURON_ID_WIDTH  = (NUM_NEURONS <= 2) ? 1 :
                                 (NUM_NEURONS <= 4) ? 2 :
                                 (NUM_NEURONS <= 8) ? 3 :
                                 (NUM_NEURONS <= 16) ? 4 :
                                 (NUM_NEURONS <= 32) ? 5 :
                                 (NUM_NEURONS <= 64) ? 6 :
                                 (NUM_NEURONS <= 128) ? 7 : 8;
    localparam DELAY_WIDTH      = 8;
    localparam FIFO_DEPTH       = 256;
    
    // Clock period (100MHz)
    localparam CLK_PERIOD = 10;
    
    // DUT signals
    reg                         clk;
    reg                         rst_n;
    
    // Input spike interface
    reg                         s_spike_valid;
    reg [NEURON_ID_WIDTH-1:0]  s_spike_neuron_id;
    wire                        s_spike_ready;
    
    // Output spike interface
    wire                        m_spike_valid;
    wire [NEURON_ID_WIDTH-1:0] m_spike_dest_id;
    wire [WEIGHT_WIDTH-1:0]    m_spike_weight;
    wire                        m_spike_exc_inh;
    reg                         m_spike_ready;
    
    // Configuration interface
    reg                         config_we;
    reg [31:0]                 config_addr;
    reg [31:0]                 config_data;
    wire [31:0]                config_readdata;
    
    // Status outputs
    wire [31:0]                routed_spike_count;
    wire                       router_busy;
    wire                       fifo_overflow;
    
    // Test variables
    integer                    test_num;
    integer                    error_count;
    integer                    spike_count;
    integer                    i, j;
    reg [255:0]               test_name;
    
    // Tracking arrays
    reg [31:0]                expected_spikes[0:NUM_NEURONS-1];
    reg [31:0]                received_spikes[0:NUM_NEURONS-1];
    reg [7:0]                 expected_weights[0:NUM_NEURONS-1];
    reg [7:0]                 received_weights[0:NUM_NEURONS-1];
    
    //-------------------------------------------------------------------------
    // DUT Instantiation
    //-------------------------------------------------------------------------
    spike_router #(
        .NUM_NEURONS(NUM_NEURONS),
        .MAX_FANOUT(MAX_FANOUT),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .FIFO_DEPTH(FIFO_DEPTH)
    ) DUT (
        .clk(clk),
        .rst_n(rst_n),
        .s_spike_valid(s_spike_valid),
        .s_spike_neuron_id(s_spike_neuron_id),
        .s_spike_ready(s_spike_ready),
        .m_spike_valid(m_spike_valid),
        .m_spike_dest_id(m_spike_dest_id),
        .m_spike_weight(m_spike_weight),
        .m_spike_exc_inh(m_spike_exc_inh),
        .m_spike_ready(m_spike_ready),
        .config_we(config_we),
        .config_addr(config_addr),
        .config_data(config_data),
        .config_readdata(config_readdata),
        .routed_spike_count(routed_spike_count),
        .router_busy(router_busy),
        .fifo_overflow(fifo_overflow)
    );
    
    //-------------------------------------------------------------------------
    // Clock Generation
    //-------------------------------------------------------------------------
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    //-------------------------------------------------------------------------
    // Output Spike Monitoring
    //-------------------------------------------------------------------------
    always @(posedge clk) begin
        if (m_spike_valid && m_spike_ready) begin
            received_spikes[m_spike_dest_id] = received_spikes[m_spike_dest_id] + 1;
            received_weights[m_spike_dest_id] = m_spike_weight;
            $display("[%0t] Output spike: dest=%0d, weight=%0d, exc/inh=%b", 
                     $time, m_spike_dest_id, m_spike_weight, m_spike_exc_inh);
        end
    end
    
    //-------------------------------------------------------------------------
    // Test Tasks
    //-------------------------------------------------------------------------
    
    // Initialize test
    task init_test(input [255:0] name);
        begin
            test_name = name;
            spike_count = 0;
            for (i = 0; i < NUM_NEURONS; i = i + 1) begin
                expected_spikes[i] = 0;
                received_spikes[i] = 0;
                expected_weights[i] = 0;
                received_weights[i] = 0;
            end
            $display("\n========================================");
            $display("Test %0d: %0s", test_num, test_name);
            $display("========================================");
            test_num = test_num + 1;
        end
    endtask
    
    // Apply reset
    task apply_reset();
        begin
            @(posedge clk);
            rst_n = 1'b0;
            repeat(5) @(posedge clk);
            rst_n = 1'b1;
            @(posedge clk);
        end
    endtask
    
    // Configure connection
    task configure_connection(
        input [5:0] source_id,
        input [5:0] dest_id,
        input [7:0] weight,
        input exc_inh,
        input [6:0] delay,
        input [15:0] conn_index
    );
        begin
            @(posedge clk);
            config_we = 1'b1;
            config_addr = {8'h00, 16'd0} | conn_index;
            config_data = {8'd0, 1'b1, exc_inh, weight, delay, 2'd0, dest_id};
            @(posedge clk);
            config_we = 1'b0;
            @(posedge clk);
            $display("  Configured connection: src=%0d -> dst=%0d, w=%0d, delay=%0d", 
                     source_id, dest_id, weight, delay);
        end
    endtask
    
    // Configure neuron connection count
    task configure_neuron_count(
        input [5:0] neuron_id,
        input [7:0] conn_count
    );
        begin
            @(posedge clk);
            config_we = 1'b1;
            config_addr = {8'h01, 16'd0, 2'd0, neuron_id};
            config_data = {24'd0, conn_count};
            @(posedge clk);
            config_we = 1'b0;
            @(posedge clk);
            $display("  Neuron %0d configured with %0d connections", neuron_id, conn_count);
        end
    endtask
    
    // Send spike
    task send_spike(input [5:0] neuron_id);
        begin
            @(posedge clk);
            wait(s_spike_ready);
            s_spike_valid = 1'b1;
            s_spike_neuron_id = neuron_id;
            @(posedge clk);
            s_spike_valid = 1'b0;
            spike_count = spike_count + 1;
        end
    endtask
    
    // Wait with timeout
    task wait_with_timeout(input integer cycles);
        integer k;
        begin
            k = 0;
            while (k < cycles && router_busy) begin
                @(posedge clk);
                k = k + 1;
            end
            if (!router_busy) begin
                repeat(10) @(posedge clk);
            end
        end
    endtask
    
    // Verify results
    task verify_results();
        integer errors;
        begin
            errors = 0;
            for (i = 0; i < NUM_NEURONS; i = i + 1) begin
                if (expected_spikes[i] != received_spikes[i]) begin
                    $display("ERROR: Neuron %0d - Expected %0d spikes, got %0d", 
                             i, expected_spikes[i], received_spikes[i]);
                    errors = errors + 1;
                end else if (expected_spikes[i] > 0) begin
                    if (expected_weights[i] != received_weights[i]) begin
                        $display("ERROR: Neuron %0d - Expected weight %0d, got %0d", 
                                 i, expected_weights[i], received_weights[i]);
                        errors = errors + 1;
                    end else begin
                        $display("PASS: Neuron %0d - Correctly received %0d spikes with weight %0d", 
                                 i, received_spikes[i], received_weights[i]);
                    end
                end
            end
            
            if (errors == 0) begin
                $display("Test PASSED");
            end else begin
                $display("Test FAILED with %0d errors", errors);
                error_count = error_count + errors;
            end
        end
    endtask
    
    //-------------------------------------------------------------------------
    // Main Test Sequence
    //-------------------------------------------------------------------------
    initial begin
        // Initialize signals
        rst_n = 1'b1;
        s_spike_valid = 1'b0;
        s_spike_neuron_id = 0;
        m_spike_ready = 1'b1;
        config_we = 1'b0;
        config_addr = 0;
        config_data = 0;
        test_num = 1;
        error_count = 0;
        
        // Create waveform dump
        $dumpfile("tb_spike_router.vcd");
        $dumpvars(0, tb_spike_router);
        
        // Initial reset
        apply_reset();
        
        //---------------------------------------------------------------------
        // Test 1: Simple One-to-One Routing
        //---------------------------------------------------------------------
        init_test("Simple One-to-One Routing");
        
        // Configure neuron 0 to connect to neuron 1 with weight 50
        configure_connection(0, 1, 8'd50, 1'b1, 7'd0, 16'd0);
        configure_neuron_count(0, 8'd1);
        
        expected_spikes[1] = 1;
        expected_weights[1] = 50;
        
        // Send spike from neuron 0
        send_spike(0);
        wait_with_timeout(100);
        verify_results();
        
        //---------------------------------------------------------------------
        // Test 2: One-to-Many Routing (Fanout)
        //---------------------------------------------------------------------
        init_test("One-to-Many Routing (Fanout)");
        apply_reset();
        
        // Configure neuron 5 to connect to neurons 10, 11, 12, 13
        configure_connection(5, 10, 8'd30, 1'b1, 7'd0, 16'd40);
        configure_connection(5, 11, 8'd40, 1'b1, 7'd0, 16'd41);
        configure_connection(5, 12, 8'd50, 1'b0, 7'd0, 16'd42);  // Inhibitory
        configure_connection(5, 13, 8'd60, 1'b1, 7'd0, 16'd43);
        configure_neuron_count(5, 8'd4);
        
        expected_spikes[10] = 1; expected_weights[10] = 30;
        expected_spikes[11] = 1; expected_weights[11] = 40;
        expected_spikes[12] = 1; expected_weights[12] = 50;
        expected_spikes[13] = 1; expected_weights[13] = 60;
        
        // Send spike from neuron 5
        send_spike(5);
        wait_with_timeout(200);
        verify_results();
        
        //---------------------------------------------------------------------
        // Test 3: Many-to-One Routing (Convergence)
        //---------------------------------------------------------------------
        init_test("Many-to-One Routing (Convergence)");
        apply_reset();
        
        // Configure neurons 20, 21, 22 to all connect to neuron 30
        configure_connection(20, 30, 8'd20, 1'b1, 7'd0, 16'd80);
        configure_neuron_count(20, 8'd1);
        
        configure_connection(21, 30, 8'd25, 1'b1, 7'd0, 16'd81);
        configure_neuron_count(21, 8'd1);
        
        configure_connection(22, 30, 8'd30, 1'b1, 7'd0, 16'd82);
        configure_neuron_count(22, 8'd1);
        
        expected_spikes[30] = 3;
        expected_weights[30] = 30; // Last weight received
        
        // Send spikes from all source neurons
        send_spike(20);
        send_spike(21);
        send_spike(22);
        wait_with_timeout(300);
        verify_results();
        
        //---------------------------------------------------------------------
        // Test 4: Delayed Routing
        //---------------------------------------------------------------------
        init_test("Delayed Spike Routing");
        apply_reset();
        
        // Configure connections with different delays
        configure_connection(40, 50, 8'd80, 1'b1, 7'd10, 16'd120);  // 10 cycle delay
        configure_connection(40, 51, 8'd81, 1'b1, 7'd50, 16'd121);  // 50 cycle delay
        configure_connection(40, 52, 8'd82, 1'b1, 7'd100, 16'd122); // 100 cycle delay
        configure_neuron_count(40, 8'd3);
        
        // Track when spikes are sent
        fork
            begin
                send_spike(40);
                $display("  Spike sent at time %0t", $time);
            end
            
            begin
                // Monitor for delayed outputs
                @(posedge m_spike_valid);
                while (m_spike_dest_id != 50) @(posedge m_spike_valid);
                $display("  Spike to neuron 50 arrived at %0t (10 cycle delay)", $time);
                
                @(posedge m_spike_valid);
                while (m_spike_dest_id != 51) @(posedge m_spike_valid);
                $display("  Spike to neuron 51 arrived at %0t (50 cycle delay)", $time);
                
                @(posedge m_spike_valid);
                while (m_spike_dest_id != 52) @(posedge m_spike_valid);
                $display("  Spike to neuron 52 arrived at %0t (100 cycle delay)", $time);
            end
        join_any
        
        wait_with_timeout(200);
        
        //---------------------------------------------------------------------
        // Test 5: FIFO Stress Test
        //---------------------------------------------------------------------
        init_test("FIFO Stress Test");
        apply_reset();
        
        // Configure simple connections
        for (i = 0; i < 10; i = i + 1) begin
            configure_connection(i, i+10, 8'd100, 1'b1, 7'd0, i*MAX_FANOUT);
            configure_neuron_count(i, 8'd1);
        end
        
        // Send many spikes rapidly
        $display("  Sending rapid spike burst...");
        fork
            begin
                for (i = 0; i < 100; i = i + 1) begin
                    if (s_spike_ready) begin
                        send_spike(i % 10);
                    end else begin
                        @(posedge s_spike_ready);
                        send_spike(i % 10);
                    end
                end
            end
            
            begin
                // Monitor for overflow
                @(posedge fifo_overflow);
                $display("  WARNING: FIFO overflow detected at time %0t", $time);
            end
            
            begin
                #(CLK_PERIOD * 2000);
            end
        join_any
        
        wait_with_timeout(500);
        $display("  Total routed spikes: %0d", routed_spike_count);
        
        //---------------------------------------------------------------------
        // Test 6: Configuration Readback
        //---------------------------------------------------------------------
        init_test("Configuration Readback");
        
        // Write configuration
        config_we = 1'b1;
        config_addr = {8'h00, 16'd200};
        config_data = {8'd0, 1'b1, 1'b1, 8'd123, 7'd45, 2'd0, 6'd63};
        @(posedge clk);
        config_we = 1'b0;
        @(posedge clk);
        
        // Read back
        config_addr = {8'h00, 16'd200};
        @(posedge clk);
        @(posedge clk);
        
        if (config_readdata[23:0] == config_data[23:0]) begin
            $display("  PASS: Configuration readback correct: 0x%06x", config_readdata[23:0]);
        end else begin
            $display("  ERROR: Configuration readback mismatch. Expected: 0x%06x, Got: 0x%06x", 
                     config_data[23:0], config_readdata[23:0]);
            error_count = error_count + 1;
        end
        
        // Read spike counter
        config_addr = {8'h10, 24'd0};
        @(posedge clk);
        @(posedge clk);
        $display("  Spike counter reads: %0d", config_readdata);
        
        //---------------------------------------------------------------------
        // Test 7: Complex Network Pattern
        //---------------------------------------------------------------------
        init_test("Complex Network Pattern");
        apply_reset();
        
        // Create a small fully connected network (neurons 0-4)
        for (i = 0; i < 5; i = i + 1) begin
            for (j = 0; j < 5; j = j + 1) begin
                if (i != j) begin
                    configure_connection(i, j, 8'd20 + i*5 + j, 1'b1, 7'd2, 
                                       i*MAX_FANOUT + (j < i ? j : j-1));
                end
            end
            configure_neuron_count(i, 8'd4); // 4 connections each
        end
        
        // Send spike from neuron 2
        send_spike(2);
        
        // Each of neurons 0,1,3,4 should receive one spike
        expected_spikes[0] = 1; expected_weights[0] = 20 + 2*5 + 0;
        expected_spikes[1] = 1; expected_weights[1] = 20 + 2*5 + 1;
        expected_spikes[3] = 1; expected_weights[3] = 20 + 2*5 + 3;
        expected_spikes[4] = 1; expected_weights[4] = 20 + 2*5 + 4;
        
        wait_with_timeout(300);
        verify_results();
        
        //---------------------------------------------------------------------
        // Test 8: Reset Statistics
        //---------------------------------------------------------------------
        init_test("Reset Statistics");
        
        $display("  Current spike count: %0d", routed_spike_count);
        
        // Reset statistics
        config_we = 1'b1;
        config_addr = {8'h02, 24'd0};
        config_data = 32'd1;
        @(posedge clk);
        config_we = 1'b0;
        @(posedge clk);
        
        // Read spike counter again
        config_addr = {8'h10, 24'd0};
        @(posedge clk);
        @(posedge clk);
        
        if (config_readdata == 0) begin
            $display("  PASS: Spike counter reset to 0");
        end else begin
            $display("  ERROR: Spike counter not reset. Value: %0d", config_readdata);
            error_count = error_count + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 9: Invalid Connections
        //---------------------------------------------------------------------
        init_test("Invalid Connections");
        apply_reset();
        
        // Configure connection with invalid bit = 0
        @(posedge clk);
        config_we = 1'b1;
        config_addr = {8'h00, 16'd300};
        config_data = {8'd0, 1'b0, 1'b1, 8'd99, 7'd0, 2'd0, 6'd55}; // Valid bit = 0
        @(posedge clk);
        config_we = 1'b0;
        
        configure_neuron_count(45, 8'd1);
        
        // This spike should not generate any output
        expected_spikes[55] = 0;
        
        send_spike(45);
        wait_with_timeout(100);
        verify_results();
        
        //---------------------------------------------------------------------
        // Test 10: Maximum Fanout Test
        //---------------------------------------------------------------------
        init_test("Maximum Fanout Test");
        apply_reset();
        
        // Configure neuron 60 with maximum fanout
        for (i = 0; i < MAX_FANOUT && i < NUM_NEURONS-1; i = i + 1) begin
            configure_connection(60, (i != 60) ? i : 61, 8'd10 + i, 
                               (i % 2), 7'd0, 60*MAX_FANOUT + i);
        end
        configure_neuron_count(60, MAX_FANOUT);
        
        // Send spike
        send_spike(60);
        
        // Count expected outputs
        j = 0;
        for (i = 0; i < MAX_FANOUT && i < NUM_NEURONS-1; i = i + 1) begin
            if (i != 60) begin
                expected_spikes[i] = 1;
                expected_weights[i] = 10 + i;
                j = j + 1;
            end else begin
                expected_spikes[61] = 1;
                expected_weights[61] = 10 + i;
            end
        end
        
        wait_with_timeout(500);
        $display("  Generated %0d output spikes from 1 input", j);
        
        //---------------------------------------------------------------------
        // Test Summary
        //---------------------------------------------------------------------
        #1000;
        $display("\n========================================");
        $display("Test Summary");
        $display("========================================");
        $display("Total Tests Run: %0d", test_num - 1);
        $display("Total Errors: %0d", error_count);
        $display("Total Spikes Routed: %0d", routed_spike_count);
        
        if (error_count == 0) begin
            $display("\nAll tests PASSED!");
        end else begin
            $display("\nTests FAILED with %0d errors!", error_count);
        end
        
        $display("========================================\n");
        $finish;
    end
    
    //-------------------------------------------------------------------------
    // Timeout Watchdog
    //-------------------------------------------------------------------------
    initial begin
        #1_000_000;  // 1ms timeout
        $display("\n*** ERROR: Testbench timeout! ***");
        $finish;
    end
    
    //-------------------------------------------------------------------------
    // Performance Monitoring
    //-------------------------------------------------------------------------
    reg [31:0] output_spike_counter;
    always @(posedge clk) begin
        if (!rst_n)
            output_spike_counter <= 0;
        else if (m_spike_valid && m_spike_ready)
            output_spike_counter <= output_spike_counter + 1;
    end
    
    // Monitor router state
    always @(posedge clk) begin
        if (router_busy && ((output_spike_counter % 100) == 0) && output_spike_counter > 0) begin
            $display("[%0t] Router busy, %0d spikes processed", $time, output_spike_counter);
        end
    end

endmodule
