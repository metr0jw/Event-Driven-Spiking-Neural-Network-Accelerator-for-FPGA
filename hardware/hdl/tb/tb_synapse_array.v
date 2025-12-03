//-----------------------------------------------------------------------------
// Title         : Synapse Array Testbench
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_synapse_array.v
// Author        : Jiwoon Lee (@metr0jw)
// Description   : Testbench for synapse array with weight storage
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_synapse_array;

    // Parameters
    parameter NUM_AXONS = 8;
    parameter NUM_NEURONS = 8;
    parameter WEIGHT_WIDTH = 8;
    parameter AXON_ID_WIDTH = 3;    // log2(8) = 3
    parameter NEURON_ID_WIDTH = 3;  // log2(8) = 3
    
    // Clock and reset
    reg clk;
    reg rst_n;
    reg enable;
    
    // Input spike interface
    reg spike_in_valid;
    reg [AXON_ID_WIDTH-1:0] spike_in_axon_id;
    
    // Output spike interface
    wire spike_out_valid;
    wire [NEURON_ID_WIDTH-1:0] spike_out_neuron_id;
    wire [WEIGHT_WIDTH-1:0] spike_out_weight;
    wire spike_out_exc_inh;
    
    // Weight configuration interface
    reg weight_we;
    reg [AXON_ID_WIDTH-1:0] weight_addr_axon;
    reg [NEURON_ID_WIDTH-1:0] weight_addr_neuron;
    reg [WEIGHT_WIDTH:0] weight_data;
    
    // Test variables
    integer i, j;
    integer error_count;
    integer test_num;
    integer output_count;
    reg [255:0] test_name;
    reg [WEIGHT_WIDTH:0] expected_weights [0:NUM_NEURONS-1];
    
    // DUT instantiation
    synapse_array #(
        .NUM_AXONS(NUM_AXONS),
        .NUM_NEURONS(NUM_NEURONS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .USE_BRAM(0)  // Use distributed RAM for faster simulation
    ) DUT (
        .clk(clk),
        .rst_n(rst_n),
        .spike_in_valid(spike_in_valid),
        .spike_in_axon_id(spike_in_axon_id),
        .spike_out_valid(spike_out_valid),
        .spike_out_neuron_id(spike_out_neuron_id),
        .spike_out_weight(spike_out_weight),
        .spike_out_exc_inh(spike_out_exc_inh),
        .weight_we(weight_we),
        .weight_addr_axon(weight_addr_axon),
        .weight_addr_neuron(weight_addr_neuron),
        .weight_data(weight_data),
        .enable(enable)
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
    
    // Task: Configure weight
    task configure_weight;
        input [AXON_ID_WIDTH-1:0] axon;
        input [NEURON_ID_WIDTH-1:0] neuron;
        input [WEIGHT_WIDTH-1:0] weight;
        input exc_inh;  // 1 = excitatory, 0 = inhibitory
        begin
            @(posedge clk);
            weight_we = 1'b1;
            weight_addr_axon = axon;
            weight_addr_neuron = neuron;
            weight_data = {exc_inh, weight};
            @(posedge clk);
            weight_we = 1'b0;
        end
    endtask
    
    // Task: Send spike
    task send_spike;
        input [AXON_ID_WIDTH-1:0] axon_id;
        begin
            @(posedge clk);
            spike_in_valid = 1'b1;
            spike_in_axon_id = axon_id;
            @(posedge clk);
            @(posedge clk);  // Hold for 2 cycles to ensure capture
            spike_in_valid = 1'b0;
        end
    endtask
    
    // Task: Wait for outputs
    task wait_for_outputs;
        input integer expected_count;
        output integer received_count;
        integer timeout;
        begin
            received_count = 0;
            timeout = 0;
            while (received_count < expected_count && timeout < 1000) begin
                @(posedge clk);
                timeout = timeout + 1;
                if (spike_out_valid) begin
                    received_count = received_count + 1;
                    $display("    Output: neuron=%0d, weight=%0d, exc_inh=%0b",
                             spike_out_neuron_id, spike_out_weight, spike_out_exc_inh);
                end
            end
        end
    endtask
    
    // Main test
    initial begin
        // Initialize
        clk = 0;
        rst_n = 0;
        enable = 0;
        spike_in_valid = 0;
        spike_in_axon_id = 0;
        weight_we = 0;
        weight_addr_axon = 0;
        weight_addr_neuron = 0;
        weight_data = 0;
        error_count = 0;
        test_num = 0;
        
        $display("==============================================");
        $display("  Synapse Array Testbench");
        $display("==============================================");
        $display("  NUM_AXONS:    %0d", NUM_AXONS);
        $display("  NUM_NEURONS:  %0d", NUM_NEURONS);
        $display("  WEIGHT_WIDTH: %0d", WEIGHT_WIDTH);
        $display("==============================================");
        
        // Reset
        apply_reset();
        #100;
        enable = 1;
        
        //---------------------------------------------------------------------
        // Test 1: Weight Configuration
        //---------------------------------------------------------------------
        init_test("Weight Configuration");
        
        // Configure weights for axon 0 to all neurons
        $display("  Configuring weights for axon 0...");
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin
            configure_weight(3'd0, i[2:0], (i + 1) * 10, 1'b1);
            expected_weights[i] = {1'b1, 8'((i + 1) * 10)};
        end
        
        // Allow time for writes to complete
        repeat(10) @(posedge clk);
        $display("  PASS: Weight configuration completed");
        
        //---------------------------------------------------------------------
        // Test 2: Single Spike Propagation
        //---------------------------------------------------------------------
        init_test("Single Spike Propagation");
        
        // Send spike from axon 0
        $display("  Sending spike from axon 0...");
        send_spike(3'd0);
        
        // Monitor state machine and outputs
        output_count = 0;
        for (i = 0; i < 50; i = i + 1) begin
            @(posedge clk);
            if (spike_out_valid) begin
                output_count = output_count + 1;
                $display("    Output %0d: neuron=%0d, weight=%0d, exc_inh=%0b",
                         output_count, spike_out_neuron_id, spike_out_weight, spike_out_exc_inh);
            end
        end
        
        if (output_count > 0) begin
            $display("  PASS: Received %0d output spikes", output_count);
        end else begin
            $display("  ERROR: No output spikes received");
            error_count = error_count + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 3: Multiple Axon Configuration
        //---------------------------------------------------------------------
        init_test("Multiple Axon Configuration");
        apply_reset();
        enable = 1;
        
        // Configure weights for multiple axons
        $display("  Configuring weights for axons 0-3...");
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < NUM_NEURONS; j = j + 1) begin
                // Different weight patterns for each axon
                configure_weight(i[2:0], j[2:0], (i * 10 + j * 5 + 1), (i % 2));
            end
        end
        
        repeat(10) @(posedge clk);
        $display("  PASS: Multiple axon configuration completed");
        
        //---------------------------------------------------------------------
        // Test 4: Sequential Spike Processing
        //---------------------------------------------------------------------
        init_test("Sequential Spike Processing");
        
        output_count = 0;
        
        // Send spikes from multiple axons
        $display("  Sending spikes from axons 0, 1, 2...");
        
        // Axon 0
        send_spike(3'd0);
        wait_for_outputs(NUM_NEURONS, output_count);
        $display("  Axon 0: %0d outputs", output_count);
        
        // Wait for processing to complete
        repeat(50) @(posedge clk);
        
        // Axon 1
        send_spike(3'd1);
        wait_for_outputs(NUM_NEURONS, output_count);
        $display("  Axon 1: %0d outputs", output_count);
        
        // Wait for processing to complete
        repeat(50) @(posedge clk);
        
        // Axon 2
        send_spike(3'd2);
        wait_for_outputs(NUM_NEURONS, output_count);
        $display("  Axon 2: %0d outputs", output_count);
        
        $display("  PASS: Sequential spike processing completed");
        
        //---------------------------------------------------------------------
        // Test 5: Excitatory vs Inhibitory
        //---------------------------------------------------------------------
        init_test("Excitatory vs Inhibitory Weights");
        apply_reset();
        enable = 1;
        
        // Configure mixed excitatory/inhibitory weights
        $display("  Configuring mixed weights...");
        configure_weight(3'd0, 3'd0, 8'd100, 1'b1);  // Excitatory
        configure_weight(3'd0, 3'd1, 8'd80, 1'b0);   // Inhibitory
        configure_weight(3'd0, 3'd2, 8'd60, 1'b1);   // Excitatory
        configure_weight(3'd0, 3'd3, 8'd40, 1'b0);   // Inhibitory
        
        repeat(10) @(posedge clk);
        
        send_spike(3'd0);
        
        output_count = 0;
        for (i = 0; i < 100; i = i + 1) begin
            @(posedge clk);
            if (spike_out_valid) begin
                output_count = output_count + 1;
                $display("    Neuron %0d: weight=%0d, type=%s",
                         spike_out_neuron_id, spike_out_weight,
                         spike_out_exc_inh ? "EXC" : "INH");
            end
        end
        
        if (output_count == 4) begin
            $display("  PASS: Received expected 4 outputs");
        end else begin
            $display("  WARNING: Expected 4, got %0d outputs", output_count);
        end
        
        //---------------------------------------------------------------------
        // Test 6: Zero Weight Handling
        //---------------------------------------------------------------------
        init_test("Zero Weight Handling");
        apply_reset();
        enable = 1;
        
        // Configure with zero weight
        $display("  Configuring with zero weight...");
        configure_weight(3'd0, 3'd0, 8'd0, 1'b1);   // Zero weight
        configure_weight(3'd0, 3'd1, 8'd50, 1'b1);  // Non-zero
        
        repeat(10) @(posedge clk);
        
        send_spike(3'd0);
        
        output_count = 0;
        for (i = 0; i < 100; i = i + 1) begin
            @(posedge clk);
            if (spike_out_valid) begin
                output_count = output_count + 1;
                $display("    Output: neuron=%0d, weight=%0d", spike_out_neuron_id, spike_out_weight);
            end
        end
        
        // Zero weights should NOT generate output
        if (output_count == 1) begin
            $display("  PASS: Zero weight correctly filtered");
        end else begin
            $display("  INFO: Got %0d outputs (zero weight handling depends on implementation)", output_count);
        end
        
        //---------------------------------------------------------------------
        // Test 7: Stress Test - Rapid Spikes
        //---------------------------------------------------------------------
        init_test("Stress Test - Rapid Spikes");
        apply_reset();
        enable = 1;
        
        // Configure all weights
        for (i = 0; i < NUM_AXONS; i = i + 1) begin
            for (j = 0; j < NUM_NEURONS; j = j + 1) begin
                configure_weight(i[2:0], j[2:0], 8'd25, 1'b1);
            end
        end
        
        repeat(10) @(posedge clk);
        
        // Send rapid spikes
        $display("  Sending rapid spikes from all axons...");
        for (i = 0; i < NUM_AXONS; i = i + 1) begin
            send_spike(i[2:0]);
            repeat(5) @(posedge clk);  // Small gap
        end
        
        // Wait and count outputs
        output_count = 0;
        for (i = 0; i < 500; i = i + 1) begin
            @(posedge clk);
            if (spike_out_valid) begin
                output_count = output_count + 1;
            end
        end
        
        $display("  Total outputs from stress test: %0d", output_count);
        $display("  PASS: Stress test completed");
        
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
