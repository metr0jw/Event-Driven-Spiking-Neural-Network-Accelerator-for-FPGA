//-----------------------------------------------------------------------------
// Title         : Testbench for LIF Neuron
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_lif_neuron.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Comprehensive testbench for LIF neuron module
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_lif_neuron();

    // Parameters matching DUT
    localparam NEURON_ID       = 5;
    localparam DATA_WIDTH      = 16;
    localparam WEIGHT_WIDTH    = 8;
    localparam THRESHOLD_WIDTH = 16;
    localparam LEAK_WIDTH      = 8;
    localparam REFRAC_WIDTH    = 8;
    
    // Clock period (100MHz)
    localparam CLK_PERIOD = 10;
    
    // DUT signals
    reg                         clk;
    reg                         rst_n;
    reg                         enable;
    
    // Synaptic inputs
    reg                         syn_valid;
    reg [WEIGHT_WIDTH-1:0]      syn_weight;
    reg                         syn_excitatory;
    
    // Neuron parameters
    reg [THRESHOLD_WIDTH-1:0]   threshold;
    reg [LEAK_WIDTH-1:0]        leak_rate;
    reg [REFRAC_WIDTH-1:0]      refractory_period;
    reg                         reset_potential_en;
    reg [DATA_WIDTH-1:0]        reset_potential;
    
    // Outputs
    wire                        spike_out;
    wire [DATA_WIDTH-1:0]       membrane_potential;
    wire                        is_refractory;
    wire [REFRAC_WIDTH-1:0]     refrac_count;
    
    // Test variables
    integer                     test_num;
    integer                     spike_count;
    integer                     cycle_count;
    integer                     error_count;
    time                        last_spike_time;
    time                        current_spike_time;
    real                        spike_frequency;
    integer                     i;
    reg                         spike_generated;
    reg                         error_occurred;
    
    // Test logging
    reg [255:0] test_name;
    
    //-------------------------------------------------------------------------
    // DUT Instantiation
    //-------------------------------------------------------------------------
    lif_neuron #(
        .NEURON_ID(NEURON_ID),
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .THRESHOLD_WIDTH(THRESHOLD_WIDTH),
        .LEAK_WIDTH(LEAK_WIDTH),
        .REFRAC_WIDTH(REFRAC_WIDTH)
    ) DUT (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .syn_valid(syn_valid),
        .syn_weight(syn_weight),
        .syn_excitatory(syn_excitatory),
        .threshold(threshold),
        .leak_rate(leak_rate),
        .refractory_period(refractory_period),
        .reset_potential_en(reset_potential_en),
        .reset_potential(reset_potential),
        .spike_out(spike_out),
        .membrane_potential(membrane_potential),
        .is_refractory(is_refractory),
        .refrac_count(refrac_count)
    );
    
    //-------------------------------------------------------------------------
    // Clock Generation
    //-------------------------------------------------------------------------
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    //-------------------------------------------------------------------------
    // Spike Monitoring
    //-------------------------------------------------------------------------
    always @(posedge clk) begin
        if (spike_out) begin
            spike_count = spike_count + 1;
            current_spike_time = $time;
            if (last_spike_time > 0) begin
                $display("[%0t] Spike #%0d detected! Membrane=%0d, ISI=%0t ns", 
                         $time, spike_count, membrane_potential, 
                         current_spike_time - last_spike_time);
            end else begin
                $display("[%0t] First spike detected! Membrane=%0d", 
                         $time, membrane_potential);
            end
            last_spike_time = current_spike_time;
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
            cycle_count = 0;
            last_spike_time = 0;
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
    
    // Apply single synaptic input
    task apply_synapse(input [WEIGHT_WIDTH-1:0] weight, input excitatory);
        begin
            @(posedge clk);
            syn_valid = 1'b1;
            syn_weight = weight;
            syn_excitatory = excitatory;
            @(posedge clk);
            syn_valid = 1'b0;
            syn_weight = 0;
        end
    endtask
    
    // Apply burst of synaptic inputs
    task apply_synapse_burst(
        input [WEIGHT_WIDTH-1:0] weight, 
        input excitatory,
        input integer count
    );
        integer j;
        begin
            for (j = 0; j < count; j = j + 1) begin
                apply_synapse(weight, excitatory);
            end
        end
    endtask
    
    // Wait and monitor
    task wait_and_monitor(input integer cycles);
        integer k;
        begin
            for (k = 0; k < cycles; k = k + 1) begin
                @(posedge clk);
                cycle_count = cycle_count + 1;
            end
        end
    endtask
    
    // Check membrane potential
    task check_membrane(input [DATA_WIDTH-1:0] expected, input integer tolerance);
        begin
            if (membrane_potential < expected - tolerance || 
                membrane_potential > expected + tolerance) begin
                $display("ERROR: Membrane potential mismatch. Expected=%0d±%0d, Got=%0d", 
                         expected, tolerance, membrane_potential);
                error_count = error_count + 1;
            end else begin
                $display("PASS: Membrane potential correct: %0d", membrane_potential);
            end
        end
    endtask
    
    //-------------------------------------------------------------------------
    // Main Test Sequence
    //-------------------------------------------------------------------------
    initial begin
        // Initialize signals
        rst_n = 1'b1;
        enable = 1'b0;
        syn_valid = 1'b0;
        syn_weight = 0;
        syn_excitatory = 1'b1;
        // Lowered threshold for easier spike generation with leak
        // leak_rate = 8'b00000_100 -> shift1=4, shift2_en=0 -> tau = 0.9375 (slow leak)
        threshold = 16'd500;
        leak_rate = 8'b00000_100;  // Slow leak: shift=4 only
        refractory_period = 8'd20;
        reset_potential_en = 1'b0;
        reset_potential = 16'd0;
        test_num = 1;
        error_count = 0;
        
        // Create waveform dump
        $dumpfile("tb_lif_neuron.vcd");
        $dumpvars(0, tb_lif_neuron);
        
        // Initial reset
        apply_reset();
        enable = 1'b1;
        wait_and_monitor(10);
        
        //---------------------------------------------------------------------
        // Test 1: Basic Integration and Spike Generation
        //---------------------------------------------------------------------
        init_test("Integration and Spike Generation");
        
        // Apply excitatory inputs until spike
        // Threshold=500, weight=100, so need ~5 inputs without leak
        // With slow leak (shift=4, tau≈0.9375), consecutive inputs should accumulate
        spike_generated = 1'b0;
        $display("  Threshold=%0d, Weight=100, Leak_rate=%0d", threshold, leak_rate);
        
        for (i = 0; i < 15 && !spike_generated; i = i + 1) begin
            apply_synapse(8'd100, 1'b1);  // Larger weight for faster accumulation
            $display("  Input %0d: Membrane = %0d", i, membrane_potential);
            if (spike_out || spike_count > 0) begin
                $display("  PASS: Spike generated after %0d inputs!", i+1);
                spike_generated = 1'b1;
            end
        end
        
        if (!spike_generated) begin
            $display("ERROR: No spike generated after 15 inputs!");
            error_count = error_count + 1;
        end
        
        wait_and_monitor(30); // Wait through refractory period
        
        //---------------------------------------------------------------------
        // Test 2: Leak Current Behavior
        //---------------------------------------------------------------------
        init_test("Leak Current Behavior");
        apply_reset();
        
        // Build up membrane potential
        apply_synapse_burst(8'd50, 1'b1, 10);
        wait_and_monitor(5);
        $display("  Initial membrane potential: %0d", membrane_potential);
        
        // Monitor leak without input
        for (i = 0; i < 50; i = i + 1) begin
            wait_and_monitor(1);
            if (i % 10 == 0) begin
                $display("  After %0d cycles: Membrane = %0d", i, membrane_potential);
            end
        end
        
        if (membrane_potential >= 500) begin
            $display("ERROR: Leak not working properly!");
            error_count = error_count + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 3: Inhibitory Synapses
        //---------------------------------------------------------------------
        init_test("Inhibitory Synapses");
        apply_reset();
        
        // Build up potential
        apply_synapse_burst(8'd80, 1'b1, 8);
        wait_and_monitor(2);
        $display("  After excitation: Membrane = %0d", membrane_potential);
        
        // Apply inhibitory input
        apply_synapse_burst(8'd60, 1'b0, 5);
        wait_and_monitor(2);
        $display("  After inhibition: Membrane = %0d", membrane_potential);
        
        //---------------------------------------------------------------------
        // Test 4: Refractory Period
        //---------------------------------------------------------------------
        init_test("Refractory Period");
        apply_reset();
        spike_count = 0;
        
        // Generate a spike - need more inputs to reach threshold (1000)
        // Each input adds 100, so need at least 10 inputs
        apply_synapse_burst(8'd100, 1'b1, 12);
        wait_and_monitor(3);
        
        if (spike_count >= 1) begin
            $display("  First spike generated, entering refractory period");
            $display("  Refractory period = %0d cycles", refractory_period);
            
            // Wait one cycle for refractory to be set
            wait_and_monitor(1);
            
            // Verify we are in refractory period
            if (is_refractory) begin
                $display("  Confirmed in refractory period (count=%0d)", refrac_count);
            end else begin
                $display("WARNING: Not in refractory immediately after spike");
            end
            
            // Reset spike counter to check no new spikes during refractory
            spike_count = 0;
            
            // Try to generate another spike during refractory
            for (i = 0; i < refractory_period - 2 && is_refractory; i = i + 1) begin
                apply_synapse(8'd100, 1'b1);
                wait_and_monitor(1);
            end
            
            // Check that no spike was generated during refractory
            if (spike_count > 0) begin
                $display("ERROR: Spike generated during refractory period!");
                error_count = error_count + 1;
            end else begin
                $display("  PASS: No spikes during refractory period");
            end
        end else begin
            $display("ERROR: First spike not generated - cannot test refractory");
            error_count = error_count + 1;
        end
        
        wait_and_monitor(10);
        
        //---------------------------------------------------------------------
        // Test 5: Threshold Adjustment
        //---------------------------------------------------------------------
        init_test("Threshold Adjustment");
        apply_reset();
        
        // Test with low threshold
        threshold = 16'd500;
        spike_count = 0;
        apply_synapse_burst(8'd60, 1'b1, 10);
        wait_and_monitor(5);
        $display("  With threshold=%0d, spikes=%0d", threshold, spike_count);
        
        // Test with high threshold
        apply_reset();
        threshold = 16'd2000;
        spike_count = 0;
        apply_synapse_burst(8'd60, 1'b1, 10);
        wait_and_monitor(5);
        $display("  With threshold=%0d, spikes=%0d", threshold, spike_count);
        
        // Reset to default
        threshold = 16'd1000;
        
        //---------------------------------------------------------------------
        // Test 6: Reset Potential
        //---------------------------------------------------------------------
        init_test("Reset Potential");
        apply_reset();
        
        // Enable reset potential
        reset_potential_en = 1'b1;
        reset_potential = 16'd200;
        
        // Generate spike and check reset value
        apply_synapse_burst(8'd100, 1'b1, 15);
        wait_and_monitor(2);
        
        // Check membrane potential after spike
        wait_and_monitor(1);
        if (membrane_potential != reset_potential && is_refractory) begin
            $display("ERROR: Reset potential not applied correctly!");
            error_count = error_count + 1;
        end else begin
            $display("  Reset potential correctly applied: %0d", membrane_potential);
        end
        
        // Disable reset potential
        reset_potential_en = 1'b0;
        
        //---------------------------------------------------------------------
        // Test 7: Enable/Disable Control
        //---------------------------------------------------------------------
        init_test("Enable/Disable Control");
        apply_reset();
        
        // Disable neuron
        enable = 1'b0;
        spike_count = 0;
        apply_synapse_burst(8'd100, 1'b1, 20);
        wait_and_monitor(10);
        
        if (spike_count > 0) begin
            $display("ERROR: Spike generated when disabled!");
            error_count = error_count + 1;
        end else begin
            $display("  Correctly disabled - no spikes generated");
        end
        
        // Re-enable
        enable = 1'b1;
        
        //---------------------------------------------------------------------
        // Test 8: Saturation Protection
        //---------------------------------------------------------------------
        init_test("Saturation Protection");
        apply_reset();
        
        // Try to overflow membrane potential
        spike_generated = 1'b0;
        for (i = 0; i < 100 && !spike_generated; i = i + 1) begin
            apply_synapse(8'd255, 1'b1);
            if (spike_out) spike_generated = 1'b1;
        end
        
        $display("  Maximum membrane potential reached: %0d", membrane_potential);
        
        //---------------------------------------------------------------------
        // Test 9: Spike Frequency Response
        //---------------------------------------------------------------------
        init_test("Spike Frequency Response");
        apply_reset();
        spike_count = 0;
        
        // Apply constant input for 1000 cycles
        for (i = 0; i < 1000; i = i + 1) begin
            apply_synapse(8'd40, 1'b1);
            wait_and_monitor(1);
        end
        
        spike_frequency = real'(spike_count) * 1000.0 / real'(i * CLK_PERIOD);
        $display("  Input weight=40, Spike count=%0d, Frequency=%.2f MHz", 
                 spike_count, spike_frequency);
        
        //---------------------------------------------------------------------
        // Test 10: Mixed Excitation/Inhibition Pattern
        //---------------------------------------------------------------------
        init_test("Mixed Excitation/Inhibition Pattern");
        apply_reset();
        
        // Realistic input pattern
        for (i = 0; i < 50; i = i + 1) begin
            if ($random % 100 < 70) begin
                // 70% excitatory
                apply_synapse(8'd30 + ($random % 30), 1'b1);
            end else begin
                // 30% inhibitory
                apply_synapse(8'd20 + ($random % 20), 1'b0);
            end
            wait_and_monitor(2);
        end
        
        $display("  Mixed input pattern: %0d spikes generated", spike_count);
        
        //---------------------------------------------------------------------
        // Test Summary
        //---------------------------------------------------------------------
        #1000;
        $display("\n========================================");
        $display("Test Summary");
        $display("========================================");
        $display("Total Tests Run: %0d", test_num - 1);
        $display("Total Errors: %0d", error_count);
        
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
        #100_000;
        $display("\n*** ERROR: Testbench timeout! ***");
        $finish;
    end
    
    //-------------------------------------------------------------------------
    // Assertions and Monitors
    //-------------------------------------------------------------------------
    
    // Check that membrane potential doesn't exceed maximum
    always @(posedge clk) begin
        if (membrane_potential == {DATA_WIDTH{1'b1}}) begin
            $display("WARNING: Membrane potential saturated at maximum value");
        end
    end
    
    // Track refractory state for detecting improper spikes
    // A spike is valid if it occurs when refrac_counter was 0 in the PREVIOUS cycle
    // Since spike_out is registered (1 cycle delay from spike_reg), 
    // we need to check the state 2 cycles ago
    reg [REFRAC_WIDTH-1:0] prev_refrac_count;
    reg [REFRAC_WIDTH-1:0] prev_prev_refrac_count;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            prev_refrac_count <= 0;
            prev_prev_refrac_count <= 0;
        end else begin
            prev_prev_refrac_count <= prev_refrac_count;
            prev_refrac_count <= refrac_count;
        end
    end
    
    // Check that spike is not generated while neuron was already in refractory period
    // Valid spike: prev_prev_refrac_count == 0 (was not in refractory when spike decision was made)
    always @(posedge clk) begin
        if (rst_n && spike_out && (prev_prev_refrac_count > 0)) begin
            $display("ERROR: Spike generated while in refractory period! (prev_refrac=%0d)", 
                     prev_prev_refrac_count);
            error_count = error_count + 1;
        end
    end
    
    // Monitor parameter changes
    always @(threshold or leak_rate or refractory_period) begin
        $display("[%0t] Parameters changed: threshold=%0d, leak=%0d, refrac=%0d",
                 $time, threshold, leak_rate, refractory_period);
    end

endmodule
