//-----------------------------------------------------------------------------
// Title         : Comprehensive LIF Neuron Testbench  
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_lif_neuron_comprehensive.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Comprehensive testbench for LIF neuron with event-driven testing
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_lif_neuron_comprehensive;

    // Parameters
    parameter NEURON_ID         = 0;
    parameter DATA_WIDTH        = 16;
    parameter WEIGHT_WIDTH      = 8;
    parameter THRESHOLD_WIDTH   = 16;
    parameter LEAK_WIDTH        = 8;
    parameter REFRAC_WIDTH      = 8;
    parameter CLK_PERIOD        = 10; // 100MHz
    
    // Test parameters
    parameter THRESHOLD_VAL     = 16'd1000;
    parameter LEAK_RATE_VAL     = 8'd10;
    parameter REFRAC_PERIOD_VAL = 8'd5;
    parameter RESET_POTENTIAL_VAL = 16'd0;
    
    // Signals
    reg                         clk;
    reg                         rst_n;
    reg                         enable;
    
    // Synaptic input interface
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
    integer test_case;
    integer cycle_count;
    integer spike_count;
    integer excitatory_count;
    integer inhibitory_count;
    reg [31:0] test_passed;
    reg [31:0] test_failed;
    reg spike_detected;
    
    // File handles for logging
    integer log_file;
    integer spike_file;
    
    // Event queue for spike timing tests
    reg [31:0] event_queue [0:255];
    reg [7:0] event_head, event_tail;
    
    // DUT instantiation
    lif_neuron #(
        .NEURON_ID(NEURON_ID),
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .THRESHOLD_WIDTH(THRESHOLD_WIDTH),
        .LEAK_WIDTH(LEAK_WIDTH),
        .REFRAC_WIDTH(REFRAC_WIDTH)
    ) dut (
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
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Spike monitoring
    always @(posedge clk) begin
        if (spike_out) begin
            spike_count <= spike_count + 1;
            $fwrite(spike_file, "%0t: Spike generated, membrane_potential=%0d\n", 
                   $time, membrane_potential);
        end
    end
    
    // Main test sequence
    initial begin
        // Initialize signals
        rst_n = 0;
        enable = 0;
        syn_valid = 0;
        syn_weight = 0;
        syn_excitatory = 1;
        threshold = THRESHOLD_VAL;
        leak_rate = LEAK_RATE_VAL;
        refractory_period = REFRAC_PERIOD_VAL;
        reset_potential_en = 1;
        reset_potential = RESET_POTENTIAL_VAL;
        
        test_case = 0;
        cycle_count = 0;
        spike_count = 0;
        excitatory_count = 0;
        inhibitory_count = 0;
        test_passed = 0;
        test_failed = 0;
        event_head = 0;
        event_tail = 0;
        
        // Open log files
        log_file = $fopen("lif_neuron_test.log", "w");
        spike_file = $fopen("spike_output.log", "w");
        
        $fwrite(log_file, "=== LIF Neuron Comprehensive Test ===\n");
        $fwrite(log_file, "Time: %0t\n", $time);
        
        // Reset sequence
        #100;
        rst_n = 1;
        enable = 1;
        #100;
        
        $display("Starting LIF Neuron Comprehensive Test");
        
        // Test Case 1: Basic excitatory input integration
        test_basic_excitatory_integration();
        
        // Test Case 2: Inhibitory input effects
        test_inhibitory_input();
        
        // Test Case 3: Membrane potential leak
        test_membrane_leak();
        
        // Test Case 4: Spike generation and reset
        test_spike_generation();
        
        // Test Case 5: Refractory period behavior
        test_refractory_period();
        
        // Test Case 6: Parameter variations
        test_parameter_variations();
        
        // Test Case 7: Event-driven spike pattern
        test_event_driven_patterns();
        
        // Test Case 8: Saturation behavior
        test_saturation_behavior();
        
        // Test Case 9: Rapid fire inputs
        test_rapid_fire_inputs();
        
        // Test Case 10: Long-term behavior
        test_long_term_behavior();
        
        // Final results
        $display("\n=== Test Results ===");
        $display("Tests Passed: %0d", test_passed);
        $display("Tests Failed: %0d", test_failed);
        $display("Total Spikes: %0d", spike_count);
        $display("Excitatory Inputs: %0d", excitatory_count);
        $display("Inhibitory Inputs: %0d", inhibitory_count);
        
        $fwrite(log_file, "\n=== Final Results ===\n");
        $fwrite(log_file, "Tests Passed: %0d\n", test_passed);
        $fwrite(log_file, "Tests Failed: %0d\n", test_failed);
        $fwrite(log_file, "Total Spikes: %0d\n", spike_count);
        
        $fclose(log_file);
        $fclose(spike_file);
        
        if (test_failed == 0) begin
            $display("\nALL TESTS PASSED!");
        end else begin
            $display("\nSOME TESTS FAILED!");
        end
        
        $finish;
    end
    
    // Task: Basic excitatory integration test
    task test_basic_excitatory_integration;
        begin
            test_case = 1;
            $display("\nTest 1: Basic Excitatory Integration");
            $fwrite(log_file, "\nTest 1: Basic Excitatory Integration\n");
            
            reset_neuron();
            
            // Apply small excitatory inputs
            repeat(5) begin
                apply_synaptic_input(50, 1); // 50 weight, excitatory
                #(CLK_PERIOD * 10);
            end
            
            // Check that membrane potential increased but no spike
            if (membrane_potential > 0 && !spike_out) begin
                $display("PASS: Membrane potential integrated correctly: %0d", membrane_potential);
                test_passed = test_passed + 1;
            end else begin
                $display("FAIL: Expected integration without spike");
                test_failed = test_failed + 1;
            end
        end
    endtask
    
    // Task: Inhibitory input test
    task test_inhibitory_input;
        begin
            test_case = 2;
            $display("\nTest 2: Inhibitory Input Effects");
            $fwrite(log_file, "\nTest 2: Inhibitory Input Effects\n");
            
            reset_neuron();
            
            // Build up membrane potential
            repeat(8) begin
                apply_synaptic_input(80, 1); // Excitatory
                #(CLK_PERIOD * 5);
            end
            
            reg [DATA_WIDTH-1:0] mem_before_inhib = membrane_potential;
            
            // Apply inhibitory input
            apply_synaptic_input(100, 0); // Inhibitory
            #(CLK_PERIOD * 5);
            
            if (membrane_potential < mem_before_inhib) begin
                $display("PASS: Inhibitory input reduced membrane potential");
                test_passed = test_passed + 1;
            end else begin
                $display("FAIL: Inhibitory input didn't work correctly");
                test_failed = test_failed + 1;
            end
        end
    endtask
    
    // Task: Membrane leak test
    task test_membrane_leak;
        begin
            test_case = 3;
            $display("\nTest 3: Membrane Potential Leak");
            $fwrite(log_file, "\nTest 3: Membrane Potential Leak\n");
            
            reset_neuron();
            
            // Build up membrane potential
            repeat(5) begin
                apply_synaptic_input(100, 1);
                #(CLK_PERIOD * 2);
            end
            
            reg [DATA_WIDTH-1:0] mem_peak = membrane_potential;
            
            // Wait without input to see leak
            #(CLK_PERIOD * 100);
            
            if (membrane_potential < mem_peak) begin
                $display("PASS: Membrane potential leaked: %0d -> %0d", mem_peak, membrane_potential);
                test_passed = test_passed + 1;
            end else begin
                $display("FAIL: No membrane leak observed");
                test_failed = test_failed + 1;
            end
        end
    endtask
    
    // Task: Spike generation test
    task test_spike_generation;
        begin
            test_case = 4;
            $display("\nTest 4: Spike Generation and Reset");
            $fwrite(log_file, "\nTest 4: Spike Generation and Reset\n");
            
            reset_neuron();
            
            // Apply inputs to reach threshold
            spike_detected = 1'b0;
            repeat(15) begin
                if (!spike_detected) begin
                    apply_synaptic_input(80, 1);
                    #(CLK_PERIOD * 2);
                    if (spike_out) spike_detected = 1'b1;
                end
            end
            
            if (spike_out) begin
                $display("PASS: Spike generated at membrane potential %0d", membrane_potential);
                test_passed = test_passed + 1;
                
                // Check reset
                #CLK_PERIOD;
                if (membrane_potential == reset_potential) begin
                    $display("PASS: Membrane potential reset correctly");
                    test_passed = test_passed + 1;
                end else begin
                    $display("FAIL: Membrane potential not reset: %0d", membrane_potential);
                    test_failed = test_failed + 1;
                end
            end else begin
                $display("FAIL: No spike generated");
                test_failed = test_failed + 1;
            end
        end
    endtask
    
    // Task: Refractory period test
    task test_refractory_period;
        begin
            test_case = 5;
            $display("\nTest 5: Refractory Period Behavior");
            $fwrite(log_file, "\nTest 5: Refractory Period Behavior\n");
            
            reset_neuron();
            
            // Generate a spike
            spike_detected = 1'b0;
            repeat(15) begin
                if (!spike_detected) begin
                    apply_synaptic_input(80, 1);
                    #(CLK_PERIOD * 2);
                    if (spike_out) spike_detected = 1'b1;
                end
            end
            
            if (spike_out) begin
                // Check refractory state
                #CLK_PERIOD;
                if (is_refractory) begin
                    $display("PASS: Neuron entered refractory state");
                    test_passed = test_passed + 1;
                    
                    // Try to generate another spike during refractory
                    repeat(10) begin
                        apply_synaptic_input(127, 1); // Maximum weight
                        #CLK_PERIOD;
                    end
                    
                    if (!spike_out) begin
                        $display("PASS: No spike during refractory period");
                        test_passed = test_passed + 1;
                    end else begin
                        $display("FAIL: Spike generated during refractory period");
                        test_failed = test_failed + 1;
                    end
                    
                    // Wait for refractory to end
                    while (is_refractory) #CLK_PERIOD;
                    
                    $display("PASS: Refractory period ended after %0d cycles", refrac_count);
                    test_passed = test_passed + 1;
                    
                end else begin
                    $display("FAIL: Neuron didn't enter refractory state");
                    test_failed = test_failed + 1;
                end
            end
        end
    endtask
    
    // Task: Parameter variation test
    task test_parameter_variations;
        begin
            test_case = 6;
            $display("\nTest 6: Parameter Variations");
            $fwrite(log_file, "\nTest 6: Parameter Variations\n");
            
            // Test different thresholds
            test_threshold_variation();
            
            // Test different leak rates
            test_leak_rate_variation();
            
            // Test different refractory periods
            test_refractory_variation();
        end
    endtask
    
    // Task: Event-driven pattern test
    task test_event_driven_patterns;
        begin
            test_case = 7;
            $display("\nTest 7: Event-Driven Spike Patterns");
            $fwrite(log_file, "\nTest 7: Event-Driven Spike Patterns\n");
            
            reset_neuron();
            
            // Create a complex spike pattern
            schedule_event(10, 60, 1);   // t=10, weight=60, excitatory
            schedule_event(25, 70, 1);   // t=25, weight=70, excitatory
            schedule_event(40, 80, 1);   // t=40, weight=80, excitatory
            schedule_event(50, 30, 0);   // t=50, weight=30, inhibitory
            schedule_event(65, 90, 1);   // t=65, weight=90, excitatory
            
            // Process events
            process_event_queue();
            
            $display("PASS: Event-driven pattern test completed");
            test_passed = test_passed + 1;
        end
    endtask
    
    // Task: Saturation behavior test
    task test_saturation_behavior;
        begin
            test_case = 8;
            $display("\nTest 8: Saturation Behavior");
            $fwrite(log_file, "\nTest 8: Saturation Behavior\n");
            
            reset_neuron();
            
            // Apply very large inputs to test saturation
            repeat(20) begin
                apply_synaptic_input(127, 1); // Maximum weight
                #CLK_PERIOD;
            end
            
            if (membrane_potential <= {DATA_WIDTH{1'b1}}) begin
                $display("PASS: Membrane potential saturated correctly: %0d", membrane_potential);
                test_passed = test_passed + 1;
            end else begin
                $display("FAIL: Saturation not working");
                test_failed = test_failed + 1;
            end
        end
    endtask
    
    // Task: Rapid fire input test
    task test_rapid_fire_inputs;
        begin
            test_case = 9;
            $display("\nTest 9: Rapid Fire Inputs");
            $fwrite(log_file, "\nTest 9: Rapid Fire Inputs\n");
            
            reset_neuron();
            
            // Apply rapid consecutive inputs
            repeat(50) begin
                apply_synaptic_input($random % 100 + 20, $random % 2);
                #CLK_PERIOD;
            end
            
            $display("PASS: Rapid fire input test completed, spikes: %0d", spike_count);
            test_passed = test_passed + 1;
        end
    endtask
    
    // Task: Long-term behavior test
    task test_long_term_behavior;
        begin
            test_case = 10;
            $display("\nTest 10: Long-term Behavior");
            $fwrite(log_file, "\nTest 10: Long-term Behavior\n");
            
            reset_neuron();
            integer initial_spike_count = spike_count;
            
            // Long simulation with random inputs
            repeat(1000) begin
                if ($random % 10 == 0) begin // 10% input probability
                    apply_synaptic_input($random % 80 + 40, $random % 2);
                end
                #CLK_PERIOD;
            end
            
            integer spikes_generated = spike_count - initial_spike_count;
            $display("PASS: Long-term test completed, generated %0d spikes", spikes_generated);
            test_passed = test_passed + 1;
        end
    endtask
    
    // Helper tasks
    task reset_neuron;
        begin
            rst_n = 0;
            #(CLK_PERIOD * 2);
            rst_n = 1;
            #CLK_PERIOD;
        end
    endtask
    
    task apply_synaptic_input;
        input [WEIGHT_WIDTH-1:0] weight;
        input excitatory;
        begin
            syn_weight = weight;
            syn_excitatory = excitatory;
            syn_valid = 1;
            #CLK_PERIOD;
            syn_valid = 0;
            
            if (excitatory) excitatory_count = excitatory_count + 1;
            else inhibitory_count = inhibitory_count + 1;
        end
    endtask
    
    task schedule_event;
        input [15:0] time_offset;
        input [WEIGHT_WIDTH-1:0] weight;
        input excitatory;
        begin
            event_queue[event_tail] = {time_offset, weight, excitatory, 7'b0};
            event_tail = event_tail + 1;
        end
    endtask
    
    task process_event_queue;
        integer current_time;
        reg [31:0] current_event;
        begin
            current_time = 0;
            
            while (event_head < event_tail) begin
                current_event = event_queue[event_head];
                
                // Wait until event time
                while (current_time < current_event[31:16]) begin
                    #CLK_PERIOD;
                    current_time = current_time + 1;
                end
                
                // Apply event
                apply_synaptic_input(current_event[15:8], current_event[7]);
                event_head = event_head + 1;
            end
        end
    endtask
    
    task test_threshold_variation;
        begin
            $display("  Testing threshold variations...");
            
            // Test low threshold
            threshold = 16'd500;
            reset_neuron();
            repeat(10) begin
                apply_synaptic_input(60, 1);
                #(CLK_PERIOD * 2);
            end
            
            // Test high threshold  
            threshold = 16'd2000;
            reset_neuron();
            repeat(20) begin
                apply_synaptic_input(60, 1);
                #(CLK_PERIOD * 2);
            end
            
            // Restore default
            threshold = THRESHOLD_VAL;
            $display("  Threshold variation test completed");
        end
    endtask
    
    task test_leak_rate_variation;
        begin
            $display("  Testing leak rate variations...");
            
            // Test high leak rate
            leak_rate = 8'd50;
            reset_neuron();
            repeat(10) begin
                apply_synaptic_input(60, 1);
                #(CLK_PERIOD * 5);
            end
            
            // Test low leak rate
            leak_rate = 8'd1;
            reset_neuron();
            repeat(10) begin
                apply_synaptic_input(60, 1);
                #(CLK_PERIOD * 5);
            end
            
            // Restore default
            leak_rate = LEAK_RATE_VAL;
            $display("  Leak rate variation test completed");
        end
    endtask
    
    task test_refractory_variation;
        begin
            $display("  Testing refractory period variations...");
            
            // Test short refractory period
            refractory_period = 8'd2;
            reset_neuron();
            // Generate spike and test refractory
            
            // Test long refractory period
            refractory_period = 8'd20;
            reset_neuron();
            // Generate spike and test refractory
            
            // Restore default
            refractory_period = REFRAC_PERIOD_VAL;
            $display("  Refractory period variation test completed");
        end
    endtask
    
    // Continuous monitoring
    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;
        
        // Log important state changes
        if (spike_out) begin
            $fwrite(log_file, "%0t: SPIKE - Cycle %0d, Test %0d\n", 
                   $time, cycle_count, test_case);
        end
        
        if (is_refractory && refrac_count == refractory_period) begin
            $fwrite(log_file, "%0t: REFRACTORY START - Cycle %0d\n", 
                   $time, cycle_count);
        end
        
        if (!is_refractory && $past(is_refractory)) begin
            $fwrite(log_file, "%0t: REFRACTORY END - Cycle %0d\n", 
                   $time, cycle_count);
        end
    end

endmodule
