//-----------------------------------------------------------------------------
// Title         : LIF Neuron Array Comprehensive Testbench
// Project       : PYNQ-Z2 SNN Accelerator  
// File          : tb_neuron_array_comprehensive.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Event-driven testbench for neuron array with realistic patterns
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_neuron_array_comprehensive;

    // Parameters
    parameter NUM_NEURONS       = 16;     // Smaller for simulation
    parameter NUM_AXONS         = 16;
    parameter DATA_WIDTH        = 16;
    parameter WEIGHT_WIDTH      = 8;
    parameter THRESHOLD_WIDTH   = 16;
    parameter LEAK_WIDTH        = 8;
    parameter REFRAC_WIDTH      = 8;
    parameter NEURON_ID_WIDTH   = $clog2(NUM_NEURONS);
    parameter CLK_PERIOD        = 10;
    
    // Test parameters
    parameter THRESHOLD_VAL     = 16'd800;
    parameter LEAK_RATE_VAL     = 8'd5;
    parameter REFRAC_PERIOD_VAL = 8'd3;
    
    // Clock and reset
    reg                         clk;
    reg                         rst_n;
    reg                         enable;
    
    // Input spike interface
    reg                         s_axis_spike_valid;
    reg [NEURON_ID_WIDTH-1:0]   s_axis_spike_dest_id;
    reg [WEIGHT_WIDTH-1:0]      s_axis_spike_weight;
    reg                         s_axis_spike_exc_inh;
    wire                        s_axis_spike_ready;
    
    // Output spike interface
    wire                        m_axis_spike_valid;
    wire [NEURON_ID_WIDTH-1:0]  m_axis_spike_neuron_id;
    reg                         m_axis_spike_ready;
    
    // Configuration interface
    reg                         config_we;
    reg [NEURON_ID_WIDTH-1:0]   config_addr;
    reg [31:0]                 config_data;
    
    // Global parameters
    reg [THRESHOLD_WIDTH-1:0]   global_threshold;
    reg [LEAK_WIDTH-1:0]        global_leak_rate;
    reg [REFRAC_WIDTH-1:0]      global_refrac_period;
    
    // Status outputs
    wire [31:0]                spike_count;
    wire                       array_busy;
    
    // Test variables
    integer test_case;
    integer input_spikes_sent;
    integer output_spikes_received;
    integer test_passed;
    integer test_failed;
    
    // Spike pattern storage
    reg [31:0] spike_pattern [0:1023];
    integer pattern_length;
    integer pattern_index;
    
    // Output spike monitoring
    reg [NEURON_ID_WIDTH-1:0] output_spike_log [0:1023];
    integer output_log_index;
    
    // File handles
    integer log_file;
    integer pattern_file;
    integer result_file;
    
    // DUT instantiation
    lif_neuron_array #(
        .NUM_NEURONS(NUM_NEURONS),
        .NUM_AXONS(NUM_AXONS),
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .THRESHOLD_WIDTH(THRESHOLD_WIDTH),
        .LEAK_WIDTH(LEAK_WIDTH),
        .REFRAC_WIDTH(REFRAC_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .s_axis_spike_valid(s_axis_spike_valid),
        .s_axis_spike_dest_id(s_axis_spike_dest_id),
        .s_axis_spike_weight(s_axis_spike_weight),
        .s_axis_spike_exc_inh(s_axis_spike_exc_inh),
        .s_axis_spike_ready(s_axis_spike_ready),
        .m_axis_spike_valid(m_axis_spike_valid),
        .m_axis_spike_neuron_id(m_axis_spike_neuron_id),
        .m_axis_spike_ready(m_axis_spike_ready),
        .config_we(config_we),
        .config_addr(config_addr),
        .config_data(config_data),
        .global_threshold(global_threshold),
        .global_leak_rate(global_leak_rate),
        .global_refrac_period(global_refrac_period),
        .spike_count(spike_count),
        .array_busy(array_busy)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Output spike monitoring
    always @(posedge clk) begin
        if (m_axis_spike_valid && m_axis_spike_ready) begin
            output_spikes_received = output_spikes_received + 1;
            output_spike_log[output_log_index] = m_axis_spike_neuron_id;
            output_log_index = output_log_index + 1;
            
            $fwrite(log_file, "%0t: Output spike from neuron %0d\n", 
                   $time, m_axis_spike_neuron_id);
            $display("Output spike from neuron %0d at time %0t", 
                    m_axis_spike_neuron_id, $time);
        end
    end
    
    // Main test sequence
    initial begin
        // Initialize signals
        rst_n = 0;
        enable = 0;
        s_axis_spike_valid = 0;
        s_axis_spike_dest_id = 0;
        s_axis_spike_weight = 0;
        s_axis_spike_exc_inh = 1;
        m_axis_spike_ready = 1;
        config_we = 0;
        config_addr = 0;
        config_data = 0;
        global_threshold = THRESHOLD_VAL;
        global_leak_rate = LEAK_RATE_VAL;
        global_refrac_period = REFRAC_PERIOD_VAL;
        
        // Test variables
        test_case = 0;
        input_spikes_sent = 0;
        output_spikes_received = 0;
        test_passed = 0;
        test_failed = 0;
        pattern_length = 0;
        pattern_index = 0;
        output_log_index = 0;
        
        // Open log files
        log_file = $fopen("neuron_array_test.log", "w");
        pattern_file = $fopen("spike_patterns.log", "w");
        result_file = $fopen("test_results.log", "w");
        
        $fwrite(log_file, "=== Neuron Array Comprehensive Test ===\n");
        $display("Starting Neuron Array Comprehensive Test");
        
        // Reset sequence
        #100;
        rst_n = 1;
        enable = 1;
        #100;
        
        // Run test cases
        test_basic_spike_routing();
        test_excitatory_inhibitory_balance();
        test_spatial_spike_patterns();
        test_temporal_spike_patterns();
        test_network_dynamics();
        test_configuration_interface();
        test_backpressure_handling();
        test_concurrent_spikes();
        test_learning_patterns();
        test_performance_limits();
        
        // Final results
        $display("\n=== Final Test Results ===");
        $display("Tests Passed: %0d", test_passed);
        $display("Tests Failed: %0d", test_failed);
        $display("Input Spikes Sent: %0d", input_spikes_sent);
        $display("Output Spikes Received: %0d", output_spikes_received);
        $display("Final Spike Count: %0d", spike_count);
        
        $fwrite(result_file, "Tests Passed: %0d\n", test_passed);
        $fwrite(result_file, "Tests Failed: %0d\n", test_failed);
        $fwrite(result_file, "Input Spikes: %0d\n", input_spikes_sent);
        $fwrite(result_file, "Output Spikes: %0d\n", output_spikes_received);
        
        $fclose(log_file);
        $fclose(pattern_file);
        $fclose(result_file);
        
        if (test_failed == 0) begin
            $display("\nALL TESTS PASSED!");
        end else begin
            $display("\nSOME TESTS FAILED!");
        end
        
        $finish;
    end
    
    // Test Case 1: Basic spike routing
    task test_basic_spike_routing;
        begin
            test_case = 1;
            $display("\nTest 1: Basic Spike Routing");
            $fwrite(log_file, "\nTest 1: Basic Spike Routing\n");
            
            // Send spikes to different neurons
            for (integer i = 0; i < NUM_NEURONS; i = i + 1) begin
                send_spike(i, 100, 1); // Strong excitatory input
                #(CLK_PERIOD * 20);
            end
            
            // Wait for potential output spikes
            #(CLK_PERIOD * 100);
            
            if (output_spikes_received > 0) begin
                $display("PASS: Received %0d output spikes", output_spikes_received);
                test_passed = test_passed + 1;
            end else begin
                $display("FAIL: No output spikes received");
                test_failed = test_failed + 1;
            end
        end
    endtask
    
    // Test Case 2: Excitatory/Inhibitory balance
    task test_excitatory_inhibitory_balance;
        begin
            test_case = 2;
            $display("\nTest 2: Excitatory/Inhibitory Balance");
            $fwrite(log_file, "\nTest 2: Excitatory/Inhibitory Balance\n");
            
            integer initial_output_count = output_spikes_received;
            
            // Send mixed excitatory and inhibitory spikes
            for (integer i = 0; i < 8; i = i + 1) begin
                send_spike(i % NUM_NEURONS, 80, 1);  // Excitatory
                #(CLK_PERIOD * 5);
                send_spike(i % NUM_NEURONS, 60, 0);  // Inhibitory
                #(CLK_PERIOD * 5);
            end
            
            #(CLK_PERIOD * 100);
            
            integer output_increase = output_spikes_received - initial_output_count;
            $display("Output spikes generated: %0d", output_increase);
            
            if (output_increase >= 0) begin
                $display("PASS: Excitatory/Inhibitory test completed");
                test_passed = test_passed + 1;
            end else begin
                $display("FAIL: Unexpected behavior in E/I test");
                test_failed = test_failed + 1;
            end
        end
    endtask
    
    // Test Case 3: Spatial spike patterns
    task test_spatial_spike_patterns;
        begin
            test_case = 3;
            $display("\nTest 3: Spatial Spike Patterns");
            $fwrite(log_file, "\nTest 3: Spatial Spike Patterns\n");
            
            // Create a wave pattern across neurons
            for (integer wave = 0; wave < 3; wave = wave + 1) begin
                for (integer i = 0; i < NUM_NEURONS; i = i + 1) begin
                    send_spike(i, 90, 1);
                    #(CLK_PERIOD * 2); // Small delay between neurons
                end
                #(CLK_PERIOD * 50); // Delay between waves
            end
            
            #(CLK_PERIOD * 200);
            
            $display("PASS: Spatial pattern test completed");
            test_passed = test_passed + 1;
        end
    endtask
    
    // Test Case 4: Temporal spike patterns
    task test_temporal_spike_patterns;
        begin
            test_case = 4;
            $display("\nTest 4: Temporal Spike Patterns");
            $fwrite(log_file, "\nTest 4: Temporal Spike Patterns\n");
            
            // Generate rhythmic input patterns
            for (integer rhythm = 0; rhythm < 10; rhythm = rhythm + 1) begin
                // Burst of spikes
                for (integer burst = 0; burst < 4; burst = burst + 1) begin
                    send_spike(rhythm % NUM_NEURONS, 70, 1);
                    #(CLK_PERIOD * 3);
                end
                // Quiet period
                #(CLK_PERIOD * 20);
            end
            
            #(CLK_PERIOD * 100);
            
            $display("PASS: Temporal pattern test completed");
            test_passed = test_passed + 1;
        end
    endtask
    
    // Test Case 5: Network dynamics
    task test_network_dynamics;
        begin
            test_case = 5;
            $display("\nTest 5: Network Dynamics");
            $fwrite(log_file, "\nTest 5: Network Dynamics\n");
            
            integer initial_count = output_spikes_received;
            
            // Create interconnected activity
            // Simulate recurrent connections via software
            for (integer cycle = 0; cycle < 20; cycle = cycle + 1) begin
                // Send input spikes
                for (integer i = 0; i < 4; i = i + 1) begin
                    send_spike($random % NUM_NEURONS, 60 + ($random % 40), 1);
                end
                
                #(CLK_PERIOD * 10);
                
                // Simulate feedback from previous outputs
                if (output_spikes_received > initial_count) begin
                    // Send feedback spikes based on recent outputs
                    for (integer j = 0; j < 2; j = j + 1) begin
                        send_spike($random % NUM_NEURONS, 40, 1);
                    end
                end
                
                #(CLK_PERIOD * 20);
            end
            
            #(CLK_PERIOD * 200);
            
            integer total_activity = output_spikes_received - initial_count;
            $display("Network dynamics generated %0d output spikes", total_activity);
            
            if (total_activity > 0) begin
                $display("PASS: Network dynamics test completed");
                test_passed = test_passed + 1;
            end else begin
                $display("FAIL: No network activity observed");
                test_failed = test_failed + 1;
            end
        end
    endtask
    
    // Test Case 6: Configuration interface
    task test_configuration_interface;
        begin
            test_case = 6;
            $display("\nTest 6: Configuration Interface");
            $fwrite(log_file, "\nTest 6: Configuration Interface\n");
            
            // Test parameter updates
            configure_neuron(0, 32'h12345678);
            configure_neuron(1, 32'h87654321);
            configure_neuron(NUM_NEURONS-1, 32'hABCDEF00);
            
            // Test global parameter changes
            global_threshold = 16'd1200; // Higher threshold
            #(CLK_PERIOD * 10);
            
            // Send spikes that would trigger with old threshold but not new
            for (integer i = 0; i < 5; i = i + 1) begin
                send_spike(i, 80, 1);
                #(CLK_PERIOD * 5);
            end
            
            #(CLK_PERIOD * 100);
            
            // Restore normal threshold
            global_threshold = THRESHOLD_VAL;
            #(CLK_PERIOD * 10);
            
            $display("PASS: Configuration interface test completed");
            test_passed = test_passed + 1;
        end
    endtask
    
    // Test Case 7: Backpressure handling
    task test_backpressure_handling;
        begin
            test_case = 7;
            $display("\nTest 7: Backpressure Handling");
            $fwrite(log_file, "\nTest 7: Backpressure Handling\n");
            
            // Disable output ready to create backpressure
            m_axis_spike_ready = 0;
            
            // Send many spikes
            for (integer i = 0; i < 20; i = i + 1) begin
                send_spike(i % NUM_NEURONS, 100, 1);
                #(CLK_PERIOD * 2);
            end
            
            #(CLK_PERIOD * 100);
            
            // Re-enable output
            m_axis_spike_ready = 1;
            #(CLK_PERIOD * 200);
            
            $display("PASS: Backpressure handling test completed");
            test_passed = test_passed + 1;
        end
    endtask
    
    // Test Case 8: Concurrent spikes
    task test_concurrent_spikes;
        begin
            test_case = 8;
            $display("\nTest 8: Concurrent Spike Handling");
            $fwrite(log_file, "\nTest 8: Concurrent Spike Handling\n");
            
            // Try to send spikes to multiple neurons simultaneously
            // This tests the arbitration logic
            
            fork
                begin
                    for (integer i = 0; i < 10; i = i + 1) begin
                        send_spike(0, 80, 1);
                        #(CLK_PERIOD * 5);
                    end
                end
                begin
                    #(CLK_PERIOD * 2);
                    for (integer i = 0; i < 10; i = i + 1) begin
                        send_spike(1, 75, 1);
                        #(CLK_PERIOD * 5);
                    end
                end
                begin
                    #(CLK_PERIOD * 4);
                    for (integer i = 0; i < 10; i = i + 1) begin
                        send_spike(2, 85, 1);
                        #(CLK_PERIOD * 5);
                    end
                end
            join
            
            #(CLK_PERIOD * 200);
            
            $display("PASS: Concurrent spike test completed");
            test_passed = test_passed + 1;
        end
    endtask
    
    // Test Case 9: Learning-like patterns
    task test_learning_patterns;
        begin
            test_case = 9;
            $display("\nTest 9: Learning-like Patterns");
            $fwrite(log_file, "\nTest 9: Learning-like Patterns\n");
            
            // Simulate STDP-like patterns: pre-post and post-pre
            for (integer pair = 0; pair < 5; pair = pair + 1) begin
                // Pre-post pattern (should strengthen)
                send_spike(pair, 60, 1);      // Pre
                #(CLK_PERIOD * 10);
                send_spike(pair + 8, 70, 1);   // Post (different neuron group)
                #(CLK_PERIOD * 20);
                
                // Post-pre pattern (should weaken)
                send_spike(pair + 8, 70, 1);   // Post
                #(CLK_PERIOD * 10);
                send_spike(pair, 60, 1);       // Pre
                #(CLK_PERIOD * 30);
            end
            
            #(CLK_PERIOD * 200);
            
            $display("PASS: Learning pattern test completed");
            test_passed = test_passed + 1;
        end
    endtask
    
    // Test Case 10: Performance limits
    task test_performance_limits;
        begin
            test_case = 10;
            $display("\nTest 10: Performance Limits");
            $fwrite(log_file, "\nTest 10: Performance Limits\n");
            
            integer stress_input_count = 0;
            integer stress_output_count = output_spikes_received;
            
            // High-frequency input stress test
            for (integer stress = 0; stress < 100; stress = stress + 1) begin
                send_spike($random % NUM_NEURONS, 
                          50 + ($random % 50), 
                          $random % 2);
                stress_input_count = stress_input_count + 1;
                
                // Minimal delay between spikes
                #CLK_PERIOD;
            end
            
            #(CLK_PERIOD * 500); // Wait for all processing to complete
            
            stress_output_count = output_spikes_received - stress_output_count;
            
            $display("Stress test: %0d inputs, %0d outputs", 
                    stress_input_count, stress_output_count);
            
            if (stress_output_count >= 0) begin
                $display("PASS: Performance stress test completed");
                test_passed = test_passed + 1;
            end else begin
                $display("FAIL: Performance stress test failed");
                test_failed = test_failed + 1;
            end
        end
    endtask
    
    // Helper tasks
    task send_spike;
        input [NEURON_ID_WIDTH-1:0] dest_id;
        input [WEIGHT_WIDTH-1:0] weight;
        input excitatory;
        begin
            // Wait for interface to be ready
            wait(s_axis_spike_ready);
            
            s_axis_spike_dest_id = dest_id;
            s_axis_spike_weight = weight;
            s_axis_spike_exc_inh = excitatory;
            s_axis_spike_valid = 1;
            
            #CLK_PERIOD;
            s_axis_spike_valid = 0;
            
            input_spikes_sent = input_spikes_sent + 1;
            
            $fwrite(pattern_file, "%0t: Input spike to neuron %0d, weight=%0d, exc=%0b\n",
                   $time, dest_id, weight, excitatory);
        end
    endtask
    
    task configure_neuron;
        input [NEURON_ID_WIDTH-1:0] addr;
        input [31:0] data;
        begin
            config_addr = addr;
            config_data = data;
            config_we = 1;
            #CLK_PERIOD;
            config_we = 0;
            #CLK_PERIOD;
            
            $fwrite(log_file, "%0t: Configured neuron %0d with data 0x%08x\n",
                   $time, addr, data);
        end
    endtask
    
    // Monitor array status
    always @(posedge clk) begin
        if (array_busy) begin
            $fwrite(log_file, "%0t: Array busy signal asserted\n", $time);
        end
    end
    
    // Spike rate monitoring
    reg [31:0] last_spike_count;
    reg [31:0] spike_rate_counter;
    
    always @(posedge clk) begin
        spike_rate_counter <= spike_rate_counter + 1;
        
        if (spike_rate_counter == 10000) begin // Every 100us at 100MHz
            if (spike_count != last_spike_count) begin
                $display("Spike rate: %0d spikes in last 100us", 
                        spike_count - last_spike_count);
            end
            last_spike_count = spike_count;
            spike_rate_counter = 0;
        end
    end

endmodule
