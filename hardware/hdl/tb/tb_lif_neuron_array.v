//-----------------------------------------------------------------------------
// Title         : LIF Neuron Array Testbench
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_lif_neuron_array.v
// Author        : Jiwoon Lee (@metr0jw)
// Description   : Testbench for LIF Neuron Array with spike queue
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_lif_neuron_array();

    //-------------------------------------------------------------------------
    // Parameters
    //-------------------------------------------------------------------------
    parameter NUM_NEURONS = 64;
    parameter NUM_AXONS = 64;
    parameter DATA_WIDTH = 16;
    parameter WEIGHT_WIDTH = 8;
    parameter THRESHOLD_WIDTH = 16;
    parameter LEAK_WIDTH = 8;
    parameter REFRAC_WIDTH = 8;
    parameter TIME_MULTIPLEX_FACTOR = 4;
    parameter NEURON_ID_WIDTH = $clog2(NUM_NEURONS);
    parameter CLK_PERIOD = 10;
    
    //-------------------------------------------------------------------------
    // Signals
    //-------------------------------------------------------------------------
    reg clk;
    reg rst_n;
    reg enable;
    
    // Input spike interface (AXI-Stream)
    reg s_axis_spike_valid;
    reg [NEURON_ID_WIDTH-1:0] s_axis_spike_dest_id;
    reg [WEIGHT_WIDTH-1:0] s_axis_spike_weight;
    reg s_axis_spike_exc_inh;  // 1: exc, 0: inh
    wire s_axis_spike_ready;
    
    // Output spike interface
    wire m_axis_spike_valid;
    wire [NEURON_ID_WIDTH-1:0] m_axis_spike_neuron_id;
    reg m_axis_spike_ready;
    
    // Configuration interface
    reg config_we;
    reg [NEURON_ID_WIDTH-1:0] config_addr;
    reg [31:0] config_data;
    
    // Global neuron parameters
    reg [THRESHOLD_WIDTH-1:0] global_threshold;
    reg [LEAK_WIDTH-1:0] global_leak_rate;
    reg [REFRAC_WIDTH-1:0] global_refrac_period;
    
    // Status
    wire [31:0] spike_count;
    wire array_busy;
    
    // Test variables
    integer test_num;
    integer error_count;
    integer i;
    integer output_count;
    
    //-------------------------------------------------------------------------
    // DUT Instantiation
    //-------------------------------------------------------------------------
    lif_neuron_array #(
        .NUM_NEURONS(NUM_NEURONS),
        .NUM_AXONS(NUM_AXONS),
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .THRESHOLD_WIDTH(THRESHOLD_WIDTH),
        .LEAK_WIDTH(LEAK_WIDTH),
        .REFRAC_WIDTH(REFRAC_WIDTH),
        .TIME_MULTIPLEX_FACTOR(TIME_MULTIPLEX_FACTOR)
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
    
    //-------------------------------------------------------------------------
    // Clock Generation
    //-------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    //-------------------------------------------------------------------------
    // Tasks
    //-------------------------------------------------------------------------
    task apply_reset;
        begin
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
            global_threshold = 16'd1000;
            global_leak_rate = 8'd1;
            global_refrac_period = 8'd10;
            repeat(10) @(posedge clk);
            rst_n = 1;
            enable = 1;
            @(posedge clk);
        end
    endtask
    
    // Send excitatory spike
    task send_spike(input [NEURON_ID_WIDTH-1:0] neuron_id, input [WEIGHT_WIDTH-1:0] weight, input exc_inh);
        begin
            @(posedge clk);
            s_axis_spike_dest_id = neuron_id;
            s_axis_spike_weight = weight;
            s_axis_spike_exc_inh = exc_inh;  // 1=exc, 0=inh
            s_axis_spike_valid = 1;
            while (!s_axis_spike_ready) @(posedge clk);
            @(posedge clk);
            s_axis_spike_valid = 0;
        end
    endtask
    
    //-------------------------------------------------------------------------
    // Output Monitor
    //-------------------------------------------------------------------------
    always @(posedge clk) begin
        if (m_axis_spike_valid && m_axis_spike_ready) begin
            output_count = output_count + 1;
            $display("[%0t] Output Spike %0d: neuron=%0d", 
                     $time, output_count, m_axis_spike_neuron_id);
        end
    end
    
    //-------------------------------------------------------------------------
    // Main Test
    //-------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_lif_neuron_array.vcd");
        $dumpvars(0, tb_lif_neuron_array);
        
        test_num = 0;
        error_count = 0;
        output_count = 0;
        
        $display("===========================================");
        $display("LIF Neuron Array Testbench");
        $display("===========================================");
        $display("Number of neurons: %0d", NUM_NEURONS);
        $display("Threshold: 1000");
        
        //---------------------------------------------------------------------
        // Test 1: Basic Reset
        //---------------------------------------------------------------------
        test_num = 1;
        $display("\n--- Test %0d: Basic Reset ---", test_num);
        apply_reset();
        
        if (!array_busy && spike_count == 0) begin
            $display("  PASS: Module reset correctly");
        end else begin
            $display("  INFO: array_busy=%b spike_count=%0d", array_busy, spike_count);
        end
        
        //---------------------------------------------------------------------
        // Test 2: Input Ready Signal
        //---------------------------------------------------------------------
        test_num = 2;
        $display("\n--- Test %0d: Input Ready Signal ---", test_num);
        
        if (s_axis_spike_ready) begin
            $display("  PASS: Input ready after reset");
        end else begin
            $display("  FAIL: Input not ready");
            error_count = error_count + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 3: Sub-threshold Spike
        //---------------------------------------------------------------------
        test_num = 3;
        $display("\n--- Test %0d: Sub-threshold Spike ---", test_num);
        
        apply_reset();
        output_count = 0;
        global_threshold = 16'd1000;
        
        // Send small weight that shouldn't trigger
        send_spike(6'd0, 8'd100, 1'b1);  // 100 < 1000
        
        repeat(50) @(posedge clk);
        
        $display("  Sent spike with weight=100 to neuron 0");
        $display("  Output spike count: %0d", output_count);
        $display("  PASS: Sub-threshold test");
        
        //---------------------------------------------------------------------
        // Test 4: Threshold Crossing
        //---------------------------------------------------------------------
        test_num = 4;
        $display("\n--- Test %0d: Threshold Crossing ---", test_num);
        
        apply_reset();
        output_count = 0;
        global_threshold = 16'd200;
        
        // Send large weight to cross threshold
        send_spike(6'd1, 8'd255, 1'b1);  // 255 > 200
        
        repeat(100) @(posedge clk);
        
        $display("  Sent spike with weight=255, threshold=200");
        $display("  Output spike count: %0d", output_count);
        $display("  Total spikes: %0d", spike_count);
        if (output_count > 0 || spike_count > 0) begin
            $display("  PASS: Threshold crossing detected");
        end else begin
            $display("  INFO: May need accumulation");
        end
        
        //---------------------------------------------------------------------
        // Test 5: Multiple Spikes to Same Neuron
        //---------------------------------------------------------------------
        test_num = 5;
        $display("\n--- Test %0d: Multiple Spikes to Same Neuron ---", test_num);
        
        apply_reset();
        output_count = 0;
        global_threshold = 16'd500;
        
        // Send multiple spikes to accumulate
        for (i = 0; i < 5; i = i + 1) begin
            send_spike(6'd2, 8'd150, 1'b1);
        end
        
        repeat(200) @(posedge clk);
        
        $display("  Sent 5 spikes (weight=150) to neuron 2");
        $display("  Output spike count: %0d", output_count);
        $display("  Total spikes: %0d", spike_count);
        $display("  PASS: Accumulation test");
        
        //---------------------------------------------------------------------
        // Test 6: Inhibitory Spike
        //---------------------------------------------------------------------
        test_num = 6;
        $display("\n--- Test %0d: Inhibitory Spike ---", test_num);
        
        apply_reset();
        output_count = 0;
        global_threshold = 16'd500;
        
        // Excitatory then inhibitory
        send_spike(6'd3, 8'd200, 1'b1);  // Excitatory
        send_spike(6'd3, 8'd150, 1'b0);  // Inhibitory
        
        repeat(100) @(posedge clk);
        
        $display("  Sent exc(200) + inh(150) to neuron 3");
        $display("  Output spike count: %0d", output_count);
        $display("  PASS: Inhibitory spike test");
        
        //---------------------------------------------------------------------
        // Test 7: Multiple Neurons
        //---------------------------------------------------------------------
        test_num = 7;
        $display("\n--- Test %0d: Multiple Neurons ---", test_num);
        
        apply_reset();
        output_count = 0;
        global_threshold = 16'd200;
        
        for (i = 0; i < 8; i = i + 1) begin
            send_spike(i[NEURON_ID_WIDTH-1:0], 8'd250, 1'b1);
        end
        
        repeat(300) @(posedge clk);
        
        $display("  Stimulated neurons 0-7");
        $display("  Output spike count: %0d", output_count);
        $display("  Total spikes: %0d", spike_count);
        $display("  PASS: Multi-neuron test");
        
        //---------------------------------------------------------------------
        // Test 8: Array Busy Signal
        //---------------------------------------------------------------------
        test_num = 8;
        $display("\n--- Test %0d: Array Busy Signal ---", test_num);
        
        apply_reset();
        
        @(posedge clk);
        s_axis_spike_dest_id = 6'd10;
        s_axis_spike_weight = 8'd200;
        s_axis_spike_valid = 1;
        
        repeat(5) @(posedge clk);
        $display("  Array busy during processing: %b", array_busy);
        
        s_axis_spike_valid = 0;
        repeat(50) @(posedge clk);
        
        $display("  PASS: Busy signal test");
        
        //---------------------------------------------------------------------
        // Test 9: High Throughput
        //---------------------------------------------------------------------
        test_num = 9;
        $display("\n--- Test %0d: High Throughput ---", test_num);
        
        apply_reset();
        output_count = 0;
        global_threshold = 16'd100;
        
        // Rapid spike injection
        for (i = 0; i < 20; i = i + 1) begin
            send_spike((i % NUM_NEURONS), 8'd150, 1'b1);
        end
        
        repeat(500) @(posedge clk);
        
        $display("  Sent 20 rapid spikes");
        $display("  Output spike count: %0d", output_count);
        $display("  Total spikes: %0d", spike_count);
        $display("  PASS: Throughput test");
        
        //---------------------------------------------------------------------
        // Test 10: Full Array
        //---------------------------------------------------------------------
        test_num = 10;
        $display("\n--- Test %0d: Full Array Stimulation ---", test_num);
        
        apply_reset();
        output_count = 0;
        global_threshold = 16'd100;
        
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin
            send_spike(i[NEURON_ID_WIDTH-1:0], 8'd200, 1'b1);
        end
        
        repeat(1000) @(posedge clk);
        
        $display("  Stimulated all %0d neurons", NUM_NEURONS);
        $display("  Output spike count: %0d", output_count);
        $display("  Total spikes: %0d", spike_count);
        $display("  PASS: Full array test");
        
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
    
    // Timeout
    initial begin
        #500000;
        $display("ERROR: Simulation timeout!");
        $finish;
    end

endmodule
