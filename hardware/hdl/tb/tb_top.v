//-----------------------------------------------------------------------------
// Title         : Top-Level Testbench for PYNQ-Z2 SNN Accelerator
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_top.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : System testbench for SNN accelerator with AXI interfaces
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_top();

    //-------------------------------------------------------------------------
    // Parameters
    //-------------------------------------------------------------------------
    localparam C_S_AXI_DATA_WIDTH = 32;
    localparam C_S_AXI_ADDR_WIDTH = 32;  // keep 32-bit addressing in TB, slice to 7 bits for DUT
    localparam C_AXIS_DATA_WIDTH = 32;
    localparam NUM_NEURONS = 64;
    
    // Clock periods
    localparam CLK_PERIOD = 10;  // 100MHz AXI clock
    
    // AXI4-Lite Address Map
    localparam ADDR_CTRL        = 32'h00000000;
    localparam ADDR_STATUS      = 32'h00000004;
    localparam ADDR_CONFIG      = 32'h00000008;
    localparam ADDR_SPIKE_COUNT = 32'h0000000C;
    localparam ADDR_LEAK_RATE   = 32'h00000010;
    localparam ADDR_THRESHOLD   = 32'h00000014;
    localparam ADDR_REFRAC      = 32'h00000018;
    localparam ADDR_VERSION     = 32'h0000001C;
    
    //-------------------------------------------------------------------------
    // DUT Signals
    //-------------------------------------------------------------------------
    // Clock and Reset
    reg                             aclk;
    reg                             aresetn;
    
    // AXI4-Lite signals
    reg [C_S_AXI_ADDR_WIDTH-1:0]   s_axi_awaddr;
    reg [2:0]                       s_axi_awprot;
    reg                             s_axi_awvalid;
    wire                            s_axi_awready;
    reg [C_S_AXI_DATA_WIDTH-1:0]   s_axi_wdata;
    reg [3:0]                       s_axi_wstrb;
    reg                             s_axi_wvalid;
    wire                            s_axi_wready;
    wire [1:0]                      s_axi_bresp;
    wire                            s_axi_bvalid;
    reg                             s_axi_bready;
    reg [C_S_AXI_ADDR_WIDTH-1:0]   s_axi_araddr;
    reg [2:0]                       s_axi_arprot;
    reg                             s_axi_arvalid;
    wire                            s_axi_arready;
    wire [C_S_AXI_DATA_WIDTH-1:0]  s_axi_rdata;
    wire [1:0]                      s_axi_rresp;
    wire                            s_axi_rvalid;
    reg                             s_axi_rready;
    
    // AXI4-Stream slave (input spikes)
    reg [C_AXIS_DATA_WIDTH-1:0]    s_axis_tdata;
    reg                             s_axis_tvalid;
    wire                            s_axis_tready;
    reg [3:0]                       s_axis_tkeep;
    reg [3:0]                       s_axis_tstrb;
    reg                             s_axis_tuser;
    reg                             s_axis_tlast;
    reg                             s_axis_tid;
    reg                             s_axis_tdest;
    
    // AXI4-Stream master (output spikes)
    wire [C_AXIS_DATA_WIDTH-1:0]   m_axis_tdata;
    wire                            m_axis_tvalid;
    reg                             m_axis_tready;
    wire [3:0]                      m_axis_tkeep;
    wire [3:0]                      m_axis_tstrb;
    wire                            m_axis_tuser;
    wire                            m_axis_tlast;
    wire                            m_axis_tid;
    wire                            m_axis_tdest;
    
    // Other signals
    wire                            interrupt;
    wire [3:0]                      led;
    reg [1:0]                       sw;
    reg [3:0]                       btn;
    wire                            led4_r, led4_g, led4_b;
    wire                            led5_r, led5_g, led5_b;
    
    // Test variables
    integer                         test_num;
    integer                         error_count;
    integer                         spike_count_in;
    integer                         spike_count_out;
    reg [255:0]                     test_name;
    reg [31:0]                      read_data;
    integer                         i, j;
    reg [13:0]                      tb_timestamp;
    
    //-------------------------------------------------------------------------
    // DUT Instantiation
    //-------------------------------------------------------------------------
    snn_accelerator_hls_top #(
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(7),
        .C_AXIS_DATA_WIDTH(C_AXIS_DATA_WIDTH),
        .NUM_NEURONS(NUM_NEURONS)
    ) DUT (
        // Clock and Reset
        .aclk(aclk),
        .aresetn(aresetn),
        
        // AXI4-Lite
        .s_axi_awaddr(s_axi_awaddr[6:0]),
        .s_axi_awprot(s_axi_awprot),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr[6:0]),
        .s_axi_arprot(s_axi_arprot),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        
        // AXI4-Stream
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tkeep(s_axis_tkeep),
        .s_axis_tstrb(s_axis_tstrb),
        .s_axis_tuser(s_axis_tuser),
        .s_axis_tlast(s_axis_tlast),
        .s_axis_tid(s_axis_tid),
        .s_axis_tdest(s_axis_tdest),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tkeep(m_axis_tkeep),
        .m_axis_tstrb(m_axis_tstrb),
        .m_axis_tuser(m_axis_tuser),
        .m_axis_tlast(m_axis_tlast),
        .m_axis_tid(m_axis_tid),
        .m_axis_tdest(m_axis_tdest),
        
        // Other I/O
        .interrupt(interrupt),
        .led(led),
        .sw(sw),
        .btn(btn),
        .led4_r(led4_r),
        .led4_g(led4_g),
        .led4_b(led4_b),
        .led5_r(led5_r),
        .led5_g(led5_g),
        .led5_b(led5_b)
    );
    
    //-------------------------------------------------------------------------
    // Clock Generation
    //-------------------------------------------------------------------------
    initial begin
        aclk = 1'b0;
        forever #(CLK_PERIOD/2) aclk = ~aclk;
    end
    
    //-------------------------------------------------------------------------
    // Test Tasks
    //-------------------------------------------------------------------------
    
    // Initialize test
    task init_test(input [255:0] name);
        begin
            test_name = name;
            spike_count_in = 0;
            spike_count_out = 0;
            $display("\n========================================");
            $display("Test %0d: %0s", test_num, test_name);
            $display("========================================");
            test_num = test_num + 1;
        end
    endtask
    
    // Apply reset
    task apply_reset();
        begin
            @(posedge aclk);
            aresetn = 1'b0;
            repeat(10) @(posedge aclk);
            aresetn = 1'b1;
            @(posedge aclk);
            $display("Reset applied");
        end
    endtask
    
    // AXI4-Lite write
    task axi_write(input [31:0] addr, input [31:0] data);
        begin
            @(posedge aclk);
            // Write address channel
            s_axi_awaddr <= addr;
            s_axi_awvalid <= 1'b1;
            s_axi_awprot <= 3'b000;
            
            // Write data channel
            s_axi_wdata <= data;
            s_axi_wvalid <= 1'b1;
            s_axi_wstrb <= 4'hF;

            $display("[%0t] TB AXI write start addr=0x%08x data=0x%08x", $time, addr, data);

            // Wait for address handshake
            while (s_axi_awready !== 1'b1) @(posedge aclk);
            s_axi_awvalid <= 1'b0;
            @(posedge aclk);
            $display("[%0t] TB AXI write address accepted", $time);

            // Wait for data handshake
            while (s_axi_wready !== 1'b1) @(posedge aclk);
            s_axi_wvalid <= 1'b0;
            @(posedge aclk);
            $display("[%0t] TB AXI write data accepted", $time);

            // Write response
            s_axi_bready <= 1'b1;
            while (s_axi_bvalid !== 1'b1) @(posedge aclk);
            $display("[%0t] TB AXI write response bresp=%0b", $time, s_axi_bresp);
            @(posedge aclk);
            s_axi_bready <= 1'b0;

            if (s_axi_bresp != 2'b00) begin
                $display("ERROR: AXI write response = %b", s_axi_bresp);
                error_count = error_count + 1;
            end
        end
    endtask
    
    // AXI4-Lite read
    task axi_read(input [31:0] addr, output [31:0] data);
        begin
            @(posedge aclk);
            // Read address channel
            s_axi_araddr <= addr;
            s_axi_arvalid <= 1'b1;
            s_axi_arprot <= 3'b000;
            
            while (s_axi_arready !== 1'b1) @(posedge aclk);
            s_axi_arvalid <= 1'b0;
            @(posedge aclk);
            
            // Read data channel
            s_axi_rready <= 1'b1;
            while (s_axi_rvalid !== 1'b1) @(posedge aclk);
            data = s_axi_rdata;
            s_axi_rready <= 1'b0;
            @(posedge aclk);
            
            if (s_axi_rresp != 2'b00) begin
                $display("ERROR: AXI read response = %b", s_axi_rresp);
                error_count = error_count + 1;
            end
        end
    endtask

    // Send spike via AXI-Stream
    task send_spike(input [9:0] neuron_id, input [7:0] weight);
        begin
            @(posedge aclk);
            wait(s_axis_tready);
            s_axis_tdata = {tb_timestamp, weight, neuron_id};
            s_axis_tkeep = 4'hF;
            s_axis_tstrb = 4'hF;
            s_axis_tuser = 1'b0;
            s_axis_tid   = 1'b0;
            s_axis_tdest = 1'b0;
            tb_timestamp = tb_timestamp + 1'b1;
            s_axis_tvalid = 1'b1;
            s_axis_tlast = 1'b1;
            @(posedge aclk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast = 1'b0;
            spike_count_in = spike_count_in + 1;
        end
    endtask
    
    // Send spike burst
    task send_spike_burst(input [9:0] start_id, input [7:0] count, input [7:0] weight);
        integer k;
        begin
            for (k = 0; k < count; k = k + 1) begin
                send_spike((start_id + k) % NUM_NEURONS, weight);
            end
        end
    endtask
    
    // Configure synapse connection
    task configure_connection(
        input [9:0] source,
        input [9:0] target,
        input [7:0] weight,
        input [15:0] index
    );
        begin
            // Write to router configuration (address space 0x3xxx_xxxx)
            axi_write(32'h30000000 | (index << 2), {8'd0, 1'b1, 1'b1, weight, 7'd0, 1'b0, target});
        end
    endtask
    
    // Configure neuron connection count
    task configure_neuron_count(input [9:0] neuron_id, input [7:0] count);
        begin
            axi_write(32'h31000000 | (neuron_id << 2), {24'd0, count});
        end
    endtask
    
    // Monitor output spikes
    always @(posedge aclk) begin
        if (m_axis_tvalid && m_axis_tready) begin
            spike_count_out = spike_count_out + 1;
            $display("[%0t] Output spike: data=0x%08x", $time, m_axis_tdata);
        end
    end

    // Temporary debug: observe early AXI handshakes
    integer dbg_cycle;
    initial begin
        dbg_cycle = 0;
        forever begin
            @(posedge aclk);
            if (dbg_cycle < 200) begin
                $display("[DBG %0t] resetn=%b awv=%b awr=%b wv=%b wr=%b arv=%b arr=%b", $time,
                         aresetn, s_axi_awvalid, s_axi_awready, s_axi_wvalid, s_axi_wready,
                         s_axi_arvalid, s_axi_arready);
                dbg_cycle = dbg_cycle + 1;
            end
        end
    end
    
    //-------------------------------------------------------------------------
    // Main Test Sequence
    //-------------------------------------------------------------------------
    initial begin
        // Initialize signals
        aresetn = 1'b1;
        s_axi_awaddr = 0;
        s_axi_awprot = 0;
        s_axi_awvalid = 0;
        s_axi_wdata = 0;
        s_axi_wstrb = 0;
        s_axi_wvalid = 0;
        s_axi_bready = 0;
        s_axi_araddr = 0;
        s_axi_arprot = 0;
        s_axi_arvalid = 0;
        s_axi_rready = 0;
        s_axis_tdata = 0;
        s_axis_tvalid = 0;
        s_axis_tkeep = 4'hF;
        s_axis_tstrb = 4'hF;
        s_axis_tuser = 1'b0;
        s_axis_tlast = 0;
        s_axis_tid = 1'b0;
        s_axis_tdest = 1'b0;
        m_axis_tready = 1;
        sw = 2'b00;
        btn = 4'b0000;
        test_num = 1;
        error_count = 0;
        tb_timestamp = 14'd0;
        
        // Create waveform dump
        $dumpfile("tb_top.vcd");
        $dumpvars(0, tb_top);
        
        // Initial reset
        apply_reset();
        #(CLK_PERIOD * 20);
        
        //---------------------------------------------------------------------
        // Test 1: Basic AXI4-Lite Access
        //---------------------------------------------------------------------
        init_test("Basic AXI4-Lite Register Access");
        
        // Read version register
        axi_read(ADDR_VERSION, read_data);
        $display("  Version register: 0x%08x", read_data);
        
        // Write and read back control register
        axi_write(ADDR_CTRL, 32'h00000001);  // Enable SNN
        axi_read(ADDR_CTRL, read_data);
        if (read_data[0] == 1'b1) begin
            $display("  PASS: Control register write/read");
        end else begin
            $display("  ERROR: Control register mismatch");
            error_count = error_count + 1;
        end
        
        // Configure neuron parameters
        axi_write(ADDR_LEAK_RATE, 32'h0000000A);    // Leak rate = 10
        axi_write(ADDR_THRESHOLD, 32'h000003E8);    // Threshold = 1000
        axi_write(ADDR_REFRAC, 32'h00000014);       // Refractory = 20
        
        // Read back parameters
        axi_read(ADDR_LEAK_RATE, read_data);
        $display("  Leak rate: %0d", read_data[15:0]);
        axi_read(ADDR_THRESHOLD, read_data);
        $display("  Threshold: %0d", read_data[15:0]);
        axi_read(ADDR_REFRAC, read_data);
        $display("  Refractory period: %0d", read_data[15:0]);
        
        //---------------------------------------------------------------------
        // Test 2: Status Register Check
        //---------------------------------------------------------------------
        init_test("Status Register and LED Check");
        
        axi_read(ADDR_STATUS, read_data);
        $display("  Initial status: 0x%08x", read_data);
        $display("  - SNN enabled: %b", read_data[0]);
        $display("  - Array busy: %b", read_data[13]);
        $display("  - Router busy: %b", read_data[14]);
        
        // Check LED outputs
        $display("  LED states: %b", led);
        $display("  RGB LED4: R=%b G=%b B=%b", led4_r, led4_g, led4_b);
        
        //---------------------------------------------------------------------
        // Test 3: Simple Spike Processing
        //---------------------------------------------------------------------
        init_test("Simple Spike Processing");
        
        // Configure a simple connection: neuron 0 -> neuron 1
        configure_connection(0, 1, 100, 0);
        configure_neuron_count(0, 1);
        
        // Send spikes to neuron 0
        $display("  Sending 5 spikes to neuron 0...");
        send_spike_burst(0, 5, 50);
        
        // Wait for processing
        #(CLK_PERIOD * 200);
        
        // Check spike count
        axi_read(ADDR_SPIKE_COUNT, read_data);
        $display("  Total spike count: %0d", read_data);
        
        //---------------------------------------------------------------------
        // Test 4: Fanout Test
        //---------------------------------------------------------------------
        init_test("Fanout Spike Routing");
        
        // Configure neuron 5 to connect to neurons 10, 11, 12
        configure_connection(5, 10, 60, 20);
        configure_connection(5, 11, 70, 21);
        configure_connection(5, 12, 80, 22);
        configure_neuron_count(5, 3);
        
        spike_count_out = 0;
        
        // Send spike to neuron 5
        $display("  Sending spike to neuron 5 (fanout to 3 neurons)...");
        send_spike(5, 100);
        
        // Wait for routing
        #(CLK_PERIOD * 300);
        
        $display("  Output spikes detected: %0d", spike_count_out);
        
        //---------------------------------------------------------------------
        // Test 5: Continuous Spike Stream
        //---------------------------------------------------------------------
        init_test("Continuous Spike Stream");
        
        // Enable configuration mode
        axi_write(ADDR_CONFIG, 32'h00000100);
        
        // Create a chain: 20->21->22->23
        for (i = 20; i < 23; i = i + 1) begin
            configure_connection(i, i+1, 80, i*32);
            configure_neuron_count(i, 1);
        end
        
        // Disable configuration mode
        axi_write(ADDR_CONFIG, 32'h00000000);
        
        spike_count_out = 0;
        
        // Send continuous spikes
        $display("  Sending continuous spike stream...");
        fork
            begin
                for (i = 0; i < 20; i = i + 1) begin
                    send_spike(20, 60);
                    #(CLK_PERIOD * 50);
                end
            end
            
            begin
                // Monitor for 2000 cycles
                #(CLK_PERIOD * 2000);
            end
        join_any
        
        $display("  Total output spikes: %0d", spike_count_out);
        
        //---------------------------------------------------------------------
        // Test 6: Reset and Recovery
        //---------------------------------------------------------------------
        init_test("Reset and Recovery");
        
    // Send clear-counters command (test expects counters to be cleared)
    $display("  Sending SNN clear-counters...");
    axi_write(ADDR_CTRL, 32'h00000004);  // Clear counters bit
    #(CLK_PERIOD * 10);
    axi_write(ADDR_CTRL, 32'h00000001);  // Enable SNN (clear bit removed)
        
        // Read spike count (should be reset)
        axi_read(ADDR_SPIKE_COUNT, read_data);
        if (read_data == 0) begin
            $display("  PASS: Spike counter reset");
        end else begin
            $display("  ERROR: Spike counter not reset: %0d", read_data);
            error_count = error_count + 1;
        end
        
        // Send test spike to verify recovery
        send_spike(0, 50);
        #(CLK_PERIOD * 100);
        
        //---------------------------------------------------------------------
        // Test 7: Button and Switch Interaction
        //---------------------------------------------------------------------
        init_test("Button and Switch Test");
        
        // Test manual reset button
        $display("  Testing manual reset button...");
        btn[0] = 1'b1;
        #(CLK_PERIOD * 10);
        btn[0] = 1'b0;
        
        // Test mode switches
        sw[0] = 1'b1;  // Enable some test mode
        #(CLK_PERIOD * 10);
        
        // Test spike injection button
        $display("  Testing spike injection button...");
        spike_count_out = 0;
        btn[2] = 1'b1;
        sw[1] = 1'b1;
        #(CLK_PERIOD * 2);
        btn[2] = 1'b0;
        #(CLK_PERIOD * 100);
        
        if (spike_count_out > 0) begin
            $display("  PASS: Manual spike injection worked");
        end
        
        sw = 2'b00;
        
        //---------------------------------------------------------------------
        // Test 8: Performance Test
        //---------------------------------------------------------------------
        init_test("Performance and Stress Test");
        
        // Configure dense connections
        $display("  Configuring dense network...");
        for (i = 0; i < 16; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                configure_connection(i, 16 + i*4 + j, 50 + j*10, i*32 + j);
            end
            configure_neuron_count(i, 4);
        end
        
        spike_count_out = 0;
        
        // Send rapid spike burst
        $display("  Sending rapid spike burst...");
        fork
            begin
                for (i = 0; i < 100; i = i + 1) begin
                    send_spike(i % 16, 80);
                    if (i % 10 == 0) begin
                        #(CLK_PERIOD * 10);
                    end
                end
            end
            
            begin
                // Monitor status
                repeat(10) begin
                    #(CLK_PERIOD * 100);
                    axi_read(ADDR_STATUS, read_data);
                    if (read_data[15]) begin
                        $display("  WARNING: FIFO overflow detected!");
                    end
                end
            end
        join
        
        #(CLK_PERIOD * 500);
        $display("  Performance test complete. Output spikes: %0d", spike_count_out);
        
        //---------------------------------------------------------------------
        // Test 9: Interrupt Test
        //---------------------------------------------------------------------
        init_test("Interrupt Generation");
        
        // Enable interrupt with threshold
        axi_write(ADDR_CTRL, 32'h00000009);  // Enable SNN + interrupt
        axi_write(32'h00000020, 32'h00000064);  // Set spike threshold to 100
        
        // Clear counters
        axi_write(ADDR_CTRL, 32'h0000000D);  // Enable + interrupt + clear
        #(CLK_PERIOD * 10);
        axi_write(ADDR_CTRL, 32'h00000009);  // Remove clear
        
        // Generate spikes to trigger interrupt
        $display("  Generating spikes to trigger interrupt...");
        if (interrupt !== 1'b0) begin
            @(negedge interrupt);
        end else begin
            @(posedge aclk);  // Ensure we align to clock before driving spikes
        end
        
        send_spike_burst(0, 50, 100);
        
        // Wait for interrupt
        fork
            begin
                @(posedge interrupt);
                $display("  Interrupt triggered!");
            end
            begin
                #(CLK_PERIOD * 5000);
                $display("  Timeout waiting for interrupt");
            end
        join_any
        
        //---------------------------------------------------------------------
        // Test 10: Full System Integration
        //---------------------------------------------------------------------
        init_test("Full System Integration");
        
        // Reset everything
        axi_write(ADDR_CTRL, 32'h00000002);
        #(CLK_PERIOD * 10);
        axi_write(ADDR_CTRL, 32'h00000001);
        
        // Configure realistic SNN parameters
        axi_write(ADDR_LEAK_RATE, 32'h00000008);
        axi_write(ADDR_THRESHOLD, 32'h00000320);  // 800
        axi_write(ADDR_REFRAC, 32'h00000019);     // 25
        
        // Create a small fully connected layer (4x4)
        $display("  Creating 4x4 fully connected network...");
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                if (i != j) begin
                    configure_connection(i, j+4, 40 + i*10 + j, i*32 + j);
                end
            end
            configure_neuron_count(i, 3);
        end
        
        spike_count_out = 0;
        
        // Send pattern
        $display("  Sending spike pattern...");
        send_spike(0, 100);
        #(CLK_PERIOD * 50);
        send_spike(1, 100);
        #(CLK_PERIOD * 50);
        send_spike(2, 100);
        #(CLK_PERIOD * 50);
        send_spike(3, 100);
        
        // Wait for network to settle
        #(CLK_PERIOD * 1000);
        
        // Read final statistics
        axi_read(ADDR_SPIKE_COUNT, read_data);
        $display("  Final spike count: %0d", read_data);
        axi_read(ADDR_STATUS, read_data);
        $display("  Final status: 0x%08x", read_data);
        
        //---------------------------------------------------------------------
        // Test Summary
        //---------------------------------------------------------------------
        #(CLK_PERIOD * 100);
        
        $display("\n========================================");
        $display("Test Summary");
        $display("========================================");
        $display("Total Tests Run: %0d", test_num - 1);
        $display("Total Errors: %0d", error_count);
        $display("Total Input Spikes: %0d", spike_count_in);
        $display("Total Output Spikes: %0d", spike_count_out);
        
        if (error_count == 0) begin
            $display("\nAll tests PASSED!");
        end else begin
            $display("\nTests FAILED with %0d errors!", error_count);
        end
        
        // Display final LED states
        $display("\nFinal LED states:");
        $display("  Regular LEDs: %b", led);
        $display("  RGB LED4: R=%b G=%b B=%b", led4_r, led4_g, led4_b);
        $display("  RGB LED5: R=%b G=%b B=%b", led5_r, led5_g, led5_b);
        
        $display("========================================\n");
        $finish;
    end
    
    //-------------------------------------------------------------------------
    // Timeout Watchdog
    //-------------------------------------------------------------------------
    localparam integer TIMEOUT_CYCLES = 5_000_000; // 50ms at 100MHz, avoids premature timeout during HLS integration

    initial begin
        #(CLK_PERIOD * TIMEOUT_CYCLES);
        $display("\n*** ERROR: Testbench timeout! ***");
        $finish;
    end
    
    //-------------------------------------------------------------------------
    // Protocol Checkers
    //-------------------------------------------------------------------------
    
    // AXI4-Lite write address channel
    always @(posedge aclk) begin
        if (s_axi_awvalid && s_axi_awready) begin
            $display("[%0t] AXI Write: addr=0x%08x", $time, s_axi_awaddr);
        end
    end
    
    // AXI4-Lite read address channel
    always @(posedge aclk) begin
        if (s_axi_arvalid && s_axi_arready) begin
            $display("[%0t] AXI Read: addr=0x%08x", $time, s_axi_araddr);
        end
    end
    
    // Check for AXI protocol violations (requires $past, not supported by Icarus)
`ifndef __ICARUS__
    always @(posedge aclk) begin
        // Check that VALID doesn't drop without READY
        if ($past(s_axi_awvalid) && !$past(s_axi_awready) && !s_axi_awvalid) begin
            $display("ERROR: AXI protocol violation - AWVALID dropped without AWREADY");
            error_count = error_count + 1;
        end
    end
`endif

endmodule
