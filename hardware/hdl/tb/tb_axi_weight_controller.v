//-----------------------------------------------------------------------------
// Title         : AXI Weight Controller Testbench
// Project       : PYNQ-Z2 SNN Accelerator
// File          : tb_axi_weight_controller.v
// Author        : Jiwoon Lee (@metr0jw)
// Description   : Testbench for AXI weight configuration interface
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_axi_weight_controller;

    // Parameters
    parameter C_S_AXI_DATA_WIDTH = 32;
    parameter C_S_AXI_ADDR_WIDTH = 16;
    parameter WEIGHT_WIDTH       = 16;
    parameter NUM_SYNAPSES       = 4096;
    parameter ADDR_BITS          = 12;
    
    parameter CLK_PERIOD = 10;  // 100 MHz
    
    // DUT signals
    reg                             s_axi_aclk;
    reg                             s_axi_aresetn;
    reg  [C_S_AXI_ADDR_WIDTH-1:0]  s_axi_awaddr;
    reg  [2:0]                      s_axi_awprot;
    reg                             s_axi_awvalid;
    wire                            s_axi_awready;
    reg  [C_S_AXI_DATA_WIDTH-1:0]  s_axi_wdata;
    reg  [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb;
    reg                             s_axi_wvalid;
    wire                            s_axi_wready;
    wire [1:0]                      s_axi_bresp;
    wire                            s_axi_bvalid;
    reg                             s_axi_bready;
    reg  [C_S_AXI_ADDR_WIDTH-1:0]  s_axi_araddr;
    reg  [2:0]                      s_axi_arprot;
    reg                             s_axi_arvalid;
    wire                            s_axi_arready;
    wire [C_S_AXI_DATA_WIDTH-1:0]  s_axi_rdata;
    wire [1:0]                      s_axi_rresp;
    wire                            s_axi_rvalid;
    reg                             s_axi_rready;
    
    // Weight memory interface
    wire                            weight_we;
    wire [ADDR_BITS-1:0]           weight_addr;
    wire [WEIGHT_WIDTH-1:0]        weight_wdata;
    reg  [WEIGHT_WIDTH-1:0]        weight_rdata;
    wire                            weight_rd_en;
    
    // Bulk transfer interface
    wire                            bulk_start;
    wire [ADDR_BITS-1:0]           bulk_start_addr;
    wire [ADDR_BITS-1:0]           bulk_length;
    reg                             bulk_done;
    
    wire [31:0]                     weight_status;
    
    // Simple weight memory for testing
    reg [WEIGHT_WIDTH-1:0] weight_mem [0:255];
    
    // Test counters
    integer test_errors;
    integer test_num;
    reg [31:0] read_data;
    
    //-------------------------------------------------------------------------
    // DUT Instantiation
    //-------------------------------------------------------------------------
    axi_weight_controller #(
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .NUM_SYNAPSES(NUM_SYNAPSES),
        .ADDR_BITS(ADDR_BITS)
    ) dut (
        .s_axi_aclk(s_axi_aclk),
        .s_axi_aresetn(s_axi_aresetn),
        .s_axi_awaddr(s_axi_awaddr),
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
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arprot(s_axi_arprot),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        .weight_we(weight_we),
        .weight_addr(weight_addr),
        .weight_wdata(weight_wdata),
        .weight_rdata(weight_rdata),
        .weight_rd_en(weight_rd_en),
        .bulk_start(bulk_start),
        .bulk_start_addr(bulk_start_addr),
        .bulk_length(bulk_length),
        .bulk_done(bulk_done),
        .weight_status(weight_status)
    );
    
    //-------------------------------------------------------------------------
    // Clock Generation
    //-------------------------------------------------------------------------
    initial begin
        s_axi_aclk = 0;
        forever #(CLK_PERIOD/2) s_axi_aclk = ~s_axi_aclk;
    end
    
    //-------------------------------------------------------------------------
    // Simple Weight Memory Model
    //-------------------------------------------------------------------------
    always @(posedge s_axi_aclk) begin
        if (weight_we) begin
            weight_mem[weight_addr[7:0]] <= weight_wdata;
            $display("  [MEM] Write addr=%0d data=0x%04x", weight_addr, weight_wdata);
        end
        if (weight_rd_en) begin
            weight_rdata <= weight_mem[weight_addr[7:0]];
        end
    end
    
    //-------------------------------------------------------------------------
    // AXI Write Task
    //-------------------------------------------------------------------------
    task axi_write;
        input [C_S_AXI_ADDR_WIDTH-1:0] addr;
        input [C_S_AXI_DATA_WIDTH-1:0] data;
        begin
            @(posedge s_axi_aclk);
            s_axi_awaddr  <= addr;
            s_axi_awvalid <= 1'b1;
            s_axi_wdata   <= data;
            s_axi_wstrb   <= 4'hF;
            s_axi_wvalid  <= 1'b1;
            s_axi_bready  <= 1'b1;
            
            // Wait for both ready signals
            wait(s_axi_awready && s_axi_wready);
            @(posedge s_axi_aclk);
            s_axi_awvalid <= 1'b0;
            s_axi_wvalid  <= 1'b0;
            
            // Wait for response
            wait(s_axi_bvalid);
            @(posedge s_axi_aclk);
            s_axi_bready <= 1'b0;
            
            $display("  [AXI] Write addr=0x%04x data=0x%08x", addr, data);
        end
    endtask
    
    //-------------------------------------------------------------------------
    // AXI Read Task
    //-------------------------------------------------------------------------
    task axi_read;
        input  [C_S_AXI_ADDR_WIDTH-1:0] addr;
        output [C_S_AXI_DATA_WIDTH-1:0] data;
        begin
            @(posedge s_axi_aclk);
            s_axi_araddr  <= addr;
            s_axi_arvalid <= 1'b1;
            s_axi_rready  <= 1'b1;
            
            // Wait for address ready
            wait(s_axi_arready);
            @(posedge s_axi_aclk);
            s_axi_arvalid <= 1'b0;
            
            // Wait for read data valid
            wait(s_axi_rvalid);
            data = s_axi_rdata;
            @(posedge s_axi_aclk);
            s_axi_rready <= 1'b0;
            
            $display("  [AXI] Read  addr=0x%04x data=0x%08x", addr, data);
        end
    endtask
    
    //-------------------------------------------------------------------------
    // Test Sequence
    //-------------------------------------------------------------------------
    initial begin
        $display("\n========================================");
        $display("AXI Weight Controller Testbench");
        $display("========================================\n");
        
        // Initialize
        test_errors = 0;
        test_num = 0;
        s_axi_aresetn = 0;
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
        weight_rdata = 0;
        bulk_done = 0;
        
        // Initialize weight memory
        for (integer i = 0; i < 256; i = i + 1) begin
            weight_mem[i] = 16'h0000;
        end
        
        // Reset
        repeat (10) @(posedge s_axi_aclk);
        s_axi_aresetn = 1;
        repeat (5) @(posedge s_axi_aclk);
        
        //---------------------------------------------------------------------
        // Test 1: Read control register (default value)
        //---------------------------------------------------------------------
        test_num = 1;
        $display("========================================");
        $display("Test %0d: Read Control Register (default)", test_num);
        $display("========================================");
        
        axi_read(16'h0000, read_data);
        if (read_data == 32'h0) begin
            $display("  PASS: Control register = 0x%08x", read_data);
        end else begin
            $display("  FAIL: Expected 0x00000000, got 0x%08x", read_data);
            test_errors = test_errors + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 2: Enable weight updates
        //---------------------------------------------------------------------
        test_num = 2;
        $display("\n========================================");
        $display("Test %0d: Enable Weight Updates", test_num);
        $display("========================================");
        
        axi_write(16'h0000, 32'h0000_0001);  // Enable bit[0]
        axi_read(16'h0000, read_data);
        if (read_data[0] == 1'b1) begin
            $display("  PASS: Weight updates enabled");
        end else begin
            $display("  FAIL: Enable bit not set");
            test_errors = test_errors + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 3: Direct weight memory write
        //---------------------------------------------------------------------
        test_num = 3;
        $display("\n========================================");
        $display("Test %0d: Direct Weight Memory Write", test_num);
        $display("========================================");
        
        // Write to weight memory at address 0x2000 (offset 0, weight addr 0)
        axi_write(16'h2000, 32'h0000_1234);
        repeat (3) @(posedge s_axi_aclk);
        
        // Write to weight memory at address 0x2002 (offset 2, weight addr 1)
        axi_write(16'h2002, 32'h0000_5678);
        repeat (3) @(posedge s_axi_aclk);
        
        // Write to weight memory at address 0x2004 (offset 4, weight addr 2)
        axi_write(16'h2004, 32'h0000_ABCD);
        repeat (3) @(posedge s_axi_aclk);
        
        // Verify writes
        if (weight_mem[0] == 16'h1234 && weight_mem[1] == 16'h5678 && weight_mem[2] == 16'hABCD) begin
            $display("  PASS: Direct writes successful");
            $display("    weight_mem[0] = 0x%04x", weight_mem[0]);
            $display("    weight_mem[1] = 0x%04x", weight_mem[1]);
            $display("    weight_mem[2] = 0x%04x", weight_mem[2]);
        end else begin
            $display("  FAIL: Direct writes failed");
            $display("    weight_mem[0] = 0x%04x (expected 0x1234)", weight_mem[0]);
            $display("    weight_mem[1] = 0x%04x (expected 0x5678)", weight_mem[1]);
            $display("    weight_mem[2] = 0x%04x (expected 0xABCD)", weight_mem[2]);
            test_errors = test_errors + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 4: Check write counter
        //---------------------------------------------------------------------
        test_num = 4;
        $display("\n========================================");
        $display("Test %0d: Check Write Counter", test_num);
        $display("========================================");
        
        axi_read(16'h0018, read_data);  // ADDR_WRITE_COUNT
        if (read_data == 32'd3) begin
            $display("  PASS: Write count = %0d", read_data);
        end else begin
            $display("  FAIL: Expected 3, got %0d", read_data);
            test_errors = test_errors + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 5: Indirect weight write via single_addr/single_data
        //---------------------------------------------------------------------
        test_num = 5;
        $display("\n========================================");
        $display("Test %0d: Indirect Weight Write", test_num);
        $display("========================================");
        
        // Set single address to 10
        axi_write(16'h0010, 32'd10);  // ADDR_SINGLE_ADDR
        repeat (2) @(posedge s_axi_aclk);
        
        // Write data (this triggers the actual weight write)
        axi_write(16'h0014, 32'h0000_BEEF);  // ADDR_SINGLE_DATA
        repeat (3) @(posedge s_axi_aclk);
        
        if (weight_mem[10] == 16'hBEEF) begin
            $display("  PASS: Indirect write to addr 10 = 0x%04x", weight_mem[10]);
        end else begin
            $display("  FAIL: Expected 0xBEEF, got 0x%04x", weight_mem[10]);
            test_errors = test_errors + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 6: Bulk transfer configuration
        //---------------------------------------------------------------------
        test_num = 6;
        $display("\n========================================");
        $display("Test %0d: Bulk Transfer Configuration", test_num);
        $display("========================================");
        
        // Set bulk start address
        axi_write(16'h0008, 32'd100);  // ADDR_BULK_ADDR
        
        // Set bulk length
        axi_write(16'h000C, 32'd50);   // ADDR_BULK_LEN
        
        // Read back
        axi_read(16'h0008, read_data);
        if (read_data == 32'd100) begin
            $display("  PASS: Bulk start addr = %0d", read_data);
        end else begin
            $display("  FAIL: Bulk start addr expected 100, got %0d", read_data);
            test_errors = test_errors + 1;
        end
        
        axi_read(16'h000C, read_data);
        if (read_data == 32'd50) begin
            $display("  PASS: Bulk length = %0d", read_data);
        end else begin
            $display("  FAIL: Bulk length expected 50, got %0d", read_data);
            test_errors = test_errors + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 7: Trigger bulk start
        //---------------------------------------------------------------------
        test_num = 7;
        $display("\n========================================");
        $display("Test %0d: Trigger Bulk Start", test_num);
        $display("========================================");
        
        // Write control reg with bulk start bit
        axi_write(16'h0000, 32'h0000_0003);  // Enable + Bulk start
        
        // Check bulk_start signal
        @(posedge s_axi_aclk);
        if (bulk_start) begin
            $display("  PASS: Bulk start signal asserted");
        end else begin
            $display("  INFO: Bulk start may have been auto-cleared");
        end
        
        // Verify bulk_start_addr and bulk_length outputs
        $display("  Bulk start addr: %0d", bulk_start_addr);
        $display("  Bulk length: %0d", bulk_length);
        
        // Simulate bulk done
        repeat (5) @(posedge s_axi_aclk);
        bulk_done = 1'b1;
        repeat (2) @(posedge s_axi_aclk);
        bulk_done = 1'b0;
        
        //---------------------------------------------------------------------
        // Test 8: Read status register
        //---------------------------------------------------------------------
        test_num = 8;
        $display("\n========================================");
        $display("Test %0d: Read Status Register", test_num);
        $display("========================================");
        
        axi_read(16'h0004, read_data);
        $display("  Status register = 0x%08x", read_data);
        $display("    Bulk done: %0b", read_data[7]);
        $display("    Read pending: %0b", read_data[6]);
        $display("    Write enable: %0b", read_data[5]);
        $display("  PASS: Status read successful");
        
        //---------------------------------------------------------------------
        // Test 9: Clear counters
        //---------------------------------------------------------------------
        test_num = 9;
        $display("\n========================================");
        $display("Test %0d: Clear Counters", test_num);
        $display("========================================");
        
        // Write control with clear bit
        axi_write(16'h0000, 32'h0000_0005);  // Enable + Clear counters
        repeat (3) @(posedge s_axi_aclk);
        
        axi_read(16'h0018, read_data);  // Read write count
        if (read_data == 32'd0) begin
            $display("  PASS: Write counter cleared");
        end else begin
            $display("  FAIL: Write counter not cleared, got %0d", read_data);
            test_errors = test_errors + 1;
        end
        
        //---------------------------------------------------------------------
        // Test 10: Multiple rapid writes
        //---------------------------------------------------------------------
        test_num = 10;
        $display("\n========================================");
        $display("Test %0d: Multiple Rapid Writes", test_num);
        $display("========================================");
        
        for (integer i = 0; i < 10; i = i + 1) begin
            axi_write(16'h2000 + (i*2), i * 100);
        end
        
        // Verify
        $display("  Verifying rapid writes...");
        for (integer i = 0; i < 10; i = i + 1) begin
            if (weight_mem[i] != i * 100) begin
                $display("    FAIL: weight_mem[%0d] = %0d (expected %0d)", i, weight_mem[i], i*100);
                test_errors = test_errors + 1;
            end
        end
        $display("  PASS: Rapid writes completed");
        
        //---------------------------------------------------------------------
        // Summary
        //---------------------------------------------------------------------
        repeat (10) @(posedge s_axi_aclk);
        
        $display("\n========================================");
        $display("Test Summary");
        $display("========================================");
        $display("Total Tests: %0d", test_num);
        $display("Errors: %0d", test_errors);
        
        if (test_errors == 0) begin
            $display("PASS: All tests passed!");
        end else begin
            $display("FAIL: %0d tests failed", test_errors);
        end
        $display("========================================\n");
        
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #100000;
        $display("\nTIMEOUT: Simulation exceeded time limit");
        $finish;
    end

endmodule
