//-----------------------------------------------------------------------------
// HLS + RTL Integration Smoke Testbench
// Instantiates the full snn_accelerator_top (HLS-based AXI) and performs
// basic AXI-Lite accesses plus a single spike transaction to verify
// handshakes and dataflow wire-up. Intent is a fast sanity check, not a
// functional correctness test.
//-----------------------------------------------------------------------------
`timescale 1ns/1ps

module tb_hls_integration;
    // Parameters
    localparam CLK_PERIOD = 10;       // 100 MHz
    localparam ADDR_CTRL  = 32'h0000_0000;
    localparam ADDR_LEAK  = 32'h0000_0010;
    localparam ADDR_THRESH= 32'h0000_0014;
    localparam ADDR_REFRAC= 32'h0000_0018;

    // Clock / reset
    reg aclk;
    reg aresetn;

    // AXI-Lite
    reg  [31:0] s_axi_awaddr;
    reg  [2:0]  s_axi_awprot;
    reg         s_axi_awvalid;
    wire        s_axi_awready;
    reg  [31:0] s_axi_wdata;
    reg  [3:0]  s_axi_wstrb;
    reg         s_axi_wvalid;
    wire        s_axi_wready;
    wire [1:0]  s_axi_bresp;
    wire        s_axi_bvalid;
    reg         s_axi_bready;
    reg  [31:0] s_axi_araddr;
    reg  [2:0]  s_axi_arprot;
    reg         s_axi_arvalid;
    wire        s_axi_arready;
    wire [31:0] s_axi_rdata;
    wire [1:0]  s_axi_rresp;
    wire        s_axi_rvalid;
    reg         s_axi_rready;

    // AXI-Stream in/out
    reg  [31:0] s_axis_tdata;
    reg         s_axis_tvalid;
    wire        s_axis_tready;
    reg  [3:0]  s_axis_tkeep;
    reg  [3:0]  s_axis_tstrb;
    reg         s_axis_tuser;
    reg         s_axis_tlast;
    reg         s_axis_tid;
    reg         s_axis_tdest;

    wire [31:0] m_axis_tdata;
    wire        m_axis_tvalid;
    reg         m_axis_tready;
    wire [3:0]  m_axis_tkeep;
    wire [3:0]  m_axis_tstrb;
    wire        m_axis_tuser;
    wire        m_axis_tlast;
    wire        m_axis_tid;
    wire        m_axis_tdest;

    // Misc
    wire        interrupt;
    wire [3:0]  led;
    reg  [1:0]  sw;
    reg  [3:0]  btn;
    wire        led4_r, led4_g, led4_b;
    wire        led5_r, led5_g, led5_b;

    integer errors;
    integer spikes_out;

    // DUT: full top wrapper
    snn_accelerator_top dut (
        .aclk(aclk),
        .aresetn(aresetn),
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

    // Clock
    initial begin
        aclk = 1'b0;
        forever #(CLK_PERIOD/2) aclk = ~aclk;
    end

    // Simple monitors
    always @(posedge aclk) begin
        if (m_axis_tvalid && m_axis_tready)
            spikes_out <= spikes_out + 1;
    end

    // AXI-Lite helpers
    task axi_write(input [31:0] addr, input [31:0] data);
        begin
            @(posedge aclk);
            s_axi_awaddr  <= addr;
            s_axi_awvalid <= 1'b1;
            s_axi_awprot  <= 3'b000;
            s_axi_wdata   <= data;
            s_axi_wstrb   <= 4'hF;
            s_axi_wvalid  <= 1'b1;
            s_axi_bready  <= 1'b1;
            // Wait for handshakes
            wait(s_axi_awready === 1'b1);
            wait(s_axi_wready  === 1'b1);
            @(posedge aclk);
            s_axi_awvalid <= 1'b0;
            s_axi_wvalid  <= 1'b0;
            wait(s_axi_bvalid === 1'b1);
            @(posedge aclk);
            s_axi_bready  <= 1'b0;
            if (s_axi_bresp != 2'b00) errors <= errors + 1;
        end
    endtask

    // Send one spike packet
    task send_spike(input [9:0] neuron_id, input [7:0] weight);
        begin
            @(posedge aclk);
            wait(s_axis_tready === 1'b1);
            s_axis_tdata  <= {14'd0, weight, neuron_id};
            s_axis_tkeep  <= 4'hF;
            s_axis_tstrb  <= 4'hF;
            s_axis_tuser  <= 1'b0;
            s_axis_tid    <= 1'b0;
            s_axis_tdest  <= 1'b0;
            s_axis_tlast  <= 1'b1;
            s_axis_tvalid <= 1'b1;
            @(posedge aclk);
            s_axis_tvalid <= 1'b0;
            s_axis_tlast  <= 1'b0;
        end
    endtask

    // Main sequence
    initial begin
        // init signals
        {s_axi_awaddr,s_axi_awprot,s_axi_awvalid} = 0;
        {s_axi_wdata,s_axi_wstrb,s_axi_wvalid}    = 0;
        {s_axi_bready,s_axi_araddr,s_axi_arprot}  = 0;
        {s_axi_arvalid,s_axi_rready}              = 0;
        {s_axis_tdata,s_axis_tkeep,s_axis_tstrb}  = 0;
        {s_axis_tuser,s_axis_tlast,s_axis_tid}    = 0;
        {s_axis_tdest,s_axis_tvalid}              = 0;
        m_axis_tready = 1'b1;
        sw  = 2'b00;
        btn = 4'b0000;
        errors = 0;
        spikes_out = 0;

        aresetn = 1'b0;
        repeat(20) @(posedge aclk);
        aresetn = 1'b1;
        repeat(5) @(posedge aclk);

        // Basic register programming
        axi_write(ADDR_LEAK,    32'd10);
        axi_write(ADDR_THRESH,  32'd200);
        axi_write(ADDR_REFRAC,  32'd8);
        axi_write(ADDR_CTRL,    32'h0000_0001); // enable

        // Wait for stream ready then push one spike
        repeat(20) @(posedge aclk);
        send_spike(10'd1, 8'd50);

        // Observe for a while
        repeat(2000) @(posedge aclk);

        $display("[TB] spikes_out=%0d errors=%0d", spikes_out, errors);
        if (errors == 0)
            $display("[TB] PASS: Integration smoke test completed");
        else
            $display("[TB] FAIL: errors=%0d", errors);
        $finish;
    end

    // Safety timeout (50 us)
    initial begin
        #(50_000 * CLK_PERIOD);
        $display("[TB] TIMEOUT");
        $finish;
    end
endmodule
