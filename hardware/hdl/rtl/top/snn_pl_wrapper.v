//-----------------------------------------------------------------------------
// Title         : SNN Accelerator PL-Only Wrapper for PYNQ-Z2
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_pl_wrapper.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : PL-only wrapper with minimal external pins for testing
//                 This version connects AXI interfaces internally
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module snn_pl_wrapper (
    //-------------------------------------------------------------------------
    // Clock and Reset from PYNQ-Z2 125MHz oscillator
    //-------------------------------------------------------------------------
    input  wire         sysclk,         // 125 MHz system clock
    
    //-------------------------------------------------------------------------
    // Board I/O - Directly available on PYNQ-Z2
    //-------------------------------------------------------------------------
    output wire [3:0]   led,            // Regular LEDs
    input  wire [1:0]   sw,             // Switches
    input  wire [3:0]   btn,            // Push buttons
    
    // RGB LEDs
    output wire         led4_r,
    output wire         led4_g,
    output wire         led4_b,
    output wire         led5_r,
    output wire         led5_g,
    output wire         led5_b
);

    //-------------------------------------------------------------------------
    // Internal Signals
    //-------------------------------------------------------------------------
    wire        clk_100mhz;
    wire        locked;
    wire        sys_rst_n;
    
    // AXI-Lite signals (directly connected to internal test logic)
    wire [31:0] s_axi_awaddr;
    wire [2:0]  s_axi_awprot;
    wire        s_axi_awvalid;
    wire        s_axi_awready;
    wire [31:0] s_axi_wdata;
    wire [3:0]  s_axi_wstrb;
    wire        s_axi_wvalid;
    wire        s_axi_wready;
    wire [1:0]  s_axi_bresp;
    wire        s_axi_bvalid;
    wire        s_axi_bready;
    wire [31:0] s_axi_araddr;
    wire [2:0]  s_axi_arprot;
    wire        s_axi_arvalid;
    wire        s_axi_arready;
    wire [31:0] s_axi_rdata;
    wire [1:0]  s_axi_rresp;
    wire        s_axi_rvalid;
    wire        s_axi_rready;
    
    // AXI-Stream signals
    wire [31:0] s_axis_tdata;
    wire        s_axis_tvalid;
    wire        s_axis_tready;
    wire [3:0]  s_axis_tkeep;
    wire [3:0]  s_axis_tstrb;
    wire        s_axis_tuser;
    wire        s_axis_tlast;
    wire        s_axis_tid;
    wire        s_axis_tdest;
    wire [31:0] m_axis_tdata;
    wire        m_axis_tvalid;
    wire        m_axis_tready;
    wire [3:0]  m_axis_tkeep;
    wire [3:0]  m_axis_tstrb;
    wire        m_axis_tuser;
    wire        m_axis_tlast;
    wire        m_axis_tid;
    wire        m_axis_tdest;
    
    wire        interrupt;

    //-------------------------------------------------------------------------
    // Clock Generation
    //-------------------------------------------------------------------------
    // Using 125MHz sysclk directly from PYNQ-Z2 oscillator
    // For production use requiring exact 100MHz, instantiate MMCM/PLL:
    //   clk_wiz_0 clk_gen (.clk_in1(sysclk), .clk_out1(clk_100mhz), .locked(locked));
    assign clk_100mhz = sysclk;
    assign locked = 1'b1;
    
    //-------------------------------------------------------------------------
    // Reset Synchronizer
    //-------------------------------------------------------------------------
    reg [3:0] rst_sync;
    
    always @(posedge clk_100mhz or negedge btn[0]) begin
        if (!btn[0])  // Use button 0 as reset
            rst_sync <= 4'b0;
        else
            rst_sync <= {rst_sync[2:0], 1'b1};
    end
    
    assign sys_rst_n = rst_sync[3] & locked;

    //-------------------------------------------------------------------------
    // Simple AXI-Lite Test Controller
    // Performs basic register writes to configure the SNN and test operation
    //-------------------------------------------------------------------------
    reg [3:0]  test_state;
    reg [31:0] test_addr;
    reg [31:0] test_data;
    reg        test_write;
    reg        test_read;
    reg [31:0] read_data;
    reg [31:0] test_counter;
    
    localparam ST_IDLE      = 4'd0;
    localparam ST_WRITE_CMD = 4'd1;
    localparam ST_WRITE_ADDR = 4'd2;
    localparam ST_WRITE_DATA = 4'd3;
    localparam ST_WRITE_RESP = 4'd4;
    localparam ST_READ_CMD  = 4'd5;
    localparam ST_READ_ADDR = 4'd6;
    localparam ST_READ_DATA = 4'd7;
    localparam ST_DONE      = 4'd8;
    localparam ST_RUN       = 4'd9;
    
    // AXI-Lite Master signals
    reg [31:0] axi_awaddr_r;
    reg        axi_awvalid_r;
    reg [31:0] axi_wdata_r;
    reg        axi_wvalid_r;
    reg        axi_bready_r;
    reg [31:0] axi_araddr_r;
    reg        axi_arvalid_r;
    reg        axi_rready_r;
    
    assign s_axi_awaddr  = axi_awaddr_r;
    assign s_axi_awprot  = 3'b000;
    assign s_axi_awvalid = axi_awvalid_r;
    assign s_axi_wdata   = axi_wdata_r;
    assign s_axi_wstrb   = 4'hF;
    assign s_axi_wvalid  = axi_wvalid_r;
    assign s_axi_bready  = axi_bready_r;
    assign s_axi_araddr  = axi_araddr_r;
    assign s_axi_arprot  = 3'b000;
    assign s_axi_arvalid = axi_arvalid_r;
    assign s_axi_rready  = axi_rready_r;
    
    // Register addresses (matching axi_lite_regs_v1_0)
    localparam ADDR_CTRL      = 32'h00;  // Control register
    localparam ADDR_CONFIG    = 32'h04;  // Config register
    localparam ADDR_LEAK      = 32'h08;  // Leak rate
    localparam ADDR_THRESHOLD = 32'h0C;  // Threshold
    localparam ADDR_STATUS    = 32'h10;  // Status (read-only)
    localparam ADDR_SPIKE_CNT = 32'h14;  // Spike count (read-only)
    
    always @(posedge clk_100mhz or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            test_state    <= ST_IDLE;
            axi_awaddr_r  <= 32'h0;
            axi_awvalid_r <= 1'b0;
            axi_wdata_r   <= 32'h0;
            axi_wvalid_r  <= 1'b0;
            axi_bready_r  <= 1'b0;
            axi_araddr_r  <= 32'h0;
            axi_arvalid_r <= 1'b0;
            axi_rready_r  <= 1'b0;
            test_counter  <= 32'h0;
            read_data     <= 32'h0;
        end else begin
            case (test_state)
                ST_IDLE: begin
                    test_counter <= test_counter + 1'b1;
                    // Wait for stabilization, then configure
                    if (test_counter == 32'h00100000) begin
                        // Write threshold = 1000
                        axi_awaddr_r <= ADDR_THRESHOLD;
                        axi_wdata_r  <= 32'd1000;
                        test_state   <= ST_WRITE_ADDR;
                    end
                end
                
                ST_WRITE_ADDR: begin
                    axi_awvalid_r <= 1'b1;
                    if (s_axi_awready) begin
                        axi_awvalid_r <= 1'b0;
                        test_state    <= ST_WRITE_DATA;
                    end
                end
                
                ST_WRITE_DATA: begin
                    axi_wvalid_r <= 1'b1;
                    if (s_axi_wready) begin
                        axi_wvalid_r <= 1'b0;
                        axi_bready_r <= 1'b1;
                        test_state   <= ST_WRITE_RESP;
                    end
                end
                
                ST_WRITE_RESP: begin
                    if (s_axi_bvalid) begin
                        axi_bready_r <= 1'b0;
                        // Write control register to enable
                        if (axi_awaddr_r == ADDR_THRESHOLD) begin
                            axi_awaddr_r <= ADDR_CTRL;
                            axi_wdata_r  <= 32'h00000001;  // Enable SNN
                            test_state   <= ST_WRITE_ADDR;
                        end else begin
                            test_state   <= ST_RUN;
                        end
                    end
                end
                
                ST_RUN: begin
                    // SNN is running, periodically read status
                    test_counter <= test_counter + 1'b1;
                    if (test_counter[23:0] == 24'hFFFFFF) begin
                        axi_araddr_r <= ADDR_SPIKE_CNT;
                        test_state   <= ST_READ_ADDR;
                    end
                end
                
                ST_READ_ADDR: begin
                    axi_arvalid_r <= 1'b1;
                    if (s_axi_arready) begin
                        axi_arvalid_r <= 1'b0;
                        axi_rready_r  <= 1'b1;
                        test_state    <= ST_READ_DATA;
                    end
                end
                
                ST_READ_DATA: begin
                    if (s_axi_rvalid) begin
                        read_data    <= s_axi_rdata;
                        axi_rready_r <= 1'b0;
                        test_state   <= ST_RUN;
                    end
                end
                
                default: test_state <= ST_IDLE;
            endcase
        end
    end
    
    // Expose read_data for debug via LED blinking pattern
    wire [31:0] debug_read_data = read_data;  // Can be probed via ILA

    //-------------------------------------------------------------------------
    // Simple Input Spike Generator (for testing)
    //-------------------------------------------------------------------------
    reg [31:0] spike_gen_counter;
    reg        spike_gen_valid;
    reg [31:0] spike_gen_data;
    reg [9:0]  spike_neuron_id;
    
    always @(posedge clk_100mhz or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            spike_gen_counter <= 32'h0;
            spike_gen_valid   <= 1'b0;
            spike_gen_data    <= 32'h0;
            spike_neuron_id   <= 8'h0;
        end else begin
            spike_gen_counter <= spike_gen_counter + 1'b1;
            
            // Generate spike every ~1M cycles when sw[1] is on
            if (sw[1] && spike_gen_counter[19:0] == 20'hFFFFF) begin
                spike_gen_valid <= 1'b1;
                spike_neuron_id <= spike_neuron_id + 1'b1;
                // Format: [31:18]=timestamp(14b), [17:10]=weight(8b), [9:0]=neuron_id(10b)
                spike_gen_data  <= {spike_gen_counter[27:14], 8'd50, spike_neuron_id};
            end else if (s_axis_tready) begin
                spike_gen_valid <= 1'b0;
            end
        end
    end
    
    assign s_axis_tdata  = spike_gen_data;
    assign s_axis_tvalid = spike_gen_valid;
    assign s_axis_tkeep  = 4'hF;
    assign s_axis_tstrb  = 4'hF;
    assign s_axis_tuser  = 1'b0;
    assign s_axis_tlast  = spike_gen_valid;  // Each spike is a complete packet
    assign s_axis_tid    = 1'b0;
    assign s_axis_tdest  = 1'b0;

    // Output stream - just accept
    assign m_axis_tready = 1'b1;

    //-------------------------------------------------------------------------
    // SNN Accelerator Instance
    //-------------------------------------------------------------------------
    snn_accelerator_hls_top #(
        .C_S_AXI_DATA_WIDTH(32),
        .C_S_AXI_ADDR_WIDTH(7),
        .C_AXIS_DATA_WIDTH(32),
        .NUM_NEURONS(256),
        .NUM_AXONS(1024),
        .NUM_PARALLEL_UNITS(8)
    ) u_snn_accelerator (
        // Clock and Reset
        .aclk           (clk_100mhz),
        .aresetn        (sys_rst_n),
        
        // AXI4-Lite Slave
        .s_axi_awaddr   (s_axi_awaddr),
        .s_axi_awprot   (s_axi_awprot),
        .s_axi_awvalid  (s_axi_awvalid),
        .s_axi_awready  (s_axi_awready),
        .s_axi_wdata    (s_axi_wdata),
        .s_axi_wstrb    (s_axi_wstrb),
        .s_axi_wvalid   (s_axi_wvalid),
        .s_axi_wready   (s_axi_wready),
        .s_axi_bresp    (s_axi_bresp),
        .s_axi_bvalid   (s_axi_bvalid),
        .s_axi_bready   (s_axi_bready),
        .s_axi_araddr   (s_axi_araddr),
        .s_axi_arprot   (s_axi_arprot),
        .s_axi_arvalid  (s_axi_arvalid),
        .s_axi_arready  (s_axi_arready),
        .s_axi_rdata    (s_axi_rdata),
        .s_axi_rresp    (s_axi_rresp),
        .s_axi_rvalid   (s_axi_rvalid),
        .s_axi_rready   (s_axi_rready),
        
        // AXI4-Stream Slave (Input Spikes)
        .s_axis_tdata   (s_axis_tdata),
        .s_axis_tvalid  (s_axis_tvalid),
        .s_axis_tready  (s_axis_tready),
        .s_axis_tkeep   (s_axis_tkeep),
        .s_axis_tstrb   (s_axis_tstrb),
        .s_axis_tuser   (s_axis_tuser),
        .s_axis_tlast   (s_axis_tlast),
        .s_axis_tid     (s_axis_tid),
        .s_axis_tdest   (s_axis_tdest),
        
        // AXI4-Stream Master (Output Spikes)
        .m_axis_tdata   (m_axis_tdata),
        .m_axis_tvalid  (m_axis_tvalid),
        .m_axis_tready  (m_axis_tready),
        .m_axis_tkeep   (m_axis_tkeep),
        .m_axis_tstrb   (m_axis_tstrb),
        .m_axis_tuser   (m_axis_tuser),
        .m_axis_tlast   (m_axis_tlast),
        .m_axis_tid     (m_axis_tid),
        .m_axis_tdest   (m_axis_tdest),
        
        // Interrupt
        .interrupt      (interrupt),
        
        // Board I/O
        .led            (led),
        .sw             (sw),
        .btn            (btn),
        .led4_r         (led4_r),
        .led4_g         (led4_g),
        .led4_b         (led4_b),
        .led5_r         (led5_r),
        .led5_g         (led5_g),
        .led5_b         (led5_b)
    );

endmodule
