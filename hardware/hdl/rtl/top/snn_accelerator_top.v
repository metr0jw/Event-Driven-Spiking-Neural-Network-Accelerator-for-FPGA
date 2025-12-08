//-----------------------------------------------------------------------------
// Title         : SNN Accelerator Top Module for PYNQ-Z2
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_accelerator_top.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Top-level IP core for Zynq-based SNN accelerator
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module snn_accelerator_top #(
    parameter C_S_AXI_DATA_WIDTH    = 32,
    parameter C_S_AXI_ADDR_WIDTH    = 32,
    parameter C_AXIS_DATA_WIDTH     = 32,
    parameter NUM_NEURONS           = 256,
    parameter NUM_AXONS             = 1024,
    parameter NUM_PARALLEL_UNITS    = 8,
    parameter SPIKE_BUFFER_DEPTH    = 64,
    parameter NEURON_ID_WIDTH       = $clog2(NUM_NEURONS),
    parameter AXON_ID_WIDTH         = 10,
    parameter DATA_WIDTH            = 16,
    parameter WEIGHT_WIDTH          = 8,
    parameter LEAK_WIDTH            = 8,
    parameter THRESHOLD_WIDTH       = 16,
    parameter REFRAC_WIDTH          = 8,
    parameter ROUTER_BUFFER_DEPTH   = 512,
    parameter USE_BRAM              = 1,
    parameter USE_DSP               = 1
)(
    input  wire                          aclk,
    input  wire                          aresetn,

    // AXI4-Lite Slave Interface (Configuration)
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    input  wire [2:0]                    s_axi_awprot,
    input  wire                          s_axi_awvalid,
    output wire                          s_axi_awready,
    input  wire [C_S_AXI_DATA_WIDTH-1:0] s_axi_wdata,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire                          s_axi_wvalid,
    output wire                          s_axi_wready,
    output wire [1:0]                    s_axi_bresp,
    output wire                          s_axi_bvalid,
    input  wire                          s_axi_bready,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    input  wire [2:0]                    s_axi_arprot,
    input  wire                          s_axi_arvalid,
    output wire                          s_axi_arready,
    output wire [C_S_AXI_DATA_WIDTH-1:0] s_axi_rdata,
    output wire [1:0]                    s_axi_rresp,
    output wire                          s_axi_rvalid,
    input  wire                          s_axi_rready,

    // AXI4-Stream Slave Interface (Input Spikes from PS)
    input  wire [C_AXIS_DATA_WIDTH-1:0]  s_axis_tdata,
    input  wire                          s_axis_tvalid,
    output wire                          s_axis_tready,
    input  wire [3:0]                    s_axis_tkeep,
    input  wire [3:0]                    s_axis_tstrb,
    input  wire                          s_axis_tuser,
    input  wire                          s_axis_tlast,
    input  wire                          s_axis_tid,
    input  wire                          s_axis_tdest,

    // AXI4-Stream Master Interface (Output Spikes to PS)
    output wire [C_AXIS_DATA_WIDTH-1:0]  m_axis_tdata,
    output wire                          m_axis_tvalid,
    input  wire                          m_axis_tready,
    output wire [3:0]                    m_axis_tkeep,
    output wire [3:0]                    m_axis_tstrb,
    output wire                          m_axis_tuser,
    output wire                          m_axis_tlast,
    output wire                          m_axis_tid,
    output wire                          m_axis_tdest,

    // Interrupt to PS
    output wire                          interrupt,

    // Board I/O (Optional)
    output wire [3:0]                    led,
    input  wire [1:0]                    sw,
    input  wire [3:0]                    btn,
    output wire                          led4_r,
    output wire                          led4_g,
    output wire                          led4_b,
    output wire                          led5_r,
    output wire                          led5_g,
    output wire                          led5_b
);

    // Reuse the HLS-based top that already encapsulates all AXI handling.
    snn_accelerator_hls_top #(
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(7),
        .C_AXIS_DATA_WIDTH(C_AXIS_DATA_WIDTH),
        .NUM_NEURONS(NUM_NEURONS),
        .NUM_AXONS(NUM_AXONS),
        .NUM_PARALLEL_UNITS(NUM_PARALLEL_UNITS),
        .SPIKE_BUFFER_DEPTH(SPIKE_BUFFER_DEPTH),
        .NEURON_ID_WIDTH(NEURON_ID_WIDTH),
        .AXON_ID_WIDTH(AXON_ID_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .LEAK_WIDTH(LEAK_WIDTH),
        .THRESHOLD_WIDTH(THRESHOLD_WIDTH),
        .REFRAC_WIDTH(REFRAC_WIDTH),
        .ROUTER_BUFFER_DEPTH(ROUTER_BUFFER_DEPTH),
        .USE_BRAM(USE_BRAM),
        .USE_DSP(USE_DSP)
    ) u_hls_top (
        .aclk(aclk),
        .aresetn(aresetn),

        // AXI4-Lite
        .s_axi_awaddr(s_axi_awaddr[6:0]),
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

        // Interrupt and board I/O
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

endmodule
