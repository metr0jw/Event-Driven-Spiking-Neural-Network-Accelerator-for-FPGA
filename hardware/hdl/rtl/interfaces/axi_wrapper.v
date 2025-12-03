//-----------------------------------------------------------------------------
// Title         : AXI4-Lite and AXI4-Stream Wrapper
// Project       : PYNQ-Z2 SNN Accelerator
// File          : axi_wrapper.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : AXI interfaces for PS-PL communication
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module axi_wrapper #(
    parameter C_S_AXI_DATA_WIDTH = 32,
    parameter C_S_AXI_ADDR_WIDTH = 32,
    parameter C_AXIS_DATA_WIDTH  = 32,
    parameter NUM_NEURONS        = 64
)(
    // AXI4-Lite Slave Interface
    input  wire                             s_axi_aclk,
    input  wire                             s_axi_aresetn,
    // Write Address Channel
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]   s_axi_awaddr,
    input  wire [2:0]                       s_axi_awprot,
    input  wire                             s_axi_awvalid,
    output wire                             s_axi_awready,
    // Write Data Channel
    input  wire [C_S_AXI_DATA_WIDTH-1:0]   s_axi_wdata,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire                             s_axi_wvalid,
    output wire                             s_axi_wready,
    // Write Response Channel
    output wire [1:0]                       s_axi_bresp,
    output wire                             s_axi_bvalid,
    input  wire                             s_axi_bready,
    // Read Address Channel
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]   s_axi_araddr,
    input  wire [2:0]                       s_axi_arprot,
    input  wire                             s_axi_arvalid,
    output wire                             s_axi_arready,
    // Read Data Channel
    output wire [C_S_AXI_DATA_WIDTH-1:0]   s_axi_rdata,
    output wire [1:0]                       s_axi_rresp,
    output wire                             s_axi_rvalid,
    input  wire                             s_axi_rready,
    
    // AXI4-Stream Slave Interface (Input spikes)
    input  wire [C_AXIS_DATA_WIDTH-1:0]     s_axis_tdata,
    input  wire                             s_axis_tvalid,
    output wire                             s_axis_tready,
    input  wire                             s_axis_tlast,
    
    // AXI4-Stream Master Interface (Output spikes)
    output wire [C_AXIS_DATA_WIDTH-1:0]     m_axis_tdata,
    output wire                             m_axis_tvalid,
    input  wire                             m_axis_tready,
    output wire                             m_axis_tlast,
    
    // SNN Control/Status Interface
    output wire [31:0]                      ctrl_reg,
    output wire [31:0]                      config_reg,
    output wire [15:0]                      leak_rate,
    output wire [15:0]                      threshold,
    output wire [15:0]                      refractory_period,
    input  wire [31:0]                      status_reg,
    input  wire [31:0]                      spike_count,
    
    // Spike Router Interface
    output wire                             spike_in_valid,
    output wire [7:0]                       spike_in_neuron_id,
    output wire [7:0]                       spike_in_weight,
    input  wire                             spike_in_ready,
    
    input  wire                             spike_out_valid,
    input  wire [7:0]                       spike_out_neuron_id,
    output wire                             spike_out_ready
);

    //-------------------------------------------------------------------------
    // AXI4-Lite Control Register IP
    //-------------------------------------------------------------------------
    wire [31:0] ctrl_reg_int;
    wire [31:0] config_reg_int;
    wire [15:0] leak_rate_int;
    wire [15:0] threshold_int;
    wire [15:0] refractory_period_int;

    wire                             axi_awready_int;
    wire                             axi_wready_int;
    wire [1:0]                       axi_bresp_int;
    wire                             axi_bvalid_int;
    wire                             axi_arready_int;
    wire [C_S_AXI_DATA_WIDTH-1:0]    axi_rdata_int;
    wire [1:0]                       axi_rresp_int;
    wire                             axi_rvalid_int;

    axi_lite_regs_v1_0 #(
        .C_S00_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH),
        .C_S00_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH)
    ) control_regs_inst (
        .ctrl_reg(ctrl_reg_int),
        .config_reg(config_reg_int),
        .leak_rate(leak_rate_int),
        .threshold(threshold_int),
        .refractory_period(refractory_period_int),
        .status_reg(status_reg),
        .spike_count(spike_count),
        .s00_axi_aclk(s_axi_aclk),
        .s00_axi_aresetn(s_axi_aresetn),
        .s00_axi_awaddr(s_axi_awaddr[C_S_AXI_ADDR_WIDTH-1:0]),
        .s00_axi_awprot(s_axi_awprot),
        .s00_axi_awvalid(s_axi_awvalid),
        .s00_axi_awready(axi_awready_int),
        .s00_axi_wdata(s_axi_wdata),
        .s00_axi_wstrb(s_axi_wstrb),
        .s00_axi_wvalid(s_axi_wvalid),
        .s00_axi_wready(axi_wready_int),
        .s00_axi_bresp(axi_bresp_int),
        .s00_axi_bvalid(axi_bvalid_int),
        .s00_axi_bready(s_axi_bready),
        .s00_axi_araddr(s_axi_araddr[C_S_AXI_ADDR_WIDTH-1:0]),
        .s00_axi_arprot(s_axi_arprot),
        .s00_axi_arvalid(s_axi_arvalid),
        .s00_axi_arready(axi_arready_int),
        .s00_axi_rdata(axi_rdata_int),
        .s00_axi_rresp(axi_rresp_int),
        .s00_axi_rvalid(axi_rvalid_int),
        .s00_axi_rready(s_axi_rready)
    );

    assign ctrl_reg = ctrl_reg_int;
    assign config_reg = config_reg_int;
    assign leak_rate = leak_rate_int;
    assign threshold = threshold_int;
    assign refractory_period = refractory_period_int;

    assign s_axi_awready = axi_awready_int;
    assign s_axi_wready  = axi_wready_int;
    assign s_axi_bresp   = axi_bresp_int;
    assign s_axi_bvalid  = axi_bvalid_int;
    assign s_axi_arready = axi_arready_int;
    assign s_axi_rdata   = axi_rdata_int;
    assign s_axi_rresp   = axi_rresp_int;
    assign s_axi_rvalid  = axi_rvalid_int;
    
    //-------------------------------------------------------------------------
    // AXI4-Stream Interface
    //-------------------------------------------------------------------------
    // Input spike parsing
    assign spike_in_valid = s_axis_tvalid;
    assign spike_in_neuron_id = s_axis_tdata[7:0];
    assign spike_in_weight = s_axis_tdata[15:8];
    assign s_axis_tready = spike_in_ready;

    // Output spike formatting
    reg axis_tvalid_reg;
    reg [31:0] axis_tdata_reg;

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axis_tvalid_reg <= 1'b0;
            axis_tdata_reg <= 32'd0;
        end else begin
            if (spike_out_valid && !axis_tvalid_reg) begin
                axis_tvalid_reg <= 1'b1;
                axis_tdata_reg <= {16'd0, 8'd0, spike_out_neuron_id};
            end else if (axis_tvalid_reg && m_axis_tready) begin
                axis_tvalid_reg <= 1'b0;
            end
        end
    end

    assign m_axis_tvalid = axis_tvalid_reg;
    assign m_axis_tdata = axis_tdata_reg;
    assign m_axis_tlast = axis_tvalid_reg;
    assign spike_out_ready = !axis_tvalid_reg || m_axis_tready;

endmodule
