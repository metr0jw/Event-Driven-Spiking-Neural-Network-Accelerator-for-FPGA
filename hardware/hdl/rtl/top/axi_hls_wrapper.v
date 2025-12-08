//-----------------------------------------------------------------------------
// Map axi_hls_wrapper instance to the HLS-generated snn_top_hls module.
// This avoids hand-written AXI logic while satisfying existing RTL hierarchy.
//-----------------------------------------------------------------------------
`timescale 1ns/1ps

module axi_hls_wrapper (
    input  wire         ap_clk,
    input  wire         ap_rst_n,
    // AXI4-Stream Spike Input
    input  wire [31:0]  s_axis_spikes_TDATA,
    input  wire         s_axis_spikes_TVALID,
    output wire         s_axis_spikes_TREADY,
    input  wire [3:0]   s_axis_spikes_TKEEP,
    input  wire [3:0]   s_axis_spikes_TSTRB,
    input  wire         s_axis_spikes_TUSER,
    input  wire         s_axis_spikes_TLAST,
    input  wire         s_axis_spikes_TID,
    input  wire         s_axis_spikes_TDEST,
    // AXI4-Stream Raw Data Input (on-chip encoder)
    input  wire [559:0] s_axis_data_TDATA,
    input  wire         s_axis_data_TVALID,
    output wire         s_axis_data_TREADY,
    input  wire [69:0]  s_axis_data_TKEEP,
    input  wire [69:0]  s_axis_data_TSTRB,
    input  wire         s_axis_data_TUSER,
    input  wire         s_axis_data_TLAST,
    input  wire         s_axis_data_TID,
    input  wire         s_axis_data_TDEST,
    // AXI4-Stream Spike Output
    output wire [31:0]  m_axis_spikes_TDATA,
    output wire         m_axis_spikes_TVALID,
    input  wire         m_axis_spikes_TREADY,
    output wire [3:0]   m_axis_spikes_TKEEP,
    output wire [3:0]   m_axis_spikes_TSTRB,
    output wire         m_axis_spikes_TUSER,
    output wire         m_axis_spikes_TLAST,
    output wire         m_axis_spikes_TID,
    output wire         m_axis_spikes_TDEST,
    // Direct spike interface to RTL core
    output wire         spike_in_valid,
    output wire [9:0]   spike_in_neuron_id,
    output wire [7:0]   spike_in_weight,
    input  wire         spike_in_ready,
    input  wire         spike_out_valid,
    input  wire [9:0]   spike_out_neuron_id,
    input  wire [7:0]   spike_out_weight,
    output wire         spike_out_ready,
    // Control/status
    output wire         snn_enable,
    output wire         snn_reset,
    output wire         clear_counters,
    output wire [15:0]  threshold_out,
    output wire [15:0]  leak_rate_out,
    output wire [15:0]  refractory_out,
    input  wire         snn_ready,
    input  wire         snn_busy,
    input  wire         snn_error,
    input  wire [31:0]  snn_spike_count,
    // AXI4-Lite control
    input  wire         s_axi_ctrl_AWVALID,
    output wire         s_axi_ctrl_AWREADY,
    input  wire [6:0]   s_axi_ctrl_AWADDR,
    input  wire         s_axi_ctrl_WVALID,
    output wire         s_axi_ctrl_WREADY,
    input  wire [31:0]  s_axi_ctrl_WDATA,
    input  wire [3:0]   s_axi_ctrl_WSTRB,
    input  wire         s_axi_ctrl_ARVALID,
    output wire         s_axi_ctrl_ARREADY,
    input  wire [6:0]   s_axi_ctrl_ARADDR,
    output wire         s_axi_ctrl_RVALID,
    input  wire         s_axi_ctrl_RREADY,
    output wire [31:0]  s_axi_ctrl_RDATA,
    output wire [1:0]   s_axi_ctrl_RRESP,
    output wire         s_axi_ctrl_BVALID,
    input  wire         s_axi_ctrl_BREADY,
    output wire [1:0]   s_axi_ctrl_BRESP,
    output wire         interrupt
);
    // Unused weight-stream from HLS IP, tie off ready and drop payload.
    wire [31:0] m_axis_weights_TDATA;
    wire        m_axis_weights_TVALID;
    wire [3:0]  m_axis_weights_TKEEP;
    wire [3:0]  m_axis_weights_TSTRB;
    wire        m_axis_weights_TUSER;
    wire        m_axis_weights_TLAST;
    wire        m_axis_weights_TID;
    wire        m_axis_weights_TDEST;
    wire [7:0]  reward_signal_unused;

    // Tie-offs for unused optional ports
    assign reward_signal_unused = 8'd0;

    // Width adapters between 10-bit RTL IDs and 8-bit HLS core IDs
    wire [7:0]  spike_in_neuron_id_hls  = spike_in_neuron_id[7:0];
    wire [7:0]  spike_out_neuron_id_hls;

    assign spike_out_neuron_id = {2'b00, spike_out_neuron_id_hls};

    snn_top_hls u_ip (
        .ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),
        .s_axis_spikes_TDATA(s_axis_spikes_TDATA),
        .s_axis_spikes_TVALID(s_axis_spikes_TVALID),
        .s_axis_spikes_TREADY(s_axis_spikes_TREADY),
        .s_axis_spikes_TKEEP(s_axis_spikes_TKEEP),
        .s_axis_spikes_TSTRB(s_axis_spikes_TSTRB),
        .s_axis_spikes_TUSER(s_axis_spikes_TUSER),
        .s_axis_spikes_TLAST(s_axis_spikes_TLAST),
        .s_axis_spikes_TID(s_axis_spikes_TID),
        .s_axis_spikes_TDEST(s_axis_spikes_TDEST),
        .s_axis_data_TDATA(s_axis_data_TDATA),
        .s_axis_data_TVALID(s_axis_data_TVALID),
        .s_axis_data_TREADY(s_axis_data_TREADY),
        .s_axis_data_TKEEP(s_axis_data_TKEEP),
        .s_axis_data_TSTRB(s_axis_data_TSTRB),
        .s_axis_data_TUSER(s_axis_data_TUSER),
        .s_axis_data_TLAST(s_axis_data_TLAST),
        .s_axis_data_TID(s_axis_data_TID),
        .s_axis_data_TDEST(s_axis_data_TDEST),
        .m_axis_spikes_TDATA(m_axis_spikes_TDATA),
        .m_axis_spikes_TVALID(m_axis_spikes_TVALID),
        .m_axis_spikes_TREADY(m_axis_spikes_TREADY),
        .m_axis_spikes_TKEEP(m_axis_spikes_TKEEP),
        .m_axis_spikes_TSTRB(m_axis_spikes_TSTRB),
        .m_axis_spikes_TUSER(m_axis_spikes_TUSER),
        .m_axis_spikes_TLAST(m_axis_spikes_TLAST),
        .m_axis_spikes_TID(m_axis_spikes_TID),
        .m_axis_spikes_TDEST(m_axis_spikes_TDEST),
        .m_axis_weights_TDATA(m_axis_weights_TDATA),
        .m_axis_weights_TVALID(m_axis_weights_TVALID),
        .m_axis_weights_TREADY(1'b1),
        .m_axis_weights_TKEEP(m_axis_weights_TKEEP),
        .m_axis_weights_TSTRB(m_axis_weights_TSTRB),
        .m_axis_weights_TUSER(m_axis_weights_TUSER),
        .m_axis_weights_TLAST(m_axis_weights_TLAST),
        .m_axis_weights_TID(m_axis_weights_TID),
        .m_axis_weights_TDEST(m_axis_weights_TDEST),
        .spike_in_valid(spike_in_valid),
        .spike_in_neuron_id(spike_in_neuron_id_hls),
        .spike_in_weight(spike_in_weight),
        .spike_in_ready(spike_in_ready),
        .spike_out_valid(spike_out_valid),
        .spike_out_neuron_id(spike_out_neuron_id_hls),
        .spike_out_weight(spike_out_weight),
        .spike_out_ready(spike_out_ready),
        .snn_enable(snn_enable),
        .snn_reset(snn_reset),
        .threshold_out(threshold_out),
        .leak_rate_out(leak_rate_out),
        .snn_ready(snn_ready),
        .snn_busy(snn_busy),
        .s_axi_ctrl_AWVALID(s_axi_ctrl_AWVALID),
        .s_axi_ctrl_AWREADY(s_axi_ctrl_AWREADY),
        .s_axi_ctrl_AWADDR(s_axi_ctrl_AWADDR),
        .s_axi_ctrl_WVALID(s_axi_ctrl_WVALID),
        .s_axi_ctrl_WREADY(s_axi_ctrl_WREADY),
        .s_axi_ctrl_WDATA(s_axi_ctrl_WDATA),
        .s_axi_ctrl_WSTRB(s_axi_ctrl_WSTRB),
        .s_axi_ctrl_ARVALID(s_axi_ctrl_ARVALID),
        .s_axi_ctrl_ARREADY(s_axi_ctrl_ARREADY),
        .s_axi_ctrl_ARADDR(s_axi_ctrl_ARADDR),
        .s_axi_ctrl_RVALID(s_axi_ctrl_RVALID),
        .s_axi_ctrl_RREADY(s_axi_ctrl_RREADY),
        .s_axi_ctrl_RDATA(s_axi_ctrl_RDATA),
        .s_axi_ctrl_RRESP(s_axi_ctrl_RRESP),
        .s_axi_ctrl_BVALID(s_axi_ctrl_BVALID),
        .s_axi_ctrl_BREADY(s_axi_ctrl_BREADY),
        .s_axi_ctrl_BRESP(s_axi_ctrl_BRESP),
        .interrupt(interrupt)
    );

    // Features not provided by the current HLS IP: tie or zero for simulation.
    assign clear_counters  = 1'b0;
    assign refractory_out  = 16'd0;
    // snn_error and snn_spike_count are status inputs into the HLS IP in RTL; tie-off here.
    // The wrapper does not consume them, so they are left unused by design.
endmodule
