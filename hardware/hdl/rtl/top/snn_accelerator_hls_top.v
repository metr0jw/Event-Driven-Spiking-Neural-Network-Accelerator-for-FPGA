//-----------------------------------------------------------------------------
// Title         : SNN Accelerator Top Module with HLS AXI Wrapper
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_accelerator_hls_top.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Top-level module using HLS-generated AXI wrapper for
//                 more reliable PS-PL communication
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module snn_accelerator_hls_top #(
    // AXI Parameters
    parameter C_S_AXI_DATA_WIDTH    = 32,
    parameter C_S_AXI_ADDR_WIDTH    = 7,  // HLS uses 7-bit address
    parameter C_AXIS_DATA_WIDTH     = 32,
    
    // SNN Parameters
    parameter NUM_NEURONS           = 256,
    parameter NUM_AXONS             = 256,
    parameter NUM_PARALLEL_UNITS    = 8,
    parameter SPIKE_BUFFER_DEPTH    = 64,
    parameter NEURON_ID_WIDTH       = 8,
    parameter AXON_ID_WIDTH         = 8,
    parameter DATA_WIDTH            = 16,
    parameter WEIGHT_WIDTH          = 8,
    parameter LEAK_WIDTH            = 8,
    parameter THRESHOLD_WIDTH       = 16,
    parameter REFRAC_WIDTH          = 8,
    parameter ROUTER_BUFFER_DEPTH   = 512,
    parameter USE_BRAM              = 1,
    parameter USE_DSP               = 1
)(
    //-------------------------------------------------------------------------
    // Clock and Reset
    //-------------------------------------------------------------------------
    input  wire                          aclk,
    input  wire                          aresetn,
    
    //-------------------------------------------------------------------------
    // AXI4-Lite Slave Interface (Configuration) - To HLS wrapper
    //-------------------------------------------------------------------------
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
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
    input  wire                          s_axi_arvalid,
    output wire                          s_axi_arready,
    output wire [C_S_AXI_DATA_WIDTH-1:0] s_axi_rdata,
    output wire [1:0]                    s_axi_rresp,
    output wire                          s_axi_rvalid,
    input  wire                          s_axi_rready,
    
    //-------------------------------------------------------------------------
    // AXI4-Stream Slave Interface (Input Spikes from PS)
    //-------------------------------------------------------------------------
    input  wire [C_AXIS_DATA_WIDTH-1:0]  s_axis_tdata,
    input  wire                          s_axis_tvalid,
    output wire                          s_axis_tready,
    input  wire [3:0]                    s_axis_tkeep,
    input  wire [3:0]                    s_axis_tstrb,
    input  wire                          s_axis_tuser,
    input  wire                          s_axis_tlast,
    input  wire                          s_axis_tid,
    input  wire                          s_axis_tdest,
    
    //-------------------------------------------------------------------------
    // AXI4-Stream Master Interface (Output Spikes to PS)
    //-------------------------------------------------------------------------
    output wire [C_AXIS_DATA_WIDTH-1:0]  m_axis_tdata,
    output wire                          m_axis_tvalid,
    input  wire                          m_axis_tready,
    output wire [3:0]                    m_axis_tkeep,
    output wire [3:0]                    m_axis_tstrb,
    output wire                          m_axis_tuser,
    output wire                          m_axis_tlast,
    output wire                          m_axis_tid,
    output wire                          m_axis_tdest,
    
    //-------------------------------------------------------------------------
    // Interrupt to PS
    //-------------------------------------------------------------------------
    output wire                          interrupt,
    
    //-------------------------------------------------------------------------
    // Board I/O (Optional)
    //-------------------------------------------------------------------------
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

    //-------------------------------------------------------------------------
    // Internal Signals
    //-------------------------------------------------------------------------
    wire                         sys_clk;
    wire                         sys_rst_n;
    
    // HLS wrapper outputs (control signals to SNN core)
    wire                         snn_enable;
    wire                         snn_reset;
    wire                         clear_counters;
    wire [15:0]                  leak_rate;
    wire [15:0]                  threshold;
    wire [15:0]                  refractory_period;
    
    // HLS wrapper inputs (status from SNN core)
    wire                         snn_ready;
    wire                         snn_busy;
    wire                         snn_error;
    wire [31:0]                  snn_spike_count;
    
    // Spike interfaces from HLS wrapper to SNN core
    wire                         spike_in_valid;
    wire [NEURON_ID_WIDTH-1:0]   spike_in_neuron_id;
    wire signed [WEIGHT_WIDTH-1:0] spike_in_weight;
    wire                         spike_in_ready;
    
    // Spike interfaces from SNN core to HLS wrapper
    wire                         spike_out_valid;
    wire [NEURON_ID_WIDTH-1:0]   spike_out_neuron_id;
    wire signed [WEIGHT_WIDTH-1:0] spike_out_weight;
    wire                         spike_out_ready;
    
    // Internal spike routing
    wire                         routed_spike_valid;
    wire [NEURON_ID_WIDTH-1:0]   routed_spike_dest_id;
    wire [WEIGHT_WIDTH-1:0]      routed_spike_weight;
    wire                         routed_spike_exc_inh;
    wire                         routed_spike_ready;
    
    wire                         neuron_spike_valid;
    wire [NEURON_ID_WIDTH-1:0]   neuron_spike_id;
    wire                         neuron_spike_ready;
    
    // Status signals
    wire [31:0]                  neuron_spike_count;
    wire [31:0]                  routed_spike_count;
    wire [31:0]                  throughput_counter;
    wire [7:0]                   active_neurons;
    wire                         router_busy;
    wire                         fifo_overflow;
    wire                         array_busy;
    wire                         synapse_busy;

    //-------------------------------------------------------------------------
    // Clock and Reset
    //-------------------------------------------------------------------------
    assign sys_clk = aclk;
    
    reset_sync reset_sync_inst (
        .clk(sys_clk),
        .async_rst_n(aresetn),
        .sync_rst_n(sys_rst_n)
    );
    
    //-------------------------------------------------------------------------
    // HLS AXI Wrapper Instance
    //-------------------------------------------------------------------------
    axi_hls_wrapper axi_hls_wrapper_inst (
        .ap_clk(aclk),
        .ap_rst_n(aresetn),
        
        // AXI4-Lite Control Interface
        .s_axi_ctrl_AWVALID(s_axi_awvalid),
        .s_axi_ctrl_AWREADY(s_axi_awready),
        .s_axi_ctrl_AWADDR(s_axi_awaddr),
        .s_axi_ctrl_WVALID(s_axi_wvalid),
        .s_axi_ctrl_WREADY(s_axi_wready),
        .s_axi_ctrl_WDATA(s_axi_wdata),
        .s_axi_ctrl_WSTRB(s_axi_wstrb),
        .s_axi_ctrl_ARVALID(s_axi_arvalid),
        .s_axi_ctrl_ARREADY(s_axi_arready),
        .s_axi_ctrl_ARADDR(s_axi_araddr),
        .s_axi_ctrl_RVALID(s_axi_rvalid),
        .s_axi_ctrl_RREADY(s_axi_rready),
        .s_axi_ctrl_RDATA(s_axi_rdata),
        .s_axi_ctrl_RRESP(s_axi_rresp),
        .s_axi_ctrl_BVALID(s_axi_bvalid),
        .s_axi_ctrl_BREADY(s_axi_bready),
        .s_axi_ctrl_BRESP(s_axi_bresp),
        
        // AXI4-Stream Spike Input
        .s_axis_spikes_TVALID(s_axis_tvalid),
        .s_axis_spikes_TREADY(s_axis_tready),
        .s_axis_spikes_TDATA(s_axis_tdata),
        .s_axis_spikes_TKEEP(s_axis_tkeep),
        .s_axis_spikes_TSTRB(s_axis_tstrb),
        .s_axis_spikes_TUSER(s_axis_tuser),
        .s_axis_spikes_TLAST(s_axis_tlast),
        .s_axis_spikes_TID(s_axis_tid),
        .s_axis_spikes_TDEST(s_axis_tdest),
        
        // AXI4-Stream Spike Output
        .m_axis_spikes_TREADY(m_axis_tready),
        .m_axis_spikes_TVALID(m_axis_tvalid),
        .m_axis_spikes_TDATA(m_axis_tdata),
        .m_axis_spikes_TKEEP(m_axis_tkeep),
        .m_axis_spikes_TSTRB(m_axis_tstrb),
        .m_axis_spikes_TUSER(m_axis_tuser),
        .m_axis_spikes_TLAST(m_axis_tlast),
        .m_axis_spikes_TID(m_axis_tid),
        .m_axis_spikes_TDEST(m_axis_tdest),
        
        // Direct wire interface to Verilog SNN core
        .spike_in_valid(spike_in_valid),
        .spike_in_neuron_id(spike_in_neuron_id),
        .spike_in_weight(spike_in_weight),
        .spike_in_ready(spike_in_ready),
        
        .spike_out_valid(spike_out_valid),
        .spike_out_neuron_id(spike_out_neuron_id),
        .spike_out_weight(spike_out_weight),
        .spike_out_ready(spike_out_ready),
        
        // Control signals
        .snn_enable(snn_enable),
        .snn_reset(snn_reset),
        .clear_counters(clear_counters),
        .leak_rate_out(leak_rate),
        .threshold_out(threshold),
        .refractory_out(refractory_period),
        
        // Status signals
        .snn_ready(snn_ready),
        .snn_busy(snn_busy),
        .snn_error(snn_error),
        .snn_spike_count(snn_spike_count),
        
        // Interrupt
        .interrupt(interrupt)
    );
    
    //-------------------------------------------------------------------------
    // Status Signal Assembly
    //-------------------------------------------------------------------------
    assign snn_ready = ~(array_busy | synapse_busy | router_busy);
    assign snn_busy = array_busy | synapse_busy | router_busy;
    assign snn_error = fifo_overflow;
    assign snn_spike_count = neuron_spike_count;
    
    //-------------------------------------------------------------------------
    // Synapse Array
    //-------------------------------------------------------------------------
    synapse_array #(
        .NUM_AXONS(NUM_AXONS),
        .NUM_NEURONS(NUM_NEURONS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .NUM_READ_PORTS(NUM_PARALLEL_UNITS),
        .USE_BRAM(USE_BRAM),
        .USE_DSP(USE_DSP)
    ) synapse_array_inst (
        .clk(sys_clk),
        .rst_n(sys_rst_n & ~snn_reset),
        
        .spike_in_valid(spike_in_valid),
        .spike_in_axon_id(spike_in_neuron_id[AXON_ID_WIDTH-1:0]),
        .spike_in_ready(spike_in_ready),
        
        .spike_out_valid(routed_spike_valid),
        .spike_out_neuron_id(routed_spike_dest_id),
        .spike_out_weight(routed_spike_weight),
        .spike_out_exc_inh(routed_spike_exc_inh),
        
        .weight_we(1'b0),
        .weight_addr_axon({AXON_ID_WIDTH{1'b0}}),
        .weight_addr_neuron({NEURON_ID_WIDTH{1'b0}}),
        .weight_data({WEIGHT_WIDTH+1{1'b0}}),
        
        .batch_we(1'b0),
        .batch_axon_id({AXON_ID_WIDTH{1'b0}}),
        .batch_weights({NUM_PARALLEL_UNITS*(WEIGHT_WIDTH+1){1'b0}}),
        .batch_start_neuron({NEURON_ID_WIDTH{1'b0}}),
        
        .enable(snn_enable),
        .busy(synapse_busy),
        .spike_count()
    );
    
    //-------------------------------------------------------------------------
    // LIF Neuron Array
    //-------------------------------------------------------------------------
    lif_neuron_array #(
        .NUM_NEURONS(NUM_NEURONS),
        .NUM_AXONS(NUM_AXONS),
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .THRESHOLD_WIDTH(THRESHOLD_WIDTH),
        .LEAK_WIDTH(LEAK_WIDTH),
        .REFRAC_WIDTH(REFRAC_WIDTH),
        .NUM_PARALLEL_UNITS(NUM_PARALLEL_UNITS),
        .SPIKE_BUFFER_DEPTH(SPIKE_BUFFER_DEPTH),
        .USE_BRAM(USE_BRAM),
        .USE_DSP(USE_DSP)
    ) neuron_array_inst (
        .clk(sys_clk),
        .rst_n(sys_rst_n & ~snn_reset),
        .enable(snn_enable),
        
        .s_axis_spike_valid(routed_spike_valid),
        .s_axis_spike_dest_id(routed_spike_dest_id),
        .s_axis_spike_weight(routed_spike_weight),
        .s_axis_spike_exc_inh(routed_spike_exc_inh),
        .s_axis_spike_ready(routed_spike_ready),
        
        .m_axis_spike_valid(neuron_spike_valid),
        .m_axis_spike_neuron_id(neuron_spike_id),
        .m_axis_spike_ready(neuron_spike_ready),
        
        .config_we(1'b0),
        .config_addr({NEURON_ID_WIDTH{1'b0}}),
        .config_data(32'd0),
        
        .global_threshold(threshold),
        .global_leak_rate(leak_rate[LEAK_WIDTH-1:0]),
        .global_refrac_period(refractory_period[REFRAC_WIDTH-1:0]),
        
        .spike_count(neuron_spike_count),
        .array_busy(array_busy),
        .throughput_counter(throughput_counter),
        .active_neurons(active_neurons)
    );
    
    //-------------------------------------------------------------------------
    // Spike Router
    //-------------------------------------------------------------------------
    spike_router #(
        .NUM_NEURONS(NUM_NEURONS),
        .MAX_FANOUT(32),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .FIFO_DEPTH(ROUTER_BUFFER_DEPTH)
    ) spike_router_inst (
        .clk(sys_clk),
        .rst_n(sys_rst_n & ~snn_reset),
        
        .s_spike_valid(neuron_spike_valid),
        .s_spike_neuron_id(neuron_spike_id),
        .s_spike_ready(neuron_spike_ready),
        
        .m_spike_valid(spike_out_valid),
        .m_spike_dest_id(spike_out_neuron_id),
        .m_spike_weight(spike_out_weight),
        .m_spike_exc_inh(),
        .m_spike_ready(spike_out_ready),
        
        .config_we(1'b0),
        .config_addr(32'd0),
        .config_data(32'd0),
        .config_readdata(),
        
        .routed_spike_count(routed_spike_count),
        .router_busy(router_busy),
        .fifo_overflow(fifo_overflow)
    );
    
    //-------------------------------------------------------------------------
    // LED Status Indicators
    //-------------------------------------------------------------------------
    reg [23:0] heartbeat_counter;
    always @(posedge sys_clk) begin
        if (!sys_rst_n)
            heartbeat_counter <= 24'd0;
        else
            heartbeat_counter <= heartbeat_counter + 1'b1;
    end
    
    assign led[0] = heartbeat_counter[23];
    assign led[1] = snn_enable;
    assign led[2] = |neuron_spike_count[10:0];
    assign led[3] = fifo_overflow;
    
    assign led4_g = snn_enable & ~snn_reset;
    assign led4_r = snn_reset;
    assign led4_b = snn_busy;
    
    reg [15:0] activity_pwm;
    always @(posedge sys_clk) begin
        if (!sys_rst_n)
            activity_pwm <= 16'd0;
        else
            activity_pwm <= activity_pwm + 1'b1;
    end
    
    wire [7:0] spike_rate = neuron_spike_count[7:0];
    assign led5_g = (activity_pwm[15:8] < spike_rate);
    assign led5_r = spike_out_valid;
    assign led5_b = spike_in_valid;

endmodule
