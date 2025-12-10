//-----------------------------------------------------------------------------
// Title         : Integrated SNN Accelerator Top (HLS + Verilog RTL)
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_integrated_top.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Top-level wrapper with FULL HLS <-> RTL integration:
//                 - HLS IP outputs spikes → Spike Router → LIF Neurons
//                 - LIF Neuron outputs → HLS IP (for STDP learning)
//                 - Bidirectional spike flow enables online learning
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module snn_integrated_top #(
    // System Parameters
    parameter NUM_NEURONS           = 256,
    parameter NUM_AXONS             = 1024,
    parameter NUM_PARALLEL_UNITS    = 8,
    parameter SPIKE_BUFFER_DEPTH    = 64,
    parameter NEURON_ID_WIDTH       = 8,
    parameter AXON_ID_WIDTH         = 10,
    parameter DATA_WIDTH            = 16,
    parameter WEIGHT_WIDTH          = 8,
    parameter LEAK_WIDTH            = 8,
    parameter THRESHOLD_WIDTH       = 16,
    parameter REFRAC_WIDTH          = 8,
    parameter ROUTER_BUFFER_DEPTH   = 512
)(
    //-------------------------------------------------------------------------
    // DDR Interface (directly from PS)
    //-------------------------------------------------------------------------
    inout  wire [14:0]  DDR_addr,
    inout  wire [2:0]   DDR_ba,
    inout  wire         DDR_cas_n,
    inout  wire         DDR_ck_n,
    inout  wire         DDR_ck_p,
    inout  wire         DDR_cke,
    inout  wire         DDR_cs_n,
    inout  wire [3:0]   DDR_dm,
    inout  wire [31:0]  DDR_dq,
    inout  wire [3:0]   DDR_dqs_n,
    inout  wire [3:0]   DDR_dqs_p,
    inout  wire         DDR_odt,
    inout  wire         DDR_ras_n,
    inout  wire         DDR_reset_n,
    inout  wire         DDR_we_n,
    
    //-------------------------------------------------------------------------
    // Fixed IO (PS)
    //-------------------------------------------------------------------------
    inout  wire         FIXED_IO_ddr_vrn,
    inout  wire         FIXED_IO_ddr_vrp,
    inout  wire [53:0]  FIXED_IO_mio,
    inout  wire         FIXED_IO_ps_clk,
    inout  wire         FIXED_IO_ps_porb,
    inout  wire         FIXED_IO_ps_srstb
);

    //=========================================================================
    // Internal Signals: Clock and Reset
    //=========================================================================
    wire clk_100mhz;
    wire rst_n_sync;
    wire debug_learning_active;
    
    //=========================================================================
    // HLS <-> RTL Interface Signals (from Block Design)
    //=========================================================================
    
    // HLS → RTL: Spikes from HLS to RTL neurons
    wire                         hls_spike_out_valid;
    wire [NEURON_ID_WIDTH-1:0]   hls_spike_out_neuron_id;
    wire [WEIGHT_WIDTH-1:0]      hls_spike_out_weight;
    wire                         rtl_spike_in_ready;
    
    // RTL → HLS: Spikes from RTL neurons to HLS (for learning)
    wire                         rtl_spike_out_valid;
    wire [NEURON_ID_WIDTH-1:0]   rtl_spike_out_neuron_id;
    wire [WEIGHT_WIDTH-1:0]      rtl_spike_out_weight;
    wire                         hls_spike_in_ready;
    
    // SNN Control
    wire                         hls_snn_enable;
    wire                         hls_snn_reset;
    wire                         rtl_snn_ready;
    wire                         rtl_snn_busy;
    
    //=========================================================================
    // Internal Spike Router Signals
    //=========================================================================
    wire                         router_spike_valid;
    wire [NEURON_ID_WIDTH-1:0]   router_spike_dest_id;
    wire [WEIGHT_WIDTH-1:0]      router_spike_weight;
    wire                         router_spike_exc_inh;
    wire                         router_spike_ready;
    
    // Router input selection (HLS or recurrent)
    wire                         router_input_valid;
    wire [NEURON_ID_WIDTH-1:0]   router_input_neuron_id;
    wire                         router_input_ready;
    
    //=========================================================================
    // Neuron Array Output Signals
    //=========================================================================
    wire                         neuron_spike_valid;
    wire [NEURON_ID_WIDTH-1:0]   neuron_spike_id;
    wire                         neuron_spike_ready;
    
    // Statistics
    wire [31:0]                  router_spike_count;
    wire [31:0]                  neuron_spike_count;
    wire                         router_busy;
    wire                         neuron_array_busy;
    
    //=========================================================================
    // Block Design Instantiation (PS + HLS IP + AXI)
    // Now with full HLS <-> RTL interface connections
    //=========================================================================
    
    design_1_wrapper u_block_design (
        // DDR Interface
        .DDR_addr           (DDR_addr),
        .DDR_ba             (DDR_ba),
        .DDR_cas_n          (DDR_cas_n),
        .DDR_ck_n           (DDR_ck_n),
        .DDR_ck_p           (DDR_ck_p),
        .DDR_cke            (DDR_cke),
        .DDR_cs_n           (DDR_cs_n),
        .DDR_dm             (DDR_dm),
        .DDR_dq             (DDR_dq),
        .DDR_dqs_n          (DDR_dqs_n),
        .DDR_dqs_p          (DDR_dqs_p),
        .DDR_odt            (DDR_odt),
        .DDR_ras_n          (DDR_ras_n),
        .DDR_reset_n        (DDR_reset_n),
        .DDR_we_n           (DDR_we_n),
        
        // Fixed IO
        .FIXED_IO_ddr_vrn   (FIXED_IO_ddr_vrn),
        .FIXED_IO_ddr_vrp   (FIXED_IO_ddr_vrp),
        .FIXED_IO_mio       (FIXED_IO_mio),
        .FIXED_IO_ps_clk    (FIXED_IO_ps_clk),
        .FIXED_IO_ps_porb   (FIXED_IO_ps_porb),
        .FIXED_IO_ps_srstb  (FIXED_IO_ps_srstb),
        
        // PL Clock/Reset outputs
        .clk_100mhz         (clk_100mhz),
        .rst_n_sync         (rst_n_sync),
        
        // Debug
        .debug_learning_active (debug_learning_active),
        
        //---------------------------------------------------------------------
        // HLS → RTL Spike Interface
        //---------------------------------------------------------------------
        .hls_spike_out_valid     (hls_spike_out_valid),
        .hls_spike_out_neuron_id (hls_spike_out_neuron_id),
        .hls_spike_out_weight    (hls_spike_out_weight),
        .rtl_spike_in_ready      (rtl_spike_in_ready),
        
        //---------------------------------------------------------------------
        // RTL → HLS Spike Interface (for STDP learning)
        //---------------------------------------------------------------------
        .rtl_spike_out_valid     (rtl_spike_out_valid),
        .rtl_spike_out_neuron_id (rtl_spike_out_neuron_id),
        .rtl_spike_out_weight    (rtl_spike_out_weight),
        .hls_spike_in_ready      (hls_spike_in_ready),
        
        //---------------------------------------------------------------------
        // SNN Control Interface
        //---------------------------------------------------------------------
        .hls_snn_enable          (hls_snn_enable),
        .hls_snn_reset           (hls_snn_reset),
        .rtl_snn_ready           (rtl_snn_ready),
        .rtl_snn_busy            (rtl_snn_busy)
    );
    
    //=========================================================================
    // Spike Input Multiplexer
    // Priority: HLS spikes > Recurrent neuron spikes
    //=========================================================================
    
    // Simple priority: HLS has priority when sending spikes
    assign router_input_valid     = hls_spike_out_valid | neuron_spike_valid;
    assign router_input_neuron_id = hls_spike_out_valid ? hls_spike_out_neuron_id : neuron_spike_id;
    
    // Ready signals
    assign rtl_spike_in_ready = router_input_ready;
    assign neuron_spike_ready = router_input_ready & ~hls_spike_out_valid;
    
    //=========================================================================
    // RTL → HLS Connection
    // Send neuron output spikes to HLS for STDP learning
    //=========================================================================
    
    assign rtl_spike_out_valid     = neuron_spike_valid;
    assign rtl_spike_out_neuron_id = neuron_spike_id;
    assign rtl_spike_out_weight    = router_spike_weight;  // Use current weight
    
    //=========================================================================
    // SNN Status to HLS
    //=========================================================================
    
    assign rtl_snn_ready = ~router_busy & ~neuron_array_busy;
    assign rtl_snn_busy  = router_busy | neuron_array_busy;
    
    //=========================================================================
    // Verilog RTL: Spike Router (AER-based)
    //=========================================================================
    
    spike_router #(
        .NUM_NEURONS        (NUM_NEURONS),
        .MAX_FANOUT         (32),
        .WEIGHT_WIDTH       (WEIGHT_WIDTH),
        .NEURON_ID_WIDTH    (NEURON_ID_WIDTH),
        .DELAY_WIDTH        (8),
        .FIFO_DEPTH         (ROUTER_BUFFER_DEPTH)
    ) u_spike_router (
        .clk                (clk_100mhz),
        .rst_n              (rst_n_sync & ~hls_snn_reset),
        
        // Input (from HLS or recurrent)
        .s_spike_valid      (router_input_valid),
        .s_spike_neuron_id  (router_input_neuron_id),
        .s_spike_ready      (router_input_ready),
        
        // Output to neuron array
        .m_spike_valid      (router_spike_valid),
        .m_spike_dest_id    (router_spike_dest_id),
        .m_spike_weight     (router_spike_weight),
        .m_spike_exc_inh    (router_spike_exc_inh),
        .m_spike_ready      (router_spike_ready),
        
        // Configuration (can be connected to HLS later)
        .config_we          (1'b0),
        .config_addr        (32'd0),
        .config_data        (32'd0),
        .config_readdata    (),
        
        // Statistics
        .routed_spike_count (router_spike_count),
        .router_busy        (router_busy),
        .fifo_overflow      ()
    );
    
    //=========================================================================
    // Verilog RTL: LIF Neuron Array (AC-based, energy-efficient)
    //=========================================================================
    
    lif_neuron_array #(
        .NUM_NEURONS        (NUM_NEURONS),
        .NUM_AXONS          (NUM_AXONS),
        .DATA_WIDTH         (DATA_WIDTH),
        .WEIGHT_WIDTH       (WEIGHT_WIDTH),
        .THRESHOLD_WIDTH    (THRESHOLD_WIDTH),
        .LEAK_WIDTH         (LEAK_WIDTH),
        .REFRAC_WIDTH       (REFRAC_WIDTH),
        .NUM_PARALLEL_UNITS (NUM_PARALLEL_UNITS),
        .SPIKE_BUFFER_DEPTH (SPIKE_BUFFER_DEPTH),
        .USE_BRAM           (1),
        .USE_DSP            (0)  // AC-based: no DSP blocks
    ) u_neuron_array (
        .clk                (clk_100mhz),
        .rst_n              (rst_n_sync & ~hls_snn_reset),
        .enable             (hls_snn_enable),
        
        // Input from spike router
        .s_axis_spike_valid     (router_spike_valid),
        .s_axis_spike_dest_id   (router_spike_dest_id),
        .s_axis_spike_weight    (router_spike_weight),
        .s_axis_spike_exc_inh   (router_spike_exc_inh),
        .s_axis_spike_ready     (router_spike_ready),
        
        // Output spikes (to HLS for learning + recurrent)
        .m_axis_spike_valid     (neuron_spike_valid),
        .m_axis_spike_neuron_id (neuron_spike_id),
        .m_axis_spike_ready     (hls_spike_in_ready),  // HLS controls flow
        
        // Configuration (can be connected to HLS later)
        .config_we              (1'b0),
        .config_addr            ({NEURON_ID_WIDTH{1'b0}}),
        .config_data            (32'd0),
        
        // Global neuron parameters (can be connected to HLS registers)
        .global_threshold       (16'd100),
        .global_leak_rate       (8'h11),
        .global_refrac_period   (8'd10),
        
        // Monitoring
        .spike_count            (neuron_spike_count),
        .array_busy             (neuron_array_busy),
        .throughput_counter     (),
        .active_neurons         ()
    );
    
endmodule
