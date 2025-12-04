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
    // AXI Parameters
    parameter C_S_AXI_DATA_WIDTH    = 32,
    parameter C_S_AXI_ADDR_WIDTH    = 32,
    parameter C_AXIS_DATA_WIDTH     = 32,
    
    // SNN Parameters - scaled up for better resource utilization
    parameter NUM_NEURONS           = 256,
    parameter NUM_AXONS             = 256,
    parameter NUM_PARALLEL_UNITS    = 8,     // Parallel processing units
    parameter SPIKE_BUFFER_DEPTH    = 64,    // Input spike buffer depth
    // Calculate log2 ceiling manually for Verilog-2001 compatibility
    parameter NEURON_ID_WIDTH      = (NUM_NEURONS <= 2) ? 1 :
                                    (NUM_NEURONS <= 4) ? 2 :
                                    (NUM_NEURONS <= 8) ? 3 :
                                    (NUM_NEURONS <= 16) ? 4 :
                                    (NUM_NEURONS <= 32) ? 5 :
                                    (NUM_NEURONS <= 64) ? 6 :
                                    (NUM_NEURONS <= 128) ? 7 : 8,
    parameter AXON_ID_WIDTH        = (NUM_AXONS <= 2) ? 1 :
                                    (NUM_AXONS <= 4) ? 2 :
                                    (NUM_AXONS <= 8) ? 3 :
                                    (NUM_AXONS <= 16) ? 4 :
                                    (NUM_AXONS <= 32) ? 5 :
                                    (NUM_AXONS <= 64) ? 6 :
                                    (NUM_AXONS <= 128) ? 7 : 8,
    parameter DATA_WIDTH           = 16,
    parameter WEIGHT_WIDTH         = 8,
    parameter LEAK_WIDTH           = 8,
    parameter THRESHOLD_WIDTH      = 16,
    parameter REFRAC_WIDTH         = 8,
    parameter ROUTER_BUFFER_DEPTH  = 512,   // Increased buffer depth
    
    // Resource hints
    parameter USE_BRAM              = 1,     // Use BRAM for storage
    parameter USE_DSP               = 1      // Use DSP for arithmetic
)(
    //-------------------------------------------------------------------------
    // Clock and Reset
    //-------------------------------------------------------------------------
    input  wire                          aclk,
    input  wire                          aresetn,
    
    //-------------------------------------------------------------------------
    // AXI4-Lite Slave Interface (Configuration)
    //-------------------------------------------------------------------------
    // Write Address Channel
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    input  wire [2:0]                    s_axi_awprot,
    input  wire                          s_axi_awvalid,
    output wire                          s_axi_awready,
    
    // Write Data Channel
    input  wire [C_S_AXI_DATA_WIDTH-1:0] s_axi_wdata,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire                          s_axi_wvalid,
    output wire                          s_axi_wready,
    
    // Write Response Channel
    output wire [1:0]                    s_axi_bresp,
    output wire                          s_axi_bvalid,
    input  wire                          s_axi_bready,
    
    // Read Address Channel
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    input  wire [2:0]                    s_axi_arprot,
    input  wire                          s_axi_arvalid,
    output wire                          s_axi_arready,
    
    // Read Data Channel
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
    input  wire                          s_axis_tlast,
    
    //-------------------------------------------------------------------------
    // AXI4-Stream Master Interface (Output Spikes to PS)
    //-------------------------------------------------------------------------
    output wire [C_AXIS_DATA_WIDTH-1:0]  m_axis_tdata,
    output wire                          m_axis_tvalid,
    input  wire                          m_axis_tready,
    output wire                          m_axis_tlast,
    
    //-------------------------------------------------------------------------
    // Interrupt to PS
    //-------------------------------------------------------------------------
    output wire                          interrupt,
    
    //-------------------------------------------------------------------------
    // Board I/O (Optional - for direct PL connections)
    //-------------------------------------------------------------------------
    output wire [3:0]                    led,        // Regular LEDs
    input  wire [1:0]                    sw,         // Switches
    input  wire [3:0]                    btn,        // Push buttons
    
    // RGB LEDs
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
    
    // Configuration registers
    wire [31:0]                 ctrl_reg;
    wire [31:0]                 config_reg;
    wire [15:0]                 leak_rate;
    wire [15:0]                 threshold;
    wire [15:0]                 refractory_period;
    wire [31:0]                 status_reg;
    wire [31:0]                 spike_count;
    
    // Control signals
    wire                        snn_enable;
    wire                        snn_reset;
    wire                        clear_counters;
    
    // Spike interfaces between modules
    wire                        input_spike_valid;
    wire [NEURON_ID_WIDTH-1:0]  input_spike_neuron_id;
    wire [WEIGHT_WIDTH-1:0]     input_spike_weight;
    wire                        input_spike_ready;
    
    wire                        neuron_spike_valid;
    wire [NEURON_ID_WIDTH-1:0]  neuron_spike_id;
    wire                        neuron_spike_ready;
    
    wire                        routed_spike_valid;
    wire [NEURON_ID_WIDTH-1:0]  routed_spike_dest_id;
    wire [WEIGHT_WIDTH-1:0]     routed_spike_weight;
    wire                        routed_spike_exc_inh;
    wire                        routed_spike_ready;
    
    wire                        output_spike_valid;
    wire [NEURON_ID_WIDTH-1:0]  output_spike_neuron_id;
    wire                        output_spike_ready;

    // Width adaptation helpers for AXI wrapper (8-bit neuron IDs)
    // axi_wrapper outputs 8-bit spike_in_neuron_id, we extend to NEURON_ID_WIDTH
    wire [7:0] spike_in_neuron_id_axi;
    wire [7:0] spike_out_neuron_id_axi;

    // Convert from AXI wrapper 8-bit output to internal NEURON_ID_WIDTH
    // spike_in_neuron_id_axi comes FROM axi_wrapper, goes TO input_spike_neuron_id
generate
    if (NEURON_ID_WIDTH > 8) begin : gen_spike_in_extend
        assign input_spike_neuron_id = {{(NEURON_ID_WIDTH-8){1'b0}}, spike_in_neuron_id_axi};
    end else begin : gen_spike_in_slice
        assign input_spike_neuron_id = spike_in_neuron_id_axi[NEURON_ID_WIDTH-1:0];
    end
endgenerate

    // Convert from internal NEURON_ID_WIDTH to AXI wrapper 8-bit input
    // output_spike_neuron_id goes TO axi_wrapper as spike_out_neuron_id_axi
generate
    if (NEURON_ID_WIDTH >= 8) begin : gen_spike_out_slice
        assign spike_out_neuron_id_axi = output_spike_neuron_id[7:0];
    end else begin : gen_spike_out_extend
        assign spike_out_neuron_id_axi = {{(8-NEURON_ID_WIDTH){1'b0}}, output_spike_neuron_id[NEURON_ID_WIDTH-1:0]};
    end
endgenerate
    
    // Status signals
    wire [31:0]                 neuron_spike_count;
    wire [31:0]                 routed_spike_count;
    wire [31:0]                 throughput_counter;
    wire [7:0]                  active_neurons;
    wire                        router_busy;
    wire                        fifo_overflow;
    wire                        array_busy;
    wire                        synapse_busy;
    
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
    // AXI Interface Wrapper
    //-------------------------------------------------------------------------
    axi_wrapper #(
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH),
        .C_AXIS_DATA_WIDTH(C_AXIS_DATA_WIDTH),
        .NUM_NEURONS(NUM_NEURONS)
    ) axi_wrapper_inst (
        // AXI4-Lite slave
        .s_axi_aclk(aclk),
        .s_axi_aresetn(aresetn),
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
        
        // AXI4-Stream interfaces
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast),
        
        // Control/Status
        .ctrl_reg(ctrl_reg),
        .config_reg(config_reg),
        .leak_rate(leak_rate),
        .threshold(threshold),
        .refractory_period(refractory_period),
        .status_reg(status_reg),
        .spike_count(spike_count),
        
        // Spike interface
        .spike_in_valid(input_spike_valid),
        .spike_in_neuron_id(spike_in_neuron_id_axi),
        .spike_in_weight(input_spike_weight),
        .spike_in_ready(input_spike_ready),
        .spike_out_valid(output_spike_valid),
        .spike_out_neuron_id(spike_out_neuron_id_axi),
        .spike_out_ready(output_spike_ready)
    );
    
    // Extract control signals
    assign snn_enable = ctrl_reg[0];
    assign snn_reset = ctrl_reg[1];
    assign clear_counters = ctrl_reg[2];
    
    //-------------------------------------------------------------------------
    // Synapse Array (BRAM-banked parallel access)
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
        
        // Input spike
        .spike_in_valid(input_spike_valid),
        .spike_in_axon_id(input_spike_neuron_id[AXON_ID_WIDTH-1:0]),
        .spike_in_ready(input_spike_ready),
        
        // Output to neurons
        .spike_out_valid(routed_spike_valid),
        .spike_out_neuron_id(routed_spike_dest_id),
        .spike_out_weight(routed_spike_weight),
        .spike_out_exc_inh(routed_spike_exc_inh),
        
        // Weight configuration
        .weight_we(config_reg[8] && (s_axi_awaddr[15:12] == 4'h1)),
        .weight_addr_axon(s_axi_awaddr[AXON_ID_WIDTH+7:8]),
        .weight_addr_neuron(s_axi_awaddr[NEURON_ID_WIDTH-1:0]),
        .weight_data({s_axi_wdata[8], s_axi_wdata[7:0]}),
        
        // Batch write interface
        .batch_we(1'b0),
        .batch_axon_id({AXON_ID_WIDTH{1'b0}}),
        .batch_weights({NUM_PARALLEL_UNITS*(WEIGHT_WIDTH+1){1'b0}}),
        .batch_start_neuron({NEURON_ID_WIDTH{1'b0}}),
        
        .enable(snn_enable),
        .busy(synapse_busy),
        .spike_count()
    );
    
    //-------------------------------------------------------------------------
    // LIF Neuron Array (BRAM/DSP optimized with parallel processing)
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
        
        // Input spikes
        .s_axis_spike_valid(routed_spike_valid),
        .s_axis_spike_dest_id(routed_spike_dest_id),
        .s_axis_spike_weight(routed_spike_weight),
        .s_axis_spike_exc_inh(routed_spike_exc_inh),
        .s_axis_spike_ready(routed_spike_ready),
        
        // Output spikes
        .m_axis_spike_valid(neuron_spike_valid),
        .m_axis_spike_neuron_id(neuron_spike_id),
        .m_axis_spike_ready(neuron_spike_ready),
        
        // Configuration
        .config_we(config_reg[9] && (s_axi_awaddr[15:12] == 4'h2)),
        .config_addr(s_axi_awaddr[NEURON_ID_WIDTH-1:0]),
        .config_data(s_axi_wdata),
        
        // Parameters
        .global_threshold(threshold),
        .global_leak_rate(leak_rate[LEAK_WIDTH-1:0]),
        .global_refrac_period(refractory_period[REFRAC_WIDTH-1:0]),
        
        // Status
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
        
        // Input from neurons
        .s_spike_valid(neuron_spike_valid),
        .s_spike_neuron_id(neuron_spike_id),
        .s_spike_ready(neuron_spike_ready),
        
        // Output (loopback to synapses or to PS)
        .m_spike_valid(output_spike_valid),
        .m_spike_dest_id(output_spike_neuron_id),
        .m_spike_weight(),
        .m_spike_exc_inh(),
        .m_spike_ready(output_spike_ready),
        
        // Configuration
        .config_we(config_reg[10] && (s_axi_awaddr[15:12] == 4'h3)),
        .config_addr(s_axi_awaddr),
        .config_data(s_axi_wdata),
        .config_readdata(),
        
        // Status
        .routed_spike_count(routed_spike_count),
        .router_busy(router_busy),
        .fifo_overflow(fifo_overflow)
    );
    
    //-------------------------------------------------------------------------
    // Status Register Assembly (enhanced with new signals)
    //-------------------------------------------------------------------------
    assign status_reg = {
        active_neurons,          // [31:24] Active neuron count
        fifo_overflow,           // [23]    FIFO overflow
        router_busy,             // [22]    Router busy
        array_busy,              // [21]    Neuron array busy
        synapse_busy,            // [20]    Synapse array busy
        3'd0,                    // [19:17] Reserved
        throughput_counter[16],  // [16]    Throughput MSB
        throughput_counter[15:8], // [15:8] Throughput indicator
        |neuron_spike_count[7:0], // [7]     Spike activity
        3'd0,                    // [6:4]   Reserved
        output_spike_valid,      // [3]     Output spike present
        neuron_spike_valid,      // [2]     Neuron spike present
        input_spike_valid,       // [1]     Input spike present
        snn_enable              // [0]     SNN enabled
    };
    
    // Spike counter with clear
    reg [31:0] total_spike_count;
    always @(posedge sys_clk) begin
        if (!sys_rst_n || clear_counters)
            total_spike_count <= 32'd0;
        else if (output_spike_valid && output_spike_ready)
            total_spike_count <= total_spike_count + 1'b1;
    end
    assign spike_count = total_spike_count;
    
    //-------------------------------------------------------------------------
    // Interrupt Generation
    //-------------------------------------------------------------------------
    reg interrupt_reg;
    reg [15:0] spike_threshold;
    
    always @(posedge sys_clk) begin
        if (!sys_rst_n) begin
            interrupt_reg <= 1'b0;
            spike_threshold <= 16'd100;
        end else begin
            // Generate interrupt when spike count exceeds threshold
            if (ctrl_reg[3]) begin // Interrupt enable
                interrupt_reg <= (total_spike_count[15:0] >= spike_threshold);
            end else begin
                interrupt_reg <= 1'b0;
            end
            
            // Update threshold
            if (config_reg[11] && (s_axi_awaddr[7:0] == 8'h20))
                spike_threshold <= s_axi_wdata[15:0];
        end
    end
    assign interrupt = interrupt_reg;
    
    //-------------------------------------------------------------------------
    // LED Status Indicators
    //-------------------------------------------------------------------------
    // Regular LEDs
    reg [23:0] heartbeat_counter;
    always @(posedge sys_clk) begin
        if (!sys_rst_n)
            heartbeat_counter <= 24'd0;
        else
            heartbeat_counter <= heartbeat_counter + 1'b1;
    end
    
    //-------------------------------------------------------------------------
    // Performance Counters (exposed via status registers)
    //-------------------------------------------------------------------------
    reg [31:0] cycle_counter;
    reg [31:0] active_cycles;
    
    always @(posedge sys_clk) begin
        if (!sys_rst_n || clear_counters) begin
            cycle_counter <= 32'd0;
            active_cycles <= 32'd0;
        end else if (snn_enable) begin
            cycle_counter <= cycle_counter + 1'b1;
            if (array_busy || router_busy)
                active_cycles <= active_cycles + 1'b1;
        end
    end
    
    // Connect performance counters to debug outputs
    wire [31:0] debug_cycle_count = cycle_counter;
    wire [31:0] debug_active_cycles = active_cycles;
    
    // Test spike injection (connected to status output via LED)
    reg inject_spike_d;
    wire test_spike_pulse;
    always @(posedge sys_clk) begin
        if (!sys_rst_n) begin
            inject_spike_d <= 1'b0;
        end else begin
            inject_spike_d <= inject_spike;
        end
    end
    assign test_spike_pulse = inject_spike & ~inject_spike_d;
    
    assign led[0] = heartbeat_counter[23];              // Heartbeat
    assign led[1] = snn_enable;                         // System enabled
    assign led[2] = |neuron_spike_count[10:0];          // Spike activity
    assign led[3] = fifo_overflow | status_reg[15] | test_spike_pulse; // Error/test indicator
    
    // RGB LED 4 - System status
    assign led4_g = snn_enable & ~snn_reset;            // Green: running
    assign led4_r = snn_reset;                          // Red: reset
    assign led4_b = array_busy | router_busy | (|debug_active_cycles[7:0]); // Blue: busy
    
    // RGB LED 5 - Activity indicator
    reg [15:0] activity_pwm;
    always @(posedge sys_clk) begin
        if (!sys_rst_n)
            activity_pwm <= 16'd0;
        else
            activity_pwm <= activity_pwm + 1'b1;
    end
    
    wire [7:0] spike_rate = neuron_spike_count[7:0];
    assign led5_g = (activity_pwm[15:8] < spike_rate);   // Green: spike rate
    assign led5_r = output_spike_valid;                   // Red: output spike
    assign led5_b = input_spike_valid;                    // Blue: input spike
    
    //-------------------------------------------------------------------------
    // Debug Features (using buttons and switches)
    //-------------------------------------------------------------------------
    // Button functions:
    // btn[0]: Manual reset
    // btn[1]: Single step mode
    // btn[2]: Inject test spike
    // btn[3]: Clear counters
    
    // Consume unused btn/sw inputs to suppress warnings
    wire btn_sw_used = |btn | |sw;  // Will be optimized but prevents warning
    
    wire manual_reset = btn[0];
    wire single_step = btn[1] & sw[0];
    wire inject_spike = btn[2] & sw[1];
    wire clear_stats = btn[3];

endmodule
