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
    output reg  [31:0]                      ctrl_reg,
    output reg  [31:0]                      config_reg,
    output reg  [15:0]                      leak_rate,
    output reg  [15:0]                      threshold,
    output reg  [15:0]                      refractory_period,
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
    // AXI4-Lite Address Map
    //-------------------------------------------------------------------------
    localparam ADDR_CTRL        = 8'h00;  // Control register
    localparam ADDR_STATUS      = 8'h04;  // Status register
    localparam ADDR_CONFIG      = 8'h08;  // Configuration register
    localparam ADDR_SPIKE_COUNT = 8'h0C;  // Spike counter
    localparam ADDR_LEAK_RATE   = 8'h10;  // Leak rate
    localparam ADDR_THRESHOLD   = 8'h14;  // Threshold
    localparam ADDR_REFRAC      = 8'h18;  // Refractory period
    localparam ADDR_VERSION     = 8'h1C;  // Version register
    
    localparam VERSION = 32'h20240100;  // Version 2024.01.00
    
    //-------------------------------------------------------------------------
    // AXI4-Lite Signals
    //-------------------------------------------------------------------------
    reg axi_awready;
    reg axi_wready;
    reg [1:0] axi_bresp;
    reg axi_bvalid;
    reg axi_arready;
    reg [C_S_AXI_DATA_WIDTH-1:0] axi_rdata;
    reg [1:0] axi_rresp;
    reg axi_rvalid;
    
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr;
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_araddr;
    
    wire slv_reg_wren;
    wire slv_reg_rden;
    
    //-------------------------------------------------------------------------
    // AXI4-Lite Write Logic
    //-------------------------------------------------------------------------
    assign s_axi_awready = axi_awready;
    assign s_axi_wready = axi_wready;
    assign s_axi_bresp = axi_bresp;
    assign s_axi_bvalid = axi_bvalid;
    
    assign slv_reg_wren = axi_wready && s_axi_wvalid && axi_awready && s_axi_awvalid;
    
    // Write Address Ready
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_awready <= 1'b0;
            axi_awaddr <= 0;
        end else begin
            if (!axi_awready && s_axi_awvalid && s_axi_wvalid) begin
                axi_awready <= 1'b1;
                axi_awaddr <= s_axi_awaddr;
            end else begin
                axi_awready <= 1'b0;
            end
        end
    end
    
    // Write Data Ready
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_wready <= 1'b0;
        end else begin
            if (!axi_wready && s_axi_wvalid && s_axi_awvalid) begin
                axi_wready <= 1'b1;
            end else begin
                axi_wready <= 1'b0;
            end
        end
    end
    
    // Write Response
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_bvalid <= 1'b0;
            axi_bresp <= 2'b00;
        end else begin
            if (axi_awready && s_axi_awvalid && axi_wready && s_axi_wvalid && !axi_bvalid) begin
                axi_bvalid <= 1'b1;
                axi_bresp <= 2'b00; // OKAY
            end else if (s_axi_bready && axi_bvalid) begin
                axi_bvalid <= 1'b0;
            end
        end
    end
    
    // Register Write
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            ctrl_reg <= 32'h00000000;
            config_reg <= 32'h00000000;
            leak_rate <= 16'd10;
            threshold <= 16'd1000;
            refractory_period <= 16'd20;
        end else begin
            if (slv_reg_wren) begin
                case (axi_awaddr[7:0])
                    ADDR_CTRL: begin
                        for (integer byte_idx = 0; byte_idx < 4; byte_idx = byte_idx + 1) begin
                            if (s_axi_wstrb[byte_idx])
                                ctrl_reg[byte_idx*8 +: 8] <= s_axi_wdata[byte_idx*8 +: 8];
                        end
                    end
                    ADDR_CONFIG: begin
                        for (integer byte_idx = 0; byte_idx < 4; byte_idx = byte_idx + 1) begin
                            if (s_axi_wstrb[byte_idx])
                                config_reg[byte_idx*8 +: 8] <= s_axi_wdata[byte_idx*8 +: 8];
                        end
                    end
                    ADDR_LEAK_RATE: begin
                        if (s_axi_wstrb[0]) leak_rate[7:0] <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) leak_rate[15:8] <= s_axi_wdata[15:8];
                    end
                    ADDR_THRESHOLD: begin
                        if (s_axi_wstrb[0]) threshold[7:0] <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) threshold[15:8] <= s_axi_wdata[15:8];
                    end
                    ADDR_REFRAC: begin
                        if (s_axi_wstrb[0]) refractory_period[7:0] <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) refractory_period[15:8] <= s_axi_wdata[15:8];
                    end
                endcase
            end
        end
    end
    
    //-------------------------------------------------------------------------
    // AXI4-Lite Read Logic
    //-------------------------------------------------------------------------
    assign s_axi_arready = axi_arready;
    assign s_axi_rdata = axi_rdata;
    assign s_axi_rresp = axi_rresp;
    assign s_axi_rvalid = axi_rvalid;
    
    assign slv_reg_rden = axi_arready && s_axi_arvalid && !axi_rvalid;
    
    // Read Address Ready
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_arready <= 1'b0;
            axi_araddr <= 0;
        end else begin
            if (!axi_arready && s_axi_arvalid) begin
                axi_arready <= 1'b1;
                axi_araddr <= s_axi_araddr;
            end else begin
                axi_arready <= 1'b0;
            end
        end
    end
    
    // Read Data Response
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_rvalid <= 1'b0;
            axi_rresp <= 2'b00;
        end else begin
            if (axi_arready && s_axi_arvalid && !axi_rvalid) begin
                axi_rvalid <= 1'b1;
                axi_rresp <= 2'b00; // OKAY
            end else if (axi_rvalid && s_axi_rready) begin
                axi_rvalid <= 1'b0;
            end
        end
    end
    
    // Read Data Mux
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_rdata <= 0;
        end else begin
            if (slv_reg_rden) begin
                case (axi_araddr[7:0])
                    ADDR_CTRL:        axi_rdata <= ctrl_reg;
                    ADDR_STATUS:      axi_rdata <= status_reg;
                    ADDR_CONFIG:      axi_rdata <= config_reg;
                    ADDR_SPIKE_COUNT: axi_rdata <= spike_count;
                    ADDR_LEAK_RATE:   axi_rdata <= {16'd0, leak_rate};
                    ADDR_THRESHOLD:   axi_rdata <= {16'd0, threshold};
                    ADDR_REFRAC:      axi_rdata <= {16'd0, refractory_period};
                    ADDR_VERSION:     axi_rdata <= VERSION;
                    default:          axi_rdata <= 32'hDEADBEEF;
                endcase
            end
        end
    end
    
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
