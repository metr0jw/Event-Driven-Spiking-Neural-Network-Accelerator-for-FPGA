`timescale 1ns / 1ps

module axi_lite_regs_v1_0_S00_AXI #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 8
)(
    input  wire                             S_AXI_ACLK,
    input  wire                             S_AXI_ARESETN,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]    S_AXI_AWADDR,
    input  wire [2:0]                       S_AXI_AWPROT,
    input  wire                             S_AXI_AWVALID,
    output wire                             S_AXI_AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1:0]    S_AXI_WDATA,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] S_AXI_WSTRB,
    input  wire                             S_AXI_WVALID,
    output wire                             S_AXI_WREADY,
    output wire [1:0]                       S_AXI_BRESP,
    output wire                             S_AXI_BVALID,
    input  wire                             S_AXI_BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]    S_AXI_ARADDR,
    input  wire [2:0]                       S_AXI_ARPROT,
    input  wire                             S_AXI_ARVALID,
    output wire                             S_AXI_ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0]    S_AXI_RDATA,
    output wire [1:0]                       S_AXI_RRESP,
    output wire                             S_AXI_RVALID,
    input  wire                             S_AXI_RREADY,
    output reg  [31:0]                      ctrl_reg,
    output reg  [31:0]                      config_reg,
    output reg  [15:0]                      leak_rate,
    output reg  [15:0]                      threshold,
    output reg  [15:0]                      refractory_period,
    input  wire [31:0]                      status_reg,
    input  wire [31:0]                      spike_count
);

    // ---------------------------------------------------------------------
    // Address map
    // ---------------------------------------------------------------------
    localparam [7:0] ADDR_CTRL        = 8'h00;
    localparam [7:0] ADDR_STATUS      = 8'h04;
    localparam [7:0] ADDR_CONFIG      = 8'h08;
    localparam [7:0] ADDR_SPIKE_COUNT = 8'h0C;
    localparam [7:0] ADDR_LEAK_RATE   = 8'h10;
    localparam [7:0] ADDR_THRESHOLD   = 8'h14;
    localparam [7:0] ADDR_REFRAC      = 8'h18;
    localparam [7:0] ADDR_VERSION     = 8'h1C;

    localparam [31:0] VERSION = 32'h20240100;

    // ---------------------------------------------------------------------
    // AXI handshake registers
    // ---------------------------------------------------------------------
    reg                             axi_awready;
    reg                             axi_wready;
    reg  [1:0]                      axi_bresp;
    reg                             axi_bvalid;
    reg                             axi_arready;
    reg  [C_S_AXI_DATA_WIDTH-1:0]   axi_rdata;
    reg  [1:0]                      axi_rresp;
    reg                             axi_rvalid;
    reg  [C_S_AXI_ADDR_WIDTH-1:0]   axi_awaddr;
    reg  [C_S_AXI_ADDR_WIDTH-1:0]   axi_araddr;

    wire                            slv_reg_wren;
    wire                            slv_reg_rden;

    assign S_AXI_AWREADY = axi_awready;
    assign S_AXI_WREADY  = axi_wready;
    assign S_AXI_BRESP   = axi_bresp;
    assign S_AXI_BVALID  = axi_bvalid;
    assign S_AXI_ARREADY = axi_arready;
    assign S_AXI_RDATA   = axi_rdata;
    assign S_AXI_RRESP   = axi_rresp;
    assign S_AXI_RVALID  = axi_rvalid;

    assign slv_reg_wren = axi_awready && S_AXI_AWVALID && axi_wready && S_AXI_WVALID;
    // Allow read decoding whenever a valid AR is presented (don't rely on arready pulse
    // because some testbenches sample AR earlier). This makes read data muxing more robust.
    assign slv_reg_rden = S_AXI_ARVALID && !axi_rvalid;

    wire base_write_region;
    wire base_read_region;

generate
    if (C_S_AXI_ADDR_WIDTH > 8) begin : gen_addr_filter
        assign base_write_region = (axi_awaddr[C_S_AXI_ADDR_WIDTH-1:8] == { (C_S_AXI_ADDR_WIDTH-8){1'b0} });
        assign base_read_region  = (axi_araddr[C_S_AXI_ADDR_WIDTH-1:8] == { (C_S_AXI_ADDR_WIDTH-8){1'b0} });
    end else begin : gen_addr_filter_passthru
        assign base_write_region = 1'b1;
        assign base_read_region  = 1'b1;
    end
endgenerate

    // ---------------------------------------------------------------------
    // Write address channel
    // ---------------------------------------------------------------------
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_awready <= 1'b0;
            axi_awaddr  <= {C_S_AXI_ADDR_WIDTH{1'b0}};
        end else begin
            if (!axi_awready && S_AXI_AWVALID) begin
                axi_awready <= 1'b1;
                axi_awaddr  <= S_AXI_AWADDR;
            end else if (axi_awready && S_AXI_AWVALID) begin
                axi_awready <= 1'b0;
            end else begin
                axi_awready <= 1'b0;
            end
        end
    end

    // ---------------------------------------------------------------------
    // Write data channel
    // ---------------------------------------------------------------------
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_wready <= 1'b0;
        end else begin
            if (!axi_wready && S_AXI_WVALID) begin
                axi_wready <= 1'b1;
            end else if (axi_wready && S_AXI_WVALID) begin
                axi_wready <= 1'b0;
            end else begin
                axi_wready <= 1'b0;
            end
        end
    end

    // ---------------------------------------------------------------------
    // Write response channel
    // ---------------------------------------------------------------------
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_bvalid <= 1'b0;
            axi_bresp  <= 2'b00;
        end else begin
            if (axi_awready && S_AXI_AWVALID && axi_wready && S_AXI_WVALID && !axi_bvalid) begin
                axi_bvalid <= 1'b1;
                axi_bresp  <= 2'b00;
            end else if (axi_bvalid && S_AXI_BREADY) begin
                axi_bvalid <= 1'b0;
            end
        end
    end

    // ---------------------------------------------------------------------
    // Register write logic
    // ---------------------------------------------------------------------
    integer byte_idx;
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            ctrl_reg          <= 32'h0000_0000;
            config_reg        <= 32'h0000_0000;
            leak_rate         <= 16'd10;
            threshold         <= 16'd1000;
            refractory_period <= 16'd20;
        end else begin
            if (slv_reg_wren) begin
                $display("[AXI_REG_DBG %0t] write addr=0x%0h data=0x%08x base_ok=%0b", $time, axi_awaddr, S_AXI_WDATA, base_write_region);
            end

            if (slv_reg_wren && base_write_region) begin
                case (axi_awaddr[7:0])
                    ADDR_CTRL: begin
                        ctrl_reg <= S_AXI_WDATA;
                        $display("[AXI_REG_DBG %0t] ctrl_reg queued update to 0x%08x", $time, S_AXI_WDATA);
                    end
                    ADDR_CONFIG: begin
                        for (byte_idx = 0; byte_idx < 4; byte_idx = byte_idx + 1) begin
                            if (S_AXI_WSTRB[byte_idx])
                                config_reg[byte_idx*8 +: 8] <= S_AXI_WDATA[byte_idx*8 +: 8];
                        end
                    end
                    ADDR_LEAK_RATE: begin
                        if (S_AXI_WSTRB[0]) leak_rate[7:0]   <= S_AXI_WDATA[7:0];
                        if (S_AXI_WSTRB[1]) leak_rate[15:8]  <= S_AXI_WDATA[15:8];
                    end
                    ADDR_THRESHOLD: begin
                        if (S_AXI_WSTRB[0]) threshold[7:0]   <= S_AXI_WDATA[7:0];
                        if (S_AXI_WSTRB[1]) threshold[15:8]  <= S_AXI_WDATA[15:8];
                    end
                    ADDR_REFRAC: begin
                        if (S_AXI_WSTRB[0]) refractory_period[7:0]  <= S_AXI_WDATA[7:0];
                        if (S_AXI_WSTRB[1]) refractory_period[15:8] <= S_AXI_WDATA[15:8];
                    end
                    default: begin
                        // Writes to read-only or undefined locations are ignored
                    end
                endcase
            end

        end
    end

    always @(ctrl_reg) begin
        $display("[AXI_REG_DBG %0t] ctrl_reg now 0x%08x", $time, ctrl_reg);
    end

    // ---------------------------------------------------------------------
    // Read address channel
    // ---------------------------------------------------------------------
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_arready <= 1'b0;
            axi_araddr  <= {C_S_AXI_ADDR_WIDTH{1'b0}};
        end else begin
            if (!axi_arready && S_AXI_ARVALID) begin
                axi_arready <= 1'b1;
                axi_araddr  <= S_AXI_ARADDR;
                $display("[AXI_REG_DBG %0t] AR accepted addr=0x%0h arready=%0b", $time, S_AXI_ARADDR, axi_arready);
            end else begin
                axi_arready <= 1'b0;
            end
        end
    end

    // ---------------------------------------------------------------------
    // Read data channel
    // ---------------------------------------------------------------------
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_rvalid <= 1'b0;
            axi_rresp  <= 2'b00;
        end else begin
            if (axi_arready && S_AXI_ARVALID && !axi_rvalid) begin
                axi_rvalid <= 1'b1;
                axi_rresp  <= 2'b00;
            end else if (axi_rvalid && S_AXI_RREADY) begin
                axi_rvalid <= 1'b0;
            end
        end
    end

    // ---------------------------------------------------------------------
    // Read data mux
    // ---------------------------------------------------------------------
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_rdata <= {C_S_AXI_DATA_WIDTH{1'b0}};
        end else if (slv_reg_rden) begin
            if (base_read_region) begin
                case (axi_araddr[7:0])
                    ADDR_CTRL:        axi_rdata <= ctrl_reg;
                    ADDR_STATUS:      axi_rdata <= status_reg;
                    ADDR_CONFIG:      axi_rdata <= config_reg;
                    ADDR_SPIKE_COUNT: axi_rdata <= spike_count;
                    ADDR_LEAK_RATE:   axi_rdata <= {16'd0, leak_rate};
                    ADDR_THRESHOLD:   axi_rdata <= {16'd0, threshold};
                    ADDR_REFRAC:      axi_rdata <= {16'd0, refractory_period};
                    ADDR_VERSION:     axi_rdata <= VERSION;
                    default:          axi_rdata <= 32'hDEAD_BEEF;
                endcase
                $display("[AXI_REG_DBG %0t] Read addr=0x%0h -> data=0x%08x base_ok=%0b", $time, axi_araddr, axi_rdata, base_read_region);
            end else begin
                axi_rdata <= 32'hDEAD_BEEF;
                $display("[AXI_REG_DBG %0t] Read addr=0x%0h outside base region -> DEAD_BEEF", $time, axi_araddr);
            end
        end
    end

endmodule

