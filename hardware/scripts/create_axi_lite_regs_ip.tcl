# -----------------------------------------------------------------------------
# create_axi_lite_regs_ip.tcl
#
# Batch script to generate a Vivado AXI4-Lite slave IP that exposes the
# accelerator control/status registers used by the SNN design.
#
# Usage:
#   vivado -mode batch -source hardware/scripts/create_axi_lite_regs_ip.tcl
# -----------------------------------------------------------------------------

# Resolve key paths (script lives in hardware/scripts)
set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file normalize [file join $script_dir .. ..]]

# Configuration ----------------------------------------------------------------
set part_name    "xc7z020clg400-1"                 ;# PYNQ-Z2
set ip_name      "axi_lite_regs"
set ip_version   "v1_0"
set vendor_name  "jiwoonlee"
set library_name "user"
set taxonomy     "/UserIP"

set tmp_dir  [file normalize [file join $repo_root hardware/ip_repo .tmp_${ip_name}_${ip_version}]]
set ip_dir   [file normalize [file join $repo_root hardware/ip_repo ${ip_name}_${ip_version}]]

# Clean previous runs ----------------------------------------------------------
if {[file exists $tmp_dir]} {
    puts "INFO: Removing previous temporary project: $tmp_dir"
    file delete -force $tmp_dir
}
if {[file exists $ip_dir]} {
    puts "INFO: Removing previous packaged IP: $ip_dir"
    file delete -force $ip_dir
}

file mkdir $tmp_dir

# Create a throw-away Vivado project and author the HDL directly
create_project -force ${ip_name}_${ip_version}_prj $tmp_dir -part $part_name

set hdl_dir [file normalize [file join $tmp_dir src]]
file mkdir $hdl_dir

set s00_axi_file [file join $hdl_dir "${ip_name}_${ip_version}_S00_AXI.v"]
set top_axi_file [file join $hdl_dir "${ip_name}_${ip_version}.v"]

puts "INFO: Writing custom AXI register HDL."

# Custom S00 AXI implementation -------------------------------------------------
set fp [open $s00_axi_file w]
puts $fp {`timescale 1ns / 1ps

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

    assign slv_reg_wren = axi_wready && S_AXI_WVALID;
    assign slv_reg_rden = axi_arready && S_AXI_ARVALID && !axi_rvalid;

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
                case (axi_awaddr[7:0])
                    ADDR_CTRL: begin
                        for (byte_idx = 0; byte_idx < 4; byte_idx = byte_idx + 1) begin
                            if (S_AXI_WSTRB[byte_idx])
                                ctrl_reg[byte_idx*8 +: 8] <= S_AXI_WDATA[byte_idx*8 +: 8];
                        end
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
        end
    end

endmodule
}
close $fp

# Custom top-level wrapper ------------------------------------------------------
set fp [open $top_axi_file w]
puts $fp {`timescale 1ns / 1ps

module axi_lite_regs_v1_0 #(
    parameter integer C_S00_AXI_DATA_WIDTH = 32,
    parameter integer C_S00_AXI_ADDR_WIDTH = 8
)(
    output wire [31:0] ctrl_reg,
    output wire [31:0] config_reg,
    output wire [15:0] leak_rate,
    output wire [15:0] threshold,
    output wire [15:0] refractory_period,
    input  wire [31:0] status_reg,
    input  wire [31:0] spike_count,
    input  wire        s00_axi_aclk,
    input  wire        s00_axi_aresetn,
    input  wire [C_S00_AXI_ADDR_WIDTH-1:0] s00_axi_awaddr,
    input  wire [2:0]  s00_axi_awprot,
    input  wire        s00_axi_awvalid,
    output wire        s00_axi_awready,
    input  wire [C_S00_AXI_DATA_WIDTH-1:0] s00_axi_wdata,
    input  wire [(C_S00_AXI_DATA_WIDTH/8)-1:0] s00_axi_wstrb,
    input  wire        s00_axi_wvalid,
    output wire        s00_axi_wready,
    output wire [1:0]  s00_axi_bresp,
    output wire        s00_axi_bvalid,
    input  wire        s00_axi_bready,
    input  wire [C_S00_AXI_ADDR_WIDTH-1:0] s00_axi_araddr,
    input  wire [2:0]  s00_axi_arprot,
    input  wire        s00_axi_arvalid,
    output wire        s00_axi_arready,
    output wire [C_S00_AXI_DATA_WIDTH-1:0] s00_axi_rdata,
    output wire [1:0]  s00_axi_rresp,
    output wire        s00_axi_rvalid,
    input  wire        s00_axi_rready
);

    axi_lite_regs_v1_0_S00_AXI #(
        .C_S_AXI_DATA_WIDTH(C_S00_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(C_S00_AXI_ADDR_WIDTH)
    ) s00_axi_inst (
        .S_AXI_ACLK(s00_axi_aclk),
        .S_AXI_ARESETN(s00_axi_aresetn),
        .S_AXI_AWADDR(s00_axi_awaddr),
        .S_AXI_AWPROT(s00_axi_awprot),
        .S_AXI_AWVALID(s00_axi_awvalid),
        .S_AXI_AWREADY(s00_axi_awready),
        .S_AXI_WDATA(s00_axi_wdata),
        .S_AXI_WSTRB(s00_axi_wstrb),
        .S_AXI_WVALID(s00_axi_wvalid),
        .S_AXI_WREADY(s00_axi_wready),
        .S_AXI_BRESP(s00_axi_bresp),
        .S_AXI_BVALID(s00_axi_bvalid),
        .S_AXI_BREADY(s00_axi_bready),
        .S_AXI_ARADDR(s00_axi_araddr),
        .S_AXI_ARPROT(s00_axi_arprot),
        .S_AXI_ARVALID(s00_axi_arvalid),
        .S_AXI_ARREADY(s00_axi_arready),
        .S_AXI_RDATA(s00_axi_rdata),
        .S_AXI_RRESP(s00_axi_rresp),
        .S_AXI_RVALID(s00_axi_rvalid),
        .S_AXI_RREADY(s00_axi_rready),
        .ctrl_reg(ctrl_reg),
        .config_reg(config_reg),
        .leak_rate(leak_rate),
        .threshold(threshold),
        .refractory_period(refractory_period),
        .status_reg(status_reg),
        .spike_count(spike_count)
    );

endmodule
}
close $fp

add_files -norecurse [list $s00_axi_file $top_axi_file]
set_property top ${ip_name}_${ip_version} [get_filesets sources_1]

update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

# Package the IP ----------------------------------------------------------------
ipx::package_project -root_dir $ip_dir -vendor $vendor_name -library $library_name \
    -taxonomy $taxonomy -import_files -force
set core [ipx::current_core]
set_property display_name "SNN AXI-Lite Register Bank" $core
set_property description "AXI4-Lite control/status register file for the SNN accelerator." $core

# Add AXI4-Lite slave interface
if {[llength [ipx::get_bus_interfaces S00_AXI -of_objects $core]] == 0} {
    set axi_if [ipx::add_bus_interface S00_AXI $core]
    set_property interface_mode {slave} $axi_if
    set_property abstraction_type_vlnv {xilinx.com:interface:aximm_rtl:1.0} $axi_if
    set_property bus_type_vlnv {xilinx.com:interface:aximm:1.0} $axi_if
} else {
    set axi_if [ipx::get_bus_interfaces S00_AXI -of_objects $core]
}

proc map_axi_port {if_name logical physical} {
    set pm [ipx::add_port_map $logical $if_name]
    set_property physical_name $physical $pm
}

map_axi_port $axi_if AWADDR  s00_axi_awaddr
map_axi_port $axi_if AWPROT  s00_axi_awprot
map_axi_port $axi_if AWVALID s00_axi_awvalid
map_axi_port $axi_if AWREADY s00_axi_awready
map_axi_port $axi_if WDATA   s00_axi_wdata
map_axi_port $axi_if WSTRB   s00_axi_wstrb
map_axi_port $axi_if WVALID  s00_axi_wvalid
map_axi_port $axi_if WREADY  s00_axi_wready
map_axi_port $axi_if BRESP   s00_axi_bresp
map_axi_port $axi_if BVALID  s00_axi_bvalid
map_axi_port $axi_if BREADY  s00_axi_bready
map_axi_port $axi_if ARADDR  s00_axi_araddr
map_axi_port $axi_if ARPROT  s00_axi_arprot
map_axi_port $axi_if ARVALID s00_axi_arvalid
map_axi_port $axi_if ARREADY s00_axi_arready
map_axi_port $axi_if RDATA   s00_axi_rdata
map_axi_port $axi_if RRESP   s00_axi_rresp
map_axi_port $axi_if RVALID  s00_axi_rvalid
map_axi_port $axi_if RREADY  s00_axi_rready

# Clock interface
if {[llength [ipx::get_bus_interfaces S00_AXI_CLK -of_objects $core]] == 0} {
    set clk_if [ipx::add_bus_interface S00_AXI_CLK $core]
    set_property interface_mode {slave} $clk_if
    set_property abstraction_type_vlnv {xilinx.com:interface:clock_rtl:1.0} $clk_if
    set_property bus_type_vlnv {xilinx.com:interface:clock:1.0} $clk_if
} else {
    set clk_if [ipx::get_bus_interfaces S00_AXI_CLK -of_objects $core]
}
set pm [ipx::add_port_map CLK $clk_if]
set_property physical_name s00_axi_aclk $pm

# Reset interface
if {[llength [ipx::get_bus_interfaces S00_AXI_RST -of_objects $core]] == 0} {
    set rst_if [ipx::add_bus_interface S00_AXI_RST $core]
    set_property interface_mode {slave} $rst_if
    set_property abstraction_type_vlnv {xilinx.com:interface:reset_rtl:1.0} $rst_if
    set_property bus_type_vlnv {xilinx.com:interface:reset:1.0} $rst_if
} else {
    set rst_if [ipx::get_bus_interfaces S00_AXI_RST -of_objects $core]
}
set pm [ipx::add_port_map RST $rst_if]
set_property physical_name s00_axi_aresetn $pm

# Memory map describing the register space
if {[llength [ipx::get_memory_maps S00_AXI -of_objects $core]] == 0} {
    set mm [ipx::add_memory_map S00_AXI $core]
} else {
    set mm [ipx::get_memory_maps S00_AXI -of_objects $core]
}
if {[llength [ipx::get_address_blocks S00_AXI_reg -of_objects $mm]] == 0} {
    set ab [ipx::add_address_block S00_AXI_reg $mm]
} else {
    set ab [ipx::get_address_blocks S00_AXI_reg -of_objects $mm]
}

# Tie the memory map to the AXI slave interface
set_property SLAVE_MEMORY_MAP_REF S00_AXI $axi_if

ipx::save_core $core
ipx::unload_core $core

close_project

puts "INFO: Packaged IP available at $ip_dir"
