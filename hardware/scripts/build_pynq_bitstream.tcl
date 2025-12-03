#------------------------------------------------------------------------------
# PYNQ-Z2 Bitstream Build Script for SNN Accelerator
# Creates complete design with PS and generates bitstream
# Target: PYNQ-Z2 (xc7z020clg400-1)
# Method: Package SNN RTL as IP, then create block design
#------------------------------------------------------------------------------

set script_dir [file dirname [file normalize [info script]]]
set proj_root [file normalize "$script_dir/../.."]
set part_name "xc7z020clg400-1"
set proj_name "snn_pynq"
set proj_dir "$proj_root/build/pynq_project"
set ip_repo_dir "$proj_root/build/ip_repo"

puts "=============================================="
puts "SNN Accelerator - PYNQ-Z2 Full Build"
puts "=============================================="
puts "Part: $part_name"
puts "Project: $proj_dir"
puts ""

#------------------------------------------------------------------------------
# Step 1: Create Project
#------------------------------------------------------------------------------
puts "Step 1: Creating project..."
file delete -force $proj_dir
file mkdir $ip_repo_dir
create_project $proj_name $proj_dir -part $part_name -force

#------------------------------------------------------------------------------
# Step 2: Package SNN Accelerator as IP
#------------------------------------------------------------------------------
puts "Step 2: Packaging SNN Accelerator IP..."
set hdl_dir "$proj_root/hardware/hdl"

# Add RTL source files to project  
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/common/*.v]
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/neurons/*.v]
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/synapses/*.v]
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/router/*.v]
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/layers/*.v]
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/interfaces/*.v]
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/top/*.v]

set_property top snn_accelerator_top [current_fileset]
update_compile_order -fileset sources_1

# Create IP packaging project
ipx::package_project -root_dir $ip_repo_dir/snn_accelerator -vendor user.org -library user -taxonomy /UserIP -force

# Configure IP interfaces
set core [ipx::current_core]
set_property vendor user.org $core
set_property library user $core
set_property name snn_accelerator $core
set_property version 1.0 $core
set_property display_name "SNN Accelerator" $core
set_property description "Event-Driven Spiking Neural Network Accelerator" $core

# Infer AXI4-Lite interface
ipx::infer_bus_interface {s_axi_awaddr s_axi_awprot s_axi_awvalid s_axi_awready s_axi_wdata s_axi_wstrb s_axi_wvalid s_axi_wready s_axi_bresp s_axi_bvalid s_axi_bready s_axi_araddr s_axi_arprot s_axi_arvalid s_axi_arready s_axi_rdata s_axi_rresp s_axi_rvalid s_axi_rready} xilinx.com:interface:aximm_rtl:1.0 [ipx::current_core]
set_property name S_AXI [ipx::get_bus_interfaces s_axi -of_objects [ipx::current_core]]

# Infer AXI4-Stream slave interface (input)
ipx::infer_bus_interface {s_axis_tdata s_axis_tvalid s_axis_tready s_axis_tlast} xilinx.com:interface:axis_rtl:1.0 [ipx::current_core]
set_property name S_AXIS [ipx::get_bus_interfaces s_axis -of_objects [ipx::current_core]]

# Infer AXI4-Stream master interface (output)
ipx::infer_bus_interface {m_axis_tdata m_axis_tvalid m_axis_tready m_axis_tlast} xilinx.com:interface:axis_rtl:1.0 [ipx::current_core]
set_property name M_AXIS [ipx::get_bus_interfaces m_axis -of_objects [ipx::current_core]]

# Associate clocks with interfaces
ipx::associate_bus_interfaces -busif S_AXI -clock aclk [ipx::current_core]
ipx::associate_bus_interfaces -busif S_AXIS -clock aclk [ipx::current_core]
ipx::associate_bus_interfaces -busif M_AXIS -clock aclk [ipx::current_core]

# Set reset polarity
set_property VALUE ACTIVE_LOW [ipx::get_bus_parameters POLARITY -of_objects [ipx::get_bus_interfaces aresetn -of_objects [ipx::current_core]]]

# Add address space
set_property CONFIG.ADDR_WIDTH 8 [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]
ipx::add_memory_map S_AXI [ipx::current_core]
set_property slave_memory_map_ref S_AXI [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]
ipx::add_address_block reg0 [ipx::get_memory_maps S_AXI -of_objects [ipx::current_core]]
set_property range 65536 [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps S_AXI -of_objects [ipx::current_core]]]

# Save and close IP
set_property core_revision 1 [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
ipx::check_integrity [ipx::current_core]
ipx::archive_core $ip_repo_dir/snn_accelerator.zip [ipx::current_core]

puts "IP packaged successfully."

#------------------------------------------------------------------------------
# Step 3: Add IP to repository  
#------------------------------------------------------------------------------
puts "Step 3: Setting up IP repository..."
set_property ip_repo_paths $ip_repo_dir [current_project]
update_ip_catalog

#------------------------------------------------------------------------------
# Step 4: Create Block Design with PS
#------------------------------------------------------------------------------
puts "Step 4: Creating block design..."
create_bd_design "design_1"

# Add ZYNQ7 Processing System
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0

# Apply PYNQ-Z2 configuration
set_property -dict [list \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_PRESET_BANK0_VOLTAGE {LVCMOS 3.3V} \
    CONFIG.PCW_PRESET_BANK1_VOLTAGE {LVCMOS 1.8V} \
    CONFIG.PCW_UIPARAM_DDR_PARTNO {MT41K256M16 RE-125} \
    CONFIG.PCW_UIPARAM_DDR_DEVICE_CAPACITY {4096 MBits} \
    CONFIG.PCW_UIPARAM_DDR_FREQ_MHZ {525} \
    CONFIG.PCW_UART0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_UART0_UART0_IO {MIO 14 .. 15} \
    CONFIG.PCW_TTC0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_SD0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_SD0_SD0_IO {MIO 40 .. 45} \
    CONFIG.PCW_USB0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_USB0_RESET_ENABLE {1} \
    CONFIG.PCW_USB0_RESET_IO {MIO 46} \
    CONFIG.PCW_ENET0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_ENET0_ENET0_IO {MIO 16 .. 27} \
    CONFIG.PCW_ENET0_GRP_MDIO_ENABLE {1} \
    CONFIG.PCW_ENET0_GRP_MDIO_IO {MIO 52 .. 53} \
] [get_bd_cells processing_system7_0]

# Add SNN Accelerator IP
create_bd_cell -type ip -vlnv user.org:user:snn_accelerator:1.0 snn_accelerator_0

# Add AXI Interconnect for GP0
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
set_property -dict [list CONFIG.NUM_MI {1}] [get_bd_cells axi_interconnect_0]

# Add DMA for AXI-Stream (optional, for high throughput)
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_mm2s_burst_size {16} \
    CONFIG.c_s2mm_burst_size {16} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} \
] [get_bd_cells axi_dma_0]

# Add AXI Interconnect for DMA
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_dma
set_property -dict [list CONFIG.NUM_MI {1}] [get_bd_cells axi_interconnect_dma]

# Add processor system reset
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_100M

#------------------------------------------------------------------------------
# Step 5: Connect Block Design
#------------------------------------------------------------------------------
puts "Step 5: Connecting block design..."

# Apply block automation for PS
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" Master "Disable" Slave "Disable" } [get_bd_cells processing_system7_0]

# Connect clocks
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins snn_accelerator_0/aclk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/S00_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/M00_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_dma/ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_dma/S00_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_dma/M00_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_dma_0/s_axi_lite_aclk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins rst_ps7_100M/slowest_sync_clk]

# Connect resets
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_100M/ext_reset_in]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins snn_accelerator_0/aresetn]
connect_bd_net [get_bd_pins rst_ps7_100M/interconnect_aresetn] [get_bd_pins axi_interconnect_0/ARESETN]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_interconnect_0/S00_ARESETN]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_interconnect_0/M00_ARESETN]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_dma_0/axi_resetn]
connect_bd_net [get_bd_pins rst_ps7_100M/interconnect_aresetn] [get_bd_pins axi_interconnect_dma/ARESETN]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_interconnect_dma/S00_ARESETN]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_interconnect_dma/M00_ARESETN]

# Connect AXI interfaces
# PS -> AXI Interconnect -> SNN Control Registers
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins axi_interconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M00_AXI] [get_bd_intf_pins snn_accelerator_0/S_AXI]

# PS -> AXI Interconnect -> DMA control
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_dma/S00_AXI] [get_bd_intf_pins processing_system7_0/M_AXI_GP0]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_dma/M00_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

# DMA to SNN (AXI Stream)
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins snn_accelerator_0/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins snn_accelerator_0/M_AXIS] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

# Connect interrupt
connect_bd_net [get_bd_pins snn_accelerator_0/interrupt] [get_bd_pins processing_system7_0/IRQ_F2P]

# Make LED outputs external
create_bd_port -dir O -from 3 -to 0 led
connect_bd_net [get_bd_pins snn_accelerator_0/led] [get_bd_ports led]

# Assign addresses
assign_bd_address

# Validate design
validate_bd_design
save_bd_design

#------------------------------------------------------------------------------
# Step 6: Generate Output Products
#------------------------------------------------------------------------------
puts "Step 6: Generating output products..."
generate_target all [get_files $proj_dir/$proj_name.srcs/sources_1/bd/design_1/design_1.bd]

#------------------------------------------------------------------------------
# Step 7: Create HDL Wrapper
#------------------------------------------------------------------------------
puts "Step 7: Creating HDL wrapper..."
make_wrapper -files [get_files $proj_dir/$proj_name.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse $proj_dir/$proj_name.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

#------------------------------------------------------------------------------
# Step 8: Add Constraints
#------------------------------------------------------------------------------
puts "Step 8: Adding constraints..."

set xdc_content {
# PYNQ-Z2 Board Constraints

# LEDs
set_property PACKAGE_PIN R14 [get_ports {led[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[0]}]
set_property PACKAGE_PIN P14 [get_ports {led[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[1]}]
set_property PACKAGE_PIN N16 [get_ports {led[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[2]}]
set_property PACKAGE_PIN M14 [get_ports {led[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[3]}]
}

set xdc_file "$proj_dir/constraints.xdc"
set fp [open $xdc_file w]
puts $fp $xdc_content
close $fp
add_files -fileset constrs_1 $xdc_file

#------------------------------------------------------------------------------
# Step 9: Run Synthesis
#------------------------------------------------------------------------------
puts "Step 9: Running synthesis..."
puts "=============================================="
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed!"
    puts "Check: $proj_dir/$proj_name.runs/synth_1/runme.log"
    exit 1
}
puts "Synthesis completed successfully."

#------------------------------------------------------------------------------
# Step 10: Run Implementation  
#------------------------------------------------------------------------------
puts "Step 10: Running implementation..."
puts "=============================================="
launch_runs impl_1 -jobs 8
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Implementation failed!"
    puts "Check: $proj_dir/$proj_name.runs/impl_1/runme.log"
    exit 1
}
puts "Implementation completed successfully."

#------------------------------------------------------------------------------
# Step 11: Generate Bitstream
#------------------------------------------------------------------------------
puts "Step 11: Generating bitstream..."
puts "=============================================="
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

# Copy output files
file mkdir "$proj_root/outputs"

set bitstream_src "$proj_dir/$proj_name.runs/impl_1/design_1_wrapper.bit"
set bitstream_dst "$proj_root/outputs/snn_accelerator.bit"
if {[file exists $bitstream_src]} {
    file copy -force $bitstream_src $bitstream_dst
    puts "Bitstream: $bitstream_dst"
}

# HWH file for PYNQ
set hwh_src "$proj_dir/$proj_name.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh"
set hwh_dst "$proj_root/outputs/snn_accelerator.hwh"
if {[file exists $hwh_src]} {
    file copy -force $hwh_src $hwh_dst
    puts "HWH file: $hwh_dst"
}

#------------------------------------------------------------------------------
# Step 12: Generate Reports
#------------------------------------------------------------------------------
puts ""
puts "=============================================="
puts "Build Complete!"
puts "=============================================="

open_run impl_1
report_utilization -file "$proj_root/outputs/utilization_report.txt"
report_timing_summary -file "$proj_root/outputs/timing_report.txt"

puts ""
puts "Output files in: $proj_root/outputs/"
puts "  - snn_accelerator.bit (bitstream)"
puts "  - snn_accelerator.hwh (hardware handoff for PYNQ)"
puts "  - utilization_report.txt"
puts "  - timing_report.txt"
puts ""

exit 0
