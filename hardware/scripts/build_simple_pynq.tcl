#------------------------------------------------------------------------------
# Simple PYNQ-Z2 Build Script for SNN Accelerator
# Creates minimal block design with PS + SNN IP
# Target: PYNQ-Z2 (xc7z020clg400-1)
#------------------------------------------------------------------------------

set script_dir [file dirname [file normalize [info script]]]
set proj_root [file normalize "$script_dir/../.."]
set part_name "xc7z020clg400-1"
set proj_name "snn_simple"
set proj_dir "$proj_root/build/simple_project"

puts "=============================================="
puts "SNN Accelerator - Simple PYNQ Build"
puts "=============================================="
puts "Part: $part_name"
puts "Project: $proj_dir"
puts ""

#------------------------------------------------------------------------------
# Step 1: Create Project and Add Sources
#------------------------------------------------------------------------------
puts "Step 1: Creating project..."
file delete -force $proj_dir
create_project $proj_name $proj_dir -part $part_name -force

set hdl_dir "$proj_root/hardware/hdl"
set ip_dir "$proj_root/hardware/ip_repo"

# Add all RTL and IP files
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/common/*.v]
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/neurons/*.v]
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/synapses/*.v]
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/router/*.v]
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/layers/*.v]
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/interfaces/*.v]
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/top/*.v]
add_files -norecurse [glob -nocomplain $ip_dir/axi_lite_regs_v1_0/src/*.v]
add_files -norecurse [glob -nocomplain $ip_dir/axi_1_0/hdl/*.v]

update_compile_order -fileset sources_1

#------------------------------------------------------------------------------
# Step 2: Create Block Design
#------------------------------------------------------------------------------
puts "Step 2: Creating block design..."
create_bd_design "design_1"

# Add ZYNQ7 PS with basic config
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7_0
set_property -dict [list \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_IRQ_F2P_INTR {1} \
    CONFIG.preset {ZC702} \
] [get_bd_cells ps7_0]

# Apply automation for DDR and FIXED_IO
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" Master "Disable" Slave "Disable"} \
    [get_bd_cells ps7_0]

# Add SNN module as RTL reference
create_bd_cell -type module -reference snn_accelerator_top snn_0

# Add proc sys reset
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_0

#------------------------------------------------------------------------------
# Step 3: Connect Design
#------------------------------------------------------------------------------
puts "Step 3: Connecting design..."

# Clock connections
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins snn_0/aclk]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins rst_0/slowest_sync_clk]

# Reset connections
connect_bd_net [get_bd_pins ps7_0/FCLK_RESET0_N] [get_bd_pins rst_0/ext_reset_in]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn] [get_bd_pins snn_0/aresetn]

# AXI connection - use automation
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/ps7_0/M_AXI_GP0" intc_ip "Auto" Clk_xbar "Auto" Clk_master "Auto" Clk_slave "Auto"} \
    [get_bd_intf_pins snn_0/s_axi]

# Connect interrupt
connect_bd_net [get_bd_pins snn_0/interrupt] [get_bd_pins ps7_0/IRQ_F2P]

# External LED port
create_bd_port -dir O -from 3 -to 0 led
connect_bd_net [get_bd_pins snn_0/led] [get_bd_ports led]

# AXI Stream - tie off for now (no DMA in simple version)
# Input stream - create constant low for not-valid
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_zero
set_property -dict [list CONFIG.CONST_VAL {0} CONFIG.CONST_WIDTH {32}] [get_bd_cells const_zero]

create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_gnd
set_property -dict [list CONFIG.CONST_VAL {0} CONFIG.CONST_WIDTH {1}] [get_bd_cells const_gnd]

connect_bd_net [get_bd_pins const_zero/dout] [get_bd_pins snn_0/s_axis_tdata]
connect_bd_net [get_bd_pins const_gnd/dout] [get_bd_pins snn_0/s_axis_tvalid]
connect_bd_net [get_bd_pins const_gnd/dout] [get_bd_pins snn_0/s_axis_tlast]

# Output stream ready
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_one
set_property -dict [list CONFIG.CONST_VAL {1} CONFIG.CONST_WIDTH {1}] [get_bd_cells const_one]
connect_bd_net [get_bd_pins const_one/dout] [get_bd_pins snn_0/m_axis_tready]

# Tie off unused inputs - create constant bus for sw and btn
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_sw
set_property -dict [list CONFIG.CONST_VAL {0} CONFIG.CONST_WIDTH {2}] [get_bd_cells const_sw]
connect_bd_net [get_bd_pins const_sw/dout] [get_bd_pins snn_0/sw]

create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_btn
set_property -dict [list CONFIG.CONST_VAL {0} CONFIG.CONST_WIDTH {4}] [get_bd_cells const_btn]
connect_bd_net [get_bd_pins const_btn/dout] [get_bd_pins snn_0/btn]

# Validate and save
regenerate_bd_layout
validate_bd_design
save_bd_design

#------------------------------------------------------------------------------
# Step 4: Generate HDL Wrapper
#------------------------------------------------------------------------------
puts "Step 4: Generating wrapper..."
make_wrapper -files [get_files $proj_dir/$proj_name.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse $proj_dir/$proj_name.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

#------------------------------------------------------------------------------
# Step 5: Add Constraints
#------------------------------------------------------------------------------
puts "Step 5: Adding constraints..."
set xdc_content {
# PYNQ-Z2 LED Constraints
set_property PACKAGE_PIN R14 [get_ports {led[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[0]}]
set_property PACKAGE_PIN P14 [get_ports {led[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[1]}]
set_property PACKAGE_PIN N16 [get_ports {led[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[2]}]
set_property PACKAGE_PIN M14 [get_ports {led[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[3]}]
}
set xdc_file "$proj_dir/pynq.xdc"
set fp [open $xdc_file w]
puts $fp $xdc_content
close $fp
add_files -fileset constrs_1 $xdc_file

#------------------------------------------------------------------------------
# Step 6: Run Synthesis
#------------------------------------------------------------------------------
puts "Step 6: Running synthesis..."
launch_runs synth_1 -jobs 8
wait_on_run synth_1

set synth_status [get_property STATUS [get_runs synth_1]]
puts "Synthesis status: $synth_status"

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed!"
    puts "Check: $proj_dir/$proj_name.runs/synth_1/runme.log"
    exit 1
}

#------------------------------------------------------------------------------
# Step 7: Run Implementation
#------------------------------------------------------------------------------
puts "Step 7: Running implementation..."
launch_runs impl_1 -jobs 8
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Implementation failed!"
    exit 1
}

#------------------------------------------------------------------------------
# Step 8: Generate Bitstream
#------------------------------------------------------------------------------
puts "Step 8: Generating bitstream..."
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

# Copy outputs
file mkdir "$proj_root/outputs"
set bit_src "$proj_dir/$proj_name.runs/impl_1/design_1_wrapper.bit"
set bit_dst "$proj_root/outputs/snn_accelerator.bit"
if {[file exists $bit_src]} {
    file copy -force $bit_src $bit_dst
    puts "\nBitstream: $bit_dst"
}

set hwh_src "$proj_dir/$proj_name.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh"
set hwh_dst "$proj_root/outputs/snn_accelerator.hwh"
if {[file exists $hwh_src]} {
    file copy -force $hwh_src $hwh_dst
    puts "HWH: $hwh_dst"
}

#------------------------------------------------------------------------------
# Step 9: Reports
#------------------------------------------------------------------------------
puts "\n=============================================="
puts "BUILD COMPLETE"
puts "=============================================="

open_run impl_1
report_utilization -file "$proj_root/outputs/utilization.txt"
report_timing_summary -file "$proj_root/outputs/timing.txt" -warn_on_violation

puts "\nOutputs in: $proj_root/outputs/"
puts "  snn_accelerator.bit"
puts "  snn_accelerator.hwh"
puts "  utilization.txt"
puts "  timing.txt"

exit 0
