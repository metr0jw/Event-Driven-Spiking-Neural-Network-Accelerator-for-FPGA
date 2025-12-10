#-----------------------------------------------------------------------------
# Title         : PYNQ Bitstream Build with HLS Learning Engine
# Project       : PYNQ-Z2 SNN Accelerator
# File          : build_pynq_with_hls.tcl
# Author        : Jiwoon Lee (@metr0jw)
# Organization  : Kwangwoon University, Seoul, South Korea
# Contact       : jwlee@linux.com
# Description   : Builds bitstream integrating HLS-based on-chip learning
#-----------------------------------------------------------------------------

# Parse command line arguments
set num_jobs 8
if {$argc >= 2} {
    if {[lindex $argv 0] == "-jobs"} {
        set num_jobs [lindex $argv 1]
        puts "Using $num_jobs parallel jobs"
    }
}

# Get script directory
set script_dir [file dirname [file normalize [info script]]]
set proj_root [file normalize "$script_dir/../.."]
set hls_ip_dir "$proj_root/hardware/ip_repo"

# Project settings
set proj_name "snn_accelerator_hls"
set proj_dir "$proj_root/build/vivado_hls"
set part "xc7z020clg400-1"

puts "=============================================="
puts "PYNQ-Z2 SNN Accelerator Build"
puts "With HLS On-Chip Learning Engine"
puts "=============================================="
puts "Project root: $proj_root"
puts "HLS IP dir: $hls_ip_dir"
puts ""

#-----------------------------------------------------------------------------
# Step 1: Check HLS IP exists
#-----------------------------------------------------------------------------
puts "Step 1: Checking HLS IP..."

# Check for v++ generated output first, then legacy location
set hls_output_path "$proj_root/hardware/hls/hls_output"
set hls_ip_path "$hls_ip_dir/snn_top_hls_1_0"

if {[file exists "$hls_output_path/hls/impl/ip"]} {
    set hls_ip_dir "$hls_output_path/hls/impl/ip"
    puts "  Found v++ HLS IP: $hls_ip_dir"
} elseif {[file exists $hls_ip_path]} {
    set hls_ip_dir $hls_ip_path
    puts "  Found legacy HLS IP: $hls_ip_dir"
} else {
    puts "ERROR: HLS IP not found"
    puts "Please run HLS synthesis first:"
    puts "  cd hardware/hls/scripts"
    puts "  ./build_hls.sh"
    puts ""
    puts "Note: vitis_hls -f script.tcl is deprecated in Vitis 2025.2+"
    exit 1
}

#-----------------------------------------------------------------------------
# Step 2: Create Vivado project
#-----------------------------------------------------------------------------
puts ""
puts "Step 2: Creating Vivado project..."

file mkdir $proj_dir
create_project $proj_name $proj_dir -part $part -force

# Set project properties
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

#-----------------------------------------------------------------------------
# Step 3: Add Verilog RTL sources
#-----------------------------------------------------------------------------
puts ""
puts "Step 3: Adding Verilog RTL sources..."

set rtl_dir "$proj_root/hardware/hdl/rtl"

# Add all Verilog source files to project
set rtl_files [list]
lappend rtl_files {*}[glob -nocomplain $rtl_dir/common/*.v]
lappend rtl_files {*}[glob -nocomplain $rtl_dir/neurons/*.v]
lappend rtl_files {*}[glob -nocomplain $rtl_dir/synapses/*.v]
lappend rtl_files {*}[glob -nocomplain $rtl_dir/router/*.v]
lappend rtl_files {*}[glob -nocomplain $rtl_dir/layers/*.v]

# Add only the specific top module we need
set top_module_file "$rtl_dir/top/snn_accelerator_hls_top.v"
if {[file exists $top_module_file]} {
    lappend rtl_files $top_module_file
}

# Add axi_hls_wrapper if it exists (needed by snn_accelerator_hls_top)
set axi_wrapper_file "$rtl_dir/top/axi_hls_wrapper.v"
if {[file exists $axi_wrapper_file]} {
    lappend rtl_files $axi_wrapper_file
}

if {[llength $rtl_files] > 0} {
    add_files -norecurse $rtl_files
    puts "  Added [llength $rtl_files] Verilog RTL files:"
    foreach f $rtl_files {
        puts "    - [file tail $f]"
    }
} else {
    puts "  WARNING: No Verilog RTL files found in $rtl_dir"
}

#-----------------------------------------------------------------------------
# Step 4: Add IP repository
#-----------------------------------------------------------------------------
puts ""
puts "Step 4: Setting up IP repository..."

set_property ip_repo_paths [list $hls_ip_dir] [current_project]
update_ip_catalog

#-----------------------------------------------------------------------------
# Step 5: Create Block Design
#-----------------------------------------------------------------------------
puts ""
puts "Step 5: Creating block design..."

create_bd_design "design_1"

# Add ZYNQ7 Processing System
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0

# Configure PS for PYNQ-Z2
set_property -dict [list \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_PRESET_BANK0_VOLTAGE {LVCMOS 3.3V} \
    CONFIG.PCW_PRESET_BANK1_VOLTAGE {LVCMOS 1.8V} \
    CONFIG.PCW_UIPARAM_DDR_PARTNO {MT41K256M16 RE-125} \
    CONFIG.PCW_UIPARAM_DDR_FREQ_MHZ {525} \
    CONFIG.PCW_UART0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_SD0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_USB0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_ENET0_PERIPHERAL_ENABLE {1} \
] [get_bd_cells processing_system7_0]

# Add HLS SNN IP
create_bd_cell -type ip -vlnv xilinx.com:hls:snn_top_hls:1.0 snn_top_hls_0

# Add Verilog SNN Core as hierarchy module
set snn_core_files [list]
lappend snn_core_files {*}[glob -nocomplain $rtl_dir/common/*.v]
lappend snn_core_files {*}[glob -nocomplain $rtl_dir/neurons/*.v]
lappend snn_core_files {*}[glob -nocomplain $rtl_dir/synapses/*.v]
lappend snn_core_files {*}[glob -nocomplain $rtl_dir/router/*.v]

if {[llength $snn_core_files] > 0} {
    puts "  Creating SNN Core hierarchy with [llength $snn_core_files] Verilog files"
    # Files are already added, just need to reference them
}

# Add AXI Interconnect for control path (GP0)
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
set_property -dict [list CONFIG.NUM_MI {1}] [get_bd_cells axi_interconnect_0]

# Add AXI SmartConnect for data path (HP0) - handles AXI4 to AXI3 conversion
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_hp0
set_property -dict [list \
    CONFIG.NUM_SI {2} \
    CONFIG.NUM_MI {1} \
] [get_bd_cells axi_interconnect_hp0]

# Add AXI DMA for spike streaming
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_m_axi_mm2s_data_width {32} \
    CONFIG.c_m_axis_mm2s_tdata_width {32} \
    CONFIG.c_m_axi_s2mm_data_width {32} \
    CONFIG.c_s_axis_s2mm_tdata_width {32} \
] [get_bd_cells axi_dma_0]

# Add Processor System Reset
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0

#-----------------------------------------------------------------------------
# Step 6: Connect Block Design
#-----------------------------------------------------------------------------
puts ""
puts "Step 6: Connecting block design..."

# Connect clocks
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins snn_top_hls_0/ap_clk] \
    [get_bd_pins axi_interconnect_0/ACLK] \
    [get_bd_pins axi_interconnect_0/S00_ACLK] \
    [get_bd_pins axi_interconnect_0/M00_ACLK] \
    [get_bd_pins axi_interconnect_hp0/ACLK] \
    [get_bd_pins axi_interconnect_hp0/S00_ACLK] \
    [get_bd_pins axi_interconnect_hp0/S01_ACLK] \
    [get_bd_pins axi_interconnect_hp0/M00_ACLK] \
    [get_bd_pins axi_dma_0/s_axi_lite_aclk] \
    [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \
    [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] \
    [get_bd_pins proc_sys_reset_0/slowest_sync_clk]

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] \
    [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK]

# Connect resets
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] \
    [get_bd_pins proc_sys_reset_0/ext_reset_in]

connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
    [get_bd_pins snn_top_hls_0/ap_rst_n] \
    [get_bd_pins axi_interconnect_0/ARESETN] \
    [get_bd_pins axi_interconnect_0/S00_ARESETN] \
    [get_bd_pins axi_interconnect_0/M00_ARESETN] \
    [get_bd_pins axi_interconnect_hp0/ARESETN] \
    [get_bd_pins axi_interconnect_hp0/S00_ARESETN] \
    [get_bd_pins axi_interconnect_hp0/S01_ARESETN] \
    [get_bd_pins axi_interconnect_hp0/M00_ARESETN] \
    [get_bd_pins axi_dma_0/axi_resetn]

# Connect AXI control interfaces (GP0 path)
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] \
    [get_bd_intf_pins axi_interconnect_0/S00_AXI]

connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M00_AXI] \
    [get_bd_intf_pins snn_top_hls_0/s_axi_ctrl]

# Connect DMA to HP port via AXI Interconnect (for AXI4 to AXI3 conversion)
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] \
    [get_bd_intf_pins axi_interconnect_hp0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] \
    [get_bd_intf_pins axi_interconnect_hp0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_hp0/M00_AXI] \
    [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

# Connect AXI-Stream interfaces
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] \
    [get_bd_intf_pins snn_top_hls_0/s_axis_spikes]
connect_bd_intf_net [get_bd_intf_pins snn_top_hls_0/m_axis_spikes] \
    [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

#-----------------------------------------------------------------------------
# Step 7: Assign Addresses
#-----------------------------------------------------------------------------
puts ""
puts "Step 7: Assigning addresses..."

assign_bd_address

# Set specific addresses
set_property offset 0x43C00000 [get_bd_addr_segs {processing_system7_0/Data/SEG_snn_top_hls_0_Reg}]
set_property range 64K [get_bd_addr_segs {processing_system7_0/Data/SEG_snn_top_hls_0_Reg}]

#-----------------------------------------------------------------------------
# Step 8: Validate and Save
#-----------------------------------------------------------------------------
puts ""
puts "Step 8: Validating block design..."

validate_bd_design
save_bd_design

# Generate wrapper
make_wrapper -files [get_files $proj_dir/$proj_name.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse $proj_dir/$proj_name.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v

# Set wrapper as top module
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

#-----------------------------------------------------------------------------
# Step 9: Add Constraints
#-----------------------------------------------------------------------------
puts ""
puts "Step 9: Adding constraints..."

set xdc_content {
# PYNQ-Z2 constraints
set_property IOSTANDARD LVCMOS33 [get_ports {led[0]}]
set_property PACKAGE_PIN R14 [get_ports {led[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[1]}]
set_property PACKAGE_PIN P14 [get_ports {led[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[2]}]
set_property PACKAGE_PIN N16 [get_ports {led[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[3]}]
set_property PACKAGE_PIN M14 [get_ports {led[3]}]
}

set xdc_file "$proj_dir/pynq.xdc"
set fp [open $xdc_file w]
puts $fp $xdc_content
close $fp
add_files -fileset constrs_1 $xdc_file

#-----------------------------------------------------------------------------
# Step 10: Synthesis
#-----------------------------------------------------------------------------
puts ""
puts "Step 10: Running synthesis with $num_jobs jobs..."

launch_runs synth_1 -jobs $num_jobs
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed!"
    exit 1
}

#-----------------------------------------------------------------------------
# Step 11: Implementation
#-----------------------------------------------------------------------------
puts ""
puts "Step 11: Running implementation with $num_jobs jobs..."

launch_runs impl_1 -jobs $num_jobs
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Implementation failed!"
    exit 1
}

#-----------------------------------------------------------------------------
# Step 12: Bitstream
#-----------------------------------------------------------------------------
puts ""
puts "Step 12: Generating bitstream with $num_jobs jobs..."

launch_runs impl_1 -to_step write_bitstream -jobs $num_jobs
wait_on_run impl_1

#-----------------------------------------------------------------------------
# Step 13: Export outputs
#-----------------------------------------------------------------------------
puts ""
puts "Step 13: Exporting outputs..."

file mkdir "$proj_root/outputs"

# Copy bitstream
set bit_src "$proj_dir/$proj_name.runs/impl_1/design_1_wrapper.bit"
set bit_dst "$proj_root/outputs/snn_accelerator_hls.bit"
if {[file exists $bit_src]} {
    file copy -force $bit_src $bit_dst
    puts "Bitstream: $bit_dst"
}

# Copy HWH
set hwh_src "$proj_dir/$proj_name.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh"
set hwh_dst "$proj_root/outputs/snn_accelerator_hls.hwh"
if {[file exists $hwh_src]} {
    file copy -force $hwh_src $hwh_dst
    puts "HWH: $hwh_dst"
}

# Generate reports
open_run impl_1
report_utilization -file "$proj_root/outputs/utilization_hls.txt"
report_timing_summary -file "$proj_root/outputs/timing_hls.txt"

#-----------------------------------------------------------------------------
# Summary
#-----------------------------------------------------------------------------
puts ""
puts "=============================================="
puts "BUILD COMPLETE!"
puts "=============================================="
puts ""
puts "Output files:"
puts "  - outputs/snn_accelerator_hls.bit"
puts "  - outputs/snn_accelerator_hls.hwh"
puts "  - outputs/utilization_hls.txt"
puts "  - outputs/timing_hls.txt"
puts ""
puts "Features:"
puts "  - On-chip STDP learning"
puts "  - R-STDP with eligibility traces"
puts "  - AXI4-Lite control interface"
puts "  - AXI4-Stream spike I/O"
puts ""

exit 0
