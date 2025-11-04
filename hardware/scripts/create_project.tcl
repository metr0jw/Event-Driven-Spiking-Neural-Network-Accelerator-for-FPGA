##-----------------------------------------------------------------------------
## Title         : Vivado Project Creation Script
## Project       : PYNQ-Z2 SNN Accelerator
## File          : create_project.tcl
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Creates Vivado project for PYNQ-Z2 SNN accelerator
##-----------------------------------------------------------------------------

# Get the directory where this script is located
set script_dir [file dirname [file normalize [info script]]]
set proj_root [file normalize "$script_dir/../.."]

# Project settings
set proj_name "snn_accelerator_pynq"
set proj_dir "$proj_root/build/vivado"
set part_name "xc7z020clg400-1"
set board_part "tul.com.tw:pynq-z2:part0:1.0"

# Create project
create_project $proj_name $proj_dir -part $part_name -force
set_property board_part $board_part [current_project]

# Set project properties
set obj [current_project]
set_property -name "default_lib" -value "xil_defaultlib" -objects $obj
set_property -name "enable_vhdl_2008" -value "1" -objects $obj
set_property -name "ip_cache_permissions" -value "read write" -objects $obj
set_property -name "ip_output_repo" -value "$proj_dir/${proj_name}.cache/ip" -objects $obj
set_property -name "mem.enable_memory_map_generation" -value "1" -objects $obj
set_property -name "platform.board_id" -value "pynq-z2" -objects $obj
set_property -name "sim.central_dir" -value "$proj_dir/${proj_name}.ip_user_files" -objects $obj
set_property -name "sim.ip.auto_export_scripts" -value "1" -objects $obj
set_property -name "simulator_language" -value "Mixed" -objects $obj

# Create filesets
if {[string equal [get_filesets -quiet sources_1] ""]} {
    create_fileset -srcset sources_1
}

if {[string equal [get_filesets -quiet constrs_1] ""]} {
    create_fileset -constrset constrs_1
}

if {[string equal [get_filesets -quiet sim_1] ""]} {
    create_fileset -simset sim_1
}

# Add HDL source files
set hdl_dir "$proj_root/hardware/hdl"

# Add RTL sources
add_files -norecurse -fileset sources_1 [glob -nocomplain $hdl_dir/rtl/top/*.v]
add_files -norecurse -fileset sources_1 [glob -nocomplain $hdl_dir/rtl/neurons/*.v]
add_files -norecurse -fileset sources_1 [glob -nocomplain $hdl_dir/rtl/synapses/*.v]
add_files -norecurse -fileset sources_1 [glob -nocomplain $hdl_dir/rtl/router/*.v]
add_files -norecurse -fileset sources_1 [glob -nocomplain $hdl_dir/rtl/interfaces/*.v]
add_files -norecurse -fileset sources_1 [glob -nocomplain $hdl_dir/rtl/common/*.v]

# Set top module
set_property top snn_accelerator_top [current_fileset]

# Add testbenches
set obj [get_filesets sim_1]
add_files -norecurse -fileset $obj [glob -nocomplain $hdl_dir/tb/*.v]
set_property top tb_top $obj
set_property top_lib xil_defaultlib $obj

# Add constraints
# Use the official PYNQ-Z2 constraints as base
add_files -fileset constrs_1 -norecurse "$proj_root/hardware/constraints/pynq_z2_v1.0.xdc"
# Add your custom constraints (only for modifications/additions)
if {[file exists "$proj_root/hardware/constraints/timing.xdc"]} {
    add_files -fileset constrs_1 -norecurse "$proj_root/hardware/constraints/timing.xdc"
}
if {[file exists "$proj_root/hardware/constraints/bitstream.xdc"]} {
    add_files -fileset constrs_1 -norecurse "$proj_root/hardware/constraints/bitstream.xdc"
}

# Set constraint processing order
set_property PROCESSING_ORDER EARLY [get_files pynq_z2_pins.xdc]
set_property PROCESSING_ORDER NORMAL [get_files timing.xdc]
set_property PROCESSING_ORDER LATE [get_files bitstream.xdc]

# Add IP repository paths (if any)
set ip_repo_list [list]
lappend ip_repo_list "$proj_root/hardware/ip/custom_ip"
lappend ip_repo_list "$proj_root/hardware/ip/hls_ip"

set_property ip_repo_paths $ip_repo_list [current_project]
update_ip_catalog

# Create block design for Zynq PS configuration
source "$script_dir/create_block_design.tcl"

# Generate wrapper for block design
make_wrapper -files [get_files "$proj_dir/${proj_name}.srcs/sources_1/bd/system/system.bd"] -top
add_files -norecurse "$proj_dir/${proj_name}.srcs/sources_1/bd/system/hdl/system_wrapper.v"
set_property top system_wrapper [current_fileset]
update_compile_order -fileset sources_1

# Configure runs
set obj [get_runs synth_1]
set_property strategy "Vivado Synthesis Defaults" $obj
set_property flow "Vivado Synthesis 2023" $obj
set_property report_strategy {Vivado Synthesis Default Reports} $obj
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {-directive default} -objects $obj

set obj [get_runs impl_1]
set_property strategy "Vivado Implementation Defaults" $obj
set_property flow "Vivado Implementation 2023" $obj
set_property report_strategy {Vivado Implementation Default Reports} $obj
set_property -name {STEPS.OPT_DESIGN.ARGS.MORE OPTIONS} -value {} -objects $obj
set_property -name {STEPS.PLACE_DESIGN.ARGS.MORE OPTIONS} -value {} -objects $obj
set_property -name {STEPS.ROUTE_DESIGN.ARGS.MORE OPTIONS} -value {} -objects $obj

puts "Project created successfully: $proj_name"
puts "Project location: $proj_dir"
