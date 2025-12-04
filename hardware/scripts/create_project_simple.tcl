##-----------------------------------------------------------------------------
## Title         : Vivado Project Creation Script (Simplified)
## Project       : PYNQ-Z2 SNN Accelerator
## File          : create_project_simple.tcl
## Description   : Creates Vivado project without board_part requirement
##-----------------------------------------------------------------------------

# Get the directory where this script is located
set script_dir [file dirname [file normalize [info script]]]
set proj_root [file normalize "$script_dir/../.."]

# Project settings
set proj_name "snn_accelerator_pynq"
set proj_dir "$proj_root/build/vivado"
set part_name "xc7z020clg400-1"

puts "=============================================="
puts "Creating SNN Accelerator Vivado Project"
puts "=============================================="
puts "Project Root: $proj_root"
puts "Project Dir: $proj_dir"
puts "Target Device: $part_name"

# Create project directory
file mkdir $proj_dir

# Create project (without board_part)
create_project $proj_name $proj_dir -part $part_name -force

# Set project properties
set obj [current_project]
set_property -name "default_lib" -value "xil_defaultlib" -objects $obj
set_property -name "enable_vhdl_2008" -value "1" -objects $obj
set_property -name "ip_cache_permissions" -value "read write" -objects $obj
set_property -name "ip_output_repo" -value "$proj_dir/${proj_name}.cache/ip" -objects $obj
set_property -name "mem.enable_memory_map_generation" -value "1" -objects $obj
set_property -name "sim.central_dir" -value "$proj_dir/${proj_name}.ip_user_files" -objects $obj
set_property -name "sim.ip.auto_export_scripts" -value "1" -objects $obj
set_property -name "simulator_language" -value "Mixed" -objects $obj
set_property -name "target_language" -value "Verilog" -objects $obj

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

puts "Adding RTL sources from: $hdl_dir"

# Add RTL sources
set src_files [list]
foreach dir {top neurons synapses router interfaces common layers} {
    set files [glob -nocomplain $hdl_dir/rtl/$dir/*.v]
    if {[llength $files] > 0} {
        lappend src_files {*}$files
        puts "  Found [llength $files] files in rtl/$dir"
    }
}

if {[llength $src_files] > 0} {
    add_files -norecurse -fileset sources_1 $src_files
}

# Set top module
set_property top snn_accelerator_top [current_fileset]

# Add testbenches
puts "Adding testbenches..."
set tb_files [glob -nocomplain $hdl_dir/tb/*.v]
if {[llength $tb_files] > 0} {
    add_files -norecurse -fileset sim_1 $tb_files
    set_property top tb_top [get_filesets sim_1]
    set_property top_lib xil_defaultlib [get_filesets sim_1]
    puts "  Added [llength $tb_files] testbench files"
}

# Add constraints
puts "Adding constraints..."
set constr_dir "$proj_root/hardware/constraints"
foreach xdc {pynq_z2_v1.0.xdc timing.xdc bitstream.xdc pynq_z2_pins.xdc} {
    set xdc_file "$constr_dir/$xdc"
    if {[file exists $xdc_file]} {
        add_files -fileset constrs_1 -norecurse $xdc_file
        puts "  Added: $xdc"
    }
}

# Configure synthesis run
set obj [get_runs synth_1]
set_property strategy "Vivado Synthesis Defaults" $obj
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {-directive default} -objects $obj

# Configure implementation run  
set obj [get_runs impl_1]
set_property strategy "Vivado Implementation Defaults" $obj

puts ""
puts "=============================================="
puts "Project Created Successfully!"
puts "=============================================="
puts "Project Name: $proj_name"
puts "Location: $proj_dir"
puts ""
puts "Next steps:"
puts "1. Open project in Vivado GUI to add PS block design"
puts "2. Run synthesis: launch_runs synth_1 -jobs 4"
puts "3. Run implementation: launch_runs impl_1 -to_step write_bitstream -jobs 4"
puts ""
