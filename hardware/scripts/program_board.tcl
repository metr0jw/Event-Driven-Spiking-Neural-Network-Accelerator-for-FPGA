##-----------------------------------------------------------------------------
## Title         : FPGA Programming Script
## Project       : PYNQ-Z2 SNN Accelerator
## File          : program_board.tcl
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Programs PYNQ-Z2 board with bitstream
##-----------------------------------------------------------------------------

# Get script directory
set script_dir [file dirname [file normalize [info script]]]
set proj_root [file normalize "$script_dir/../.."]

# Default bitstream location
set default_bit "$proj_root/build/bitstreams/snn_accelerator.bit"
set default_ltx "$proj_root/build/bitstreams/debug_nets.ltx"

# Parse command line arguments
if {[info exists argc] && $argc > 0} {
    set bit_file [lindex $argv 0]
} else {
    set bit_file $default_bit
}

# Check if bitstream exists
if {![file exists $bit_file]} {
    puts "ERROR: Bitstream file not found: $bit_file"
    exit 1
}

puts "----------------------------------------"
puts "PYNQ-Z2 Programming Script"
puts "Bitstream: $bit_file"
puts "----------------------------------------"

# Open hardware manager
open_hw_manager

# Connect to hardware server
puts "Connecting to hardware server..."
connect_hw_server -allow_non_jtag

# Get hardware targets
set hw_targets [get_hw_targets]
if {[llength $hw_targets] == 0} {
    puts "ERROR: No hardware targets found. Please check board connection."
    exit 1
}

# Open first target
puts "Opening hardware target..."
open_hw_target

# Get devices
set hw_devices [get_hw_devices]
puts "Found devices: $hw_devices"

# Find Zynq device
set zynq_device ""
foreach device $hw_devices {
    if {[string match "*xc7z020*" $device]} {
        set zynq_device $device
        break
    }
}

if {$zynq_device == ""} {
    puts "ERROR: No Zynq device found"
    exit 1
}

puts "Programming device: $zynq_device"

# Set current device
current_hw_device $zynq_device
refresh_hw_device -update_hw_probes false $zynq_device

# Set programming properties
set_property PROGRAM.FILE $bit_file $zynq_device

# Check for debug probes file
if {[file exists $default_ltx]} {
    set_property PROBES.FILE $default_ltx $zynq_device
    set_property FULL_PROBES.FILE $default_ltx $zynq_device
    puts "Debug probes file loaded: $default_ltx"
}

# Program the device
puts "Programming FPGA..."
program_hw_devices $zynq_device

# Verify programming
refresh_hw_device $zynq_device
set prog_done [get_property PROGRAM.IS_DONE $zynq_device]

if {$prog_done} {
    puts "----------------------------------------"
    puts "FPGA programmed successfully!"
    puts "----------------------------------------"
} else {
    puts "ERROR: FPGA programming failed!"
    exit 1
}

# Close hardware target
close_hw_target

# Close hardware manager
close_hw_manager

puts "\nProgramming completed."
