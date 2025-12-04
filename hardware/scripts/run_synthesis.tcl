##-----------------------------------------------------------------------------
## Title         : Run Synthesis and Implementation Script
## Project       : PYNQ-Z2 SNN Accelerator
## File          : run_synthesis.tcl
## Description   : Runs synthesis and optionally implementation
##-----------------------------------------------------------------------------

# Get project directory
set proj_dir "/home/jwlee/workspace/Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA/build/vivado"
set proj_name "snn_accelerator_pynq"

puts "=============================================="
puts "Opening Project..."
puts "=============================================="
open_project "$proj_dir/$proj_name.xpr"

puts "=============================================="
puts "Running Synthesis..."
puts "=============================================="

# Reset synthesis run
reset_run synth_1

# Launch synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Check synthesis results
set synth_status [get_property STATUS [get_runs synth_1]]
puts "Synthesis Status: $synth_status"

if {$synth_status ne "synth_design Complete!"} {
    puts "ERROR: Synthesis failed!"
    close_project
    exit 1
}

# Get synthesis report
puts ""
puts "=============================================="
puts "Synthesis Report Summary"
puts "=============================================="

# Open synthesized design to get utilization
open_run synth_1 -name synth_1

# Report utilization
report_utilization -file "$proj_dir/utilization_synth.rpt" -hierarchical
puts "Utilization report written to: utilization_synth.rpt"

# Report timing summary
report_timing_summary -file "$proj_dir/timing_synth.rpt"
puts "Timing report written to: timing_synth.rpt"

close_design

puts ""
puts "=============================================="
puts "Synthesis Complete!"
puts "=============================================="
puts ""
puts "To run implementation, execute run_implementation.tcl"
puts ""

close_project
exit 0
