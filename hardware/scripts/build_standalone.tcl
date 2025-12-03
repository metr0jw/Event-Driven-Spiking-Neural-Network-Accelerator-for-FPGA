#------------------------------------------------------------------------------
# Standalone Build Script for SNN Accelerator
# Creates project from scratch and generates bitstream
# Target: PYNQ-Z2 (xc7z020clg400-1)
#------------------------------------------------------------------------------

set script_dir [file dirname [file normalize [info script]]]
set proj_root [file normalize "$script_dir/../.."]
set part_name "xc7z020clg400-1"
set output_dir "$proj_root/build/impl"
set proj_name "snn_accel"

# Create output directory
file mkdir $output_dir

puts "=============================================="
puts "SNN Accelerator - Full Build"
puts "=============================================="
puts "Part: $part_name"
puts "Output: $output_dir"
puts ""

#------------------------------------------------------------------------------
# Step 1: Create In-Memory Project
#------------------------------------------------------------------------------
puts "Step 1: Creating project..."
create_project -in_memory -part $part_name

#------------------------------------------------------------------------------
# Step 2: Add Source Files
#------------------------------------------------------------------------------
puts "Step 2: Adding source files..."
set hdl_dir "$proj_root/hardware/hdl"

# Read all RTL files
read_verilog [glob -nocomplain $hdl_dir/rtl/common/*.v]
read_verilog [glob -nocomplain $hdl_dir/rtl/neurons/*.v]
read_verilog [glob -nocomplain $hdl_dir/rtl/synapses/*.v]
read_verilog [glob -nocomplain $hdl_dir/rtl/router/*.v]
read_verilog [glob -nocomplain $hdl_dir/rtl/layers/*.v]
read_verilog [glob -nocomplain $hdl_dir/rtl/interfaces/*.v]
read_verilog [glob -nocomplain $hdl_dir/rtl/top/*.v]
read_verilog [glob -nocomplain $proj_root/hardware/ip_repo/*/src/*.v]

# Set top module
set_property top snn_accelerator_top [current_fileset]

#------------------------------------------------------------------------------
# Step 3: Add Timing Constraints
#------------------------------------------------------------------------------
puts "Step 3: Adding constraints..."

# Create timing constraints XDC
set xdc_content {
# Timing Constraints for SNN Accelerator
# Target: PYNQ-Z2 @ 100MHz

# Primary clock from AXI interface
create_clock -period 10.000 -name aclk [get_ports aclk]

# Input delays (relative to clock)
set_input_delay -clock aclk -max 3.0 [get_ports -filter {DIRECTION == IN && NAME !~ "aclk"}]
set_input_delay -clock aclk -min 0.5 [get_ports -filter {DIRECTION == IN && NAME !~ "aclk"}]

# Output delays
set_output_delay -clock aclk -max 3.0 [get_ports -filter {DIRECTION == OUT}]
set_output_delay -clock aclk -min 0.5 [get_ports -filter {DIRECTION == OUT}]

# False paths for async reset
set_false_path -from [get_ports aresetn]

# Multicycle paths (if needed for large accumulators)
# set_multicycle_path 2 -setup -from [get_cells */membrane_potential_reg*] -to [get_cells */membrane_potential_reg*]
}

set xdc_file "$output_dir/timing.xdc"
set fp [open $xdc_file w]
puts $fp $xdc_content
close $fp

read_xdc $xdc_file

#------------------------------------------------------------------------------
# Step 4: Run Synthesis
#------------------------------------------------------------------------------
puts ""
puts "Step 4: Running synthesis..."
puts "=============================================="

synth_design -top snn_accelerator_top -part $part_name -flatten_hierarchy rebuilt -retiming

# Generate post-synth reports
puts "Generating synthesis reports..."
report_utilization -file $output_dir/post_synth_util.rpt
report_timing_summary -file $output_dir/post_synth_timing.rpt

# Save checkpoint
write_checkpoint -force $output_dir/post_synth.dcp

puts "Synthesis completed."
puts ""

#------------------------------------------------------------------------------
# Step 5: Run Optimization
#------------------------------------------------------------------------------
puts "Step 5: Running optimization..."
puts "=============================================="

opt_design -directive Explore

write_checkpoint -force $output_dir/post_opt.dcp

puts "Optimization completed."
puts ""

#------------------------------------------------------------------------------
# Step 6: Run Placement
#------------------------------------------------------------------------------
puts "Step 6: Running placement..."
puts "=============================================="

place_design -directive Explore

# Physical optimization after placement
phys_opt_design -directive AggressiveExplore

write_checkpoint -force $output_dir/post_place.dcp
report_utilization -file $output_dir/post_place_util.rpt
report_timing_summary -file $output_dir/post_place_timing.rpt

puts "Placement completed."
puts ""

#------------------------------------------------------------------------------
# Step 7: Run Routing
#------------------------------------------------------------------------------
puts "Step 7: Running routing..."
puts "=============================================="

route_design -directive Explore

# Post-route optimization
phys_opt_design -directive Explore

write_checkpoint -force $output_dir/post_route.dcp

puts "Routing completed."
puts ""

#------------------------------------------------------------------------------
# Step 8: Generate Reports
#------------------------------------------------------------------------------
puts "Step 8: Generating final reports..."
puts "=============================================="

report_utilization -file $output_dir/final_util.rpt -hierarchical
report_timing_summary -file $output_dir/final_timing.rpt -max_paths 20
report_power -file $output_dir/final_power.rpt
report_drc -file $output_dir/final_drc.rpt
report_methodology -file $output_dir/final_methodology.rpt

#------------------------------------------------------------------------------
# Step 9: Generate Bitstream
#------------------------------------------------------------------------------
puts "Step 9: Generating bitstream..."
puts "=============================================="

write_bitstream -force $output_dir/snn_accelerator.bit

# Generate hardware handoff for PYNQ
write_hw_platform -fixed -include_bit -force $output_dir/snn_accelerator.xsa

puts ""
puts "=============================================="
puts "BUILD COMPLETE!"
puts "=============================================="
puts ""
puts "Output files:"
puts "  Bitstream:  $output_dir/snn_accelerator.bit"
puts "  HW Platform: $output_dir/snn_accelerator.xsa"
puts ""
puts "Resource utilization:"
report_utilization -hierarchical -hierarchical_depth 2
puts ""

# Print timing summary
puts "Timing Summary:"
puts "=============================================="
set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -hold]]
puts "  WNS (Worst Negative Slack): $wns ns"
puts "  WHS (Worst Hold Slack): $whs ns"
if {$wns >= 0} {
    puts "  Timing: MET"
} else {
    puts "  Timing: NOT MET - Need optimization"
}
puts "=============================================="

exit
