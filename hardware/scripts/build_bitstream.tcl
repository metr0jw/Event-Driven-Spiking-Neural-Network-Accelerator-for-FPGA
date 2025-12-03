##-----------------------------------------------------------------------------
## Title         : Bitstream Generation Script
## Project       : PYNQ-Z2 SNN Accelerator
## File          : build_bitstream.tcl
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Automated synthesis, implementation, and bitstream generation
##-----------------------------------------------------------------------------

# Script options
set run_synthesis 1
set run_implementation 1
set run_bitstream 1
set use_incremental 0
set generate_reports 1

# Get script directory
set script_dir [file dirname [file normalize [info script]]]
set proj_root [file normalize "$script_dir/../.."]
set proj_name "snn_accelerator_pynq"
set proj_dir "$proj_root/build/vivado"

# Open project if not already open
if {[catch {current_project}]} {
    open_project "$proj_dir/$proj_name.xpr"
}

# Set number of jobs for parallel processing
set num_jobs 4

# Update compile order
update_compile_order -fileset sources_1

# Reset runs if requested
if {[info exists reset_runs] && $reset_runs == 1} {
    reset_run synth_1
    reset_run impl_1
}

# Synthesis
if {$run_synthesis} {
    puts "----------------------------------------"
    puts "Starting Synthesis..."
    puts "----------------------------------------"
    
    # Launch synthesis
    launch_runs synth_1 -jobs $num_jobs
    wait_on_run synth_1
    
    # Check synthesis results
    set synth_status [get_property STATUS [get_runs synth_1]]
    set synth_progress [get_property PROGRESS [get_runs synth_1]]
    
    if {$synth_status != "synth_design Complete!"} {
        puts "ERROR: Synthesis failed with status: $synth_status"
        exit 1
    }
    
    puts "Synthesis completed successfully"
    
    # Open synthesized design for analysis
    open_run synth_1 -name synth_1
    
    # Generate synthesis reports
    if {$generate_reports} {
        set report_dir "$proj_dir/reports/synthesis"
        file mkdir $report_dir
        
        report_utilization -file "$report_dir/post_synth_utilization.rpt" -hierarchical
        report_timing_summary -file "$report_dir/post_synth_timing_summary.rpt" -max_paths 10
        report_power -file "$report_dir/post_synth_power.rpt"
        report_drc -file "$report_dir/post_synth_drc.rpt"
    }
}

# Implementation
if {$run_implementation} {
    puts "----------------------------------------"
    puts "Starting Implementation..."
    puts "----------------------------------------"
    
    # Set implementation strategies for better timing closure
    set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
    
    # Enable incremental implementation if requested
    if {$use_incremental && [file exists "$proj_dir/checkpoint/post_route.dcp"]} {
        set_property incremental_checkpoint "$proj_dir/checkpoint/post_route.dcp" [get_runs impl_1]
    }
    
    # Launch implementation
    launch_runs impl_1 -jobs $num_jobs
    wait_on_run impl_1
    
    # Check implementation results
    set impl_status [get_property STATUS [get_runs impl_1]]
    set impl_progress [get_property PROGRESS [get_runs impl_1]]
    
    if {$impl_status != "route_design Complete!"} {
        puts "ERROR: Implementation failed with status: $impl_status"
        exit 1
    }
    
    puts "Implementation completed successfully"
    
    # Open implemented design
    open_run impl_1
    
    # Save checkpoint for incremental flow
    if {$use_incremental} {
        file mkdir "$proj_dir/checkpoint"
        write_checkpoint -force "$proj_dir/checkpoint/post_route.dcp"
    }
    
    # Generate implementation reports
    if {$generate_reports} {
        set report_dir "$proj_dir/reports/implementation"
        file mkdir $report_dir
        
        report_utilization -file "$report_dir/post_impl_utilization.rpt" -hierarchical
        report_timing_summary -file "$report_dir/post_impl_timing_summary.rpt" -max_paths 10 -warn_on_violation
        report_power -file "$report_dir/post_impl_power.rpt"
        report_drc -file "$report_dir/post_impl_drc.rpt"
        report_methodology -file "$report_dir/post_impl_methodology.rpt"
        report_io -file "$report_dir/post_impl_io.rpt"
        
        # Check timing
        set timing_met [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
        if {$timing_met < 0} {
            puts "WARNING: Timing not met! Worst negative slack: $timing_met"
        } else {
            puts "Timing requirements met. Worst slack: $timing_met"
        }
    }
}

# Bitstream generation
if {$run_bitstream} {
    puts "----------------------------------------"
    puts "Starting Bitstream Generation..."
    puts "----------------------------------------"
    
    # Launch bitstream generation
    launch_runs impl_1 -to_step write_bitstream -jobs $num_jobs
    wait_on_run impl_1
    
    # Check if bitstream was generated
    set bit_file "$proj_dir/${proj_name}.runs/impl_1/system_wrapper.bit"
    if {![file exists $bit_file]} {
        puts "ERROR: Bitstream generation failed!"
        exit 1
    }
    
    puts "Bitstream generated successfully"
    
    # Copy bitstream to output directory
    set output_dir "$proj_root/build/bitstreams"
    file mkdir $output_dir
    
    set timestamp [clock format [clock seconds] -format "%Y%m%d_%H%M%S"]
    set output_bit "$output_dir/snn_accelerator_${timestamp}.bit"
    file copy -force $bit_file $output_bit
    
    # Also copy to a fixed name for easy access
    file copy -force $bit_file "$output_dir/snn_accelerator.bit"
    
    # Generate hardware handoff for PYNQ
    set hwh_file "$proj_dir/${proj_name}.runs/impl_1/system_wrapper.hwh"
    if {[file exists $hwh_file]} {
        file copy -force $hwh_file "$output_dir/snn_accelerator.hwh"
        file copy -force $hwh_file "$output_dir/snn_accelerator_${timestamp}.hwh"
    }
    
    # Export hardware for SDK/Vitis
    write_hw_platform -fixed -include_bit -force -file "$output_dir/snn_accelerator.xsa"
    
    puts "----------------------------------------"
    puts "Build Summary:"
    puts "  Bitstream: $output_bit"
    puts "  Hardware handoff: $output_dir/snn_accelerator.hwh"
    puts "  XSA file: $output_dir/snn_accelerator.xsa"
    puts "----------------------------------------"
}

# Final summary
puts "\nBuild completed successfully!"
puts "Output files are in: $output_dir"
