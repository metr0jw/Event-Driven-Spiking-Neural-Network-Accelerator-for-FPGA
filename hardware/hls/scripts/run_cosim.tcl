##-----------------------------------------------------------------------------
## Title         : HLS Co-simulation Script
## Project       : PYNQ-Z2 SNN Accelerator
## File          : run_cosim.tcl
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Runs C/RTL co-simulation for verification
##-----------------------------------------------------------------------------

# Function to run co-simulation
proc run_cosim_for_project {proj_name options} {
    puts "\n=========================================="
    puts "Running co-simulation for: $proj_name"
    puts "=========================================="
    
    # Open project
    open_project $proj_name
    open_solution "solution1"
    
    # Run co-simulation with specified options
    puts "Co-simulation options: $options"
    eval cosim_design $options
    
    # Parse results
    set report_file "$proj_name/solution1/sim/report/${proj_name}_cosim.rpt"
    if {[file exists $report_file]} {
        puts "\n----- Co-simulation Results -----"
        set fp [open $report_file r]
        set file_data [read $fp]
        close $fp
        
        # Check for pass/fail
        if {[regexp {Pass} $file_data]} {
            puts "Co-simulation: PASSED"
        } else {
            puts "Co-simulation: FAILED"
        }
        
        # Extract performance metrics
        if {[regexp {Average II\s*:\s*(\d+)} $file_data -> ii]} {
            puts "Average II: $ii"
        }
        if {[regexp {Latency\s*:\s*(\d+)} $file_data -> latency]} {
            puts "Latency: $latency cycles"
        }
    }
    
    close_project
}

# Co-simulation options for each module
# Format: -O (optimize compile) -ldflags "-lm" (link math library)

# Basic options
set basic_opts "-O -ldflags {-lm}"

# With waveform dump (slower but useful for debugging)
set wave_opts "-O -ldflags {-lm} -trace_level all -wave_debug"

# Run co-simulation for each project
puts "Note: Co-simulation may take several minutes per module..."

# Learning Engine - needs waveform for STDP verification
run_cosim_for_project "snn_learning_engine_prj" $wave_opts

# Spike Encoder - basic verification sufficient
run_cosim_for_project "spike_encoder_prj" $basic_opts

# Weight Updater - verify memory access patterns
run_cosim_for_project "weight_updater_prj" $wave_opts

# Network Controller - verify control flow
run_cosim_for_project "network_controller_prj" $basic_opts

puts "\n=========================================="
puts "All co-simulations completed!"
puts "Check individual reports for detailed results"
puts "=========================================="
