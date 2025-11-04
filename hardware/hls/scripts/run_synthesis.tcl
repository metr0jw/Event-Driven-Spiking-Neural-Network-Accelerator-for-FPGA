##-----------------------------------------------------------------------------
## Title         : HLS Synthesis Script
## Project       : PYNQ-Z2 SNN Accelerator
## File          : run_synthesis.tcl
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Runs synthesis for all HLS modules with optimizations
##-----------------------------------------------------------------------------

# Function to run synthesis for a project
proc run_synthesis_for_project {proj_name directives} {
    puts "\n=========================================="
    puts "Running synthesis for: $proj_name"
    puts "=========================================="
    
    # Open project
    open_project $proj_name
    open_solution "solution1"
    
    # Apply optimization directives
    foreach directive $directives {
        puts "Applying directive: $directive"
        eval $directive
    }
    
    # Run C simulation
    puts "\nRunning C simulation..."
    csim_design -clean
    
    # Run synthesis
    puts "\nRunning synthesis..."
    csynth_design
    
    # Close project
    close_project
    
    # Parse and display results
    set report_file "$proj_name/solution1/syn/report/${proj_name}_csynth.rpt"
    if {[file exists $report_file]} {
        puts "\n----- Synthesis Results -----"
        set fp [open $report_file r]
        set file_data [read $fp]
        close $fp
        
        # Extract key metrics
        if {[regexp {Timing \(ns\):\s+\* Summary[\s\S]*?slack\s+(\S+)} $file_data -> slack]} {
            puts "Timing slack: $slack ns"
        }
        if {[regexp {Latency \(clock cycles\):\s+\* Summary[\s\S]*?max\s+(\d+)} $file_data -> latency]} {
            puts "Max latency: $latency cycles"
        }
        if {[regexp {== Utilization Estimates[\s\S]*?BRAM_18K\s+\|\s+(\d+)} $file_data -> bram]} {
            puts "BRAM usage: $bram"
        }
        if {[regexp {== Utilization Estimates[\s\S]*?DSP48E\s+\|\s+(\d+)} $file_data -> dsp]} {
            puts "DSP usage: $dsp"
        }
        if {[regexp {== Utilization Estimates[\s\S]*?FF\s+\|\s+(\d+)} $file_data -> ff]} {
            puts "FF usage: $ff"
        }
        if {[regexp {== Utilization Estimates[\s\S]*?LUT\s+\|\s+(\d+)} $file_data -> lut]} {
            puts "LUT usage: $lut"
        }
    }
    
    puts "\nSynthesis completed for $proj_name"
}

# Define optimization directives for each module

# Learning Engine directives
set learning_directives {
    "set_directive_interface -mode s_axilite snn_learning_engine"
    "set_directive_interface -mode axis -register -register_mode both snn_learning_engine pre_spikes"
    "set_directive_interface -mode axis -register -register_mode both snn_learning_engine post_spikes"
    "set_directive_interface -mode axis -register -register_mode both snn_learning_engine weight_updates"
    "set_directive_array_partition -type cyclic -factor 8 snn_learning_engine pre_spike_times"
    "set_directive_array_partition -type cyclic -factor 8 snn_learning_engine post_spike_times"
    "set_directive_pipeline -II 2 snn_learning_engine/LTD_LOOP"
    "set_directive_pipeline -II 2 snn_learning_engine/LTP_LOOP"
    "set_directive_dataflow snn_learning_engine"
}

# Spike Encoder directives
set encoder_directives {
    "set_directive_interface -mode s_axilite spike_encoder"
    "set_directive_interface -mode axis -register -register_mode both spike_encoder data_in"
    "set_directive_interface -mode axis -register -register_mode both spike_encoder spikes_out"
    "set_directive_array_partition -type cyclic -factor 16 spike_encoder phase_accumulator"
    "set_directive_unroll -factor 8 spike_encoder/ENCODE_LOOP"
    "set_directive_pipeline spike_encoder/ENCODE_LOOP"
    "set_directive_inline encode_rate"
    "set_directive_inline encode_temporal"
    "set_directive_inline encode_phase"
}

# Weight Updater directives
set weight_directives {
    "set_directive_interface -mode s_axilite weight_updater"
    "set_directive_interface -mode axis -register -register_mode both weight_updater updates_in"
    "set_directive_interface -mode m_axi -depth 65536 -offset slave weight_updater weight_memory"
    "set_directive_pipeline weight_updater"
    "set_directive_inline apply_decay"
}

# Spike Decoder directives
set decoder_directives {
    "set_directive_interface -mode s_axilite spike_decoder"
    "set_directive_interface -mode axis -register -register_mode both spike_decoder spikes_in"
    "set_directive_interface -mode axis -register -register_mode both spike_decoder data_out"
    "set_directive_array_partition -type cyclic -factor 8 spike_decoder spike_counts"
    "set_directive_array_partition -type cyclic -factor 8 spike_decoder spike_rates"
    "set_directive_pipeline spike_decoder/RESET_LOOP"
    "set_directive_unroll -factor 8 spike_decoder/RESET_LOOP"
    "set_directive_inline decode_spike_count"
    "set_directive_inline decode_spike_rate"
}

# Network Controller directives
set controller_directives {
    "set_directive_interface -mode s_axilite network_controller"
    "set_directive_interface -mode axis -register -register_mode both network_controller input_data"
    "set_directive_interface -mode axis -register -register_mode both network_controller output_data"
    "set_directive_interface -mode axis -register -register_mode both network_controller encoder_ctrl"
    "set_directive_interface -mode axis -register -register_mode both network_controller learning_ctrl"
    "set_directive_interface -mode axis -register -register_mode both network_controller decoder_ctrl"
    "set_directive_inline send_control_packet"
}

# Run synthesis for all projects
run_synthesis_for_project "snn_learning_engine_prj" $learning_directives
run_synthesis_for_project "spike_encoder_prj" $encoder_directives
run_synthesis_for_project "weight_updater_prj" $weight_directives
run_synthesis_for_project "spike_decoder_prj" $decoder_directives
run_synthesis_for_project "network_controller_prj" $controller_directives

puts "\n=========================================="
puts "All synthesis runs completed!"
puts "=========================================="
