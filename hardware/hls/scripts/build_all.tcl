##-----------------------------------------------------------------------------
## Title         : Master Build Script for HLS
## Project       : PYNQ-Z2 SNN Accelerator
## File          : build_all.tcl
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Runs complete HLS flow for all modules
##-----------------------------------------------------------------------------

# Get command line arguments
set do_create 1
set do_synthesis 1
set do_cosim 0
set do_export 1
set clean_first 0

# Parse arguments
if { $::argc > 0 } {
    for {set i 0} {$i < $::argc} {incr i} {
        set arg [lindex $::argv $i]
        switch -glob -- $arg {
            "-no_create"    { set do_create 0 }
            "-no_synthesis" { set do_synthesis 0 }
            "-cosim"        { set do_cosim 1 }
            "-no_export"    { set do_export 0 }
            "-clean"        { set clean_first 1 }
            "-help"         {
                puts "Usage: vitis_hls -f build_all.tcl \[options\]"
                puts "Options:"
                puts "  -no_create     Skip project creation"
                puts "  -no_synthesis  Skip synthesis"
                puts "  -cosim         Run co-simulation (slow)"
                puts "  -no_export     Skip IP export"
                puts "  -clean         Clean before build"
                puts "  -help          Show this message"
                exit 0
            }
        }
    }
}

# Start timer
set start_time [clock seconds]

puts "\n======================================================"
puts "PYNQ-Z2 SNN Accelerator - HLS Build Script"
puts "======================================================"
puts "Configuration:"
puts "  Create projects: $do_create"
puts "  Run synthesis:   $do_synthesis"
puts "  Run co-sim:      $do_cosim"
puts "  Export IP:       $do_export"
puts "  Clean first:     $clean_first"
puts ""

# Clean if requested
if {$clean_first} {
    puts "Cleaning previous builds..."
    set projects {
        "snn_learning_engine_prj"
        "spike_encoder_prj"
        "weight_updater_prj"
        "spike_decoder_prj"
        "network_controller_prj"
    }
    
    foreach proj $projects {
        if {[file exists $proj]} {
            puts "  Removing $proj"
            file delete -force $proj
        }
    }
    
    if {[file exists "../../ip/custom_ip"]} {
        puts "  Removing IP repository"
        file delete -force "../../ip/custom_ip"
    }
}

# Step 1: Create projects
if {$do_create} {
    puts "\n----- Step 1: Creating HLS Projects -----"
    source create_project.tcl
} else {
    puts "\n----- Step 1: Skipping project creation -----"
}

# Step 2: Run synthesis
if {$do_synthesis} {
    puts "\n----- Step 2: Running Synthesis -----"
    source run_synthesis.tcl
} else {
    puts "\n----- Step 2: Skipping synthesis -----"
}

# Step 3: Run co-simulation (optional)
if {$do_cosim} {
    puts "\n----- Step 3: Running Co-simulation -----"
    puts "Warning: This may take 30+ minutes"
    source run_cosim.tcl
} else {
    puts "\n----- Step 3: Skipping co-simulation -----"
}

# Step 4: Export IP cores
if {$do_export} {
    puts "\n----- Step 4: Exporting IP Cores -----"
    source export_ip.tcl
} else {
    puts "\n----- Step 4: Skipping IP export -----"
}

# Calculate elapsed time
set end_time [clock seconds]
set elapsed [expr $end_time - $start_time]
set minutes [expr $elapsed / 60]
set seconds [expr $elapsed % 60]

puts "\n======================================================"
puts "Build completed in $minutes minutes $seconds seconds"
puts "======================================================"

# Generate summary report
set report_file "build_summary.txt"
set fp [open $report_file w]
puts $fp "PYNQ-Z2 SNN Accelerator HLS Build Summary"
puts $fp "=========================================="
puts $fp "Build date: [clock format [clock seconds]]"
puts $fp "Build time: $minutes minutes $seconds seconds"
puts $fp ""

# List generated IP cores
if {$do_export && [file exists "../../ip/custom_ip"]} {
    puts $fp "Generated IP cores:"
    set ip_dirs [glob -nocomplain -directory "../../ip/custom_ip" -type d *]
    foreach ip_dir $ip_dirs {
        puts $fp "  - [file tail $ip_dir]"
    }
}

close $fp

puts "\nBuild summary written to: $report_file"
puts "\nNext steps:"
puts "1. Open Vivado"
puts "2. Add IP repository: [file normalize ../../ip/custom_ip]"
puts "3. Create block design with HLS IP cores"
puts "4. Generate bitstream for PYNQ-Z2"
