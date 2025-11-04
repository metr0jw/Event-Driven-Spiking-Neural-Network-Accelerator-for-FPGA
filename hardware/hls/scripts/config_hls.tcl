##-----------------------------------------------------------------------------
## Title         : HLS Configuration Settings
## Project       : PYNQ-Z2 SNN Accelerator
## File          : config_hls.tcl
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Common configuration settings for all HLS projects
##-----------------------------------------------------------------------------

# PYNQ-Z2 specific settings
set PYNQ_Z2_PART "xc7z020clg400-1"
set PYNQ_Z2_BOARD "tul.com.tw:pynq-z2:part0:1.0"

# Clock settings (100MHz system clock)
set SYSTEM_CLOCK_PERIOD 10
set CLOCK_UNCERTAINTY 1.25

# Resource constraints based on PYNQ-Z2
# Total resources: 53,200 LUTs, 106,400 FFs, 220 DSPs, 140 BRAMs
# Reserve ~50% for other logic
set MAX_LUT 26000
set MAX_FF 53000
set MAX_DSP 110
set MAX_BRAM 70

# Optimization settings
set DEFAULT_OPTIMIZATION_LEVEL "3"
set ENABLE_DATAFLOW 1
set PIPELINE_STYLE "flp"  # Flatten loop pipeline

# Interface settings
set USE_AXI_LITE_CONTROL 1
set USE_AXI_STREAM_DATA 1
set AXI_LITE_ADDR_WIDTH 32
set AXI_LITE_DATA_WIDTH 32
set AXI_STREAM_DATA_WIDTH 32

# Common directives for all projects
proc apply_common_directives {top_function} {
    # Set latency constraint
    set_directive_latency -min 1 -max 1000 $top_function
    
    # Set interface defaults
    if {$::USE_AXI_LITE_CONTROL} {
        set_directive_interface -mode s_axilite -register $top_function
    }
    
    # Resource allocation
    config_rtl -prefix "${top_function}_"
    
    # Vivado settings for implementation
    config_compile -name_max_length 80
    config_schedule -enable_dsp_full_reg
}

# Function to check resource usage
proc check_resource_usage {project_name} {
    set report_file "$project_name/solution1/syn/report/[set project_name]_csynth.rpt"
    
    if {[file exists $report_file]} {
        set fp [open $report_file r]
        set content [read $fp]
        close $fp
        
        # Extract resource usage
        regexp {LUT\s+\|\s+(\d+)} $content -> lut_usage
        regexp {FF\s+\|\s+(\d+)} $content -> ff_usage
        regexp {DSP\s+\|\s+(\d+)} $content -> dsp_usage
        regexp {BRAM\s+\|\s+(\d+)} $content -> bram_usage
        
        puts "Resource usage for $project_name:"
        puts "  LUT:  $lut_usage / $::MAX_LUT"
        puts "  FF:   $ff_usage / $::MAX_FF"
        puts "  DSP:  $dsp_usage / $::MAX_DSP"
        puts "  BRAM: $bram_usage / $::MAX_BRAM"
        
        # Check if within limits
        if {$lut_usage > $::MAX_LUT} {
            puts "WARNING: LUT usage exceeds limit!"
        }
        if {$dsp_usage > $::MAX_DSP} {
            puts "WARNING: DSP usage exceeds limit!"
        }
    }
}

puts "HLS configuration loaded for PYNQ-Z2"
