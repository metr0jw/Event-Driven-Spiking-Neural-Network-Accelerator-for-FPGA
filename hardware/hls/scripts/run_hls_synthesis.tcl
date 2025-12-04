#-----------------------------------------------------------------------------
# Vitis HLS Synthesis Script for AXI HLS Wrapper
# Target: PYNQ-Z2 (xc7z020clg400-1)
#-----------------------------------------------------------------------------

# Set project parameters
set PROJECT_NAME "axi_hls_wrapper"
set TOP_FUNCTION "axi_hls_wrapper"
set PART "xc7z020clg400-1"
set CLOCK_PERIOD 10.0
set CLOCK_UNCERTAINTY 1.25

# Get script directory
set SCRIPT_DIR [file dirname [info script]]
set HLS_DIR [file normalize "$SCRIPT_DIR/.."]
set OUTPUT_DIR "$HLS_DIR/output"

# Clean and create output directory
file delete -force $OUTPUT_DIR/$PROJECT_NAME
file mkdir $OUTPUT_DIR

# Create HLS project
open_project -reset $OUTPUT_DIR/$PROJECT_NAME

# Set top-level function
set_top $TOP_FUNCTION

# Add source files
add_files $HLS_DIR/src/axi_hls_wrapper.cpp -cflags "-I$HLS_DIR/include"
add_files $HLS_DIR/include/axi_hls_wrapper.h
add_files $HLS_DIR/include/axi_interfaces.h
add_files $HLS_DIR/include/snn_types.h

# Add testbench (optional)
# add_files -tb $HLS_DIR/test/tb_axi_hls_wrapper.cpp -cflags "-I$HLS_DIR/include"

# Create solution
open_solution -reset "solution1" -flow_target vivado

# Set target device
set_part $PART

# Set clock
create_clock -period $CLOCK_PERIOD -name default
set_clock_uncertainty $CLOCK_UNCERTAINTY

# Set directives for better QoR
config_interface -m_axi_latency 0
config_compile -pipeline_loops 0

# Run synthesis
puts "========================================="
puts "Starting C Synthesis..."
puts "========================================="
csynth_design

# Report results
puts "========================================="
puts "C Synthesis Complete"
puts "========================================="

# Export RTL as IP
puts "========================================="
puts "Exporting IP..."
puts "========================================="
export_design -rtl verilog -format ip_catalog \
    -vendor "kw.ac.kr" \
    -library "snn" \
    -ipname "axi_hls_wrapper" \
    -version "1.0" \
    -description "HLS-based AXI4-Lite/Stream wrapper for SNN Accelerator" \
    -display_name "AXI HLS Wrapper" \
    -taxonomy "/SNN"

puts "========================================="
puts "IP Export Complete!"
puts "IP Location: $OUTPUT_DIR/$PROJECT_NAME/solution1/impl/ip"
puts "========================================="

# Close project
close_project

exit
