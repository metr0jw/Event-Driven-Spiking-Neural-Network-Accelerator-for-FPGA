##-----------------------------------------------------------------------------
## Title         : HLS Project Creation Script
## Project       : PYNQ-Z2 SNN Accelerator
## File          : create_project.tcl
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Creates and configures all HLS projects
##-----------------------------------------------------------------------------

# Set project parameters
set PROJ_ROOT [pwd]/..
set SRC_DIR $PROJ_ROOT/src
set INC_DIR $PROJ_ROOT/include
set TEST_DIR $PROJ_ROOT/test
set SCRIPT_DIR $PROJ_ROOT/script

# Target device - PYNQ-Z2 uses Zynq-7020
set PART_NAME "xc7z020clg400-1"
set CLOCK_PERIOD 10
set CLOCK_UNCERTAINTY 1.25

# Create individual projects
proc create_hls_project {proj_name top_function src_files tb_files} {
    global PROJ_ROOT SRC_DIR INC_DIR TEST_DIR PART_NAME CLOCK_PERIOD CLOCK_UNCERTAINTY
    
    puts "=========================================="
    puts "Creating project: $proj_name"
    puts "=========================================="
    
    # Open/create project
    open_project -reset $proj_name
    
    # Add source files
    foreach src $src_files {
        add_files $SRC_DIR/$src -cflags "-I$INC_DIR"
    }
    
    # Add testbench files
    foreach tb $tb_files {
        add_files -tb $TEST_DIR/$tb -cflags "-I$INC_DIR -I$TEST_DIR"
    }
    
    # Set top function
    set_top $top_function
    
    # Create solution
    open_solution -reset "solution1"
    
    # Set part
    set_part $PART_NAME
    
    # Set clock
    create_clock -period ${CLOCK_PERIOD}ns -name default
    set_clock_uncertainty $CLOCK_UNCERTAINTY
    
    # Save project
    close_project
    
    puts "Project $proj_name created successfully\n"
}

# Create Learning Engine project
create_hls_project \
    "snn_learning_engine_prj" \
    "snn_learning_engine" \
    {snn_learning_engine.cpp} \
    {tb_snn_learning_engine.cpp test_utils.h}

# Create Spike Encoder project
create_hls_project \
    "spike_encoder_prj" \
    "spike_encoder" \
    {spike_encoder.cpp} \
    {tb_spike_encoder.cpp test_utils.h}

# Create Weight Updater project
create_hls_project \
    "weight_updater_prj" \
    "weight_updater" \
    {weight_updater.cpp} \
    {tb_weight_updater.cpp test_utils.h}

# Create Network Controller project
create_hls_project \
    "network_controller_prj" \
    "network_controller" \
    {network_controller.cpp} \
    {test_utils.h}

puts "All HLS projects created successfully!"
