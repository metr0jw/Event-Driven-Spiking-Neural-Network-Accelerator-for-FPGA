# Create HLS project for SNN Accelerator
open_project snn_accelerator_hls
set_top network_controller

# Add source files
add_files hardware/hls/src/network_controller.cpp
add_files hardware/hls/src/snn_learning_engine.cpp
add_files hardware/hls/src/spike_encoder.cpp
add_files hardware/hls/src/spike_decoder.cpp
add_files hardware/hls/src/weight_updater.cpp
add_files hardware/hls/src/pc_interface.cpp

# Add header files
add_files hardware/hls/include/network_controller.h
add_files hardware/hls/include/snn_types.h
add_files hardware/hls/include/snn_config.h
add_files hardware/hls/include/snn_learning_engine.h
add_files hardware/hls/include/spike_encoder.h
add_files hardware/hls/include/spike_decoder.h
add_files hardware/hls/include/weight_updater.h
add_files hardware/hls/include/axi_interfaces.h
add_files hardware/hls/include/pc_interface.h
add_files hardware/hls/include/sh_utils.h

# Add testbench files
add_files -tb hardware/hls/test/tb_network_controller.cpp
add_files -tb hardware/hls/test/tb_snn_learning_engine.cpp
add_files -tb hardware/hls/test/tb_spike_encoder.cpp
add_files -tb hardware/hls/test/tb_spike_decoder.cpp
add_files -tb hardware/hls/test/tb_weight_updater.cpp
add_files -tb hardware/hls/test/test_utils.h

# Create solution for PYNQ-Z2 (Zynq-7020)
open_solution "solution1"
set_part {xc7z020clg400-1}
create_clock -period 10 -name default

# Set interface directives for AXI
set_directive_interface -mode s_axilite network_controller command
set_directive_interface -mode s_axilite network_controller config
set_directive_interface -mode axis -register network_controller input_data
set_directive_interface -mode axis -register network_controller output_data
set_directive_interface -mode s_axilite -bundle control network_controller return

# Run C simulation (optional)
# csim_design

# Run synthesis
csynth_design

# Export RTL as IP
export_design -format ip_catalog -description "SNN Network Controller IP" -version "1.0"
