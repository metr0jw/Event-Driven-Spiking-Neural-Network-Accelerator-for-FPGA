# Create Vivado project for SNN Accelerator
create_project snn_accelerator ./snn_accelerator -part xc7z020clg400-1

# Add Verilog RTL sources - Core modules
add_files {
    hardware/hdl/rtl/top/snn_accelerator_top.v
    hardware/hdl/rtl/neurons/lif_neuron_array.v
    hardware/hdl/rtl/neurons/lif_neuron.v
    hardware/hdl/rtl/router/spike_router.v
    hardware/hdl/rtl/synapses/synapse_array.v
    hardware/hdl/rtl/synapses/weight_memory.v
    hardware/hdl/rtl/interfaces/axi_wrapper.v
    hardware/hdl/rtl/common/fifo.v
    hardware/hdl/rtl/common/reset_sync.v
    hardware/hdl/rtl/common/sync_pulse.v
}

# Add AC-based (energy-efficient) modules - NO MAC, only AC operations!
# Energy savings: ~5x per operation + sparsity bonus
add_files {
    hardware/hdl/rtl/neurons/lif_neuron_ac.v
    hardware/hdl/rtl/synapses/synapse_array_ac.v
    hardware/hdl/rtl/layers/snn_conv1d_ac.v
    hardware/hdl/rtl/layers/snn_conv2d_ac.v
    hardware/hdl/rtl/layers/snn_fc_ac.v
    hardware/hdl/rtl/common/snn_energy_monitor.v
}

# Add all layer implementations (convolution, pooling, etc.)
add_files {
    hardware/hdl/rtl/layers/snn_conv1d.v
    hardware/hdl/rtl/layers/snn_conv2d.v
    hardware/hdl/rtl/layers/snn_avgpool1d.v
    hardware/hdl/rtl/layers/snn_avgpool2d.v
    hardware/hdl/rtl/layers/snn_maxpool1d.v
    hardware/hdl/rtl/layers/snn_maxpool2d.v
    hardware/hdl/rtl/layers/snn_pooling.v
    hardware/hdl/rtl/layers/snn_layer_manager.v
}

# Add constraint files
add_files -fileset constrs_1 {
    hardware/constraints/pynq_z2_pins.xdc
    hardware/constraints/timing.xdc
    hardware/constraints/bitstream.xdc
}

# Add HLS IP repository (after HLS synthesis)
set_property ip_repo_paths ./snn_accelerator_hls/solution1/impl/ip [current_project]
update_ip_catalog -rebuild

# Create block design for PYNQ-Z2
create_bd_design "snn_system"

# Add Zynq PS for PYNQ-Z2
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
endgroup

# Configure PS for PYNQ-Z2
set_property -dict [list CONFIG.PCW_PRESET_BANK1_VOLTAGE {LVCMOS18} \
                        CONFIG.PCW_PACKAGE_NAME {clg400} \
                        CONFIG.PCW_USE_S_AXI_HP0 {1} \
                        CONFIG.PCW_USE_S_AXI_HP1 {1} \
                        CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
                        CONFIG.PCW_EN_CLK0_PORT {1} \
                        CONFIG.PCW_EN_RST0_PORT {1}] [get_bd_cells processing_system7_0]

# Add HLS IP to block design (uncomment after HLS is run)
# create_bd_cell -type ip -vlnv xilinx.com:hls:network_controller:1.0 network_controller_0

# Add AXI Interconnect
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
endgroup

# Configure AXI Interconnect
set_property -dict [list CONFIG.NUM_MI {2}] [get_bd_cells axi_interconnect_0]

# Connect clocks and resets
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/S00_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/M00_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/M01_ACLK]

connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins axi_interconnect_0/ARESETN]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins axi_interconnect_0/S00_ARESETN]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins axi_interconnect_0/M00_ARESETN]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins axi_interconnect_0/M01_ARESETN]

# Connect AXI interfaces
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins axi_interconnect_0/S00_AXI]

# Generate addresses
assign_bd_address

# Validate design
validate_bd_design

# Generate wrapper
make_wrapper -files [get_files ./snn_accelerator/snn_accelerator.srcs/sources_1/bd/snn_system/snn_system.bd] -top
add_files -norecurse ./snn_accelerator/snn_accelerator.srcs/sources_1/bd/snn_system/hdl/snn_system_wrapper.v

# Set top module
set_property top snn_system_wrapper [current_fileset]

# Run synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed"
    exit 1
}

# Run implementation
launch_runs impl_1 -jobs 4
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Implementation failed"
    exit 1
}

# Generate bitstream
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Bitstream generation failed"
    exit 1
}

puts "SUCCESS: Bitstream generated successfully"
