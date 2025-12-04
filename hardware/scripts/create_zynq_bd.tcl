#-----------------------------------------------------------------------------
# Title         : Create Zynq Block Design for PYNQ-Z2
# Project       : PYNQ-Z2 SNN Accelerator
# File          : create_zynq_bd.tcl
# Author        : Jiwoon Lee (@metr0jw)
# Description   : Creates a block design with Zynq PS and SNN accelerator
#-----------------------------------------------------------------------------

# Set project directory
set proj_dir [pwd]
set proj_name "Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA"

# Open existing project
open_project hardware/vivado/${proj_name}.xpr

# Set IP repository paths
set_property ip_repo_paths [list \
    "$proj_dir/hardware/ip_repo/axi_1_0" \
    "$proj_dir/hardware/ip_repo/axi_lite_regs_v1_0" \
] [current_project]
update_ip_catalog -rebuild

# Delete existing block design if exists
set bd_name "snn_zynq_system"
if {[get_bd_designs -quiet $bd_name] != ""} {
    close_bd_design [get_bd_designs $bd_name]
    delete_files -quiet [get_files ${bd_name}.bd]
}

# Create new block design
create_bd_design $bd_name

# Create Zynq Processing System
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0

# Configure Zynq PS for PYNQ-Z2
set_property -dict [list \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_USE_S_AXI_GP0 {1} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_IRQ_F2P_INTR {1} \
    CONFIG.PCW_PRESET_BANK0_VOLTAGE {LVCMOS 3.3V} \
    CONFIG.PCW_PRESET_BANK1_VOLTAGE {LVCMOS 1.8V} \
    CONFIG.PCW_QSPI_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_QSPI_GRP_SINGLE_SS_ENABLE {1} \
    CONFIG.PCW_SD0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_UART0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_USB0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_ENET0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_ENET0_ENET0_IO {MIO 16 .. 27} \
    CONFIG.PCW_ENET0_GRP_MDIO_ENABLE {1} \
    CONFIG.PCW_TTC0_PERIPHERAL_ENABLE {1} \
] [get_bd_cells processing_system7_0]

# Create AXI Interconnect
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
set_property -dict [list CONFIG.NUM_MI {1}] [get_bd_cells axi_interconnect_0]

# Create Processor System Reset
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0

# Add SNN Accelerator as RTL module (without external AXI ports)
# Create a wrapper for the SNN core
create_bd_cell -type module -reference snn_accelerator_top snn_accelerator_0

# Create AXI GPIO for LEDs and buttons (optional board I/O)
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0
set_property -dict [list \
    CONFIG.C_GPIO_WIDTH {4} \
    CONFIG.C_GPIO2_WIDTH {4} \
    CONFIG.C_IS_DUAL {1} \
    CONFIG.C_ALL_OUTPUTS {1} \
    CONFIG.C_ALL2_INPUTS {1} \
] [get_bd_cells axi_gpio_0]

# Create clock and reset connections
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins axi_interconnect_0/ACLK] \
    [get_bd_pins axi_interconnect_0/S00_ACLK] \
    [get_bd_pins axi_interconnect_0/M00_ACLK] \
    [get_bd_pins snn_accelerator_0/aclk] \
    [get_bd_pins proc_sys_reset_0/slowest_sync_clk] \
    [get_bd_pins axi_gpio_0/s_axi_aclk]

connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] \
    [get_bd_pins proc_sys_reset_0/ext_reset_in]

connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
    [get_bd_pins axi_interconnect_0/ARESETN] \
    [get_bd_pins axi_interconnect_0/S00_ARESETN] \
    [get_bd_pins axi_interconnect_0/M00_ARESETN] \
    [get_bd_pins snn_accelerator_0/aresetn] \
    [get_bd_pins axi_gpio_0/s_axi_aresetn]

# Connect AXI interfaces
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] \
    [get_bd_intf_pins axi_interconnect_0/S00_AXI]

connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M00_AXI] \
    [get_bd_intf_pins snn_accelerator_0/S_AXI]

# Connect interrupt
connect_bd_net [get_bd_pins snn_accelerator_0/interrupt] \
    [get_bd_pins processing_system7_0/IRQ_F2P]

# Create external ports for LEDs
create_bd_port -dir O -from 3 -to 0 led
connect_bd_net [get_bd_pins snn_accelerator_0/led] [get_bd_ports led]

# Create external ports for RGB LEDs
create_bd_port -dir O led4_r
create_bd_port -dir O led4_g
create_bd_port -dir O led4_b
create_bd_port -dir O led5_r
create_bd_port -dir O led5_g
create_bd_port -dir O led5_b
connect_bd_net [get_bd_pins snn_accelerator_0/led4_r] [get_bd_ports led4_r]
connect_bd_net [get_bd_pins snn_accelerator_0/led4_g] [get_bd_ports led4_g]
connect_bd_net [get_bd_pins snn_accelerator_0/led4_b] [get_bd_ports led4_b]
connect_bd_net [get_bd_pins snn_accelerator_0/led5_r] [get_bd_ports led5_r]
connect_bd_net [get_bd_pins snn_accelerator_0/led5_g] [get_bd_ports led5_g]
connect_bd_net [get_bd_pins snn_accelerator_0/led5_b] [get_bd_ports led5_b]

# Assign addresses
assign_bd_address

# Validate design
validate_bd_design

# Save and generate output products
save_bd_design
generate_target all [get_files ${bd_name}.bd]

# Create HDL wrapper
make_wrapper -files [get_files ${bd_name}.bd] -top
add_files -norecurse [get_files ${bd_name}_wrapper.v]

# Set wrapper as top
set_property top ${bd_name}_wrapper [current_fileset]

# Update compile order
update_compile_order -fileset sources_1

puts "Block design created successfully!"
puts "Next step: Run synthesis, implementation, and bitstream generation"

close_project
