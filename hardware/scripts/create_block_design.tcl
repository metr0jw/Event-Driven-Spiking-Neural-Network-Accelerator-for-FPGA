##-----------------------------------------------------------------------------
## Title         : Block Design Creation Script
## Project       : PYNQ-Z2 SNN Accelerator
## File          : create_block_design.tcl
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Creates Zynq block design for PYNQ-Z2
##-----------------------------------------------------------------------------

# Get script directory
set script_dir [file dirname [info script]]

# Create block design
create_bd_design "system"

# Add Zynq PS
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0

# Apply board automation first
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "0" Master "Disable" Slave "Disable" } [get_bd_cells processing_system7_0]

# Load PYNQ-Z2 specific configuration from preset file
source "$script_dir/pynq_z2_ps_config.tcl"

# Apply the PYNQ-Z2 preset configuration
set ps_config [apply_preset processing_system7_0]
set_property -dict $ps_config [get_bd_cells processing_system7_0]

# Configure PS for accelerator-specific settings
set_property -dict [list \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_M_AXI_GP1 {0} \
    CONFIG.PCW_USE_S_AXI_GP0 {0} \
    CONFIG.PCW_USE_S_AXI_GP1 {0} \
    CONFIG.PCW_USE_S_AXI_ACP {0} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP1 {0} \
    CONFIG.PCW_USE_S_AXI_HP2 {0} \
    CONFIG.PCW_USE_S_AXI_HP3 {0} \
    CONFIG.PCW_M_AXI_GP0_ENABLE_STATIC_REMAP {0} \
    CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_IRQ_F2P_INTR {1} \
] [get_bd_cells processing_system7_0]

# Add AXI Interconnect
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] [get_bd_cells axi_interconnect_0]

# Add AXI DMA for high-speed data transfer
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_m_axi_mm2s_data_width {64} \
    CONFIG.c_m_axis_mm2s_tdata_width {32} \
    CONFIG.c_mm2s_burst_size {256} \
    CONFIG.c_m_axi_s2mm_data_width {64} \
    CONFIG.c_s_axis_s2mm_tdata_width {32} \
    CONFIG.c_s2mm_burst_size {256} \
] [get_bd_cells axi_dma_0]

# Add AXI BRAM Controller for weight storage
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 axi_bram_ctrl_0
set_property -dict [list CONFIG.SINGLE_PORT_BRAM {1}] [get_bd_cells axi_bram_ctrl_0]

# Add Block Memory Generator
create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 blk_mem_gen_0
set_property -dict [list \
    CONFIG.Memory_Type {True_Dual_Port_RAM} \
    CONFIG.Use_Byte_Write_Enable {true} \
    CONFIG.Byte_Size {8} \
    CONFIG.Write_Width_A {32} \
    CONFIG.Write_Depth_A {16384} \
    CONFIG.Read_Width_A {32} \
    CONFIG.Write_Width_B {32} \
    CONFIG.Read_Width_B {32} \
    CONFIG.Register_PortA_Output_of_Memory_Primitives {false} \
    CONFIG.Register_PortB_Output_of_Memory_Primitives {false} \
] [get_bd_cells blk_mem_gen_0]

# Connect AXI interfaces
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {/processing_system7_0/FCLK_CLK0 (100 MHz)} \
    Clk_slave {Auto} \
    Clk_xbar {/processing_system7_0/FCLK_CLK0 (100 MHz)} \
    Master {/processing_system7_0/M_AXI_GP0} \
    Slave {/axi_dma_0/S_AXI_LITE} \
    intc_ip {/axi_interconnect_0} \
    master_apm {0} \
} [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {/processing_system7_0/FCLK_CLK0 (100 MHz)} \
    Clk_slave {Auto} \
    Clk_xbar {/processing_system7_0/FCLK_CLK0 (100 MHz)} \
    Master {/processing_system7_0/M_AXI_GP0} \
    Slave {/axi_bram_ctrl_0/S_AXI} \
    intc_ip {/axi_interconnect_0} \
    master_apm {0} \
} [get_bd_intf_pins axi_bram_ctrl_0/S_AXI]

# Connect DMA to HP port
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {/processing_system7_0/FCLK_CLK0 (100 MHz)} \
    Clk_slave {/processing_system7_0/FCLK_CLK0 (100 MHz)} \
    Clk_xbar {/processing_system7_0/FCLK_CLK0 (100 MHz)} \
    Master {/axi_dma_0/M_AXI_MM2S} \
    Slave {/processing_system7_0/S_AXI_HP0} \
    intc_ip {Auto} \
    master_apm {0} \
} [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {/processing_system7_0/FCLK_CLK0 (100 MHz)} \
    Clk_slave {/processing_system7_0/FCLK_CLK0 (100 MHz)} \
    Clk_xbar {/processing_system7_0/FCLK_CLK0 (100 MHz)} \
    Master {/axi_dma_0/M_AXI_S2MM} \
    Slave {/processing_system7_0/S_AXI_HP0} \
    intc_ip {/axi_smc} \
    master_apm {0} \
} [get_bd_intf_pins axi_dma_0/M_AXI_S2MM]

# Connect BRAM
connect_bd_intf_net [get_bd_intf_pins axi_bram_ctrl_0/BRAM_PORTA] [get_bd_intf_pins blk_mem_gen_0/BRAM_PORTA]

# Create DMA loopback for testing (no external ports to avoid DRC errors)
# Connect MM2S directly to S2MM for loopback test
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

# Add interrupt concat
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0
set_property -dict [list CONFIG.NUM_PORTS {2}] [get_bd_cells xlconcat_0]

# Connect interrupts
connect_bd_net [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins xlconcat_0/In0]
connect_bd_net [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins xlconcat_0/In1]
connect_bd_net [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

# Create address map
assign_bd_address

# Validate design
validate_bd_design

# Save block design
save_bd_design

puts "Block design 'system' created successfully"
