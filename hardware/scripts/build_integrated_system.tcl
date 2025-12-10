##-----------------------------------------------------------------------------
## Title         : Build Integrated SNN System (HLS + Verilog RTL)
## Project       : PYNQ-Z2 SNN Accelerator
## File          : build_integrated_system.tcl
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Creates Block Design with exported ports for RTL integration
##-----------------------------------------------------------------------------

set script_dir [file dirname [file normalize [info script]]]
set proj_root [file normalize "$script_dir/../.."]
set build_dir "$proj_root/build/vivado_integrated"
set proj_name "snn_integrated"

# Create project
puts "Step 1: Creating Vivado project..."
create_project $proj_name $build_dir -part xc7z020clg400-1 -force
# Note: PYNQ-Z2 board files not installed, using part only

# Add IP repository (HLS IP)
puts "Step 2: Adding IP repositories..."
set_property ip_repo_paths [list \
    "$proj_root/hardware/ip_repo" \
] [current_project]
update_ip_catalog
puts "  Added IP repository: $proj_root/hardware/ip_repo"

#==============================================================================
# Step 3: Add Verilog RTL sources
#==============================================================================
puts "Step 3: Adding Verilog RTL sources..."
set hdl_dir "$proj_root/hardware/hdl/rtl"

# Add files by category
foreach dir {common neurons synapses router layers top} {
    set files [glob -nocomplain $hdl_dir/$dir/*.v]
    if {[llength $files] > 0} {
        add_files -norecurse $files
        puts "  Added [llength $files] files from $dir/"
    }
}

# Set top module to the integrated wrapper
set_property top snn_integrated_top [current_fileset]
update_compile_order -fileset sources_1

#==============================================================================
# Step 4: Create Block Design (with exported ports)
#==============================================================================
puts "Step 4: Creating Block Design with exported interfaces..."

create_bd_design "design_1"

# Create Processing System
puts "  Creating Zynq PS..."
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0

# Apply PS configuration
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} \
    [get_bd_cells processing_system7_0]

# Enable HP0 and GP0
set_property -dict [list \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
] [get_bd_cells processing_system7_0]

# Create HLS IP
puts "  Creating HLS learning engine..."
create_bd_cell -type ip -vlnv xilinx.com:hls:snn_top_hls:1.0 snn_top_hls_0

# Create AXI DMA for spike streaming (MM2S only for input spikes)
puts "  Creating AXI DMA..."
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_include_s2mm {0} \
    CONFIG.c_m_axi_mm2s_data_width {64} \
    CONFIG.c_m_axis_mm2s_tdata_width {32} \
    CONFIG.c_mm2s_burst_size {16} \
] [get_bd_cells axi_dma_0]

# Create AXI Interconnects
puts "  Creating AXI interconnects..."
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
set_property -dict [list CONFIG.NUM_MI {2}] [get_bd_cells axi_interconnect_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_hp0
set_property -dict [list \
    CONFIG.NUM_SI {1} \
    CONFIG.NUM_MI {1} \
] [get_bd_cells axi_interconnect_hp0]

# Reset management
puts "  Creating reset controller..."
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0

#==============================================================================
# Step 5: Make External Ports (for RTL integration)
#==============================================================================
puts "Step 5: Creating external ports for RTL integration..."

# Clock and reset outputs (for RTL wrapper)
create_bd_port -dir O -type clk clk_100mhz
set_property CONFIG.FREQ_HZ 100000000 [get_bd_ports clk_100mhz]

create_bd_port -dir O -type rst rst_n_sync
set_property CONFIG.POLARITY ACTIVE_LOW [get_bd_ports rst_n_sync]

# Debug output (interrupt from HLS)
create_bd_port -dir O debug_learning_active

#------------------------------------------------------------------------------
# HLS → RTL Interface (Spikes from HLS to RTL neurons)
#------------------------------------------------------------------------------
# HLS outputs (spikes to RTL)
create_bd_port -dir O hls_spike_out_valid
create_bd_port -dir O -from 7 -to 0 hls_spike_out_neuron_id
create_bd_port -dir O -from 7 -to 0 hls_spike_out_weight
# RTL input ready signal to HLS
create_bd_port -dir I rtl_spike_in_ready

#------------------------------------------------------------------------------
# RTL → HLS Interface (Spikes from RTL neurons to HLS for learning)
#------------------------------------------------------------------------------
# RTL outputs (spikes to HLS)
create_bd_port -dir I rtl_spike_out_valid
create_bd_port -dir I -from 7 -to 0 rtl_spike_out_neuron_id
create_bd_port -dir I -from 7 -to 0 rtl_spike_out_weight
# HLS ready signal to RTL
create_bd_port -dir O hls_spike_in_ready

#------------------------------------------------------------------------------
# SNN Control Interface
#------------------------------------------------------------------------------
create_bd_port -dir O hls_snn_enable
create_bd_port -dir O hls_snn_reset
create_bd_port -dir I rtl_snn_ready
create_bd_port -dir I rtl_snn_busy

puts "  Created HLS <-> RTL spike interface ports"

#==============================================================================
# Step 6: Connect Block Design
#==============================================================================
puts "Step 6: Connecting Block Design..."

# Connect clocks
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] \
    [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] \
    [get_bd_pins snn_top_hls_0/ap_clk] \
    [get_bd_pins axi_dma_0/s_axi_lite_aclk] \
    [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \
    [get_bd_pins axi_interconnect_0/ACLK] \
    [get_bd_pins axi_interconnect_0/S00_ACLK] \
    [get_bd_pins axi_interconnect_0/M00_ACLK] \
    [get_bd_pins axi_interconnect_0/M01_ACLK] \
    [get_bd_pins axi_interconnect_hp0/ACLK] \
    [get_bd_pins axi_interconnect_hp0/S00_ACLK] \
    [get_bd_pins axi_interconnect_hp0/M00_ACLK] \
    [get_bd_pins proc_sys_reset_0/slowest_sync_clk] \
    [get_bd_ports clk_100mhz]

# Connect resets
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] \
    [get_bd_pins proc_sys_reset_0/ext_reset_in]
    
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
    [get_bd_pins snn_top_hls_0/ap_rst_n] \
    [get_bd_pins axi_dma_0/axi_resetn] \
    [get_bd_pins axi_interconnect_0/ARESETN] \
    [get_bd_pins axi_interconnect_0/S00_ARESETN] \
    [get_bd_pins axi_interconnect_0/M00_ARESETN] \
    [get_bd_pins axi_interconnect_0/M01_ARESETN] \
    [get_bd_pins axi_interconnect_hp0/ARESETN] \
    [get_bd_pins axi_interconnect_hp0/S00_ARESETN] \
    [get_bd_pins axi_interconnect_hp0/M00_ARESETN] \
    [get_bd_ports rst_n_sync]

# Connect AXI interfaces
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] \
    [get_bd_intf_pins axi_interconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M00_AXI] \
    [get_bd_intf_pins snn_top_hls_0/s_axi_ctrl]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M01_AXI] \
    [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] \
    [get_bd_intf_pins axi_interconnect_hp0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_hp0/M00_AXI] \
    [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

# Connect AXI-Stream (DMA → HLS)
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] \
    [get_bd_intf_pins snn_top_hls_0/s_axis_spikes]

#==============================================================================
# Step 7: Export signals for RTL integration
#==============================================================================
puts "Step 7: Exporting signals for RTL integration..."

# Export interrupt signal as debug
connect_bd_net [get_bd_pins snn_top_hls_0/interrupt] \
    [get_bd_ports debug_learning_active]

#------------------------------------------------------------------------------
# Connect HLS spike output interface (HLS → RTL)
#------------------------------------------------------------------------------
# HLS outputs spikes via spike_in_* ports (confusing naming in HLS)
connect_bd_net [get_bd_pins snn_top_hls_0/spike_in_valid] \
    [get_bd_ports hls_spike_out_valid]
connect_bd_net [get_bd_pins snn_top_hls_0/spike_in_neuron_id] \
    [get_bd_ports hls_spike_out_neuron_id]
connect_bd_net [get_bd_pins snn_top_hls_0/spike_in_weight] \
    [get_bd_ports hls_spike_out_weight]
# RTL ready signal
connect_bd_net [get_bd_ports rtl_spike_in_ready] \
    [get_bd_pins snn_top_hls_0/spike_in_ready]

#------------------------------------------------------------------------------
# Connect HLS spike input interface (RTL → HLS for learning)
#------------------------------------------------------------------------------
connect_bd_net [get_bd_ports rtl_spike_out_valid] \
    [get_bd_pins snn_top_hls_0/spike_out_valid]
connect_bd_net [get_bd_ports rtl_spike_out_neuron_id] \
    [get_bd_pins snn_top_hls_0/spike_out_neuron_id]
connect_bd_net [get_bd_ports rtl_spike_out_weight] \
    [get_bd_pins snn_top_hls_0/spike_out_weight]
# HLS ready signal
connect_bd_net [get_bd_pins snn_top_hls_0/spike_out_ready] \
    [get_bd_ports hls_spike_in_ready]

#------------------------------------------------------------------------------
# Connect SNN control interface
#------------------------------------------------------------------------------
connect_bd_net [get_bd_pins snn_top_hls_0/snn_enable] \
    [get_bd_ports hls_snn_enable]
connect_bd_net [get_bd_pins snn_top_hls_0/snn_reset] \
    [get_bd_ports hls_snn_reset]
connect_bd_net [get_bd_ports rtl_snn_ready] \
    [get_bd_pins snn_top_hls_0/snn_ready]
connect_bd_net [get_bd_ports rtl_snn_busy] \
    [get_bd_pins snn_top_hls_0/snn_busy]

puts "  Connected HLS <-> RTL spike interface"
puts "  Exported debug_learning_active (interrupt signal)"

#==============================================================================
# Step 8: Assign addresses
#==============================================================================
puts "Step 8: Assigning addresses..."
assign_bd_address

# Set HLS IP address
set_property range 128 [get_bd_addr_segs {processing_system7_0/Data/SEG_snn_top_hls_0_Reg}]
set_property offset 0x43C00000 [get_bd_addr_segs {processing_system7_0/Data/SEG_snn_top_hls_0_Reg}]

#==============================================================================
# Step 9: Validate and save
#==============================================================================
puts "Step 9: Validating Block Design..."
regenerate_bd_layout
validate_bd_design
save_bd_design

# Create HDL wrapper
puts "Step 10: Creating HDL wrapper..."
make_wrapper -files [get_files $build_dir/$proj_name.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse $build_dir/$proj_name.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v

# Update to use integrated top instead
set_property top snn_integrated_top [current_fileset]
update_compile_order -fileset sources_1

#==============================================================================
# Step 11: Run Synthesis
#==============================================================================
puts "Step 11: Running synthesis..."
launch_runs synth_1 -jobs 8
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed!"
    exit 1
}

puts "Synthesis complete!"

#==============================================================================
# Step 12: Run Implementation
#==============================================================================
puts "Step 12: Running implementation..."
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Implementation failed!"
    exit 1
}

puts "Implementation complete!"

#==============================================================================
# Step 13: Generate reports
#==============================================================================
puts "Step 13: Generating reports..."
open_run impl_1

report_utilization -file $proj_root/outputs/integrated_utilization.rpt
report_timing_summary -file $proj_root/outputs/integrated_timing.rpt
report_power -file $proj_root/outputs/integrated_power.rpt

# Copy outputs
puts "Step 14: Copying outputs..."
file mkdir $proj_root/outputs
file copy -force \
    $build_dir/$proj_name.runs/impl_1/snn_integrated_top.bit \
    $proj_root/outputs/snn_integrated.bit
    
puts "===================================================================="
puts "BUILD COMPLETE: Integrated SNN System (HLS + Verilog RTL)"
puts "===================================================================="
puts "Bitstream: outputs/snn_integrated.bit"
puts "Reports:   outputs/integrated_*.rpt"
puts "===================================================================="
