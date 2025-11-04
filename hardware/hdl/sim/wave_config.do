##-----------------------------------------------------------------------------
## Title         : Waveform Configuration for SNN Accelerator
## Project       : PYNQ-Z2 SNN Accelerator
## File          : wave_config.do
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Date          : 2025-01-23
## Description   : Waveform display configuration for simulation
##-----------------------------------------------------------------------------

# Remove all existing waves
delete wave *

# Set default radix
radix hex

# Create groups for better organization
add wave -divider "SYSTEM SIGNALS"
add wave -label "Clock" -color yellow /tb_top/aclk
add wave -label "Reset_n" -color orange /tb_top/aresetn
add wave -divider ""

# AXI4-Lite Interface
add wave -divider "AXI4-LITE INTERFACE"
add wave -group "AXI Write Address" \
    -label "AWADDR" /tb_top/s_axi_awaddr \
    -label "AWVALID" /tb_top/s_axi_awvalid \
    -label "AWREADY" /tb_top/s_axi_awready

add wave -group "AXI Write Data" \
    -label "WDATA" /tb_top/s_axi_wdata \
    -label "WSTRB" /tb_top/s_axi_wstrb \
    -label "WVALID" /tb_top/s_axi_wvalid \
    -label "WREADY" /tb_top/s_axi_wready

add wave -group "AXI Write Response" \
    -label "BRESP" /tb_top/s_axi_bresp \
    -label "BVALID" /tb_top/s_axi_bvalid \
    -label "BREADY" /tb_top/s_axi_bready

add wave -group "AXI Read Address" \
    -label "ARADDR" /tb_top/s_axi_araddr \
    -label "ARVALID" /tb_top/s_axi_arvalid \
    -label "ARREADY" /tb_top/s_axi_arready

add wave -group "AXI Read Data" \
    -label "RDATA" /tb_top/s_axi_rdata \
    -label "RRESP" /tb_top/s_axi_rresp \
    -label "RVALID" /tb_top/s_axi_rvalid \
    -label "RREADY" /tb_top/s_axi_rready

# AXI-Stream Interfaces
add wave -divider "AXI-STREAM INTERFACES"
add wave -group "Input Spikes (S_AXIS)" \
    -label "TDATA" /tb_top/s_axis_tdata \
    -label "TVALID" /tb_top/s_axis_tvalid \
    -label "TREADY" /tb_top/s_axis_tready \
    -label "TLAST" /tb_top/s_axis_tlast

add wave -group "Output Spikes (M_AXIS)" \
    -label "TDATA" /tb_top/m_axis_tdata \
    -label "TVALID" /tb_top/m_axis_tvalid \
    -label "TREADY" /tb_top/m_axis_tready \
    -label "TLAST" /tb_top/m_axis_tlast

# SNN Core Signals
add wave -divider "SNN CORE"

# Configuration Registers
add wave -group "Configuration" \
    -label "Enable" /tb_top/DUT/snn_enable \
    -label "Reset" /tb_top/DUT/snn_reset \
    -label "Leak Rate" -radix unsigned /tb_top/DUT/leak_rate \
    -label "Threshold" -radix unsigned /tb_top/DUT/threshold \
    -label "Refractory" -radix unsigned /tb_top/DUT/refractory_period

# Neuron Array Signals
add wave -divider "NEURON ARRAY"
add wave -group "Neuron Array" \
    -label "Input Valid" /tb_top/DUT/neuron_array_inst/s_axis_spike_valid \
    -label "Input Neuron ID" -radix unsigned /tb_top/DUT/neuron_array_inst/s_axis_spike_dest_id \
    -label "Input Weight" -radix unsigned /tb_top/DUT/neuron_array_inst/s_axis_spike_weight \
    -label "Output Valid" /tb_top/DUT/neuron_array_inst/m_axis_spike_valid \
    -label "Output Neuron ID" -radix unsigned /tb_top/DUT/neuron_array_inst/m_axis_spike_neuron_id \
    -label "Spike Count" -radix unsigned /tb_top/DUT/neuron_array_inst/spike_count

# Individual Neuron Monitoring (first 4 neurons)
add wave -group "Neuron States" \
    -label "Neuron[0] Membrane" -radix unsigned -analog -min 0 -max 65535 /tb_top/DUT/neuron_array_inst/membrane_potential[0] \
    -label "Neuron[1] Membrane" -radix unsigned -analog -min 0 -max 65535 /tb_top/DUT/neuron_array_inst/membrane_potential[1] \
    -label "Neuron[2] Membrane" -radix unsigned -analog -min 0 -max 65535 /tb_top/DUT/neuron_array_inst/membrane_potential[2] \
    -label "Neuron[3] Membrane" -radix unsigned -analog -min 0 -max 65535 /tb_top/DUT/neuron_array_inst/membrane_potential[3]

# Spike Router Signals
add wave -divider "SPIKE ROUTER"
add wave -group "Router" \
    -label "Input Valid" /tb_top/DUT/spike_router_inst/s_spike_valid \
    -label "Input Neuron" -radix unsigned /tb_top/DUT/spike_router_inst/s_spike_neuron_id \
    -label "Output Valid" /tb_top/DUT/spike_router_inst/m_spike_valid \
    -label "Output Dest" -radix unsigned /tb_top/DUT/spike_router_inst/m_spike_dest_id \
    -label "Output Weight" -radix unsigned /tb_top/DUT/spike_router_inst/m_spike_weight \
    -label "Router Busy" /tb_top/DUT/spike_router_inst/router_busy \
    -label "FIFO Overflow" -color red /tb_top/DUT/spike_router_inst/fifo_overflow

# Status and Debug
add wave -divider "STATUS & DEBUG"
add wave -group "Status" \
    -label "Status Register" /tb_top/DUT/status_reg \
    -label "Total Spikes" -radix unsigned /tb_top/DUT/spike_count \
    -label "Interrupt" -color red /tb_top/interrupt

add wave -group "LEDs" \
    -label "Regular LEDs" -radix binary /tb_top/led \
    -label "RGB LED4 (R/G/B)" -radix binary {/tb_top/led4_r /tb_top/led4_g /tb_top/led4_b} \
    -label "RGB LED5 (R/G/B)" -radix binary {/tb_top/led5_r /tb_top/led5_g /tb_top/led5_b}

# Test Control
add wave -divider "TEST CONTROL"
add wave -label "Test Number" -radix unsigned /tb_top/test_num
add wave -label "Error Count" -radix unsigned /tb_top/error_count
add wave -label "Spike Count In" -radix unsigned /tb_top/spike_count_in
add wave -label "Spike Count Out" -radix unsigned /tb_top/spike_count_out

# Configure wave window
configure wave -namecolwidth 250
configure wave -valuecolwidth 150
configure wave -justifyvalue left
configure wave -signalnamewidth 1
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2

# Zoom to fit
wave zoom full

# Alternative configuration for specific testbenches
proc load_neuron_tb_waves {} {
    delete wave *
    
    add wave -divider "NEURON TESTBENCH"
    add wave -label "Clock" -color yellow /tb_lif_neuron/clk
    add wave -label "Reset_n" -color orange /tb_lif_neuron/rst_n
    add wave -label "Enable" /tb_lif_neuron/enable
    
    add wave -divider "INPUTS"
    add wave -label "Syn Valid" /tb_lif_neuron/syn_valid
    add wave -label "Syn Weight" -radix unsigned /tb_lif_neuron/syn_weight
    add wave -label "Syn Excitatory" /tb_lif_neuron/syn_excitatory
    
    add wave -divider "PARAMETERS"
    add wave -label "Threshold" -radix unsigned /tb_lif_neuron/threshold
    add wave -label "Leak Rate" -radix unsigned /tb_lif_neuron/leak_rate
    add wave -label "Refractory Period" -radix unsigned /tb_lif_neuron/refractory_period
    
    add wave -divider "OUTPUTS"
    add wave -label "Spike Out" -color red /tb_lif_neuron/spike_out
    add wave -label "Membrane Potential" -radix unsigned -analog -min 0 -max 65535 /tb_lif_neuron/membrane_potential
    add wave -label "Is Refractory" /tb_lif_neuron/is_refractory
    
    add wave -divider "TEST INFO"
    add wave -label "Test Name" /tb_lif_neuron/test_name
    add wave -label "Spike Count" -radix unsigned /tb_lif_neuron/spike_count
    
    wave zoom full
}

proc load_router_tb_waves {} {
    delete wave *
    
    add wave -divider "ROUTER TESTBENCH"
    add wave -label "Clock" -color yellow /tb_spike_router/clk
    add wave -label "Reset_n" -color orange /tb_spike_router/rst_n
    
    add wave -divider "INPUT INTERFACE"
    add wave -label "Input Valid" /tb_spike_router/s_spike_valid
    add wave -label "Input Neuron ID" -radix unsigned /tb_spike_router/s_spike_neuron_id
    add wave -label "Input Ready" /tb_spike_router/s_spike_ready
    
    add wave -divider "OUTPUT INTERFACE"
    add wave -label "Output Valid" /tb_spike_router/m_spike_valid
    add wave -label "Output Dest ID" -radix unsigned /tb_spike_router/m_spike_dest_id
    add wave -label "Output Weight" -radix unsigned /tb_spike_router/m_spike_weight
    add wave -label "Output Ready" /tb_spike_router/m_spike_ready
    
    add wave -divider "CONFIGURATION"
    add wave -label "Config Write" /tb_spike_router/config_we
    add wave -label "Config Addr" /tb_spike_router/config_addr
    add wave -label "Config Data" /tb_spike_router/config_data
    
    add wave -divider "STATUS"
    add wave -label "Routed Spikes" -radix unsigned /tb_spike_router/routed_spike_count
    add wave -label "Router Busy" /tb_spike_router/router_busy
    add wave -label "FIFO Overflow" -color red /tb_spike_router/fifo_overflow
    
    wave zoom full
}

# Print usage instructions
puts "Wave configuration loaded."
puts "Additional commands available:"
puts "  load_neuron_tb_waves  - Load waves for neuron testbench"
puts "  load_router_tb_waves  - Load waves for router testbench"
puts "  radix hex/dec/bin     - Change radix for all signals"
