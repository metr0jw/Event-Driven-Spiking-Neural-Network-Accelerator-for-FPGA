##-----------------------------------------------------------------------------
## File: timing.xdc
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description: Timing constraints for SNN Accelerator
##-----------------------------------------------------------------------------

## Clock constraints are already in pynq_z2_pins.xdc for sysclk

## AXI Clock (from PS) - typically 100MHz
create_clock -period 10.000 -name axi_clk [get_ports aclk]

## Clock domain crossings (if any)
# set_false_path -from [get_clocks axi_clk] -to [get_clocks sys_clk_pin]
# set_false_path -from [get_clocks sys_clk_pin] -to [get_clocks axi_clk]

## Input/Output delays (example - adjust based on actual requirements)
# set_input_delay -clock [get_clocks axi_clk] -min 1.0 [get_ports s_axi_*]
# set_input_delay -clock [get_clocks axi_clk] -max 3.0 [get_ports s_axi_*]
# set_output_delay -clock [get_clocks axi_clk] -min 1.0 [get_ports m_axi_*]
# set_output_delay -clock [get_clocks axi_clk] -max 3.0 [get_ports m_axi_*]
