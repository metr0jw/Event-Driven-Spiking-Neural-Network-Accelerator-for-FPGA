##-----------------------------------------------------------------------------
## Title         : Pin Constraints for PYNQ-Z2 SNN Accelerator
## Project       : PYNQ-Z2 SNN Accelerator
## File          : pynq_z2_pins.xdc
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Additional constraints for SNN-specific pins
##-----------------------------------------------------------------------------

## This file only contains constraints for pins actually used by the SNN accelerator
## The base PYNQ-Z2 v1.0.xdc file contains all available pin definitions


## Example: If you're using specific pins not in your design, uncomment and modify:


## Custom LED assignments for SNN status (if different from default)
# set_property -dict { PACKAGE_PIN R14   IOSTANDARD LVCMOS33 } [get_ports { snn_status_led[0] }]


## Note: Most pin assignments are already in pynq_z2_v1.0.xdc
## Only add constraints here for:
## 1. Pins with different names than in the base file
## 2. Custom timing constraints
## 3. Pin configuration changes