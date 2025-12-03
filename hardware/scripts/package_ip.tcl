##-----------------------------------------------------------------------------
## Title         : IP Packaging Script
## Project       : PYNQ-Z2 SNN Accelerator
## File          : package_ip.tcl
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Packages the SNN accelerator as a reusable IP core
##-----------------------------------------------------------------------------

# Get script directory
set script_dir [file dirname [file normalize [info script]]]
set proj_root [file normalize "$script_dir/../.."]

# IP properties
set ip_name "snn_accelerator_ip"
set ip_version "1.0"
set ip_display_name "SNN Accelerator IP"
set ip_description "Spiking Neural Network Accelerator for PYNQ-Z2"
set ip_vendor "PYNQ-Z2-SNN"
set ip_library "user"
set ip_taxonomy "/UserIP"

# Create temporary project for IP packaging
set temp_proj_name "ip_packager_temp"
set temp_proj_dir "$proj_root/build/temp_ip_packager"

create_project $temp_proj_name $temp_proj_dir -part xc7z020clg400-1 -force

# Add source files
set hdl_dir "$proj_root/hardware/hdl"

# Add all RTL sources
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/top/*.v]
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/neurons/*.v]
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/synapses/*.v]
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/router/*.v]
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/interfaces/*.v]
add_files -norecurse [glob -nocomplain $hdl_dir/rtl/common/*.v]

# Set top module
set_property top snn_accelerator_top [current_fileset]

# Update compile order
update_compile_order -fileset sources_1

# Package IP
ipx::package_project -root_dir "$proj_root/hardware/ip/custom_ip/$ip_name" -vendor $ip_vendor -library $ip_library -taxonomy $ip_taxonomy -import_files -set_current false

# Open the IP for editing
ipx::open_ipxact_file "$proj_root/hardware/ip/custom_ip/$ip_name/component.xml"

# Set IP identification
set_property vendor $ip_vendor [ipx::current_core]
set_property library $ip_library [ipx::current_core]
set_property name $ip_name [ipx::current_core]
set_property version $ip_version [ipx::current_core]
set_property display_name $ip_display_name [ipx::current_core]
set_property description $ip_description [ipx::current_core]
set_property vendor_display_name "PYNQ-Z2 SNN Team" [ipx::current_core]
set_property company_url "http://www.pynq.io" [ipx::current_core]
set_property supported_families {zynq Production} [ipx::current_core]

# Configure AXI interfaces
ipx::add_bus_interface S_AXI [ipx::current_core]
set_property abstraction_type_vlnv xilinx.com:interface:aximm_rtl:1.0 [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]
set_property bus_type_vlnv xilinx.com:interface:aximm:1.0 [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]
set_property interface_mode slave [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]

# Map AXI ports
set_property physical_name s_axi_awaddr [ipx::get_port_maps AWADDR -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_awprot [ipx::get_port_maps AWPROT -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_awvalid [ipx::get_port_maps AWVALID -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_awready [ipx::get_port_maps AWREADY -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_wdata [ipx::get_port_maps WDATA -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_wstrb [ipx::get_port_maps WSTRB -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_wvalid [ipx::get_port_maps WVALID -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_wready [ipx::get_port_maps WREADY -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_bresp [ipx::get_port_maps BRESP -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_bvalid [ipx::get_port_maps BVALID -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_bready [ipx::get_port_maps BREADY -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_araddr [ipx::get_port_maps ARADDR -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_arprot [ipx::get_port_maps ARPROT -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_arvalid [ipx::get_port_maps ARVALID -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_arready [ipx::get_port_maps ARREADY -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_rdata [ipx::get_port_maps RDATA -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_rresp [ipx::get_port_maps RRESP -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_rvalid [ipx::get_port_maps RVALID -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
set_property physical_name s_axi_rready [ipx::get_port_maps RREADY -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]

# Configure AXI-Stream interfaces
ipx::add_bus_interface S_AXIS_SPIKE [ipx::current_core]
set_property abstraction_type_vlnv xilinx.com:interface:axis_rtl:1.0 [ipx::get_bus_interfaces S_AXIS_SPIKE -of_objects [ipx::current_core]]
set_property bus_type_vlnv xilinx.com:interface:axis:1.0 [ipx::get_bus_interfaces S_AXIS_SPIKE -of_objects [ipx::current_core]]
set_property interface_mode slave [ipx::get_bus_interfaces S_AXIS_SPIKE -of_objects [ipx::current_core]]

ipx::add_bus_interface M_AXIS_SPIKE [ipx::current_core]
set_property abstraction_type_vlnv xilinx.com:interface:axis_rtl:1.0 [ipx::get_bus_interfaces M_AXIS_SPIKE -of_objects [ipx::current_core]]
set_property bus_type_vlnv xilinx.com:interface:axis:1.0 [ipx::get_bus_interfaces M_AXIS_SPIKE -of_objects [ipx::current_core]]
set_property interface_mode master [ipx::get_bus_interfaces M_AXIS_SPIKE -of_objects [ipx::current_core]]

# Associate clocks
ipx::associate_bus_interfaces -busif S_AXI -clock aclk [ipx::current_core]
ipx::associate_bus_interfaces -busif S_AXIS_SPIKE -clock aclk [ipx::current_core]
ipx::associate_bus_interfaces -busif M_AXIS_SPIKE -clock aclk [ipx::current_core]

# Add memory maps
ipx::add_memory_map S_AXI [ipx::current_core]
set_property slave_memory_map_ref S_AXI [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]

ipx::add_address_block axi_lite [ipx::get_memory_maps S_AXI -of_objects [ipx::current_core]]
set_property usage register [ipx::get_address_blocks axi_lite -of_objects [ipx::get_memory_maps S_AXI -of_objects [ipx::current_core]]]
set_property range 4096 [ipx::get_address_blocks axi_lite -of_objects [ipx::get_memory_maps S_AXI -of_objects [ipx::current_core]]]

# Set core parameters
ipx::add_user_parameter NUM_NEURONS [ipx::current_core]
set_property value_resolve_type user [ipx::get_user_parameters NUM_NEURONS -of_objects [ipx::current_core]]
set_property display_name {Number of Neurons} [ipx::get_user_parameters NUM_NEURONS -of_objects [ipx::current_core]]
set_property value 64 [ipx::get_user_parameters NUM_NEURONS -of_objects [ipx::current_core]]
set_property value_format long [ipx::get_user_parameters NUM_NEURONS -of_objects [ipx::current_core]]

# Create GUI customization
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]

# Package IP
ipx::archive_core "$proj_root/hardware/ip/custom_ip/${ip_name}_${ip_version}.zip" [ipx::current_core]

# Close IP core
ipx::unload_core "$proj_root/hardware/ip/custom_ip/$ip_name/component.xml"

# Clean up temporary project
close_project
file delete -force $temp_proj_dir

puts "----------------------------------------"
puts "IP packaging completed successfully!"
puts "IP location: $proj_root/hardware/ip/custom_ip/$ip_name"
puts "IP archive: $proj_root/hardware/ip/custom_ip/${ip_name}_${ip_version}.zip"
puts "----------------------------------------"
