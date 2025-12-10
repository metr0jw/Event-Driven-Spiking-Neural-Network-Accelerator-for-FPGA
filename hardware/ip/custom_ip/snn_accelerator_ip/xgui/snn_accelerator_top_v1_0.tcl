# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "AXON_ID_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C_AXIS_DATA_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C_S_AXI_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C_S_AXI_DATA_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "DATA_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "LEAK_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "NEURON_ID_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "NUM_AXONS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "NUM_NEURONS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "NUM_PARALLEL_UNITS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "REFRAC_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "ROUTER_BUFFER_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SPIKE_BUFFER_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "THRESHOLD_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "USE_BRAM" -parent ${Page_0}
  ipgui::add_param $IPINST -name "USE_DSP" -parent ${Page_0}
  ipgui::add_param $IPINST -name "WEIGHT_WIDTH" -parent ${Page_0}


}

proc update_PARAM_VALUE.AXON_ID_WIDTH { PARAM_VALUE.AXON_ID_WIDTH } {
	# Procedure called to update AXON_ID_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.AXON_ID_WIDTH { PARAM_VALUE.AXON_ID_WIDTH } {
	# Procedure called to validate AXON_ID_WIDTH
	return true
}

proc update_PARAM_VALUE.C_AXIS_DATA_WIDTH { PARAM_VALUE.C_AXIS_DATA_WIDTH } {
	# Procedure called to update C_AXIS_DATA_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_AXIS_DATA_WIDTH { PARAM_VALUE.C_AXIS_DATA_WIDTH } {
	# Procedure called to validate C_AXIS_DATA_WIDTH
	return true
}

proc update_PARAM_VALUE.C_S_AXI_ADDR_WIDTH { PARAM_VALUE.C_S_AXI_ADDR_WIDTH } {
	# Procedure called to update C_S_AXI_ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_S_AXI_ADDR_WIDTH { PARAM_VALUE.C_S_AXI_ADDR_WIDTH } {
	# Procedure called to validate C_S_AXI_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.C_S_AXI_DATA_WIDTH { PARAM_VALUE.C_S_AXI_DATA_WIDTH } {
	# Procedure called to update C_S_AXI_DATA_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_S_AXI_DATA_WIDTH { PARAM_VALUE.C_S_AXI_DATA_WIDTH } {
	# Procedure called to validate C_S_AXI_DATA_WIDTH
	return true
}

proc update_PARAM_VALUE.DATA_WIDTH { PARAM_VALUE.DATA_WIDTH } {
	# Procedure called to update DATA_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DATA_WIDTH { PARAM_VALUE.DATA_WIDTH } {
	# Procedure called to validate DATA_WIDTH
	return true
}

proc update_PARAM_VALUE.LEAK_WIDTH { PARAM_VALUE.LEAK_WIDTH } {
	# Procedure called to update LEAK_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.LEAK_WIDTH { PARAM_VALUE.LEAK_WIDTH } {
	# Procedure called to validate LEAK_WIDTH
	return true
}

proc update_PARAM_VALUE.NEURON_ID_WIDTH { PARAM_VALUE.NEURON_ID_WIDTH } {
	# Procedure called to update NEURON_ID_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.NEURON_ID_WIDTH { PARAM_VALUE.NEURON_ID_WIDTH } {
	# Procedure called to validate NEURON_ID_WIDTH
	return true
}

proc update_PARAM_VALUE.NUM_AXONS { PARAM_VALUE.NUM_AXONS } {
	# Procedure called to update NUM_AXONS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.NUM_AXONS { PARAM_VALUE.NUM_AXONS } {
	# Procedure called to validate NUM_AXONS
	return true
}

proc update_PARAM_VALUE.NUM_NEURONS { PARAM_VALUE.NUM_NEURONS } {
	# Procedure called to update NUM_NEURONS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.NUM_NEURONS { PARAM_VALUE.NUM_NEURONS } {
	# Procedure called to validate NUM_NEURONS
	return true
}

proc update_PARAM_VALUE.NUM_PARALLEL_UNITS { PARAM_VALUE.NUM_PARALLEL_UNITS } {
	# Procedure called to update NUM_PARALLEL_UNITS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.NUM_PARALLEL_UNITS { PARAM_VALUE.NUM_PARALLEL_UNITS } {
	# Procedure called to validate NUM_PARALLEL_UNITS
	return true
}

proc update_PARAM_VALUE.REFRAC_WIDTH { PARAM_VALUE.REFRAC_WIDTH } {
	# Procedure called to update REFRAC_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.REFRAC_WIDTH { PARAM_VALUE.REFRAC_WIDTH } {
	# Procedure called to validate REFRAC_WIDTH
	return true
}

proc update_PARAM_VALUE.ROUTER_BUFFER_DEPTH { PARAM_VALUE.ROUTER_BUFFER_DEPTH } {
	# Procedure called to update ROUTER_BUFFER_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ROUTER_BUFFER_DEPTH { PARAM_VALUE.ROUTER_BUFFER_DEPTH } {
	# Procedure called to validate ROUTER_BUFFER_DEPTH
	return true
}

proc update_PARAM_VALUE.SPIKE_BUFFER_DEPTH { PARAM_VALUE.SPIKE_BUFFER_DEPTH } {
	# Procedure called to update SPIKE_BUFFER_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SPIKE_BUFFER_DEPTH { PARAM_VALUE.SPIKE_BUFFER_DEPTH } {
	# Procedure called to validate SPIKE_BUFFER_DEPTH
	return true
}

proc update_PARAM_VALUE.THRESHOLD_WIDTH { PARAM_VALUE.THRESHOLD_WIDTH } {
	# Procedure called to update THRESHOLD_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.THRESHOLD_WIDTH { PARAM_VALUE.THRESHOLD_WIDTH } {
	# Procedure called to validate THRESHOLD_WIDTH
	return true
}

proc update_PARAM_VALUE.USE_BRAM { PARAM_VALUE.USE_BRAM } {
	# Procedure called to update USE_BRAM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.USE_BRAM { PARAM_VALUE.USE_BRAM } {
	# Procedure called to validate USE_BRAM
	return true
}

proc update_PARAM_VALUE.USE_DSP { PARAM_VALUE.USE_DSP } {
	# Procedure called to update USE_DSP when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.USE_DSP { PARAM_VALUE.USE_DSP } {
	# Procedure called to validate USE_DSP
	return true
}

proc update_PARAM_VALUE.WEIGHT_WIDTH { PARAM_VALUE.WEIGHT_WIDTH } {
	# Procedure called to update WEIGHT_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.WEIGHT_WIDTH { PARAM_VALUE.WEIGHT_WIDTH } {
	# Procedure called to validate WEIGHT_WIDTH
	return true
}


proc update_MODELPARAM_VALUE.C_S_AXI_DATA_WIDTH { MODELPARAM_VALUE.C_S_AXI_DATA_WIDTH PARAM_VALUE.C_S_AXI_DATA_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_S_AXI_DATA_WIDTH}] ${MODELPARAM_VALUE.C_S_AXI_DATA_WIDTH}
}

proc update_MODELPARAM_VALUE.C_S_AXI_ADDR_WIDTH { MODELPARAM_VALUE.C_S_AXI_ADDR_WIDTH PARAM_VALUE.C_S_AXI_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_S_AXI_ADDR_WIDTH}] ${MODELPARAM_VALUE.C_S_AXI_ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.C_AXIS_DATA_WIDTH { MODELPARAM_VALUE.C_AXIS_DATA_WIDTH PARAM_VALUE.C_AXIS_DATA_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_AXIS_DATA_WIDTH}] ${MODELPARAM_VALUE.C_AXIS_DATA_WIDTH}
}

proc update_MODELPARAM_VALUE.NUM_NEURONS { MODELPARAM_VALUE.NUM_NEURONS PARAM_VALUE.NUM_NEURONS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.NUM_NEURONS}] ${MODELPARAM_VALUE.NUM_NEURONS}
}

proc update_MODELPARAM_VALUE.NUM_AXONS { MODELPARAM_VALUE.NUM_AXONS PARAM_VALUE.NUM_AXONS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.NUM_AXONS}] ${MODELPARAM_VALUE.NUM_AXONS}
}

proc update_MODELPARAM_VALUE.NUM_PARALLEL_UNITS { MODELPARAM_VALUE.NUM_PARALLEL_UNITS PARAM_VALUE.NUM_PARALLEL_UNITS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.NUM_PARALLEL_UNITS}] ${MODELPARAM_VALUE.NUM_PARALLEL_UNITS}
}

proc update_MODELPARAM_VALUE.SPIKE_BUFFER_DEPTH { MODELPARAM_VALUE.SPIKE_BUFFER_DEPTH PARAM_VALUE.SPIKE_BUFFER_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SPIKE_BUFFER_DEPTH}] ${MODELPARAM_VALUE.SPIKE_BUFFER_DEPTH}
}

proc update_MODELPARAM_VALUE.NEURON_ID_WIDTH { MODELPARAM_VALUE.NEURON_ID_WIDTH PARAM_VALUE.NEURON_ID_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.NEURON_ID_WIDTH}] ${MODELPARAM_VALUE.NEURON_ID_WIDTH}
}

proc update_MODELPARAM_VALUE.AXON_ID_WIDTH { MODELPARAM_VALUE.AXON_ID_WIDTH PARAM_VALUE.AXON_ID_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.AXON_ID_WIDTH}] ${MODELPARAM_VALUE.AXON_ID_WIDTH}
}

proc update_MODELPARAM_VALUE.DATA_WIDTH { MODELPARAM_VALUE.DATA_WIDTH PARAM_VALUE.DATA_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DATA_WIDTH}] ${MODELPARAM_VALUE.DATA_WIDTH}
}

proc update_MODELPARAM_VALUE.WEIGHT_WIDTH { MODELPARAM_VALUE.WEIGHT_WIDTH PARAM_VALUE.WEIGHT_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.WEIGHT_WIDTH}] ${MODELPARAM_VALUE.WEIGHT_WIDTH}
}

proc update_MODELPARAM_VALUE.LEAK_WIDTH { MODELPARAM_VALUE.LEAK_WIDTH PARAM_VALUE.LEAK_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.LEAK_WIDTH}] ${MODELPARAM_VALUE.LEAK_WIDTH}
}

proc update_MODELPARAM_VALUE.THRESHOLD_WIDTH { MODELPARAM_VALUE.THRESHOLD_WIDTH PARAM_VALUE.THRESHOLD_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.THRESHOLD_WIDTH}] ${MODELPARAM_VALUE.THRESHOLD_WIDTH}
}

proc update_MODELPARAM_VALUE.REFRAC_WIDTH { MODELPARAM_VALUE.REFRAC_WIDTH PARAM_VALUE.REFRAC_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.REFRAC_WIDTH}] ${MODELPARAM_VALUE.REFRAC_WIDTH}
}

proc update_MODELPARAM_VALUE.ROUTER_BUFFER_DEPTH { MODELPARAM_VALUE.ROUTER_BUFFER_DEPTH PARAM_VALUE.ROUTER_BUFFER_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ROUTER_BUFFER_DEPTH}] ${MODELPARAM_VALUE.ROUTER_BUFFER_DEPTH}
}

proc update_MODELPARAM_VALUE.USE_BRAM { MODELPARAM_VALUE.USE_BRAM PARAM_VALUE.USE_BRAM } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.USE_BRAM}] ${MODELPARAM_VALUE.USE_BRAM}
}

proc update_MODELPARAM_VALUE.USE_DSP { MODELPARAM_VALUE.USE_DSP PARAM_VALUE.USE_DSP } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.USE_DSP}] ${MODELPARAM_VALUE.USE_DSP}
}

