# This script segment is generated automatically by AutoPilot

set axilite_register_dict [dict create]
set port_ctrl {
ctrl_reg { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 16
	offset_end 23
}
config_reg { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 24
	offset_end 31
}
learning_params { 
	dir I
	width 144
	depth 1
	mode ap_none
	offset 32
	offset_end 55
}
status_reg { 
	dir O
	width 32
	depth 1
	mode ap_vld
	offset 56
	offset_end 63
}
spike_count_reg { 
	dir O
	width 32
	depth 1
	mode ap_vld
	offset 72
	offset_end 79
}
weight_sum_reg { 
	dir O
	width 32
	depth 1
	mode ap_vld
	offset 88
	offset_end 95
}
version_reg { 
	dir O
	width 32
	depth 1
	mode ap_vld
	offset 104
	offset_end 111
}
reward_signal { 
	dir I
	width 8
	depth 1
	mode ap_none
	offset 120
	offset_end 127
}
ap_start { }
ap_done { }
ap_ready { }
ap_idle { }
interrupt {
}
}
dict set axilite_register_dict ctrl $port_ctrl


