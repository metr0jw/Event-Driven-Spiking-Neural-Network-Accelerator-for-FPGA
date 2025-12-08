set moduleName process_post_spike_Pipeline_STDP_LTP_LOOP
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 1
set isPipelined_legacy 1
set pipeline_type loop_auto_rewind
set FunctionProtocol ap_ctrl_hs
set restart_counter_num 0
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set svuvm_can_support 1
set cdfgNum 31
set C_modelName {process_post_spike_Pipeline_STDP_LTP_LOOP}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict p_ZL15pre_spike_times_7 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL15pre_spike_times_6 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL15pre_spike_times_5 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL15pre_spike_times_4 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL15pre_spike_times_3 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL15pre_spike_times_2 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL15pre_spike_times_1 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL15pre_spike_times_0 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
set C_modelArgList {
	{ params_stdp_window_val int 16 regular  }
	{ current_time int 32 regular  }
	{ sext_ln55 int 16 regular  }
	{ zext_ln46 int 16 regular  }
	{ sext_ln151 int 16 regular  }
	{ sext_ln58 int 16 regular  }
	{ sext_ln134 int 16 regular  }
	{ post_id int 8 regular  }
	{ p_ZL15pre_spike_times_7 int 32 regular {array 8 { 1 } 1 1 } {global 0}  }
	{ p_ZL15pre_spike_times_6 int 32 regular {array 8 { 1 } 1 1 } {global 0}  }
	{ p_ZL15pre_spike_times_5 int 32 regular {array 8 { 1 } 1 1 } {global 0}  }
	{ p_ZL15pre_spike_times_4 int 32 regular {array 8 { 1 } 1 1 } {global 0}  }
	{ p_ZL15pre_spike_times_3 int 32 regular {array 8 { 1 } 1 1 } {global 0}  }
	{ p_ZL15pre_spike_times_2 int 32 regular {array 8 { 1 } 1 1 } {global 0}  }
	{ p_ZL15pre_spike_times_1 int 32 regular {array 8 { 1 } 1 1 } {global 0}  }
	{ p_ZL15pre_spike_times_0 int 32 regular {array 8 { 1 } 1 1 } {global 0}  }
	{ weight_update_fifo int 64 regular {fifo 1 volatile } {global 1}  }
}
set hasAXIMCache 0
set l_AXIML2Cache [list]
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "params_stdp_window_val", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "current_time", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "sext_ln55", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "zext_ln46", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "sext_ln151", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "sext_ln58", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "sext_ln134", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "post_id", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "p_ZL15pre_spike_times_7", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_spike_times_6", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_spike_times_5", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_spike_times_4", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_spike_times_3", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_spike_times_2", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_spike_times_1", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_spike_times_0", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "extern" : 0} , 
 	{ "Name" : "weight_update_fifo", "interface" : "fifo", "bitwidth" : 64, "direction" : "WRITEONLY", "extern" : 0} ]}
# RTL Port declarations: 
set portNum 41
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ weight_update_fifo_din sc_out sc_lv 64 signal 16 } 
	{ weight_update_fifo_full_n sc_in sc_logic 1 signal 16 } 
	{ weight_update_fifo_write sc_out sc_logic 1 signal 16 } 
	{ params_stdp_window_val sc_in sc_lv 16 signal 0 } 
	{ current_time sc_in sc_lv 32 signal 1 } 
	{ sext_ln55 sc_in sc_lv 16 signal 2 } 
	{ zext_ln46 sc_in sc_lv 16 signal 3 } 
	{ sext_ln151 sc_in sc_lv 16 signal 4 } 
	{ sext_ln58 sc_in sc_lv 16 signal 5 } 
	{ sext_ln134 sc_in sc_lv 16 signal 6 } 
	{ post_id sc_in sc_lv 8 signal 7 } 
	{ p_ZL15pre_spike_times_7_address0 sc_out sc_lv 3 signal 8 } 
	{ p_ZL15pre_spike_times_7_ce0 sc_out sc_logic 1 signal 8 } 
	{ p_ZL15pre_spike_times_7_q0 sc_in sc_lv 32 signal 8 } 
	{ p_ZL15pre_spike_times_6_address0 sc_out sc_lv 3 signal 9 } 
	{ p_ZL15pre_spike_times_6_ce0 sc_out sc_logic 1 signal 9 } 
	{ p_ZL15pre_spike_times_6_q0 sc_in sc_lv 32 signal 9 } 
	{ p_ZL15pre_spike_times_5_address0 sc_out sc_lv 3 signal 10 } 
	{ p_ZL15pre_spike_times_5_ce0 sc_out sc_logic 1 signal 10 } 
	{ p_ZL15pre_spike_times_5_q0 sc_in sc_lv 32 signal 10 } 
	{ p_ZL15pre_spike_times_4_address0 sc_out sc_lv 3 signal 11 } 
	{ p_ZL15pre_spike_times_4_ce0 sc_out sc_logic 1 signal 11 } 
	{ p_ZL15pre_spike_times_4_q0 sc_in sc_lv 32 signal 11 } 
	{ p_ZL15pre_spike_times_3_address0 sc_out sc_lv 3 signal 12 } 
	{ p_ZL15pre_spike_times_3_ce0 sc_out sc_logic 1 signal 12 } 
	{ p_ZL15pre_spike_times_3_q0 sc_in sc_lv 32 signal 12 } 
	{ p_ZL15pre_spike_times_2_address0 sc_out sc_lv 3 signal 13 } 
	{ p_ZL15pre_spike_times_2_ce0 sc_out sc_logic 1 signal 13 } 
	{ p_ZL15pre_spike_times_2_q0 sc_in sc_lv 32 signal 13 } 
	{ p_ZL15pre_spike_times_1_address0 sc_out sc_lv 3 signal 14 } 
	{ p_ZL15pre_spike_times_1_ce0 sc_out sc_logic 1 signal 14 } 
	{ p_ZL15pre_spike_times_1_q0 sc_in sc_lv 32 signal 14 } 
	{ p_ZL15pre_spike_times_0_address0 sc_out sc_lv 3 signal 15 } 
	{ p_ZL15pre_spike_times_0_ce0 sc_out sc_logic 1 signal 15 } 
	{ p_ZL15pre_spike_times_0_q0 sc_in sc_lv 32 signal 15 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "weight_update_fifo_din", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "weight_update_fifo", "role": "din" }} , 
 	{ "name": "weight_update_fifo_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "weight_update_fifo", "role": "full_n" }} , 
 	{ "name": "weight_update_fifo_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "weight_update_fifo", "role": "write" }} , 
 	{ "name": "params_stdp_window_val", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "params_stdp_window_val", "role": "default" }} , 
 	{ "name": "current_time", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "current_time", "role": "default" }} , 
 	{ "name": "sext_ln55", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "sext_ln55", "role": "default" }} , 
 	{ "name": "zext_ln46", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "zext_ln46", "role": "default" }} , 
 	{ "name": "sext_ln151", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "sext_ln151", "role": "default" }} , 
 	{ "name": "sext_ln58", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "sext_ln58", "role": "default" }} , 
 	{ "name": "sext_ln134", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "sext_ln134", "role": "default" }} , 
 	{ "name": "post_id", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "post_id", "role": "default" }} , 
 	{ "name": "p_ZL15pre_spike_times_7_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_7", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_spike_times_7_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_7", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_spike_times_7_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_7", "role": "q0" }} , 
 	{ "name": "p_ZL15pre_spike_times_6_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_6", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_spike_times_6_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_6", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_spike_times_6_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_6", "role": "q0" }} , 
 	{ "name": "p_ZL15pre_spike_times_5_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_5", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_spike_times_5_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_5", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_spike_times_5_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_5", "role": "q0" }} , 
 	{ "name": "p_ZL15pre_spike_times_4_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_4", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_spike_times_4_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_4", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_spike_times_4_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_4", "role": "q0" }} , 
 	{ "name": "p_ZL15pre_spike_times_3_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_3", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_spike_times_3_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_3", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_spike_times_3_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_3", "role": "q0" }} , 
 	{ "name": "p_ZL15pre_spike_times_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_2", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_spike_times_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_2", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_spike_times_2_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_2", "role": "q0" }} , 
 	{ "name": "p_ZL15pre_spike_times_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_1", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_spike_times_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_1", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_spike_times_1_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_1", "role": "q0" }} , 
 	{ "name": "p_ZL15pre_spike_times_0_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_0", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_spike_times_0_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_0", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_spike_times_0_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_0", "role": "q0" }}  ]}

set ArgLastReadFirstWriteLatency {
	process_post_spike_Pipeline_STDP_LTP_LOOP {
		params_stdp_window_val {Type I LastRead 0 FirstWrite -1}
		current_time {Type I LastRead 0 FirstWrite -1}
		sext_ln55 {Type I LastRead 0 FirstWrite -1}
		zext_ln46 {Type I LastRead 0 FirstWrite -1}
		sext_ln151 {Type I LastRead 0 FirstWrite -1}
		sext_ln58 {Type I LastRead 0 FirstWrite -1}
		sext_ln134 {Type I LastRead 0 FirstWrite -1}
		post_id {Type I LastRead 0 FirstWrite -1}
		p_ZL15pre_spike_times_7 {Type I LastRead 0 FirstWrite -1}
		p_ZL15pre_spike_times_6 {Type I LastRead 0 FirstWrite -1}
		p_ZL15pre_spike_times_5 {Type I LastRead 0 FirstWrite -1}
		p_ZL15pre_spike_times_4 {Type I LastRead 0 FirstWrite -1}
		p_ZL15pre_spike_times_3 {Type I LastRead 0 FirstWrite -1}
		p_ZL15pre_spike_times_2 {Type I LastRead 0 FirstWrite -1}
		p_ZL15pre_spike_times_1 {Type I LastRead 0 FirstWrite -1}
		p_ZL15pre_spike_times_0 {Type I LastRead 0 FirstWrite -1}
		weight_update_fifo {Type O LastRead 42 FirstWrite 35}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "100", "Max" : "100"}
	, {"Name" : "Interval", "Min" : "72", "Max" : "72"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	params_stdp_window_val { ap_none {  { params_stdp_window_val in_data 0 16 } } }
	current_time { ap_none {  { current_time in_data 0 32 } } }
	sext_ln55 { ap_none {  { sext_ln55 in_data 0 16 } } }
	zext_ln46 { ap_none {  { zext_ln46 in_data 0 16 } } }
	sext_ln151 { ap_none {  { sext_ln151 in_data 0 16 } } }
	sext_ln58 { ap_none {  { sext_ln58 in_data 0 16 } } }
	sext_ln134 { ap_none {  { sext_ln134 in_data 0 16 } } }
	post_id { ap_none {  { post_id in_data 0 8 } } }
	p_ZL15pre_spike_times_7 { ap_memory {  { p_ZL15pre_spike_times_7_address0 mem_address 1 3 }  { p_ZL15pre_spike_times_7_ce0 mem_ce 1 1 }  { p_ZL15pre_spike_times_7_q0 mem_dout 0 32 } } }
	p_ZL15pre_spike_times_6 { ap_memory {  { p_ZL15pre_spike_times_6_address0 mem_address 1 3 }  { p_ZL15pre_spike_times_6_ce0 mem_ce 1 1 }  { p_ZL15pre_spike_times_6_q0 mem_dout 0 32 } } }
	p_ZL15pre_spike_times_5 { ap_memory {  { p_ZL15pre_spike_times_5_address0 mem_address 1 3 }  { p_ZL15pre_spike_times_5_ce0 mem_ce 1 1 }  { p_ZL15pre_spike_times_5_q0 mem_dout 0 32 } } }
	p_ZL15pre_spike_times_4 { ap_memory {  { p_ZL15pre_spike_times_4_address0 mem_address 1 3 }  { p_ZL15pre_spike_times_4_ce0 mem_ce 1 1 }  { p_ZL15pre_spike_times_4_q0 mem_dout 0 32 } } }
	p_ZL15pre_spike_times_3 { ap_memory {  { p_ZL15pre_spike_times_3_address0 mem_address 1 3 }  { p_ZL15pre_spike_times_3_ce0 mem_ce 1 1 }  { p_ZL15pre_spike_times_3_q0 mem_dout 0 32 } } }
	p_ZL15pre_spike_times_2 { ap_memory {  { p_ZL15pre_spike_times_2_address0 mem_address 1 3 }  { p_ZL15pre_spike_times_2_ce0 mem_ce 1 1 }  { p_ZL15pre_spike_times_2_q0 mem_dout 0 32 } } }
	p_ZL15pre_spike_times_1 { ap_memory {  { p_ZL15pre_spike_times_1_address0 mem_address 1 3 }  { p_ZL15pre_spike_times_1_ce0 mem_ce 1 1 }  { p_ZL15pre_spike_times_1_q0 mem_dout 0 32 } } }
	p_ZL15pre_spike_times_0 { ap_memory {  { p_ZL15pre_spike_times_0_address0 mem_address 1 3 }  { p_ZL15pre_spike_times_0_ce0 mem_ce 1 1 }  { p_ZL15pre_spike_times_0_q0 mem_dout 0 32 } } }
	weight_update_fifo { ap_fifo {  { weight_update_fifo_din fifo_data_out 1 64 }  { weight_update_fifo_full_n fifo_status_empty 0 1 }  { weight_update_fifo_write fifo_data_in 1 1 } } }
}
