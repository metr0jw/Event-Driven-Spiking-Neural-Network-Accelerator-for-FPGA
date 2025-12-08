set moduleName snn_top_hls_Pipeline_RESET_TRACE_OUTER_RESET_TRACE_INNER
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
set C_modelName {snn_top_hls_Pipeline_RESET_TRACE_OUTER_RESET_TRACE_INNER}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict p_ZL18eligibility_traces_0 { MEM_WIDTH 16 MEM_SIZE 2048 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL18eligibility_traces_1 { MEM_WIDTH 16 MEM_SIZE 2048 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL18eligibility_traces_2 { MEM_WIDTH 16 MEM_SIZE 2048 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL18eligibility_traces_3 { MEM_WIDTH 16 MEM_SIZE 2048 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
set C_modelArgList {
	{ p_ZL18eligibility_traces_0 int 16 regular {array 1024 { 3 0 } 0 1 } {global 1}  }
	{ p_ZL18eligibility_traces_1 int 16 regular {array 1024 { 3 0 } 0 1 } {global 1}  }
	{ p_ZL18eligibility_traces_2 int 16 regular {array 1024 { 3 0 } 0 1 } {global 1}  }
	{ p_ZL18eligibility_traces_3 int 16 regular {array 1024 { 3 0 } 0 1 } {global 1}  }
}
set hasAXIMCache 0
set l_AXIML2Cache [list]
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "p_ZL18eligibility_traces_0", "interface" : "memory", "bitwidth" : 16, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL18eligibility_traces_1", "interface" : "memory", "bitwidth" : 16, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL18eligibility_traces_2", "interface" : "memory", "bitwidth" : 16, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL18eligibility_traces_3", "interface" : "memory", "bitwidth" : 16, "direction" : "WRITEONLY", "extern" : 0} ]}
# RTL Port declarations: 
set portNum 22
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ p_ZL18eligibility_traces_0_address1 sc_out sc_lv 10 signal 0 } 
	{ p_ZL18eligibility_traces_0_ce1 sc_out sc_logic 1 signal 0 } 
	{ p_ZL18eligibility_traces_0_we1 sc_out sc_logic 1 signal 0 } 
	{ p_ZL18eligibility_traces_0_d1 sc_out sc_lv 16 signal 0 } 
	{ p_ZL18eligibility_traces_1_address1 sc_out sc_lv 10 signal 1 } 
	{ p_ZL18eligibility_traces_1_ce1 sc_out sc_logic 1 signal 1 } 
	{ p_ZL18eligibility_traces_1_we1 sc_out sc_logic 1 signal 1 } 
	{ p_ZL18eligibility_traces_1_d1 sc_out sc_lv 16 signal 1 } 
	{ p_ZL18eligibility_traces_2_address1 sc_out sc_lv 10 signal 2 } 
	{ p_ZL18eligibility_traces_2_ce1 sc_out sc_logic 1 signal 2 } 
	{ p_ZL18eligibility_traces_2_we1 sc_out sc_logic 1 signal 2 } 
	{ p_ZL18eligibility_traces_2_d1 sc_out sc_lv 16 signal 2 } 
	{ p_ZL18eligibility_traces_3_address1 sc_out sc_lv 10 signal 3 } 
	{ p_ZL18eligibility_traces_3_ce1 sc_out sc_logic 1 signal 3 } 
	{ p_ZL18eligibility_traces_3_we1 sc_out sc_logic 1 signal 3 } 
	{ p_ZL18eligibility_traces_3_d1 sc_out sc_lv 16 signal 3 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "p_ZL18eligibility_traces_0_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":10, "type": "signal", "bundle":{"name": "p_ZL18eligibility_traces_0", "role": "address1" }} , 
 	{ "name": "p_ZL18eligibility_traces_0_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL18eligibility_traces_0", "role": "ce1" }} , 
 	{ "name": "p_ZL18eligibility_traces_0_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL18eligibility_traces_0", "role": "we1" }} , 
 	{ "name": "p_ZL18eligibility_traces_0_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "p_ZL18eligibility_traces_0", "role": "d1" }} , 
 	{ "name": "p_ZL18eligibility_traces_1_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":10, "type": "signal", "bundle":{"name": "p_ZL18eligibility_traces_1", "role": "address1" }} , 
 	{ "name": "p_ZL18eligibility_traces_1_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL18eligibility_traces_1", "role": "ce1" }} , 
 	{ "name": "p_ZL18eligibility_traces_1_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL18eligibility_traces_1", "role": "we1" }} , 
 	{ "name": "p_ZL18eligibility_traces_1_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "p_ZL18eligibility_traces_1", "role": "d1" }} , 
 	{ "name": "p_ZL18eligibility_traces_2_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":10, "type": "signal", "bundle":{"name": "p_ZL18eligibility_traces_2", "role": "address1" }} , 
 	{ "name": "p_ZL18eligibility_traces_2_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL18eligibility_traces_2", "role": "ce1" }} , 
 	{ "name": "p_ZL18eligibility_traces_2_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL18eligibility_traces_2", "role": "we1" }} , 
 	{ "name": "p_ZL18eligibility_traces_2_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "p_ZL18eligibility_traces_2", "role": "d1" }} , 
 	{ "name": "p_ZL18eligibility_traces_3_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":10, "type": "signal", "bundle":{"name": "p_ZL18eligibility_traces_3", "role": "address1" }} , 
 	{ "name": "p_ZL18eligibility_traces_3_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL18eligibility_traces_3", "role": "ce1" }} , 
 	{ "name": "p_ZL18eligibility_traces_3_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL18eligibility_traces_3", "role": "we1" }} , 
 	{ "name": "p_ZL18eligibility_traces_3_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "p_ZL18eligibility_traces_3", "role": "d1" }}  ]}

set ArgLastReadFirstWriteLatency {
	snn_top_hls_Pipeline_RESET_TRACE_OUTER_RESET_TRACE_INNER {
		p_ZL18eligibility_traces_0 {Type O LastRead -1 FirstWrite 1}
		p_ZL18eligibility_traces_1 {Type O LastRead -1 FirstWrite 1}
		p_ZL18eligibility_traces_2 {Type O LastRead -1 FirstWrite 1}
		p_ZL18eligibility_traces_3 {Type O LastRead -1 FirstWrite 1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "1027", "Max" : "1027"}
	, {"Name" : "Interval", "Min" : "1026", "Max" : "1026"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	p_ZL18eligibility_traces_0 { ap_memory {  { p_ZL18eligibility_traces_0_address1 MemPortADDR2 1 10 }  { p_ZL18eligibility_traces_0_ce1 MemPortCE2 1 1 }  { p_ZL18eligibility_traces_0_we1 MemPortWE2 1 1 }  { p_ZL18eligibility_traces_0_d1 MemPortDIN2 1 16 } } }
	p_ZL18eligibility_traces_1 { ap_memory {  { p_ZL18eligibility_traces_1_address1 MemPortADDR2 1 10 }  { p_ZL18eligibility_traces_1_ce1 MemPortCE2 1 1 }  { p_ZL18eligibility_traces_1_we1 MemPortWE2 1 1 }  { p_ZL18eligibility_traces_1_d1 MemPortDIN2 1 16 } } }
	p_ZL18eligibility_traces_2 { ap_memory {  { p_ZL18eligibility_traces_2_address1 MemPortADDR2 1 10 }  { p_ZL18eligibility_traces_2_ce1 MemPortCE2 1 1 }  { p_ZL18eligibility_traces_2_we1 MemPortWE2 1 1 }  { p_ZL18eligibility_traces_2_d1 MemPortDIN2 1 16 } } }
	p_ZL18eligibility_traces_3 { ap_memory {  { p_ZL18eligibility_traces_3_address1 MemPortADDR2 1 10 }  { p_ZL18eligibility_traces_3_ce1 MemPortCE2 1 1 }  { p_ZL18eligibility_traces_3_we1 MemPortWE2 1 1 }  { p_ZL18eligibility_traces_3_d1 MemPortDIN2 1 16 } } }
}
