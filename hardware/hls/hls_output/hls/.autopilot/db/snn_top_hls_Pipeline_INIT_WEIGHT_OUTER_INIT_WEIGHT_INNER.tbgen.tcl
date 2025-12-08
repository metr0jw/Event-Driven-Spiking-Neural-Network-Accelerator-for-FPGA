set moduleName snn_top_hls_Pipeline_INIT_WEIGHT_OUTER_INIT_WEIGHT_INNER
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
set cdfgNum 16
set C_modelName {snn_top_hls_Pipeline_INIT_WEIGHT_OUTER_INIT_WEIGHT_INNER}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict p_ZL13weight_memory_0_0 { MEM_WIDTH 8 MEM_SIZE 512 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL13weight_memory_0_1 { MEM_WIDTH 8 MEM_SIZE 512 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL13weight_memory_1_0 { MEM_WIDTH 8 MEM_SIZE 512 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL13weight_memory_1_1 { MEM_WIDTH 8 MEM_SIZE 512 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL13weight_memory_2_0 { MEM_WIDTH 8 MEM_SIZE 512 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL13weight_memory_2_1 { MEM_WIDTH 8 MEM_SIZE 512 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL13weight_memory_3_0 { MEM_WIDTH 8 MEM_SIZE 512 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL13weight_memory_3_1 { MEM_WIDTH 8 MEM_SIZE 512 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
set C_modelArgList {
	{ p_ZL13weight_memory_0_0 int 8 regular {array 512 { 3 0 } 0 1 } {global 1}  }
	{ p_ZL13weight_memory_0_1 int 8 regular {array 512 { 3 0 } 0 1 } {global 1}  }
	{ p_ZL13weight_memory_1_0 int 8 regular {array 512 { 3 0 } 0 1 } {global 1}  }
	{ p_ZL13weight_memory_1_1 int 8 regular {array 512 { 3 0 } 0 1 } {global 1}  }
	{ p_ZL13weight_memory_2_0 int 8 regular {array 512 { 3 0 } 0 1 } {global 1}  }
	{ p_ZL13weight_memory_2_1 int 8 regular {array 512 { 3 0 } 0 1 } {global 1}  }
	{ p_ZL13weight_memory_3_0 int 8 regular {array 512 { 3 0 } 0 1 } {global 1}  }
	{ p_ZL13weight_memory_3_1 int 8 regular {array 512 { 3 0 } 0 1 } {global 1}  }
}
set hasAXIMCache 0
set l_AXIML2Cache [list]
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "p_ZL13weight_memory_0_0", "interface" : "memory", "bitwidth" : 8, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL13weight_memory_0_1", "interface" : "memory", "bitwidth" : 8, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL13weight_memory_1_0", "interface" : "memory", "bitwidth" : 8, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL13weight_memory_1_1", "interface" : "memory", "bitwidth" : 8, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL13weight_memory_2_0", "interface" : "memory", "bitwidth" : 8, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL13weight_memory_2_1", "interface" : "memory", "bitwidth" : 8, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL13weight_memory_3_0", "interface" : "memory", "bitwidth" : 8, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL13weight_memory_3_1", "interface" : "memory", "bitwidth" : 8, "direction" : "WRITEONLY", "extern" : 0} ]}
# RTL Port declarations: 
set portNum 38
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ p_ZL13weight_memory_0_0_address1 sc_out sc_lv 9 signal 0 } 
	{ p_ZL13weight_memory_0_0_ce1 sc_out sc_logic 1 signal 0 } 
	{ p_ZL13weight_memory_0_0_we1 sc_out sc_logic 1 signal 0 } 
	{ p_ZL13weight_memory_0_0_d1 sc_out sc_lv 8 signal 0 } 
	{ p_ZL13weight_memory_0_1_address1 sc_out sc_lv 9 signal 1 } 
	{ p_ZL13weight_memory_0_1_ce1 sc_out sc_logic 1 signal 1 } 
	{ p_ZL13weight_memory_0_1_we1 sc_out sc_logic 1 signal 1 } 
	{ p_ZL13weight_memory_0_1_d1 sc_out sc_lv 8 signal 1 } 
	{ p_ZL13weight_memory_1_0_address1 sc_out sc_lv 9 signal 2 } 
	{ p_ZL13weight_memory_1_0_ce1 sc_out sc_logic 1 signal 2 } 
	{ p_ZL13weight_memory_1_0_we1 sc_out sc_logic 1 signal 2 } 
	{ p_ZL13weight_memory_1_0_d1 sc_out sc_lv 8 signal 2 } 
	{ p_ZL13weight_memory_1_1_address1 sc_out sc_lv 9 signal 3 } 
	{ p_ZL13weight_memory_1_1_ce1 sc_out sc_logic 1 signal 3 } 
	{ p_ZL13weight_memory_1_1_we1 sc_out sc_logic 1 signal 3 } 
	{ p_ZL13weight_memory_1_1_d1 sc_out sc_lv 8 signal 3 } 
	{ p_ZL13weight_memory_2_0_address1 sc_out sc_lv 9 signal 4 } 
	{ p_ZL13weight_memory_2_0_ce1 sc_out sc_logic 1 signal 4 } 
	{ p_ZL13weight_memory_2_0_we1 sc_out sc_logic 1 signal 4 } 
	{ p_ZL13weight_memory_2_0_d1 sc_out sc_lv 8 signal 4 } 
	{ p_ZL13weight_memory_2_1_address1 sc_out sc_lv 9 signal 5 } 
	{ p_ZL13weight_memory_2_1_ce1 sc_out sc_logic 1 signal 5 } 
	{ p_ZL13weight_memory_2_1_we1 sc_out sc_logic 1 signal 5 } 
	{ p_ZL13weight_memory_2_1_d1 sc_out sc_lv 8 signal 5 } 
	{ p_ZL13weight_memory_3_0_address1 sc_out sc_lv 9 signal 6 } 
	{ p_ZL13weight_memory_3_0_ce1 sc_out sc_logic 1 signal 6 } 
	{ p_ZL13weight_memory_3_0_we1 sc_out sc_logic 1 signal 6 } 
	{ p_ZL13weight_memory_3_0_d1 sc_out sc_lv 8 signal 6 } 
	{ p_ZL13weight_memory_3_1_address1 sc_out sc_lv 9 signal 7 } 
	{ p_ZL13weight_memory_3_1_ce1 sc_out sc_logic 1 signal 7 } 
	{ p_ZL13weight_memory_3_1_we1 sc_out sc_logic 1 signal 7 } 
	{ p_ZL13weight_memory_3_1_d1 sc_out sc_lv 8 signal 7 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "p_ZL13weight_memory_0_0_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_0_0", "role": "address1" }} , 
 	{ "name": "p_ZL13weight_memory_0_0_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_0_0", "role": "ce1" }} , 
 	{ "name": "p_ZL13weight_memory_0_0_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_0_0", "role": "we1" }} , 
 	{ "name": "p_ZL13weight_memory_0_0_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_0_0", "role": "d1" }} , 
 	{ "name": "p_ZL13weight_memory_0_1_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_0_1", "role": "address1" }} , 
 	{ "name": "p_ZL13weight_memory_0_1_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_0_1", "role": "ce1" }} , 
 	{ "name": "p_ZL13weight_memory_0_1_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_0_1", "role": "we1" }} , 
 	{ "name": "p_ZL13weight_memory_0_1_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_0_1", "role": "d1" }} , 
 	{ "name": "p_ZL13weight_memory_1_0_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_1_0", "role": "address1" }} , 
 	{ "name": "p_ZL13weight_memory_1_0_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_1_0", "role": "ce1" }} , 
 	{ "name": "p_ZL13weight_memory_1_0_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_1_0", "role": "we1" }} , 
 	{ "name": "p_ZL13weight_memory_1_0_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_1_0", "role": "d1" }} , 
 	{ "name": "p_ZL13weight_memory_1_1_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_1_1", "role": "address1" }} , 
 	{ "name": "p_ZL13weight_memory_1_1_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_1_1", "role": "ce1" }} , 
 	{ "name": "p_ZL13weight_memory_1_1_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_1_1", "role": "we1" }} , 
 	{ "name": "p_ZL13weight_memory_1_1_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_1_1", "role": "d1" }} , 
 	{ "name": "p_ZL13weight_memory_2_0_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_2_0", "role": "address1" }} , 
 	{ "name": "p_ZL13weight_memory_2_0_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_2_0", "role": "ce1" }} , 
 	{ "name": "p_ZL13weight_memory_2_0_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_2_0", "role": "we1" }} , 
 	{ "name": "p_ZL13weight_memory_2_0_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_2_0", "role": "d1" }} , 
 	{ "name": "p_ZL13weight_memory_2_1_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_2_1", "role": "address1" }} , 
 	{ "name": "p_ZL13weight_memory_2_1_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_2_1", "role": "ce1" }} , 
 	{ "name": "p_ZL13weight_memory_2_1_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_2_1", "role": "we1" }} , 
 	{ "name": "p_ZL13weight_memory_2_1_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_2_1", "role": "d1" }} , 
 	{ "name": "p_ZL13weight_memory_3_0_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_3_0", "role": "address1" }} , 
 	{ "name": "p_ZL13weight_memory_3_0_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_3_0", "role": "ce1" }} , 
 	{ "name": "p_ZL13weight_memory_3_0_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_3_0", "role": "we1" }} , 
 	{ "name": "p_ZL13weight_memory_3_0_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_3_0", "role": "d1" }} , 
 	{ "name": "p_ZL13weight_memory_3_1_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_3_1", "role": "address1" }} , 
 	{ "name": "p_ZL13weight_memory_3_1_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_3_1", "role": "ce1" }} , 
 	{ "name": "p_ZL13weight_memory_3_1_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_3_1", "role": "we1" }} , 
 	{ "name": "p_ZL13weight_memory_3_1_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL13weight_memory_3_1", "role": "d1" }}  ]}

set ArgLastReadFirstWriteLatency {
	snn_top_hls_Pipeline_INIT_WEIGHT_OUTER_INIT_WEIGHT_INNER {
		p_ZL13weight_memory_0_0 {Type O LastRead -1 FirstWrite 2}
		p_ZL13weight_memory_0_1 {Type O LastRead -1 FirstWrite 2}
		p_ZL13weight_memory_1_0 {Type O LastRead -1 FirstWrite 2}
		p_ZL13weight_memory_1_1 {Type O LastRead -1 FirstWrite 2}
		p_ZL13weight_memory_2_0 {Type O LastRead -1 FirstWrite 2}
		p_ZL13weight_memory_2_1 {Type O LastRead -1 FirstWrite 2}
		p_ZL13weight_memory_3_0 {Type O LastRead -1 FirstWrite 2}
		p_ZL13weight_memory_3_1 {Type O LastRead -1 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "4099", "Max" : "4099"}
	, {"Name" : "Interval", "Min" : "4097", "Max" : "4097"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	p_ZL13weight_memory_0_0 { ap_memory {  { p_ZL13weight_memory_0_0_address1 MemPortADDR2 1 9 }  { p_ZL13weight_memory_0_0_ce1 MemPortCE2 1 1 }  { p_ZL13weight_memory_0_0_we1 MemPortWE2 1 1 }  { p_ZL13weight_memory_0_0_d1 MemPortDIN2 1 8 } } }
	p_ZL13weight_memory_0_1 { ap_memory {  { p_ZL13weight_memory_0_1_address1 MemPortADDR2 1 9 }  { p_ZL13weight_memory_0_1_ce1 MemPortCE2 1 1 }  { p_ZL13weight_memory_0_1_we1 MemPortWE2 1 1 }  { p_ZL13weight_memory_0_1_d1 MemPortDIN2 1 8 } } }
	p_ZL13weight_memory_1_0 { ap_memory {  { p_ZL13weight_memory_1_0_address1 MemPortADDR2 1 9 }  { p_ZL13weight_memory_1_0_ce1 MemPortCE2 1 1 }  { p_ZL13weight_memory_1_0_we1 MemPortWE2 1 1 }  { p_ZL13weight_memory_1_0_d1 MemPortDIN2 1 8 } } }
	p_ZL13weight_memory_1_1 { ap_memory {  { p_ZL13weight_memory_1_1_address1 MemPortADDR2 1 9 }  { p_ZL13weight_memory_1_1_ce1 MemPortCE2 1 1 }  { p_ZL13weight_memory_1_1_we1 MemPortWE2 1 1 }  { p_ZL13weight_memory_1_1_d1 MemPortDIN2 1 8 } } }
	p_ZL13weight_memory_2_0 { ap_memory {  { p_ZL13weight_memory_2_0_address1 MemPortADDR2 1 9 }  { p_ZL13weight_memory_2_0_ce1 MemPortCE2 1 1 }  { p_ZL13weight_memory_2_0_we1 MemPortWE2 1 1 }  { p_ZL13weight_memory_2_0_d1 MemPortDIN2 1 8 } } }
	p_ZL13weight_memory_2_1 { ap_memory {  { p_ZL13weight_memory_2_1_address1 MemPortADDR2 1 9 }  { p_ZL13weight_memory_2_1_ce1 MemPortCE2 1 1 }  { p_ZL13weight_memory_2_1_we1 MemPortWE2 1 1 }  { p_ZL13weight_memory_2_1_d1 MemPortDIN2 1 8 } } }
	p_ZL13weight_memory_3_0 { ap_memory {  { p_ZL13weight_memory_3_0_address1 MemPortADDR2 1 9 }  { p_ZL13weight_memory_3_0_ce1 MemPortCE2 1 1 }  { p_ZL13weight_memory_3_0_we1 MemPortWE2 1 1 }  { p_ZL13weight_memory_3_0_d1 MemPortDIN2 1 8 } } }
	p_ZL13weight_memory_3_1 { ap_memory {  { p_ZL13weight_memory_3_1_address1 MemPortADDR2 1 9 }  { p_ZL13weight_memory_3_1_ce1 MemPortCE2 1 1 }  { p_ZL13weight_memory_3_1_we1 MemPortWE2 1 1 }  { p_ZL13weight_memory_3_1_d1 MemPortDIN2 1 8 } } }
}
