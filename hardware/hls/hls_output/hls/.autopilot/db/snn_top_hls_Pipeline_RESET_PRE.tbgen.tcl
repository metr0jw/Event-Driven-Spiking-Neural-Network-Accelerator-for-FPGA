set moduleName snn_top_hls_Pipeline_RESET_PRE
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
set C_modelName {snn_top_hls_Pipeline_RESET_PRE}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict p_ZL15pre_spike_times_0 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL15pre_spike_times_1 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL15pre_spike_times_2 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL15pre_spike_times_3 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL15pre_spike_times_4 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL15pre_spike_times_5 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL15pre_spike_times_6 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL15pre_spike_times_7 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL16post_spike_times_0 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL16post_spike_times_1 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL16post_spike_times_2 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL16post_spike_times_3 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL16post_spike_times_4 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL16post_spike_times_5 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL16post_spike_times_6 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict p_ZL16post_spike_times_7 { MEM_WIDTH 32 MEM_SIZE 32 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
set C_modelArgList {
	{ p_ZL15pre_spike_times_0 int 32 regular {array 8 { 0 } 0 1 } {global 1}  }
	{ p_ZL15pre_spike_times_1 int 32 regular {array 8 { 0 } 0 1 } {global 1}  }
	{ p_ZL15pre_spike_times_2 int 32 regular {array 8 { 0 } 0 1 } {global 1}  }
	{ p_ZL15pre_spike_times_3 int 32 regular {array 8 { 0 } 0 1 } {global 1}  }
	{ p_ZL15pre_spike_times_4 int 32 regular {array 8 { 0 } 0 1 } {global 1}  }
	{ p_ZL15pre_spike_times_5 int 32 regular {array 8 { 0 } 0 1 } {global 1}  }
	{ p_ZL15pre_spike_times_6 int 32 regular {array 8 { 0 } 0 1 } {global 1}  }
	{ p_ZL15pre_spike_times_7 int 32 regular {array 8 { 0 } 0 1 } {global 1}  }
	{ p_ZL16post_spike_times_0 int 32 regular {array 8 { 0 } 0 1 } {global 1}  }
	{ p_ZL16post_spike_times_1 int 32 regular {array 8 { 0 } 0 1 } {global 1}  }
	{ p_ZL16post_spike_times_2 int 32 regular {array 8 { 0 } 0 1 } {global 1}  }
	{ p_ZL16post_spike_times_3 int 32 regular {array 8 { 0 } 0 1 } {global 1}  }
	{ p_ZL16post_spike_times_4 int 32 regular {array 8 { 0 } 0 1 } {global 1}  }
	{ p_ZL16post_spike_times_5 int 32 regular {array 8 { 0 } 0 1 } {global 1}  }
	{ p_ZL16post_spike_times_6 int 32 regular {array 8 { 0 } 0 1 } {global 1}  }
	{ p_ZL16post_spike_times_7 int 32 regular {array 8 { 0 } 0 1 } {global 1}  }
}
set hasAXIMCache 0
set l_AXIML2Cache [list]
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "p_ZL15pre_spike_times_0", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_spike_times_1", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_spike_times_2", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_spike_times_3", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_spike_times_4", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_spike_times_5", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_spike_times_6", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_spike_times_7", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL16post_spike_times_0", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL16post_spike_times_1", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL16post_spike_times_2", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL16post_spike_times_3", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL16post_spike_times_4", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL16post_spike_times_5", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL16post_spike_times_6", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "p_ZL16post_spike_times_7", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "extern" : 0} ]}
# RTL Port declarations: 
set portNum 70
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ p_ZL15pre_spike_times_0_address0 sc_out sc_lv 3 signal 0 } 
	{ p_ZL15pre_spike_times_0_ce0 sc_out sc_logic 1 signal 0 } 
	{ p_ZL15pre_spike_times_0_we0 sc_out sc_logic 1 signal 0 } 
	{ p_ZL15pre_spike_times_0_d0 sc_out sc_lv 32 signal 0 } 
	{ p_ZL15pre_spike_times_1_address0 sc_out sc_lv 3 signal 1 } 
	{ p_ZL15pre_spike_times_1_ce0 sc_out sc_logic 1 signal 1 } 
	{ p_ZL15pre_spike_times_1_we0 sc_out sc_logic 1 signal 1 } 
	{ p_ZL15pre_spike_times_1_d0 sc_out sc_lv 32 signal 1 } 
	{ p_ZL15pre_spike_times_2_address0 sc_out sc_lv 3 signal 2 } 
	{ p_ZL15pre_spike_times_2_ce0 sc_out sc_logic 1 signal 2 } 
	{ p_ZL15pre_spike_times_2_we0 sc_out sc_logic 1 signal 2 } 
	{ p_ZL15pre_spike_times_2_d0 sc_out sc_lv 32 signal 2 } 
	{ p_ZL15pre_spike_times_3_address0 sc_out sc_lv 3 signal 3 } 
	{ p_ZL15pre_spike_times_3_ce0 sc_out sc_logic 1 signal 3 } 
	{ p_ZL15pre_spike_times_3_we0 sc_out sc_logic 1 signal 3 } 
	{ p_ZL15pre_spike_times_3_d0 sc_out sc_lv 32 signal 3 } 
	{ p_ZL15pre_spike_times_4_address0 sc_out sc_lv 3 signal 4 } 
	{ p_ZL15pre_spike_times_4_ce0 sc_out sc_logic 1 signal 4 } 
	{ p_ZL15pre_spike_times_4_we0 sc_out sc_logic 1 signal 4 } 
	{ p_ZL15pre_spike_times_4_d0 sc_out sc_lv 32 signal 4 } 
	{ p_ZL15pre_spike_times_5_address0 sc_out sc_lv 3 signal 5 } 
	{ p_ZL15pre_spike_times_5_ce0 sc_out sc_logic 1 signal 5 } 
	{ p_ZL15pre_spike_times_5_we0 sc_out sc_logic 1 signal 5 } 
	{ p_ZL15pre_spike_times_5_d0 sc_out sc_lv 32 signal 5 } 
	{ p_ZL15pre_spike_times_6_address0 sc_out sc_lv 3 signal 6 } 
	{ p_ZL15pre_spike_times_6_ce0 sc_out sc_logic 1 signal 6 } 
	{ p_ZL15pre_spike_times_6_we0 sc_out sc_logic 1 signal 6 } 
	{ p_ZL15pre_spike_times_6_d0 sc_out sc_lv 32 signal 6 } 
	{ p_ZL15pre_spike_times_7_address0 sc_out sc_lv 3 signal 7 } 
	{ p_ZL15pre_spike_times_7_ce0 sc_out sc_logic 1 signal 7 } 
	{ p_ZL15pre_spike_times_7_we0 sc_out sc_logic 1 signal 7 } 
	{ p_ZL15pre_spike_times_7_d0 sc_out sc_lv 32 signal 7 } 
	{ p_ZL16post_spike_times_0_address0 sc_out sc_lv 3 signal 8 } 
	{ p_ZL16post_spike_times_0_ce0 sc_out sc_logic 1 signal 8 } 
	{ p_ZL16post_spike_times_0_we0 sc_out sc_logic 1 signal 8 } 
	{ p_ZL16post_spike_times_0_d0 sc_out sc_lv 32 signal 8 } 
	{ p_ZL16post_spike_times_1_address0 sc_out sc_lv 3 signal 9 } 
	{ p_ZL16post_spike_times_1_ce0 sc_out sc_logic 1 signal 9 } 
	{ p_ZL16post_spike_times_1_we0 sc_out sc_logic 1 signal 9 } 
	{ p_ZL16post_spike_times_1_d0 sc_out sc_lv 32 signal 9 } 
	{ p_ZL16post_spike_times_2_address0 sc_out sc_lv 3 signal 10 } 
	{ p_ZL16post_spike_times_2_ce0 sc_out sc_logic 1 signal 10 } 
	{ p_ZL16post_spike_times_2_we0 sc_out sc_logic 1 signal 10 } 
	{ p_ZL16post_spike_times_2_d0 sc_out sc_lv 32 signal 10 } 
	{ p_ZL16post_spike_times_3_address0 sc_out sc_lv 3 signal 11 } 
	{ p_ZL16post_spike_times_3_ce0 sc_out sc_logic 1 signal 11 } 
	{ p_ZL16post_spike_times_3_we0 sc_out sc_logic 1 signal 11 } 
	{ p_ZL16post_spike_times_3_d0 sc_out sc_lv 32 signal 11 } 
	{ p_ZL16post_spike_times_4_address0 sc_out sc_lv 3 signal 12 } 
	{ p_ZL16post_spike_times_4_ce0 sc_out sc_logic 1 signal 12 } 
	{ p_ZL16post_spike_times_4_we0 sc_out sc_logic 1 signal 12 } 
	{ p_ZL16post_spike_times_4_d0 sc_out sc_lv 32 signal 12 } 
	{ p_ZL16post_spike_times_5_address0 sc_out sc_lv 3 signal 13 } 
	{ p_ZL16post_spike_times_5_ce0 sc_out sc_logic 1 signal 13 } 
	{ p_ZL16post_spike_times_5_we0 sc_out sc_logic 1 signal 13 } 
	{ p_ZL16post_spike_times_5_d0 sc_out sc_lv 32 signal 13 } 
	{ p_ZL16post_spike_times_6_address0 sc_out sc_lv 3 signal 14 } 
	{ p_ZL16post_spike_times_6_ce0 sc_out sc_logic 1 signal 14 } 
	{ p_ZL16post_spike_times_6_we0 sc_out sc_logic 1 signal 14 } 
	{ p_ZL16post_spike_times_6_d0 sc_out sc_lv 32 signal 14 } 
	{ p_ZL16post_spike_times_7_address0 sc_out sc_lv 3 signal 15 } 
	{ p_ZL16post_spike_times_7_ce0 sc_out sc_logic 1 signal 15 } 
	{ p_ZL16post_spike_times_7_we0 sc_out sc_logic 1 signal 15 } 
	{ p_ZL16post_spike_times_7_d0 sc_out sc_lv 32 signal 15 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "p_ZL15pre_spike_times_0_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_0", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_spike_times_0_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_0", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_spike_times_0_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_0", "role": "we0" }} , 
 	{ "name": "p_ZL15pre_spike_times_0_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_0", "role": "d0" }} , 
 	{ "name": "p_ZL15pre_spike_times_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_1", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_spike_times_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_1", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_spike_times_1_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_1", "role": "we0" }} , 
 	{ "name": "p_ZL15pre_spike_times_1_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_1", "role": "d0" }} , 
 	{ "name": "p_ZL15pre_spike_times_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_2", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_spike_times_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_2", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_spike_times_2_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_2", "role": "we0" }} , 
 	{ "name": "p_ZL15pre_spike_times_2_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_2", "role": "d0" }} , 
 	{ "name": "p_ZL15pre_spike_times_3_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_3", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_spike_times_3_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_3", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_spike_times_3_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_3", "role": "we0" }} , 
 	{ "name": "p_ZL15pre_spike_times_3_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_3", "role": "d0" }} , 
 	{ "name": "p_ZL15pre_spike_times_4_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_4", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_spike_times_4_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_4", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_spike_times_4_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_4", "role": "we0" }} , 
 	{ "name": "p_ZL15pre_spike_times_4_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_4", "role": "d0" }} , 
 	{ "name": "p_ZL15pre_spike_times_5_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_5", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_spike_times_5_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_5", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_spike_times_5_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_5", "role": "we0" }} , 
 	{ "name": "p_ZL15pre_spike_times_5_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_5", "role": "d0" }} , 
 	{ "name": "p_ZL15pre_spike_times_6_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_6", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_spike_times_6_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_6", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_spike_times_6_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_6", "role": "we0" }} , 
 	{ "name": "p_ZL15pre_spike_times_6_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_6", "role": "d0" }} , 
 	{ "name": "p_ZL15pre_spike_times_7_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_7", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_spike_times_7_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_7", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_spike_times_7_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_7", "role": "we0" }} , 
 	{ "name": "p_ZL15pre_spike_times_7_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL15pre_spike_times_7", "role": "d0" }} , 
 	{ "name": "p_ZL16post_spike_times_0_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_0", "role": "address0" }} , 
 	{ "name": "p_ZL16post_spike_times_0_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_0", "role": "ce0" }} , 
 	{ "name": "p_ZL16post_spike_times_0_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_0", "role": "we0" }} , 
 	{ "name": "p_ZL16post_spike_times_0_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_0", "role": "d0" }} , 
 	{ "name": "p_ZL16post_spike_times_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_1", "role": "address0" }} , 
 	{ "name": "p_ZL16post_spike_times_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_1", "role": "ce0" }} , 
 	{ "name": "p_ZL16post_spike_times_1_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_1", "role": "we0" }} , 
 	{ "name": "p_ZL16post_spike_times_1_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_1", "role": "d0" }} , 
 	{ "name": "p_ZL16post_spike_times_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_2", "role": "address0" }} , 
 	{ "name": "p_ZL16post_spike_times_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_2", "role": "ce0" }} , 
 	{ "name": "p_ZL16post_spike_times_2_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_2", "role": "we0" }} , 
 	{ "name": "p_ZL16post_spike_times_2_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_2", "role": "d0" }} , 
 	{ "name": "p_ZL16post_spike_times_3_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_3", "role": "address0" }} , 
 	{ "name": "p_ZL16post_spike_times_3_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_3", "role": "ce0" }} , 
 	{ "name": "p_ZL16post_spike_times_3_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_3", "role": "we0" }} , 
 	{ "name": "p_ZL16post_spike_times_3_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_3", "role": "d0" }} , 
 	{ "name": "p_ZL16post_spike_times_4_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_4", "role": "address0" }} , 
 	{ "name": "p_ZL16post_spike_times_4_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_4", "role": "ce0" }} , 
 	{ "name": "p_ZL16post_spike_times_4_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_4", "role": "we0" }} , 
 	{ "name": "p_ZL16post_spike_times_4_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_4", "role": "d0" }} , 
 	{ "name": "p_ZL16post_spike_times_5_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_5", "role": "address0" }} , 
 	{ "name": "p_ZL16post_spike_times_5_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_5", "role": "ce0" }} , 
 	{ "name": "p_ZL16post_spike_times_5_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_5", "role": "we0" }} , 
 	{ "name": "p_ZL16post_spike_times_5_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_5", "role": "d0" }} , 
 	{ "name": "p_ZL16post_spike_times_6_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_6", "role": "address0" }} , 
 	{ "name": "p_ZL16post_spike_times_6_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_6", "role": "ce0" }} , 
 	{ "name": "p_ZL16post_spike_times_6_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_6", "role": "we0" }} , 
 	{ "name": "p_ZL16post_spike_times_6_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_6", "role": "d0" }} , 
 	{ "name": "p_ZL16post_spike_times_7_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_7", "role": "address0" }} , 
 	{ "name": "p_ZL16post_spike_times_7_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_7", "role": "ce0" }} , 
 	{ "name": "p_ZL16post_spike_times_7_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_7", "role": "we0" }} , 
 	{ "name": "p_ZL16post_spike_times_7_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_ZL16post_spike_times_7", "role": "d0" }}  ]}

set ArgLastReadFirstWriteLatency {
	snn_top_hls_Pipeline_RESET_PRE {
		p_ZL15pre_spike_times_0 {Type O LastRead -1 FirstWrite 0}
		p_ZL15pre_spike_times_1 {Type O LastRead -1 FirstWrite 0}
		p_ZL15pre_spike_times_2 {Type O LastRead -1 FirstWrite 0}
		p_ZL15pre_spike_times_3 {Type O LastRead -1 FirstWrite 0}
		p_ZL15pre_spike_times_4 {Type O LastRead -1 FirstWrite 0}
		p_ZL15pre_spike_times_5 {Type O LastRead -1 FirstWrite 0}
		p_ZL15pre_spike_times_6 {Type O LastRead -1 FirstWrite 0}
		p_ZL15pre_spike_times_7 {Type O LastRead -1 FirstWrite 0}
		p_ZL16post_spike_times_0 {Type O LastRead -1 FirstWrite 0}
		p_ZL16post_spike_times_1 {Type O LastRead -1 FirstWrite 0}
		p_ZL16post_spike_times_2 {Type O LastRead -1 FirstWrite 0}
		p_ZL16post_spike_times_3 {Type O LastRead -1 FirstWrite 0}
		p_ZL16post_spike_times_4 {Type O LastRead -1 FirstWrite 0}
		p_ZL16post_spike_times_5 {Type O LastRead -1 FirstWrite 0}
		p_ZL16post_spike_times_6 {Type O LastRead -1 FirstWrite 0}
		p_ZL16post_spike_times_7 {Type O LastRead -1 FirstWrite 0}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "10", "Max" : "10"}
	, {"Name" : "Interval", "Min" : "9", "Max" : "9"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	p_ZL15pre_spike_times_0 { ap_memory {  { p_ZL15pre_spike_times_0_address0 mem_address 1 3 }  { p_ZL15pre_spike_times_0_ce0 mem_ce 1 1 }  { p_ZL15pre_spike_times_0_we0 mem_we 1 1 }  { p_ZL15pre_spike_times_0_d0 mem_din 1 32 } } }
	p_ZL15pre_spike_times_1 { ap_memory {  { p_ZL15pre_spike_times_1_address0 mem_address 1 3 }  { p_ZL15pre_spike_times_1_ce0 mem_ce 1 1 }  { p_ZL15pre_spike_times_1_we0 mem_we 1 1 }  { p_ZL15pre_spike_times_1_d0 mem_din 1 32 } } }
	p_ZL15pre_spike_times_2 { ap_memory {  { p_ZL15pre_spike_times_2_address0 mem_address 1 3 }  { p_ZL15pre_spike_times_2_ce0 mem_ce 1 1 }  { p_ZL15pre_spike_times_2_we0 mem_we 1 1 }  { p_ZL15pre_spike_times_2_d0 mem_din 1 32 } } }
	p_ZL15pre_spike_times_3 { ap_memory {  { p_ZL15pre_spike_times_3_address0 mem_address 1 3 }  { p_ZL15pre_spike_times_3_ce0 mem_ce 1 1 }  { p_ZL15pre_spike_times_3_we0 mem_we 1 1 }  { p_ZL15pre_spike_times_3_d0 mem_din 1 32 } } }
	p_ZL15pre_spike_times_4 { ap_memory {  { p_ZL15pre_spike_times_4_address0 mem_address 1 3 }  { p_ZL15pre_spike_times_4_ce0 mem_ce 1 1 }  { p_ZL15pre_spike_times_4_we0 mem_we 1 1 }  { p_ZL15pre_spike_times_4_d0 mem_din 1 32 } } }
	p_ZL15pre_spike_times_5 { ap_memory {  { p_ZL15pre_spike_times_5_address0 mem_address 1 3 }  { p_ZL15pre_spike_times_5_ce0 mem_ce 1 1 }  { p_ZL15pre_spike_times_5_we0 mem_we 1 1 }  { p_ZL15pre_spike_times_5_d0 mem_din 1 32 } } }
	p_ZL15pre_spike_times_6 { ap_memory {  { p_ZL15pre_spike_times_6_address0 mem_address 1 3 }  { p_ZL15pre_spike_times_6_ce0 mem_ce 1 1 }  { p_ZL15pre_spike_times_6_we0 mem_we 1 1 }  { p_ZL15pre_spike_times_6_d0 mem_din 1 32 } } }
	p_ZL15pre_spike_times_7 { ap_memory {  { p_ZL15pre_spike_times_7_address0 mem_address 1 3 }  { p_ZL15pre_spike_times_7_ce0 mem_ce 1 1 }  { p_ZL15pre_spike_times_7_we0 mem_we 1 1 }  { p_ZL15pre_spike_times_7_d0 mem_din 1 32 } } }
	p_ZL16post_spike_times_0 { ap_memory {  { p_ZL16post_spike_times_0_address0 mem_address 1 3 }  { p_ZL16post_spike_times_0_ce0 mem_ce 1 1 }  { p_ZL16post_spike_times_0_we0 mem_we 1 1 }  { p_ZL16post_spike_times_0_d0 mem_din 1 32 } } }
	p_ZL16post_spike_times_1 { ap_memory {  { p_ZL16post_spike_times_1_address0 mem_address 1 3 }  { p_ZL16post_spike_times_1_ce0 mem_ce 1 1 }  { p_ZL16post_spike_times_1_we0 mem_we 1 1 }  { p_ZL16post_spike_times_1_d0 mem_din 1 32 } } }
	p_ZL16post_spike_times_2 { ap_memory {  { p_ZL16post_spike_times_2_address0 mem_address 1 3 }  { p_ZL16post_spike_times_2_ce0 mem_ce 1 1 }  { p_ZL16post_spike_times_2_we0 mem_we 1 1 }  { p_ZL16post_spike_times_2_d0 mem_din 1 32 } } }
	p_ZL16post_spike_times_3 { ap_memory {  { p_ZL16post_spike_times_3_address0 mem_address 1 3 }  { p_ZL16post_spike_times_3_ce0 mem_ce 1 1 }  { p_ZL16post_spike_times_3_we0 mem_we 1 1 }  { p_ZL16post_spike_times_3_d0 mem_din 1 32 } } }
	p_ZL16post_spike_times_4 { ap_memory {  { p_ZL16post_spike_times_4_address0 mem_address 1 3 }  { p_ZL16post_spike_times_4_ce0 mem_ce 1 1 }  { p_ZL16post_spike_times_4_we0 mem_we 1 1 }  { p_ZL16post_spike_times_4_d0 mem_din 1 32 } } }
	p_ZL16post_spike_times_5 { ap_memory {  { p_ZL16post_spike_times_5_address0 mem_address 1 3 }  { p_ZL16post_spike_times_5_ce0 mem_ce 1 1 }  { p_ZL16post_spike_times_5_we0 mem_we 1 1 }  { p_ZL16post_spike_times_5_d0 mem_din 1 32 } } }
	p_ZL16post_spike_times_6 { ap_memory {  { p_ZL16post_spike_times_6_address0 mem_address 1 3 }  { p_ZL16post_spike_times_6_ce0 mem_ce 1 1 }  { p_ZL16post_spike_times_6_we0 mem_we 1 1 }  { p_ZL16post_spike_times_6_d0 mem_din 1 32 } } }
	p_ZL16post_spike_times_7 { ap_memory {  { p_ZL16post_spike_times_7_address0 mem_address 1 3 }  { p_ZL16post_spike_times_7_ce0 mem_ce 1 1 }  { p_ZL16post_spike_times_7_we0 mem_we 1 1 }  { p_ZL16post_spike_times_7_d0 mem_din 1 32 } } }
}
