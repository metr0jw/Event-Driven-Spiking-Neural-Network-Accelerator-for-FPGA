set moduleName decay_eligibility_traces
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set isPipelined_legacy 0
set pipeline_type none
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
set C_modelName {decay_eligibility_traces}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict p_ZL15pre_eligibility_0 { MEM_WIDTH 8 MEM_SIZE 8 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL15pre_eligibility_1 { MEM_WIDTH 8 MEM_SIZE 8 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL15pre_eligibility_2 { MEM_WIDTH 8 MEM_SIZE 8 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL15pre_eligibility_3 { MEM_WIDTH 8 MEM_SIZE 8 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL15pre_eligibility_4 { MEM_WIDTH 8 MEM_SIZE 8 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL15pre_eligibility_5 { MEM_WIDTH 8 MEM_SIZE 8 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL15pre_eligibility_6 { MEM_WIDTH 8 MEM_SIZE 8 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL15pre_eligibility_7 { MEM_WIDTH 8 MEM_SIZE 8 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL16post_eligibility_0 { MEM_WIDTH 8 MEM_SIZE 8 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL16post_eligibility_1 { MEM_WIDTH 8 MEM_SIZE 8 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL16post_eligibility_2 { MEM_WIDTH 8 MEM_SIZE 8 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL16post_eligibility_3 { MEM_WIDTH 8 MEM_SIZE 8 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL16post_eligibility_4 { MEM_WIDTH 8 MEM_SIZE 8 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL16post_eligibility_5 { MEM_WIDTH 8 MEM_SIZE 8 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL16post_eligibility_6 { MEM_WIDTH 8 MEM_SIZE 8 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict p_ZL16post_eligibility_7 { MEM_WIDTH 8 MEM_SIZE 8 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
set C_modelArgList {
	{ p_ZL15pre_eligibility_0 int 8 regular {array 8 { 1 0 } 1 1 } {global 2}  }
	{ p_ZL15pre_eligibility_1 int 8 regular {array 8 { 1 0 } 1 1 } {global 2}  }
	{ p_ZL15pre_eligibility_2 int 8 regular {array 8 { 1 0 } 1 1 } {global 2}  }
	{ p_ZL15pre_eligibility_3 int 8 regular {array 8 { 1 0 } 1 1 } {global 2}  }
	{ p_ZL15pre_eligibility_4 int 8 regular {array 8 { 1 0 } 1 1 } {global 2}  }
	{ p_ZL15pre_eligibility_5 int 8 regular {array 8 { 1 0 } 1 1 } {global 2}  }
	{ p_ZL15pre_eligibility_6 int 8 regular {array 8 { 1 0 } 1 1 } {global 2}  }
	{ p_ZL15pre_eligibility_7 int 8 regular {array 8 { 1 0 } 1 1 } {global 2}  }
	{ p_ZL16post_eligibility_0 int 8 regular {array 8 { 1 0 } 1 1 } {global 2}  }
	{ p_ZL16post_eligibility_1 int 8 regular {array 8 { 1 0 } 1 1 } {global 2}  }
	{ p_ZL16post_eligibility_2 int 8 regular {array 8 { 1 0 } 1 1 } {global 2}  }
	{ p_ZL16post_eligibility_3 int 8 regular {array 8 { 1 0 } 1 1 } {global 2}  }
	{ p_ZL16post_eligibility_4 int 8 regular {array 8 { 1 0 } 1 1 } {global 2}  }
	{ p_ZL16post_eligibility_5 int 8 regular {array 8 { 1 0 } 1 1 } {global 2}  }
	{ p_ZL16post_eligibility_6 int 8 regular {array 8 { 1 0 } 1 1 } {global 2}  }
	{ p_ZL16post_eligibility_7 int 8 regular {array 8 { 1 0 } 1 1 } {global 2}  }
}
set hasAXIMCache 0
set l_AXIML2Cache [list]
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "p_ZL15pre_eligibility_0", "interface" : "memory", "bitwidth" : 8, "direction" : "READWRITE", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_eligibility_1", "interface" : "memory", "bitwidth" : 8, "direction" : "READWRITE", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_eligibility_2", "interface" : "memory", "bitwidth" : 8, "direction" : "READWRITE", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_eligibility_3", "interface" : "memory", "bitwidth" : 8, "direction" : "READWRITE", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_eligibility_4", "interface" : "memory", "bitwidth" : 8, "direction" : "READWRITE", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_eligibility_5", "interface" : "memory", "bitwidth" : 8, "direction" : "READWRITE", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_eligibility_6", "interface" : "memory", "bitwidth" : 8, "direction" : "READWRITE", "extern" : 0} , 
 	{ "Name" : "p_ZL15pre_eligibility_7", "interface" : "memory", "bitwidth" : 8, "direction" : "READWRITE", "extern" : 0} , 
 	{ "Name" : "p_ZL16post_eligibility_0", "interface" : "memory", "bitwidth" : 8, "direction" : "READWRITE", "extern" : 0} , 
 	{ "Name" : "p_ZL16post_eligibility_1", "interface" : "memory", "bitwidth" : 8, "direction" : "READWRITE", "extern" : 0} , 
 	{ "Name" : "p_ZL16post_eligibility_2", "interface" : "memory", "bitwidth" : 8, "direction" : "READWRITE", "extern" : 0} , 
 	{ "Name" : "p_ZL16post_eligibility_3", "interface" : "memory", "bitwidth" : 8, "direction" : "READWRITE", "extern" : 0} , 
 	{ "Name" : "p_ZL16post_eligibility_4", "interface" : "memory", "bitwidth" : 8, "direction" : "READWRITE", "extern" : 0} , 
 	{ "Name" : "p_ZL16post_eligibility_5", "interface" : "memory", "bitwidth" : 8, "direction" : "READWRITE", "extern" : 0} , 
 	{ "Name" : "p_ZL16post_eligibility_6", "interface" : "memory", "bitwidth" : 8, "direction" : "READWRITE", "extern" : 0} , 
 	{ "Name" : "p_ZL16post_eligibility_7", "interface" : "memory", "bitwidth" : 8, "direction" : "READWRITE", "extern" : 0} ]}
# RTL Port declarations: 
set portNum 118
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ p_ZL15pre_eligibility_0_address0 sc_out sc_lv 3 signal 0 } 
	{ p_ZL15pre_eligibility_0_ce0 sc_out sc_logic 1 signal 0 } 
	{ p_ZL15pre_eligibility_0_q0 sc_in sc_lv 8 signal 0 } 
	{ p_ZL15pre_eligibility_0_address1 sc_out sc_lv 3 signal 0 } 
	{ p_ZL15pre_eligibility_0_ce1 sc_out sc_logic 1 signal 0 } 
	{ p_ZL15pre_eligibility_0_we1 sc_out sc_logic 1 signal 0 } 
	{ p_ZL15pre_eligibility_0_d1 sc_out sc_lv 8 signal 0 } 
	{ p_ZL15pre_eligibility_1_address0 sc_out sc_lv 3 signal 1 } 
	{ p_ZL15pre_eligibility_1_ce0 sc_out sc_logic 1 signal 1 } 
	{ p_ZL15pre_eligibility_1_q0 sc_in sc_lv 8 signal 1 } 
	{ p_ZL15pre_eligibility_1_address1 sc_out sc_lv 3 signal 1 } 
	{ p_ZL15pre_eligibility_1_ce1 sc_out sc_logic 1 signal 1 } 
	{ p_ZL15pre_eligibility_1_we1 sc_out sc_logic 1 signal 1 } 
	{ p_ZL15pre_eligibility_1_d1 sc_out sc_lv 8 signal 1 } 
	{ p_ZL15pre_eligibility_2_address0 sc_out sc_lv 3 signal 2 } 
	{ p_ZL15pre_eligibility_2_ce0 sc_out sc_logic 1 signal 2 } 
	{ p_ZL15pre_eligibility_2_q0 sc_in sc_lv 8 signal 2 } 
	{ p_ZL15pre_eligibility_2_address1 sc_out sc_lv 3 signal 2 } 
	{ p_ZL15pre_eligibility_2_ce1 sc_out sc_logic 1 signal 2 } 
	{ p_ZL15pre_eligibility_2_we1 sc_out sc_logic 1 signal 2 } 
	{ p_ZL15pre_eligibility_2_d1 sc_out sc_lv 8 signal 2 } 
	{ p_ZL15pre_eligibility_3_address0 sc_out sc_lv 3 signal 3 } 
	{ p_ZL15pre_eligibility_3_ce0 sc_out sc_logic 1 signal 3 } 
	{ p_ZL15pre_eligibility_3_q0 sc_in sc_lv 8 signal 3 } 
	{ p_ZL15pre_eligibility_3_address1 sc_out sc_lv 3 signal 3 } 
	{ p_ZL15pre_eligibility_3_ce1 sc_out sc_logic 1 signal 3 } 
	{ p_ZL15pre_eligibility_3_we1 sc_out sc_logic 1 signal 3 } 
	{ p_ZL15pre_eligibility_3_d1 sc_out sc_lv 8 signal 3 } 
	{ p_ZL15pre_eligibility_4_address0 sc_out sc_lv 3 signal 4 } 
	{ p_ZL15pre_eligibility_4_ce0 sc_out sc_logic 1 signal 4 } 
	{ p_ZL15pre_eligibility_4_q0 sc_in sc_lv 8 signal 4 } 
	{ p_ZL15pre_eligibility_4_address1 sc_out sc_lv 3 signal 4 } 
	{ p_ZL15pre_eligibility_4_ce1 sc_out sc_logic 1 signal 4 } 
	{ p_ZL15pre_eligibility_4_we1 sc_out sc_logic 1 signal 4 } 
	{ p_ZL15pre_eligibility_4_d1 sc_out sc_lv 8 signal 4 } 
	{ p_ZL15pre_eligibility_5_address0 sc_out sc_lv 3 signal 5 } 
	{ p_ZL15pre_eligibility_5_ce0 sc_out sc_logic 1 signal 5 } 
	{ p_ZL15pre_eligibility_5_q0 sc_in sc_lv 8 signal 5 } 
	{ p_ZL15pre_eligibility_5_address1 sc_out sc_lv 3 signal 5 } 
	{ p_ZL15pre_eligibility_5_ce1 sc_out sc_logic 1 signal 5 } 
	{ p_ZL15pre_eligibility_5_we1 sc_out sc_logic 1 signal 5 } 
	{ p_ZL15pre_eligibility_5_d1 sc_out sc_lv 8 signal 5 } 
	{ p_ZL15pre_eligibility_6_address0 sc_out sc_lv 3 signal 6 } 
	{ p_ZL15pre_eligibility_6_ce0 sc_out sc_logic 1 signal 6 } 
	{ p_ZL15pre_eligibility_6_q0 sc_in sc_lv 8 signal 6 } 
	{ p_ZL15pre_eligibility_6_address1 sc_out sc_lv 3 signal 6 } 
	{ p_ZL15pre_eligibility_6_ce1 sc_out sc_logic 1 signal 6 } 
	{ p_ZL15pre_eligibility_6_we1 sc_out sc_logic 1 signal 6 } 
	{ p_ZL15pre_eligibility_6_d1 sc_out sc_lv 8 signal 6 } 
	{ p_ZL15pre_eligibility_7_address0 sc_out sc_lv 3 signal 7 } 
	{ p_ZL15pre_eligibility_7_ce0 sc_out sc_logic 1 signal 7 } 
	{ p_ZL15pre_eligibility_7_q0 sc_in sc_lv 8 signal 7 } 
	{ p_ZL15pre_eligibility_7_address1 sc_out sc_lv 3 signal 7 } 
	{ p_ZL15pre_eligibility_7_ce1 sc_out sc_logic 1 signal 7 } 
	{ p_ZL15pre_eligibility_7_we1 sc_out sc_logic 1 signal 7 } 
	{ p_ZL15pre_eligibility_7_d1 sc_out sc_lv 8 signal 7 } 
	{ p_ZL16post_eligibility_0_address0 sc_out sc_lv 3 signal 8 } 
	{ p_ZL16post_eligibility_0_ce0 sc_out sc_logic 1 signal 8 } 
	{ p_ZL16post_eligibility_0_q0 sc_in sc_lv 8 signal 8 } 
	{ p_ZL16post_eligibility_0_address1 sc_out sc_lv 3 signal 8 } 
	{ p_ZL16post_eligibility_0_ce1 sc_out sc_logic 1 signal 8 } 
	{ p_ZL16post_eligibility_0_we1 sc_out sc_logic 1 signal 8 } 
	{ p_ZL16post_eligibility_0_d1 sc_out sc_lv 8 signal 8 } 
	{ p_ZL16post_eligibility_1_address0 sc_out sc_lv 3 signal 9 } 
	{ p_ZL16post_eligibility_1_ce0 sc_out sc_logic 1 signal 9 } 
	{ p_ZL16post_eligibility_1_q0 sc_in sc_lv 8 signal 9 } 
	{ p_ZL16post_eligibility_1_address1 sc_out sc_lv 3 signal 9 } 
	{ p_ZL16post_eligibility_1_ce1 sc_out sc_logic 1 signal 9 } 
	{ p_ZL16post_eligibility_1_we1 sc_out sc_logic 1 signal 9 } 
	{ p_ZL16post_eligibility_1_d1 sc_out sc_lv 8 signal 9 } 
	{ p_ZL16post_eligibility_2_address0 sc_out sc_lv 3 signal 10 } 
	{ p_ZL16post_eligibility_2_ce0 sc_out sc_logic 1 signal 10 } 
	{ p_ZL16post_eligibility_2_q0 sc_in sc_lv 8 signal 10 } 
	{ p_ZL16post_eligibility_2_address1 sc_out sc_lv 3 signal 10 } 
	{ p_ZL16post_eligibility_2_ce1 sc_out sc_logic 1 signal 10 } 
	{ p_ZL16post_eligibility_2_we1 sc_out sc_logic 1 signal 10 } 
	{ p_ZL16post_eligibility_2_d1 sc_out sc_lv 8 signal 10 } 
	{ p_ZL16post_eligibility_3_address0 sc_out sc_lv 3 signal 11 } 
	{ p_ZL16post_eligibility_3_ce0 sc_out sc_logic 1 signal 11 } 
	{ p_ZL16post_eligibility_3_q0 sc_in sc_lv 8 signal 11 } 
	{ p_ZL16post_eligibility_3_address1 sc_out sc_lv 3 signal 11 } 
	{ p_ZL16post_eligibility_3_ce1 sc_out sc_logic 1 signal 11 } 
	{ p_ZL16post_eligibility_3_we1 sc_out sc_logic 1 signal 11 } 
	{ p_ZL16post_eligibility_3_d1 sc_out sc_lv 8 signal 11 } 
	{ p_ZL16post_eligibility_4_address0 sc_out sc_lv 3 signal 12 } 
	{ p_ZL16post_eligibility_4_ce0 sc_out sc_logic 1 signal 12 } 
	{ p_ZL16post_eligibility_4_q0 sc_in sc_lv 8 signal 12 } 
	{ p_ZL16post_eligibility_4_address1 sc_out sc_lv 3 signal 12 } 
	{ p_ZL16post_eligibility_4_ce1 sc_out sc_logic 1 signal 12 } 
	{ p_ZL16post_eligibility_4_we1 sc_out sc_logic 1 signal 12 } 
	{ p_ZL16post_eligibility_4_d1 sc_out sc_lv 8 signal 12 } 
	{ p_ZL16post_eligibility_5_address0 sc_out sc_lv 3 signal 13 } 
	{ p_ZL16post_eligibility_5_ce0 sc_out sc_logic 1 signal 13 } 
	{ p_ZL16post_eligibility_5_q0 sc_in sc_lv 8 signal 13 } 
	{ p_ZL16post_eligibility_5_address1 sc_out sc_lv 3 signal 13 } 
	{ p_ZL16post_eligibility_5_ce1 sc_out sc_logic 1 signal 13 } 
	{ p_ZL16post_eligibility_5_we1 sc_out sc_logic 1 signal 13 } 
	{ p_ZL16post_eligibility_5_d1 sc_out sc_lv 8 signal 13 } 
	{ p_ZL16post_eligibility_6_address0 sc_out sc_lv 3 signal 14 } 
	{ p_ZL16post_eligibility_6_ce0 sc_out sc_logic 1 signal 14 } 
	{ p_ZL16post_eligibility_6_q0 sc_in sc_lv 8 signal 14 } 
	{ p_ZL16post_eligibility_6_address1 sc_out sc_lv 3 signal 14 } 
	{ p_ZL16post_eligibility_6_ce1 sc_out sc_logic 1 signal 14 } 
	{ p_ZL16post_eligibility_6_we1 sc_out sc_logic 1 signal 14 } 
	{ p_ZL16post_eligibility_6_d1 sc_out sc_lv 8 signal 14 } 
	{ p_ZL16post_eligibility_7_address0 sc_out sc_lv 3 signal 15 } 
	{ p_ZL16post_eligibility_7_ce0 sc_out sc_logic 1 signal 15 } 
	{ p_ZL16post_eligibility_7_q0 sc_in sc_lv 8 signal 15 } 
	{ p_ZL16post_eligibility_7_address1 sc_out sc_lv 3 signal 15 } 
	{ p_ZL16post_eligibility_7_ce1 sc_out sc_logic 1 signal 15 } 
	{ p_ZL16post_eligibility_7_we1 sc_out sc_logic 1 signal 15 } 
	{ p_ZL16post_eligibility_7_d1 sc_out sc_lv 8 signal 15 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "p_ZL15pre_eligibility_0_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_0", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_eligibility_0_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_0", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_eligibility_0_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_0", "role": "q0" }} , 
 	{ "name": "p_ZL15pre_eligibility_0_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_0", "role": "address1" }} , 
 	{ "name": "p_ZL15pre_eligibility_0_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_0", "role": "ce1" }} , 
 	{ "name": "p_ZL15pre_eligibility_0_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_0", "role": "we1" }} , 
 	{ "name": "p_ZL15pre_eligibility_0_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_0", "role": "d1" }} , 
 	{ "name": "p_ZL15pre_eligibility_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_1", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_eligibility_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_1", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_eligibility_1_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_1", "role": "q0" }} , 
 	{ "name": "p_ZL15pre_eligibility_1_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_1", "role": "address1" }} , 
 	{ "name": "p_ZL15pre_eligibility_1_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_1", "role": "ce1" }} , 
 	{ "name": "p_ZL15pre_eligibility_1_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_1", "role": "we1" }} , 
 	{ "name": "p_ZL15pre_eligibility_1_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_1", "role": "d1" }} , 
 	{ "name": "p_ZL15pre_eligibility_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_2", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_eligibility_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_2", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_eligibility_2_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_2", "role": "q0" }} , 
 	{ "name": "p_ZL15pre_eligibility_2_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_2", "role": "address1" }} , 
 	{ "name": "p_ZL15pre_eligibility_2_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_2", "role": "ce1" }} , 
 	{ "name": "p_ZL15pre_eligibility_2_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_2", "role": "we1" }} , 
 	{ "name": "p_ZL15pre_eligibility_2_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_2", "role": "d1" }} , 
 	{ "name": "p_ZL15pre_eligibility_3_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_3", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_eligibility_3_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_3", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_eligibility_3_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_3", "role": "q0" }} , 
 	{ "name": "p_ZL15pre_eligibility_3_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_3", "role": "address1" }} , 
 	{ "name": "p_ZL15pre_eligibility_3_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_3", "role": "ce1" }} , 
 	{ "name": "p_ZL15pre_eligibility_3_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_3", "role": "we1" }} , 
 	{ "name": "p_ZL15pre_eligibility_3_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_3", "role": "d1" }} , 
 	{ "name": "p_ZL15pre_eligibility_4_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_4", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_eligibility_4_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_4", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_eligibility_4_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_4", "role": "q0" }} , 
 	{ "name": "p_ZL15pre_eligibility_4_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_4", "role": "address1" }} , 
 	{ "name": "p_ZL15pre_eligibility_4_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_4", "role": "ce1" }} , 
 	{ "name": "p_ZL15pre_eligibility_4_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_4", "role": "we1" }} , 
 	{ "name": "p_ZL15pre_eligibility_4_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_4", "role": "d1" }} , 
 	{ "name": "p_ZL15pre_eligibility_5_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_5", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_eligibility_5_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_5", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_eligibility_5_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_5", "role": "q0" }} , 
 	{ "name": "p_ZL15pre_eligibility_5_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_5", "role": "address1" }} , 
 	{ "name": "p_ZL15pre_eligibility_5_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_5", "role": "ce1" }} , 
 	{ "name": "p_ZL15pre_eligibility_5_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_5", "role": "we1" }} , 
 	{ "name": "p_ZL15pre_eligibility_5_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_5", "role": "d1" }} , 
 	{ "name": "p_ZL15pre_eligibility_6_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_6", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_eligibility_6_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_6", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_eligibility_6_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_6", "role": "q0" }} , 
 	{ "name": "p_ZL15pre_eligibility_6_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_6", "role": "address1" }} , 
 	{ "name": "p_ZL15pre_eligibility_6_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_6", "role": "ce1" }} , 
 	{ "name": "p_ZL15pre_eligibility_6_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_6", "role": "we1" }} , 
 	{ "name": "p_ZL15pre_eligibility_6_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_6", "role": "d1" }} , 
 	{ "name": "p_ZL15pre_eligibility_7_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_7", "role": "address0" }} , 
 	{ "name": "p_ZL15pre_eligibility_7_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_7", "role": "ce0" }} , 
 	{ "name": "p_ZL15pre_eligibility_7_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_7", "role": "q0" }} , 
 	{ "name": "p_ZL15pre_eligibility_7_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_7", "role": "address1" }} , 
 	{ "name": "p_ZL15pre_eligibility_7_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_7", "role": "ce1" }} , 
 	{ "name": "p_ZL15pre_eligibility_7_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_7", "role": "we1" }} , 
 	{ "name": "p_ZL15pre_eligibility_7_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL15pre_eligibility_7", "role": "d1" }} , 
 	{ "name": "p_ZL16post_eligibility_0_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_0", "role": "address0" }} , 
 	{ "name": "p_ZL16post_eligibility_0_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_0", "role": "ce0" }} , 
 	{ "name": "p_ZL16post_eligibility_0_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_0", "role": "q0" }} , 
 	{ "name": "p_ZL16post_eligibility_0_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_0", "role": "address1" }} , 
 	{ "name": "p_ZL16post_eligibility_0_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_0", "role": "ce1" }} , 
 	{ "name": "p_ZL16post_eligibility_0_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_0", "role": "we1" }} , 
 	{ "name": "p_ZL16post_eligibility_0_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_0", "role": "d1" }} , 
 	{ "name": "p_ZL16post_eligibility_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_1", "role": "address0" }} , 
 	{ "name": "p_ZL16post_eligibility_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_1", "role": "ce0" }} , 
 	{ "name": "p_ZL16post_eligibility_1_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_1", "role": "q0" }} , 
 	{ "name": "p_ZL16post_eligibility_1_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_1", "role": "address1" }} , 
 	{ "name": "p_ZL16post_eligibility_1_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_1", "role": "ce1" }} , 
 	{ "name": "p_ZL16post_eligibility_1_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_1", "role": "we1" }} , 
 	{ "name": "p_ZL16post_eligibility_1_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_1", "role": "d1" }} , 
 	{ "name": "p_ZL16post_eligibility_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_2", "role": "address0" }} , 
 	{ "name": "p_ZL16post_eligibility_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_2", "role": "ce0" }} , 
 	{ "name": "p_ZL16post_eligibility_2_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_2", "role": "q0" }} , 
 	{ "name": "p_ZL16post_eligibility_2_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_2", "role": "address1" }} , 
 	{ "name": "p_ZL16post_eligibility_2_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_2", "role": "ce1" }} , 
 	{ "name": "p_ZL16post_eligibility_2_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_2", "role": "we1" }} , 
 	{ "name": "p_ZL16post_eligibility_2_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_2", "role": "d1" }} , 
 	{ "name": "p_ZL16post_eligibility_3_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_3", "role": "address0" }} , 
 	{ "name": "p_ZL16post_eligibility_3_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_3", "role": "ce0" }} , 
 	{ "name": "p_ZL16post_eligibility_3_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_3", "role": "q0" }} , 
 	{ "name": "p_ZL16post_eligibility_3_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_3", "role": "address1" }} , 
 	{ "name": "p_ZL16post_eligibility_3_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_3", "role": "ce1" }} , 
 	{ "name": "p_ZL16post_eligibility_3_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_3", "role": "we1" }} , 
 	{ "name": "p_ZL16post_eligibility_3_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_3", "role": "d1" }} , 
 	{ "name": "p_ZL16post_eligibility_4_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_4", "role": "address0" }} , 
 	{ "name": "p_ZL16post_eligibility_4_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_4", "role": "ce0" }} , 
 	{ "name": "p_ZL16post_eligibility_4_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_4", "role": "q0" }} , 
 	{ "name": "p_ZL16post_eligibility_4_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_4", "role": "address1" }} , 
 	{ "name": "p_ZL16post_eligibility_4_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_4", "role": "ce1" }} , 
 	{ "name": "p_ZL16post_eligibility_4_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_4", "role": "we1" }} , 
 	{ "name": "p_ZL16post_eligibility_4_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_4", "role": "d1" }} , 
 	{ "name": "p_ZL16post_eligibility_5_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_5", "role": "address0" }} , 
 	{ "name": "p_ZL16post_eligibility_5_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_5", "role": "ce0" }} , 
 	{ "name": "p_ZL16post_eligibility_5_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_5", "role": "q0" }} , 
 	{ "name": "p_ZL16post_eligibility_5_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_5", "role": "address1" }} , 
 	{ "name": "p_ZL16post_eligibility_5_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_5", "role": "ce1" }} , 
 	{ "name": "p_ZL16post_eligibility_5_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_5", "role": "we1" }} , 
 	{ "name": "p_ZL16post_eligibility_5_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_5", "role": "d1" }} , 
 	{ "name": "p_ZL16post_eligibility_6_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_6", "role": "address0" }} , 
 	{ "name": "p_ZL16post_eligibility_6_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_6", "role": "ce0" }} , 
 	{ "name": "p_ZL16post_eligibility_6_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_6", "role": "q0" }} , 
 	{ "name": "p_ZL16post_eligibility_6_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_6", "role": "address1" }} , 
 	{ "name": "p_ZL16post_eligibility_6_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_6", "role": "ce1" }} , 
 	{ "name": "p_ZL16post_eligibility_6_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_6", "role": "we1" }} , 
 	{ "name": "p_ZL16post_eligibility_6_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_6", "role": "d1" }} , 
 	{ "name": "p_ZL16post_eligibility_7_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_7", "role": "address0" }} , 
 	{ "name": "p_ZL16post_eligibility_7_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_7", "role": "ce0" }} , 
 	{ "name": "p_ZL16post_eligibility_7_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_7", "role": "q0" }} , 
 	{ "name": "p_ZL16post_eligibility_7_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_7", "role": "address1" }} , 
 	{ "name": "p_ZL16post_eligibility_7_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_7", "role": "ce1" }} , 
 	{ "name": "p_ZL16post_eligibility_7_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_7", "role": "we1" }} , 
 	{ "name": "p_ZL16post_eligibility_7_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "p_ZL16post_eligibility_7", "role": "d1" }}  ]}

set ArgLastReadFirstWriteLatency {
	decay_eligibility_traces {
		p_ZL15pre_eligibility_0 {Type IO LastRead 0 FirstWrite 2}
		p_ZL15pre_eligibility_1 {Type IO LastRead 0 FirstWrite 2}
		p_ZL15pre_eligibility_2 {Type IO LastRead 0 FirstWrite 2}
		p_ZL15pre_eligibility_3 {Type IO LastRead 0 FirstWrite 2}
		p_ZL15pre_eligibility_4 {Type IO LastRead 0 FirstWrite 2}
		p_ZL15pre_eligibility_5 {Type IO LastRead 0 FirstWrite 2}
		p_ZL15pre_eligibility_6 {Type IO LastRead 0 FirstWrite 2}
		p_ZL15pre_eligibility_7 {Type IO LastRead 0 FirstWrite 2}
		p_ZL16post_eligibility_0 {Type IO LastRead 0 FirstWrite 2}
		p_ZL16post_eligibility_1 {Type IO LastRead 0 FirstWrite 2}
		p_ZL16post_eligibility_2 {Type IO LastRead 0 FirstWrite 2}
		p_ZL16post_eligibility_3 {Type IO LastRead 0 FirstWrite 2}
		p_ZL16post_eligibility_4 {Type IO LastRead 0 FirstWrite 2}
		p_ZL16post_eligibility_5 {Type IO LastRead 0 FirstWrite 2}
		p_ZL16post_eligibility_6 {Type IO LastRead 0 FirstWrite 2}
		p_ZL16post_eligibility_7 {Type IO LastRead 0 FirstWrite 2}}
	decay_eligibility_traces_Pipeline_DECAY_PRE {
		p_ZL15pre_eligibility_0 {Type IO LastRead 0 FirstWrite 2}
		p_ZL15pre_eligibility_1 {Type IO LastRead 0 FirstWrite 2}
		p_ZL15pre_eligibility_2 {Type IO LastRead 0 FirstWrite 2}
		p_ZL15pre_eligibility_3 {Type IO LastRead 0 FirstWrite 2}
		p_ZL15pre_eligibility_4 {Type IO LastRead 0 FirstWrite 2}
		p_ZL15pre_eligibility_5 {Type IO LastRead 0 FirstWrite 2}
		p_ZL15pre_eligibility_6 {Type IO LastRead 0 FirstWrite 2}
		p_ZL15pre_eligibility_7 {Type IO LastRead 0 FirstWrite 2}}
	decay_eligibility_traces_Pipeline_DECAY_POST {
		p_ZL16post_eligibility_0 {Type IO LastRead 0 FirstWrite 2}
		p_ZL16post_eligibility_1 {Type IO LastRead 0 FirstWrite 2}
		p_ZL16post_eligibility_2 {Type IO LastRead 0 FirstWrite 2}
		p_ZL16post_eligibility_3 {Type IO LastRead 0 FirstWrite 2}
		p_ZL16post_eligibility_4 {Type IO LastRead 0 FirstWrite 2}
		p_ZL16post_eligibility_5 {Type IO LastRead 0 FirstWrite 2}
		p_ZL16post_eligibility_6 {Type IO LastRead 0 FirstWrite 2}
		p_ZL16post_eligibility_7 {Type IO LastRead 0 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "12", "Max" : "12"}
	, {"Name" : "Interval", "Min" : "12", "Max" : "12"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	p_ZL15pre_eligibility_0 { ap_memory {  { p_ZL15pre_eligibility_0_address0 mem_address 1 3 }  { p_ZL15pre_eligibility_0_ce0 mem_ce 1 1 }  { p_ZL15pre_eligibility_0_q0 mem_dout 0 8 }  { p_ZL15pre_eligibility_0_address1 MemPortADDR2 1 3 }  { p_ZL15pre_eligibility_0_ce1 MemPortCE2 1 1 }  { p_ZL15pre_eligibility_0_we1 MemPortWE2 1 1 }  { p_ZL15pre_eligibility_0_d1 MemPortDIN2 1 8 } } }
	p_ZL15pre_eligibility_1 { ap_memory {  { p_ZL15pre_eligibility_1_address0 mem_address 1 3 }  { p_ZL15pre_eligibility_1_ce0 mem_ce 1 1 }  { p_ZL15pre_eligibility_1_q0 mem_dout 0 8 }  { p_ZL15pre_eligibility_1_address1 MemPortADDR2 1 3 }  { p_ZL15pre_eligibility_1_ce1 MemPortCE2 1 1 }  { p_ZL15pre_eligibility_1_we1 MemPortWE2 1 1 }  { p_ZL15pre_eligibility_1_d1 MemPortDIN2 1 8 } } }
	p_ZL15pre_eligibility_2 { ap_memory {  { p_ZL15pre_eligibility_2_address0 mem_address 1 3 }  { p_ZL15pre_eligibility_2_ce0 mem_ce 1 1 }  { p_ZL15pre_eligibility_2_q0 mem_dout 0 8 }  { p_ZL15pre_eligibility_2_address1 MemPortADDR2 1 3 }  { p_ZL15pre_eligibility_2_ce1 MemPortCE2 1 1 }  { p_ZL15pre_eligibility_2_we1 MemPortWE2 1 1 }  { p_ZL15pre_eligibility_2_d1 MemPortDIN2 1 8 } } }
	p_ZL15pre_eligibility_3 { ap_memory {  { p_ZL15pre_eligibility_3_address0 mem_address 1 3 }  { p_ZL15pre_eligibility_3_ce0 mem_ce 1 1 }  { p_ZL15pre_eligibility_3_q0 mem_dout 0 8 }  { p_ZL15pre_eligibility_3_address1 MemPortADDR2 1 3 }  { p_ZL15pre_eligibility_3_ce1 MemPortCE2 1 1 }  { p_ZL15pre_eligibility_3_we1 MemPortWE2 1 1 }  { p_ZL15pre_eligibility_3_d1 MemPortDIN2 1 8 } } }
	p_ZL15pre_eligibility_4 { ap_memory {  { p_ZL15pre_eligibility_4_address0 mem_address 1 3 }  { p_ZL15pre_eligibility_4_ce0 mem_ce 1 1 }  { p_ZL15pre_eligibility_4_q0 mem_dout 0 8 }  { p_ZL15pre_eligibility_4_address1 MemPortADDR2 1 3 }  { p_ZL15pre_eligibility_4_ce1 MemPortCE2 1 1 }  { p_ZL15pre_eligibility_4_we1 MemPortWE2 1 1 }  { p_ZL15pre_eligibility_4_d1 MemPortDIN2 1 8 } } }
	p_ZL15pre_eligibility_5 { ap_memory {  { p_ZL15pre_eligibility_5_address0 mem_address 1 3 }  { p_ZL15pre_eligibility_5_ce0 mem_ce 1 1 }  { p_ZL15pre_eligibility_5_q0 mem_dout 0 8 }  { p_ZL15pre_eligibility_5_address1 MemPortADDR2 1 3 }  { p_ZL15pre_eligibility_5_ce1 MemPortCE2 1 1 }  { p_ZL15pre_eligibility_5_we1 MemPortWE2 1 1 }  { p_ZL15pre_eligibility_5_d1 MemPortDIN2 1 8 } } }
	p_ZL15pre_eligibility_6 { ap_memory {  { p_ZL15pre_eligibility_6_address0 mem_address 1 3 }  { p_ZL15pre_eligibility_6_ce0 mem_ce 1 1 }  { p_ZL15pre_eligibility_6_q0 mem_dout 0 8 }  { p_ZL15pre_eligibility_6_address1 MemPortADDR2 1 3 }  { p_ZL15pre_eligibility_6_ce1 MemPortCE2 1 1 }  { p_ZL15pre_eligibility_6_we1 MemPortWE2 1 1 }  { p_ZL15pre_eligibility_6_d1 MemPortDIN2 1 8 } } }
	p_ZL15pre_eligibility_7 { ap_memory {  { p_ZL15pre_eligibility_7_address0 mem_address 1 3 }  { p_ZL15pre_eligibility_7_ce0 mem_ce 1 1 }  { p_ZL15pre_eligibility_7_q0 mem_dout 0 8 }  { p_ZL15pre_eligibility_7_address1 MemPortADDR2 1 3 }  { p_ZL15pre_eligibility_7_ce1 MemPortCE2 1 1 }  { p_ZL15pre_eligibility_7_we1 MemPortWE2 1 1 }  { p_ZL15pre_eligibility_7_d1 MemPortDIN2 1 8 } } }
	p_ZL16post_eligibility_0 { ap_memory {  { p_ZL16post_eligibility_0_address0 mem_address 1 3 }  { p_ZL16post_eligibility_0_ce0 mem_ce 1 1 }  { p_ZL16post_eligibility_0_q0 mem_dout 0 8 }  { p_ZL16post_eligibility_0_address1 MemPortADDR2 1 3 }  { p_ZL16post_eligibility_0_ce1 MemPortCE2 1 1 }  { p_ZL16post_eligibility_0_we1 MemPortWE2 1 1 }  { p_ZL16post_eligibility_0_d1 MemPortDIN2 1 8 } } }
	p_ZL16post_eligibility_1 { ap_memory {  { p_ZL16post_eligibility_1_address0 mem_address 1 3 }  { p_ZL16post_eligibility_1_ce0 mem_ce 1 1 }  { p_ZL16post_eligibility_1_q0 mem_dout 0 8 }  { p_ZL16post_eligibility_1_address1 MemPortADDR2 1 3 }  { p_ZL16post_eligibility_1_ce1 MemPortCE2 1 1 }  { p_ZL16post_eligibility_1_we1 MemPortWE2 1 1 }  { p_ZL16post_eligibility_1_d1 MemPortDIN2 1 8 } } }
	p_ZL16post_eligibility_2 { ap_memory {  { p_ZL16post_eligibility_2_address0 mem_address 1 3 }  { p_ZL16post_eligibility_2_ce0 mem_ce 1 1 }  { p_ZL16post_eligibility_2_q0 mem_dout 0 8 }  { p_ZL16post_eligibility_2_address1 MemPortADDR2 1 3 }  { p_ZL16post_eligibility_2_ce1 MemPortCE2 1 1 }  { p_ZL16post_eligibility_2_we1 MemPortWE2 1 1 }  { p_ZL16post_eligibility_2_d1 MemPortDIN2 1 8 } } }
	p_ZL16post_eligibility_3 { ap_memory {  { p_ZL16post_eligibility_3_address0 mem_address 1 3 }  { p_ZL16post_eligibility_3_ce0 mem_ce 1 1 }  { p_ZL16post_eligibility_3_q0 mem_dout 0 8 }  { p_ZL16post_eligibility_3_address1 MemPortADDR2 1 3 }  { p_ZL16post_eligibility_3_ce1 MemPortCE2 1 1 }  { p_ZL16post_eligibility_3_we1 MemPortWE2 1 1 }  { p_ZL16post_eligibility_3_d1 MemPortDIN2 1 8 } } }
	p_ZL16post_eligibility_4 { ap_memory {  { p_ZL16post_eligibility_4_address0 mem_address 1 3 }  { p_ZL16post_eligibility_4_ce0 mem_ce 1 1 }  { p_ZL16post_eligibility_4_q0 mem_dout 0 8 }  { p_ZL16post_eligibility_4_address1 MemPortADDR2 1 3 }  { p_ZL16post_eligibility_4_ce1 MemPortCE2 1 1 }  { p_ZL16post_eligibility_4_we1 MemPortWE2 1 1 }  { p_ZL16post_eligibility_4_d1 MemPortDIN2 1 8 } } }
	p_ZL16post_eligibility_5 { ap_memory {  { p_ZL16post_eligibility_5_address0 mem_address 1 3 }  { p_ZL16post_eligibility_5_ce0 mem_ce 1 1 }  { p_ZL16post_eligibility_5_q0 mem_dout 0 8 }  { p_ZL16post_eligibility_5_address1 MemPortADDR2 1 3 }  { p_ZL16post_eligibility_5_ce1 MemPortCE2 1 1 }  { p_ZL16post_eligibility_5_we1 MemPortWE2 1 1 }  { p_ZL16post_eligibility_5_d1 MemPortDIN2 1 8 } } }
	p_ZL16post_eligibility_6 { ap_memory {  { p_ZL16post_eligibility_6_address0 mem_address 1 3 }  { p_ZL16post_eligibility_6_ce0 mem_ce 1 1 }  { p_ZL16post_eligibility_6_q0 mem_dout 0 8 }  { p_ZL16post_eligibility_6_address1 MemPortADDR2 1 3 }  { p_ZL16post_eligibility_6_ce1 MemPortCE2 1 1 }  { p_ZL16post_eligibility_6_we1 MemPortWE2 1 1 }  { p_ZL16post_eligibility_6_d1 MemPortDIN2 1 8 } } }
	p_ZL16post_eligibility_7 { ap_memory {  { p_ZL16post_eligibility_7_address0 mem_address 1 3 }  { p_ZL16post_eligibility_7_ce0 mem_ce 1 1 }  { p_ZL16post_eligibility_7_q0 mem_dout 0 8 }  { p_ZL16post_eligibility_7_address1 MemPortADDR2 1 3 }  { p_ZL16post_eligibility_7_ce1 MemPortCE2 1 1 }  { p_ZL16post_eligibility_7_we1 MemPortWE2 1 1 }  { p_ZL16post_eligibility_7_d1 MemPortDIN2 1 8 } } }
}
