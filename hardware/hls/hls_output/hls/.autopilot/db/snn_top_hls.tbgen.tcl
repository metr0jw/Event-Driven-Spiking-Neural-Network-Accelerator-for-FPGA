set moduleName snn_top_hls
set isTopModule 1
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
set C_modelName {snn_top_hls}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
set C_modelArgList {
	{ ctrl_reg int 32 regular {axi_slave 0}  }
	{ config_reg int 32 regular {axi_slave 0}  }
	{ learning_params int 144 regular {axi_slave 0}  }
	{ status_reg int 32 regular {axi_slave 1}  }
	{ spike_count_reg int 32 regular {axi_slave 1}  }
	{ weight_sum_reg int 32 regular {axi_slave 1}  }
	{ version_reg int 32 regular {axi_slave 1}  }
	{ s_axis_spikes_V_data_V int 32 regular {axi_s 0 volatile  { s_axis_spikes Data } }  }
	{ s_axis_spikes_V_keep_V int 4 regular {axi_s 0 volatile  { s_axis_spikes Keep } }  }
	{ s_axis_spikes_V_strb_V int 4 regular {axi_s 0 volatile  { s_axis_spikes Strb } }  }
	{ s_axis_spikes_V_user_V int 1 regular {axi_s 0 volatile  { s_axis_spikes User } }  }
	{ s_axis_spikes_V_last_V int 1 regular {axi_s 0 volatile  { s_axis_spikes Last } }  }
	{ s_axis_spikes_V_id_V int 1 regular {axi_s 0 volatile  { s_axis_spikes ID } }  }
	{ s_axis_spikes_V_dest_V int 1 regular {axi_s 0 volatile  { s_axis_spikes Dest } }  }
	{ m_axis_spikes_V_data_V int 32 regular {axi_s 1 volatile  { m_axis_spikes Data } }  }
	{ m_axis_spikes_V_keep_V int 4 regular {axi_s 1 volatile  { m_axis_spikes Keep } }  }
	{ m_axis_spikes_V_strb_V int 4 regular {axi_s 1 volatile  { m_axis_spikes Strb } }  }
	{ m_axis_spikes_V_user_V int 1 regular {axi_s 1 volatile  { m_axis_spikes User } }  }
	{ m_axis_spikes_V_last_V int 1 regular {axi_s 1 volatile  { m_axis_spikes Last } }  }
	{ m_axis_spikes_V_id_V int 1 regular {axi_s 1 volatile  { m_axis_spikes ID } }  }
	{ m_axis_spikes_V_dest_V int 1 regular {axi_s 1 volatile  { m_axis_spikes Dest } }  }
	{ m_axis_weights_V_data_V int 32 regular {axi_s 1 volatile  { m_axis_weights Data } }  }
	{ m_axis_weights_V_keep_V int 4 regular {axi_s 1 volatile  { m_axis_weights Keep } }  }
	{ m_axis_weights_V_strb_V int 4 regular {axi_s 1 volatile  { m_axis_weights Strb } }  }
	{ m_axis_weights_V_user_V int 1 regular {axi_s 1 volatile  { m_axis_weights User } }  }
	{ m_axis_weights_V_last_V int 1 regular {axi_s 1 volatile  { m_axis_weights Last } }  }
	{ m_axis_weights_V_id_V int 1 regular {axi_s 1 volatile  { m_axis_weights ID } }  }
	{ m_axis_weights_V_dest_V int 1 regular {axi_s 1 volatile  { m_axis_weights Dest } }  }
	{ reward_signal int 8 regular {axi_slave 0}  }
	{ spike_in_valid int 1 regular {pointer 1}  }
	{ spike_in_neuron_id int 8 regular {pointer 1}  }
	{ spike_in_weight int 8 regular {pointer 1}  }
	{ spike_in_ready int 1 regular  }
	{ spike_out_valid int 1 regular  }
	{ spike_out_neuron_id int 8 regular  }
	{ spike_out_weight int 8 regular  }
	{ spike_out_ready int 1 regular {pointer 1}  }
	{ snn_enable int 1 regular {pointer 1}  }
	{ snn_reset int 1 regular {pointer 1}  }
	{ threshold_out int 16 regular {pointer 1}  }
	{ leak_rate_out int 16 regular {pointer 1}  }
	{ snn_ready int 1 regular  }
	{ snn_busy int 1 regular  }
}
set hasAXIMCache 0
set l_AXIML2Cache [list]
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "ctrl_reg", "interface" : "axi_slave", "bundle":"ctrl","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":16}, "offset_end" : {"in":23}} , 
 	{ "Name" : "config_reg", "interface" : "axi_slave", "bundle":"ctrl","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":24}, "offset_end" : {"in":31}} , 
 	{ "Name" : "learning_params", "interface" : "axi_slave", "bundle":"ctrl","type":"ap_none","bitwidth" : 144, "direction" : "READONLY", "offset" : {"in":32}, "offset_end" : {"in":55}} , 
 	{ "Name" : "status_reg", "interface" : "axi_slave", "bundle":"ctrl","type":"ap_vld","bitwidth" : 32, "direction" : "WRITEONLY", "offset" : {"out":56}, "offset_end" : {"out":63}} , 
 	{ "Name" : "spike_count_reg", "interface" : "axi_slave", "bundle":"ctrl","type":"ap_vld","bitwidth" : 32, "direction" : "WRITEONLY", "offset" : {"out":72}, "offset_end" : {"out":79}} , 
 	{ "Name" : "weight_sum_reg", "interface" : "axi_slave", "bundle":"ctrl","type":"ap_vld","bitwidth" : 32, "direction" : "WRITEONLY", "offset" : {"out":88}, "offset_end" : {"out":95}} , 
 	{ "Name" : "version_reg", "interface" : "axi_slave", "bundle":"ctrl","type":"ap_vld","bitwidth" : 32, "direction" : "WRITEONLY", "offset" : {"out":104}, "offset_end" : {"out":111}} , 
 	{ "Name" : "s_axis_spikes_V_data_V", "interface" : "axis", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "s_axis_spikes_V_keep_V", "interface" : "axis", "bitwidth" : 4, "direction" : "READONLY"} , 
 	{ "Name" : "s_axis_spikes_V_strb_V", "interface" : "axis", "bitwidth" : 4, "direction" : "READONLY"} , 
 	{ "Name" : "s_axis_spikes_V_user_V", "interface" : "axis", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "s_axis_spikes_V_last_V", "interface" : "axis", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "s_axis_spikes_V_id_V", "interface" : "axis", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "s_axis_spikes_V_dest_V", "interface" : "axis", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "m_axis_spikes_V_data_V", "interface" : "axis", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "m_axis_spikes_V_keep_V", "interface" : "axis", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "m_axis_spikes_V_strb_V", "interface" : "axis", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "m_axis_spikes_V_user_V", "interface" : "axis", "bitwidth" : 1, "direction" : "WRITEONLY"} , 
 	{ "Name" : "m_axis_spikes_V_last_V", "interface" : "axis", "bitwidth" : 1, "direction" : "WRITEONLY"} , 
 	{ "Name" : "m_axis_spikes_V_id_V", "interface" : "axis", "bitwidth" : 1, "direction" : "WRITEONLY"} , 
 	{ "Name" : "m_axis_spikes_V_dest_V", "interface" : "axis", "bitwidth" : 1, "direction" : "WRITEONLY"} , 
 	{ "Name" : "m_axis_weights_V_data_V", "interface" : "axis", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "m_axis_weights_V_keep_V", "interface" : "axis", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "m_axis_weights_V_strb_V", "interface" : "axis", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "m_axis_weights_V_user_V", "interface" : "axis", "bitwidth" : 1, "direction" : "WRITEONLY"} , 
 	{ "Name" : "m_axis_weights_V_last_V", "interface" : "axis", "bitwidth" : 1, "direction" : "WRITEONLY"} , 
 	{ "Name" : "m_axis_weights_V_id_V", "interface" : "axis", "bitwidth" : 1, "direction" : "WRITEONLY"} , 
 	{ "Name" : "m_axis_weights_V_dest_V", "interface" : "axis", "bitwidth" : 1, "direction" : "WRITEONLY"} , 
 	{ "Name" : "reward_signal", "interface" : "axi_slave", "bundle":"ctrl","type":"ap_none","bitwidth" : 8, "direction" : "READONLY", "offset" : {"in":120}, "offset_end" : {"in":127}} , 
 	{ "Name" : "spike_in_valid", "interface" : "wire", "bitwidth" : 1, "direction" : "WRITEONLY"} , 
 	{ "Name" : "spike_in_neuron_id", "interface" : "wire", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "spike_in_weight", "interface" : "wire", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "spike_in_ready", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "spike_out_valid", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "spike_out_neuron_id", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "spike_out_weight", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "spike_out_ready", "interface" : "wire", "bitwidth" : 1, "direction" : "WRITEONLY"} , 
 	{ "Name" : "snn_enable", "interface" : "wire", "bitwidth" : 1, "direction" : "WRITEONLY"} , 
 	{ "Name" : "snn_reset", "interface" : "wire", "bitwidth" : 1, "direction" : "WRITEONLY"} , 
 	{ "Name" : "threshold_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "leak_rate_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "snn_ready", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "snn_busy", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} ]}
# RTL Port declarations: 
set portNum 61
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst_n sc_in sc_logic 1 reset -1 active_low_sync } 
	{ s_axis_spikes_TDATA sc_in sc_lv 32 signal 7 } 
	{ s_axis_spikes_TVALID sc_in sc_logic 1 invld 13 } 
	{ s_axis_spikes_TREADY sc_out sc_logic 1 inacc 13 } 
	{ s_axis_spikes_TKEEP sc_in sc_lv 4 signal 8 } 
	{ s_axis_spikes_TSTRB sc_in sc_lv 4 signal 9 } 
	{ s_axis_spikes_TUSER sc_in sc_lv 1 signal 10 } 
	{ s_axis_spikes_TLAST sc_in sc_lv 1 signal 11 } 
	{ s_axis_spikes_TID sc_in sc_lv 1 signal 12 } 
	{ s_axis_spikes_TDEST sc_in sc_lv 1 signal 13 } 
	{ m_axis_spikes_TDATA sc_out sc_lv 32 signal 14 } 
	{ m_axis_spikes_TVALID sc_out sc_logic 1 outvld 20 } 
	{ m_axis_spikes_TREADY sc_in sc_logic 1 outacc 20 } 
	{ m_axis_spikes_TKEEP sc_out sc_lv 4 signal 15 } 
	{ m_axis_spikes_TSTRB sc_out sc_lv 4 signal 16 } 
	{ m_axis_spikes_TUSER sc_out sc_lv 1 signal 17 } 
	{ m_axis_spikes_TLAST sc_out sc_lv 1 signal 18 } 
	{ m_axis_spikes_TID sc_out sc_lv 1 signal 19 } 
	{ m_axis_spikes_TDEST sc_out sc_lv 1 signal 20 } 
	{ m_axis_weights_TDATA sc_out sc_lv 32 signal 21 } 
	{ m_axis_weights_TVALID sc_out sc_logic 1 outvld 27 } 
	{ m_axis_weights_TREADY sc_in sc_logic 1 outacc 27 } 
	{ m_axis_weights_TKEEP sc_out sc_lv 4 signal 22 } 
	{ m_axis_weights_TSTRB sc_out sc_lv 4 signal 23 } 
	{ m_axis_weights_TUSER sc_out sc_lv 1 signal 24 } 
	{ m_axis_weights_TLAST sc_out sc_lv 1 signal 25 } 
	{ m_axis_weights_TID sc_out sc_lv 1 signal 26 } 
	{ m_axis_weights_TDEST sc_out sc_lv 1 signal 27 } 
	{ spike_in_valid sc_out sc_lv 1 signal 29 } 
	{ spike_in_neuron_id sc_out sc_lv 8 signal 30 } 
	{ spike_in_weight sc_out sc_lv 8 signal 31 } 
	{ spike_in_ready sc_in sc_lv 1 signal 32 } 
	{ spike_out_valid sc_in sc_lv 1 signal 33 } 
	{ spike_out_neuron_id sc_in sc_lv 8 signal 34 } 
	{ spike_out_weight sc_in sc_lv 8 signal 35 } 
	{ spike_out_ready sc_out sc_lv 1 signal 36 } 
	{ snn_enable sc_out sc_lv 1 signal 37 } 
	{ snn_reset sc_out sc_lv 1 signal 38 } 
	{ threshold_out sc_out sc_lv 16 signal 39 } 
	{ leak_rate_out sc_out sc_lv 16 signal 40 } 
	{ snn_ready sc_in sc_lv 1 signal 41 } 
	{ snn_busy sc_in sc_lv 1 signal 42 } 
	{ s_axi_ctrl_AWVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_ctrl_AWREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_ctrl_AWADDR sc_in sc_lv 7 signal -1 } 
	{ s_axi_ctrl_WVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_ctrl_WREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_ctrl_WDATA sc_in sc_lv 32 signal -1 } 
	{ s_axi_ctrl_WSTRB sc_in sc_lv 4 signal -1 } 
	{ s_axi_ctrl_ARVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_ctrl_ARREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_ctrl_ARADDR sc_in sc_lv 7 signal -1 } 
	{ s_axi_ctrl_RVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_ctrl_RREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_ctrl_RDATA sc_out sc_lv 32 signal -1 } 
	{ s_axi_ctrl_RRESP sc_out sc_lv 2 signal -1 } 
	{ s_axi_ctrl_BVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_ctrl_BREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_ctrl_BRESP sc_out sc_lv 2 signal -1 } 
	{ interrupt sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "s_axi_ctrl_AWADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "ctrl", "role": "AWADDR" },"address":[{"name":"snn_top_hls","role":"start","value":"0","valid_bit":"0"},{"name":"snn_top_hls","role":"continue","value":"0","valid_bit":"4"},{"name":"snn_top_hls","role":"auto_start","value":"0","valid_bit":"7"},{"name":"ctrl_reg","role":"data","value":"16"},{"name":"config_reg","role":"data","value":"24"},{"name":"learning_params","role":"data","value":"32"},{"name":"reward_signal","role":"data","value":"120"}] },
	{ "name": "s_axi_ctrl_AWVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ctrl", "role": "AWVALID" } },
	{ "name": "s_axi_ctrl_AWREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ctrl", "role": "AWREADY" } },
	{ "name": "s_axi_ctrl_WVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ctrl", "role": "WVALID" } },
	{ "name": "s_axi_ctrl_WREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ctrl", "role": "WREADY" } },
	{ "name": "s_axi_ctrl_WDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "ctrl", "role": "WDATA" } },
	{ "name": "s_axi_ctrl_WSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "ctrl", "role": "WSTRB" } },
	{ "name": "s_axi_ctrl_ARADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "ctrl", "role": "ARADDR" },"address":[{"name":"snn_top_hls","role":"start","value":"0","valid_bit":"0"},{"name":"snn_top_hls","role":"done","value":"0","valid_bit":"1"},{"name":"snn_top_hls","role":"idle","value":"0","valid_bit":"2"},{"name":"snn_top_hls","role":"ready","value":"0","valid_bit":"3"},{"name":"snn_top_hls","role":"auto_start","value":"0","valid_bit":"7"},{"name":"status_reg","role":"data","value":"56"}, {"name":"status_reg","role":"valid","value":"60","valid_bit":"0"},{"name":"spike_count_reg","role":"data","value":"72"}, {"name":"spike_count_reg","role":"valid","value":"76","valid_bit":"0"},{"name":"weight_sum_reg","role":"data","value":"88"}, {"name":"weight_sum_reg","role":"valid","value":"92","valid_bit":"0"},{"name":"version_reg","role":"data","value":"104"}, {"name":"version_reg","role":"valid","value":"108","valid_bit":"0"}] },
	{ "name": "s_axi_ctrl_ARVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ctrl", "role": "ARVALID" } },
	{ "name": "s_axi_ctrl_ARREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ctrl", "role": "ARREADY" } },
	{ "name": "s_axi_ctrl_RVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ctrl", "role": "RVALID" } },
	{ "name": "s_axi_ctrl_RREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ctrl", "role": "RREADY" } },
	{ "name": "s_axi_ctrl_RDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "ctrl", "role": "RDATA" } },
	{ "name": "s_axi_ctrl_RRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "ctrl", "role": "RRESP" } },
	{ "name": "s_axi_ctrl_BVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ctrl", "role": "BVALID" } },
	{ "name": "s_axi_ctrl_BREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ctrl", "role": "BREADY" } },
	{ "name": "s_axi_ctrl_BRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "ctrl", "role": "BRESP" } },
	{ "name": "interrupt", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ctrl", "role": "interrupt" } }, 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst_n", "role": "default" }} , 
 	{ "name": "s_axis_spikes_TDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "s_axis_spikes_V_data_V", "role": "default" }} , 
 	{ "name": "s_axis_spikes_TVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "s_axis_spikes_V_dest_V", "role": "default" }} , 
 	{ "name": "s_axis_spikes_TREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "inacc", "bundle":{"name": "s_axis_spikes_V_dest_V", "role": "default" }} , 
 	{ "name": "s_axis_spikes_TKEEP", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "s_axis_spikes_V_keep_V", "role": "default" }} , 
 	{ "name": "s_axis_spikes_TSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "s_axis_spikes_V_strb_V", "role": "default" }} , 
 	{ "name": "s_axis_spikes_TUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "s_axis_spikes_V_user_V", "role": "default" }} , 
 	{ "name": "s_axis_spikes_TLAST", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "s_axis_spikes_V_last_V", "role": "default" }} , 
 	{ "name": "s_axis_spikes_TID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "s_axis_spikes_V_id_V", "role": "default" }} , 
 	{ "name": "s_axis_spikes_TDEST", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "s_axis_spikes_V_dest_V", "role": "default" }} , 
 	{ "name": "m_axis_spikes_TDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "m_axis_spikes_V_data_V", "role": "default" }} , 
 	{ "name": "m_axis_spikes_TVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "m_axis_spikes_V_dest_V", "role": "default" }} , 
 	{ "name": "m_axis_spikes_TREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "outacc", "bundle":{"name": "m_axis_spikes_V_dest_V", "role": "default" }} , 
 	{ "name": "m_axis_spikes_TKEEP", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "m_axis_spikes_V_keep_V", "role": "default" }} , 
 	{ "name": "m_axis_spikes_TSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "m_axis_spikes_V_strb_V", "role": "default" }} , 
 	{ "name": "m_axis_spikes_TUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "m_axis_spikes_V_user_V", "role": "default" }} , 
 	{ "name": "m_axis_spikes_TLAST", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "m_axis_spikes_V_last_V", "role": "default" }} , 
 	{ "name": "m_axis_spikes_TID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "m_axis_spikes_V_id_V", "role": "default" }} , 
 	{ "name": "m_axis_spikes_TDEST", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "m_axis_spikes_V_dest_V", "role": "default" }} , 
 	{ "name": "m_axis_weights_TDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "m_axis_weights_V_data_V", "role": "default" }} , 
 	{ "name": "m_axis_weights_TVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "m_axis_weights_V_dest_V", "role": "default" }} , 
 	{ "name": "m_axis_weights_TREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "outacc", "bundle":{"name": "m_axis_weights_V_dest_V", "role": "default" }} , 
 	{ "name": "m_axis_weights_TKEEP", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "m_axis_weights_V_keep_V", "role": "default" }} , 
 	{ "name": "m_axis_weights_TSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "m_axis_weights_V_strb_V", "role": "default" }} , 
 	{ "name": "m_axis_weights_TUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "m_axis_weights_V_user_V", "role": "default" }} , 
 	{ "name": "m_axis_weights_TLAST", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "m_axis_weights_V_last_V", "role": "default" }} , 
 	{ "name": "m_axis_weights_TID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "m_axis_weights_V_id_V", "role": "default" }} , 
 	{ "name": "m_axis_weights_TDEST", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "m_axis_weights_V_dest_V", "role": "default" }} , 
 	{ "name": "spike_in_valid", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "spike_in_valid", "role": "default" }} , 
 	{ "name": "spike_in_neuron_id", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "spike_in_neuron_id", "role": "default" }} , 
 	{ "name": "spike_in_weight", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "spike_in_weight", "role": "default" }} , 
 	{ "name": "spike_in_ready", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "spike_in_ready", "role": "default" }} , 
 	{ "name": "spike_out_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "spike_out_valid", "role": "default" }} , 
 	{ "name": "spike_out_neuron_id", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "spike_out_neuron_id", "role": "default" }} , 
 	{ "name": "spike_out_weight", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "spike_out_weight", "role": "default" }} , 
 	{ "name": "spike_out_ready", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "spike_out_ready", "role": "default" }} , 
 	{ "name": "snn_enable", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "snn_enable", "role": "default" }} , 
 	{ "name": "snn_reset", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "snn_reset", "role": "default" }} , 
 	{ "name": "threshold_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "threshold_out", "role": "default" }} , 
 	{ "name": "leak_rate_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "leak_rate_out", "role": "default" }} , 
 	{ "name": "snn_ready", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "snn_ready", "role": "default" }} , 
 	{ "name": "snn_busy", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "snn_busy", "role": "default" }}  ]}

set ArgLastReadFirstWriteLatency {
	snn_top_hls {
		ctrl_reg {Type I LastRead 0 FirstWrite -1}
		config_reg {Type I LastRead 0 FirstWrite -1}
		learning_params {Type I LastRead 0 FirstWrite -1}
		status_reg {Type O LastRead -1 FirstWrite 16}
		spike_count_reg {Type O LastRead -1 FirstWrite 16}
		weight_sum_reg {Type O LastRead -1 FirstWrite 18}
		version_reg {Type O LastRead -1 FirstWrite 16}
		s_axis_spikes_V_data_V {Type I LastRead 2 FirstWrite -1}
		s_axis_spikes_V_keep_V {Type I LastRead 2 FirstWrite -1}
		s_axis_spikes_V_strb_V {Type I LastRead 2 FirstWrite -1}
		s_axis_spikes_V_user_V {Type I LastRead 2 FirstWrite -1}
		s_axis_spikes_V_last_V {Type I LastRead 2 FirstWrite -1}
		s_axis_spikes_V_id_V {Type I LastRead 2 FirstWrite -1}
		s_axis_spikes_V_dest_V {Type I LastRead 2 FirstWrite -1}
		m_axis_spikes_V_data_V {Type O LastRead 5 FirstWrite 5}
		m_axis_spikes_V_keep_V {Type O LastRead 5 FirstWrite 5}
		m_axis_spikes_V_strb_V {Type O LastRead 5 FirstWrite 5}
		m_axis_spikes_V_user_V {Type O LastRead 5 FirstWrite 5}
		m_axis_spikes_V_last_V {Type O LastRead 5 FirstWrite 5}
		m_axis_spikes_V_id_V {Type O LastRead 5 FirstWrite 5}
		m_axis_spikes_V_dest_V {Type O LastRead 5 FirstWrite 5}
		m_axis_weights_V_data_V {Type O LastRead 13 FirstWrite 15}
		m_axis_weights_V_keep_V {Type O LastRead 13 FirstWrite 15}
		m_axis_weights_V_strb_V {Type O LastRead 13 FirstWrite 15}
		m_axis_weights_V_user_V {Type O LastRead 13 FirstWrite 15}
		m_axis_weights_V_last_V {Type O LastRead 13 FirstWrite 15}
		m_axis_weights_V_id_V {Type O LastRead 13 FirstWrite 15}
		m_axis_weights_V_dest_V {Type O LastRead 13 FirstWrite 15}
		reward_signal {Type I LastRead 0 FirstWrite -1}
		spike_in_valid {Type O LastRead -1 FirstWrite 16}
		spike_in_neuron_id {Type O LastRead -1 FirstWrite 16}
		spike_in_weight {Type O LastRead -1 FirstWrite 16}
		spike_in_ready {Type I LastRead 0 FirstWrite -1}
		spike_out_valid {Type I LastRead 0 FirstWrite -1}
		spike_out_neuron_id {Type I LastRead 0 FirstWrite -1}
		spike_out_weight {Type I LastRead 0 FirstWrite -1}
		spike_out_ready {Type O LastRead -1 FirstWrite 5}
		snn_enable {Type O LastRead -1 FirstWrite 2}
		snn_reset {Type O LastRead -1 FirstWrite 2}
		threshold_out {Type O LastRead -1 FirstWrite 2}
		leak_rate_out {Type O LastRead -1 FirstWrite 2}
		snn_ready {Type I LastRead 0 FirstWrite -1}
		snn_busy {Type I LastRead 0 FirstWrite -1}
		initialized {Type IO LastRead -1 FirstWrite -1}
		timestamp {Type IO LastRead -1 FirstWrite -1}
		spike_counter {Type IO LastRead -1 FirstWrite -1}
		p_ZL15pre_eligibility_0 {Type IO LastRead -1 FirstWrite -1}
		p_ZL16post_eligibility_0 {Type IO LastRead -1 FirstWrite -1}
		p_ZL15pre_eligibility_1 {Type IO LastRead -1 FirstWrite -1}
		p_ZL16post_eligibility_1 {Type IO LastRead -1 FirstWrite -1}
		p_ZL15pre_eligibility_2 {Type IO LastRead -1 FirstWrite -1}
		p_ZL16post_eligibility_2 {Type IO LastRead -1 FirstWrite -1}
		p_ZL15pre_eligibility_3 {Type IO LastRead -1 FirstWrite -1}
		p_ZL16post_eligibility_3 {Type IO LastRead -1 FirstWrite -1}
		p_ZL15pre_eligibility_4 {Type IO LastRead -1 FirstWrite -1}
		p_ZL16post_eligibility_4 {Type IO LastRead -1 FirstWrite -1}
		p_ZL15pre_eligibility_5 {Type IO LastRead -1 FirstWrite -1}
		p_ZL16post_eligibility_5 {Type IO LastRead -1 FirstWrite -1}
		p_ZL15pre_eligibility_6 {Type IO LastRead -1 FirstWrite -1}
		p_ZL16post_eligibility_6 {Type IO LastRead -1 FirstWrite -1}
		p_ZL15pre_eligibility_7 {Type IO LastRead -1 FirstWrite -1}
		p_ZL16post_eligibility_7 {Type IO LastRead -1 FirstWrite -1}
		pre_traces_trace_0 {Type IO LastRead -1 FirstWrite -1}
		pre_traces_last_spike_time_0 {Type IO LastRead -1 FirstWrite -1}
		post_traces_trace_0 {Type IO LastRead -1 FirstWrite -1}
		post_traces_last_spike_time_0 {Type IO LastRead -1 FirstWrite -1}
		pre_traces_trace_1 {Type IO LastRead -1 FirstWrite -1}
		pre_traces_last_spike_time_1 {Type IO LastRead -1 FirstWrite -1}
		post_traces_trace_1 {Type IO LastRead -1 FirstWrite -1}
		post_traces_last_spike_time_1 {Type IO LastRead -1 FirstWrite -1}
		pre_traces_trace_2 {Type IO LastRead -1 FirstWrite -1}
		pre_traces_last_spike_time_2 {Type IO LastRead -1 FirstWrite -1}
		post_traces_trace_2 {Type IO LastRead -1 FirstWrite -1}
		post_traces_last_spike_time_2 {Type IO LastRead -1 FirstWrite -1}
		pre_traces_trace_3 {Type IO LastRead -1 FirstWrite -1}
		pre_traces_last_spike_time_3 {Type IO LastRead -1 FirstWrite -1}
		post_traces_trace_3 {Type IO LastRead -1 FirstWrite -1}
		post_traces_last_spike_time_3 {Type IO LastRead -1 FirstWrite -1}
		pre_traces_trace_4 {Type IO LastRead -1 FirstWrite -1}
		pre_traces_last_spike_time_4 {Type IO LastRead -1 FirstWrite -1}
		post_traces_trace_4 {Type IO LastRead -1 FirstWrite -1}
		post_traces_last_spike_time_4 {Type IO LastRead -1 FirstWrite -1}
		pre_traces_trace_5 {Type IO LastRead -1 FirstWrite -1}
		pre_traces_last_spike_time_5 {Type IO LastRead -1 FirstWrite -1}
		post_traces_trace_5 {Type IO LastRead -1 FirstWrite -1}
		post_traces_last_spike_time_5 {Type IO LastRead -1 FirstWrite -1}
		pre_traces_trace_6 {Type IO LastRead -1 FirstWrite -1}
		pre_traces_last_spike_time_6 {Type IO LastRead -1 FirstWrite -1}
		post_traces_trace_6 {Type IO LastRead -1 FirstWrite -1}
		post_traces_last_spike_time_6 {Type IO LastRead -1 FirstWrite -1}
		pre_traces_trace_7 {Type IO LastRead -1 FirstWrite -1}
		pre_traces_last_spike_time_7 {Type IO LastRead -1 FirstWrite -1}
		post_traces_trace_7 {Type IO LastRead -1 FirstWrite -1}
		post_traces_last_spike_time_7 {Type IO LastRead -1 FirstWrite -1}
		p_ZL13weight_memory_0_0 {Type IO LastRead -1 FirstWrite -1}
		p_ZL13weight_memory_0_1 {Type IO LastRead -1 FirstWrite -1}
		p_ZL13weight_memory_1_0 {Type IO LastRead -1 FirstWrite -1}
		p_ZL13weight_memory_1_1 {Type IO LastRead -1 FirstWrite -1}
		p_ZL13weight_memory_2_0 {Type IO LastRead -1 FirstWrite -1}
		p_ZL13weight_memory_2_1 {Type IO LastRead -1 FirstWrite -1}
		p_ZL13weight_memory_3_0 {Type IO LastRead -1 FirstWrite -1}
		p_ZL13weight_memory_3_1 {Type IO LastRead -1 FirstWrite -1}
		EXP_DECAY_LUT {Type I LastRead -1 FirstWrite -1}
		read_row {Type IO LastRead -1 FirstWrite -1}
		read_col {Type IO LastRead -1 FirstWrite -1}}
	snn_top_hls_Pipeline_RESET_ELIG {
		p_ZL15pre_eligibility_0 {Type O LastRead -1 FirstWrite 0}
		p_ZL16post_eligibility_0 {Type O LastRead -1 FirstWrite 0}
		p_ZL15pre_eligibility_1 {Type O LastRead -1 FirstWrite 0}
		p_ZL16post_eligibility_1 {Type O LastRead -1 FirstWrite 0}
		p_ZL15pre_eligibility_2 {Type O LastRead -1 FirstWrite 0}
		p_ZL16post_eligibility_2 {Type O LastRead -1 FirstWrite 0}
		p_ZL15pre_eligibility_3 {Type O LastRead -1 FirstWrite 0}
		p_ZL16post_eligibility_3 {Type O LastRead -1 FirstWrite 0}
		p_ZL15pre_eligibility_4 {Type O LastRead -1 FirstWrite 0}
		p_ZL16post_eligibility_4 {Type O LastRead -1 FirstWrite 0}
		p_ZL15pre_eligibility_5 {Type O LastRead -1 FirstWrite 0}
		p_ZL16post_eligibility_5 {Type O LastRead -1 FirstWrite 0}
		p_ZL15pre_eligibility_6 {Type O LastRead -1 FirstWrite 0}
		p_ZL16post_eligibility_6 {Type O LastRead -1 FirstWrite 0}
		p_ZL15pre_eligibility_7 {Type O LastRead -1 FirstWrite 0}
		p_ZL16post_eligibility_7 {Type O LastRead -1 FirstWrite 0}}
	snn_top_hls_Pipeline_RESET_TRACES {
		pre_traces_trace_0 {Type O LastRead -1 FirstWrite 0}
		pre_traces_last_spike_time_0 {Type O LastRead -1 FirstWrite 0}
		post_traces_trace_0 {Type O LastRead -1 FirstWrite 0}
		post_traces_last_spike_time_0 {Type O LastRead -1 FirstWrite 0}
		pre_traces_trace_1 {Type O LastRead -1 FirstWrite 0}
		pre_traces_last_spike_time_1 {Type O LastRead -1 FirstWrite 0}
		post_traces_trace_1 {Type O LastRead -1 FirstWrite 0}
		post_traces_last_spike_time_1 {Type O LastRead -1 FirstWrite 0}
		pre_traces_trace_2 {Type O LastRead -1 FirstWrite 0}
		pre_traces_last_spike_time_2 {Type O LastRead -1 FirstWrite 0}
		post_traces_trace_2 {Type O LastRead -1 FirstWrite 0}
		post_traces_last_spike_time_2 {Type O LastRead -1 FirstWrite 0}
		pre_traces_trace_3 {Type O LastRead -1 FirstWrite 0}
		pre_traces_last_spike_time_3 {Type O LastRead -1 FirstWrite 0}
		post_traces_trace_3 {Type O LastRead -1 FirstWrite 0}
		post_traces_last_spike_time_3 {Type O LastRead -1 FirstWrite 0}
		pre_traces_trace_4 {Type O LastRead -1 FirstWrite 0}
		pre_traces_last_spike_time_4 {Type O LastRead -1 FirstWrite 0}
		post_traces_trace_4 {Type O LastRead -1 FirstWrite 0}
		post_traces_last_spike_time_4 {Type O LastRead -1 FirstWrite 0}
		pre_traces_trace_5 {Type O LastRead -1 FirstWrite 0}
		pre_traces_last_spike_time_5 {Type O LastRead -1 FirstWrite 0}
		post_traces_trace_5 {Type O LastRead -1 FirstWrite 0}
		post_traces_last_spike_time_5 {Type O LastRead -1 FirstWrite 0}
		pre_traces_trace_6 {Type O LastRead -1 FirstWrite 0}
		pre_traces_last_spike_time_6 {Type O LastRead -1 FirstWrite 0}
		post_traces_trace_6 {Type O LastRead -1 FirstWrite 0}
		post_traces_last_spike_time_6 {Type O LastRead -1 FirstWrite 0}
		pre_traces_trace_7 {Type O LastRead -1 FirstWrite 0}
		pre_traces_last_spike_time_7 {Type O LastRead -1 FirstWrite 0}
		post_traces_trace_7 {Type O LastRead -1 FirstWrite 0}
		post_traces_last_spike_time_7 {Type O LastRead -1 FirstWrite 0}}
	snn_top_hls_Pipeline_INIT_WEIGHT_OUTER_INIT_WEIGHT_INNER {
		p_ZL13weight_memory_0_0 {Type O LastRead -1 FirstWrite 2}
		p_ZL13weight_memory_0_1 {Type O LastRead -1 FirstWrite 2}
		p_ZL13weight_memory_1_0 {Type O LastRead -1 FirstWrite 2}
		p_ZL13weight_memory_1_1 {Type O LastRead -1 FirstWrite 2}
		p_ZL13weight_memory_2_0 {Type O LastRead -1 FirstWrite 2}
		p_ZL13weight_memory_2_1 {Type O LastRead -1 FirstWrite 2}
		p_ZL13weight_memory_3_0 {Type O LastRead -1 FirstWrite 2}
		p_ZL13weight_memory_3_1 {Type O LastRead -1 FirstWrite 2}}
	process_pre_spike_aer {
		pre_id {Type I LastRead 0 FirstWrite -1}
		current_time {Type I LastRead 2 FirstWrite -1}
		pre_traces_trace_0 {Type IO LastRead 2 FirstWrite 5}
		pre_traces_trace_1 {Type IO LastRead 2 FirstWrite 5}
		pre_traces_trace_2 {Type IO LastRead 2 FirstWrite 5}
		pre_traces_trace_3 {Type IO LastRead 2 FirstWrite 5}
		pre_traces_trace_4 {Type IO LastRead 2 FirstWrite 5}
		pre_traces_trace_5 {Type IO LastRead 2 FirstWrite 5}
		pre_traces_trace_6 {Type IO LastRead 2 FirstWrite 5}
		pre_traces_trace_7 {Type IO LastRead 2 FirstWrite 5}
		pre_traces_last_spike_time_0 {Type IO LastRead 0 FirstWrite 4}
		pre_traces_last_spike_time_1 {Type IO LastRead 0 FirstWrite 4}
		pre_traces_last_spike_time_2 {Type IO LastRead 0 FirstWrite 4}
		pre_traces_last_spike_time_3 {Type IO LastRead 0 FirstWrite 4}
		pre_traces_last_spike_time_4 {Type IO LastRead 0 FirstWrite 4}
		pre_traces_last_spike_time_5 {Type IO LastRead 0 FirstWrite 4}
		pre_traces_last_spike_time_6 {Type IO LastRead 0 FirstWrite 4}
		pre_traces_last_spike_time_7 {Type IO LastRead 0 FirstWrite 4}
		EXP_DECAY_LUT {Type I LastRead 2 FirstWrite -1}
		post_traces_trace_3 {Type I LastRead 2 FirstWrite -1}
		post_traces_trace_7 {Type I LastRead 2 FirstWrite -1}
		post_traces_last_spike_time_3 {Type I LastRead 0 FirstWrite -1}
		post_traces_last_spike_time_7 {Type I LastRead 0 FirstWrite -1}
		post_traces_trace_2 {Type I LastRead 2 FirstWrite -1}
		post_traces_trace_6 {Type I LastRead 2 FirstWrite -1}
		post_traces_last_spike_time_2 {Type I LastRead 0 FirstWrite -1}
		post_traces_last_spike_time_6 {Type I LastRead 0 FirstWrite -1}
		post_traces_trace_1 {Type I LastRead 0 FirstWrite -1}
		post_traces_trace_5 {Type I LastRead 0 FirstWrite -1}
		post_traces_last_spike_time_1 {Type I LastRead 0 FirstWrite -1}
		post_traces_last_spike_time_5 {Type I LastRead 0 FirstWrite -1}
		post_traces_trace_0 {Type I LastRead 0 FirstWrite -1}
		post_traces_trace_4 {Type I LastRead 0 FirstWrite -1}
		post_traces_last_spike_time_0 {Type I LastRead 0 FirstWrite -1}
		post_traces_last_spike_time_4 {Type I LastRead 0 FirstWrite -1}
		p_ZL13weight_memory_0_0 {Type IO LastRead 4 FirstWrite 6}
		p_ZL13weight_memory_0_1 {Type IO LastRead 4 FirstWrite 6}
		p_ZL13weight_memory_1_0 {Type IO LastRead 4 FirstWrite 6}
		p_ZL13weight_memory_1_1 {Type IO LastRead 4 FirstWrite 6}
		p_ZL13weight_memory_2_0 {Type IO LastRead 5 FirstWrite 7}
		p_ZL13weight_memory_2_1 {Type IO LastRead 5 FirstWrite 7}
		p_ZL13weight_memory_3_0 {Type IO LastRead 5 FirstWrite 7}
		p_ZL13weight_memory_3_1 {Type IO LastRead 5 FirstWrite 7}}
	process_pre_spike_aer_Pipeline_LTD_LOOP {
		current_time {Type I LastRead 0 FirstWrite -1}
		pre_id {Type I LastRead 0 FirstWrite -1}
		empty {Type I LastRead 0 FirstWrite -1}
		post_traces_trace_3 {Type I LastRead 2 FirstWrite -1}
		post_traces_trace_7 {Type I LastRead 2 FirstWrite -1}
		post_traces_last_spike_time_3 {Type I LastRead 0 FirstWrite -1}
		post_traces_last_spike_time_7 {Type I LastRead 0 FirstWrite -1}
		EXP_DECAY_LUT {Type I LastRead -1 FirstWrite -1}
		post_traces_trace_2 {Type I LastRead 2 FirstWrite -1}
		post_traces_trace_6 {Type I LastRead 2 FirstWrite -1}
		post_traces_last_spike_time_2 {Type I LastRead 0 FirstWrite -1}
		post_traces_last_spike_time_6 {Type I LastRead 0 FirstWrite -1}
		post_traces_trace_1 {Type I LastRead 0 FirstWrite -1}
		post_traces_trace_5 {Type I LastRead 0 FirstWrite -1}
		post_traces_last_spike_time_1 {Type I LastRead 0 FirstWrite -1}
		post_traces_last_spike_time_5 {Type I LastRead 0 FirstWrite -1}
		post_traces_trace_0 {Type I LastRead 0 FirstWrite -1}
		post_traces_trace_4 {Type I LastRead 0 FirstWrite -1}
		post_traces_last_spike_time_0 {Type I LastRead 0 FirstWrite -1}
		post_traces_last_spike_time_4 {Type I LastRead 0 FirstWrite -1}
		p_ZL13weight_memory_0_0 {Type IO LastRead 4 FirstWrite 6}
		p_ZL13weight_memory_0_1 {Type IO LastRead 4 FirstWrite 6}
		p_ZL13weight_memory_1_0 {Type IO LastRead 4 FirstWrite 6}
		p_ZL13weight_memory_1_1 {Type IO LastRead 4 FirstWrite 6}
		p_ZL13weight_memory_2_0 {Type IO LastRead 5 FirstWrite 7}
		p_ZL13weight_memory_2_1 {Type IO LastRead 5 FirstWrite 7}
		p_ZL13weight_memory_3_0 {Type IO LastRead 5 FirstWrite 7}
		p_ZL13weight_memory_3_1 {Type IO LastRead 5 FirstWrite 7}}
	process_post_spike_aer {
		post_id {Type I LastRead 0 FirstWrite -1}
		current_time {Type I LastRead 2 FirstWrite -1}
		post_traces_trace_0 {Type IO LastRead 2 FirstWrite 5}
		post_traces_trace_1 {Type IO LastRead 2 FirstWrite 5}
		post_traces_trace_2 {Type IO LastRead 2 FirstWrite 5}
		post_traces_trace_3 {Type IO LastRead 2 FirstWrite 5}
		post_traces_trace_4 {Type IO LastRead 2 FirstWrite 5}
		post_traces_trace_5 {Type IO LastRead 2 FirstWrite 5}
		post_traces_trace_6 {Type IO LastRead 2 FirstWrite 5}
		post_traces_trace_7 {Type IO LastRead 2 FirstWrite 5}
		post_traces_last_spike_time_0 {Type IO LastRead 0 FirstWrite 4}
		post_traces_last_spike_time_1 {Type IO LastRead 0 FirstWrite 4}
		post_traces_last_spike_time_2 {Type IO LastRead 0 FirstWrite 4}
		post_traces_last_spike_time_3 {Type IO LastRead 0 FirstWrite 4}
		post_traces_last_spike_time_4 {Type IO LastRead 0 FirstWrite 4}
		post_traces_last_spike_time_5 {Type IO LastRead 0 FirstWrite 4}
		post_traces_last_spike_time_6 {Type IO LastRead 0 FirstWrite 4}
		post_traces_last_spike_time_7 {Type IO LastRead 0 FirstWrite 4}
		EXP_DECAY_LUT {Type I LastRead 2 FirstWrite -1}
		pre_traces_trace_3 {Type I LastRead 2 FirstWrite -1}
		pre_traces_trace_7 {Type I LastRead 2 FirstWrite -1}
		pre_traces_last_spike_time_3 {Type I LastRead 0 FirstWrite -1}
		pre_traces_last_spike_time_7 {Type I LastRead 0 FirstWrite -1}
		pre_traces_trace_2 {Type I LastRead 2 FirstWrite -1}
		pre_traces_trace_6 {Type I LastRead 2 FirstWrite -1}
		pre_traces_last_spike_time_2 {Type I LastRead 0 FirstWrite -1}
		pre_traces_last_spike_time_6 {Type I LastRead 0 FirstWrite -1}
		pre_traces_trace_1 {Type I LastRead 0 FirstWrite -1}
		pre_traces_trace_5 {Type I LastRead 0 FirstWrite -1}
		pre_traces_last_spike_time_1 {Type I LastRead 0 FirstWrite -1}
		pre_traces_last_spike_time_5 {Type I LastRead 0 FirstWrite -1}
		pre_traces_trace_0 {Type I LastRead 0 FirstWrite -1}
		pre_traces_trace_4 {Type I LastRead 0 FirstWrite -1}
		pre_traces_last_spike_time_0 {Type I LastRead 0 FirstWrite -1}
		pre_traces_last_spike_time_4 {Type I LastRead 0 FirstWrite -1}
		p_ZL13weight_memory_0_0 {Type IO LastRead 5 FirstWrite 6}
		p_ZL13weight_memory_1_0 {Type IO LastRead 5 FirstWrite 6}
		p_ZL13weight_memory_2_0 {Type IO LastRead 5 FirstWrite 6}
		p_ZL13weight_memory_3_0 {Type IO LastRead 5 FirstWrite 6}
		p_ZL13weight_memory_0_1 {Type IO LastRead 5 FirstWrite 6}
		p_ZL13weight_memory_1_1 {Type IO LastRead 5 FirstWrite 6}
		p_ZL13weight_memory_2_1 {Type IO LastRead 5 FirstWrite 6}
		p_ZL13weight_memory_3_1 {Type IO LastRead 5 FirstWrite 6}}
	process_post_spike_aer_Pipeline_LTP_LOOP {
		current_time {Type I LastRead 0 FirstWrite -1}
		zext_ln162 {Type I LastRead 0 FirstWrite -1}
		empty {Type I LastRead 0 FirstWrite -1}
		pre_traces_trace_3 {Type I LastRead 2 FirstWrite -1}
		pre_traces_trace_7 {Type I LastRead 2 FirstWrite -1}
		pre_traces_last_spike_time_3 {Type I LastRead 0 FirstWrite -1}
		pre_traces_last_spike_time_7 {Type I LastRead 0 FirstWrite -1}
		EXP_DECAY_LUT {Type I LastRead -1 FirstWrite -1}
		pre_traces_trace_2 {Type I LastRead 2 FirstWrite -1}
		pre_traces_trace_6 {Type I LastRead 2 FirstWrite -1}
		pre_traces_last_spike_time_2 {Type I LastRead 0 FirstWrite -1}
		pre_traces_last_spike_time_6 {Type I LastRead 0 FirstWrite -1}
		pre_traces_trace_1 {Type I LastRead 0 FirstWrite -1}
		pre_traces_trace_5 {Type I LastRead 0 FirstWrite -1}
		pre_traces_last_spike_time_1 {Type I LastRead 0 FirstWrite -1}
		pre_traces_last_spike_time_5 {Type I LastRead 0 FirstWrite -1}
		pre_traces_trace_0 {Type I LastRead 0 FirstWrite -1}
		pre_traces_trace_4 {Type I LastRead 0 FirstWrite -1}
		pre_traces_last_spike_time_0 {Type I LastRead 0 FirstWrite -1}
		pre_traces_last_spike_time_4 {Type I LastRead 0 FirstWrite -1}
		p_ZL13weight_memory_0_0 {Type IO LastRead 5 FirstWrite 6}
		p_ZL13weight_memory_1_0 {Type IO LastRead 5 FirstWrite 6}
		p_ZL13weight_memory_2_0 {Type IO LastRead 5 FirstWrite 6}
		p_ZL13weight_memory_3_0 {Type IO LastRead 5 FirstWrite 6}
		p_ZL13weight_memory_0_1 {Type IO LastRead 5 FirstWrite 6}
		p_ZL13weight_memory_1_1 {Type IO LastRead 5 FirstWrite 6}
		p_ZL13weight_memory_2_1 {Type IO LastRead 5 FirstWrite 6}
		p_ZL13weight_memory_3_1 {Type IO LastRead 5 FirstWrite 6}}
	apply_rstdp_reward {
		reward_signal {Type I LastRead 0 FirstWrite -1}
		p_ZL15pre_eligibility_0 {Type I LastRead 2 FirstWrite -1}
		p_ZL15pre_eligibility_1 {Type I LastRead 2 FirstWrite -1}
		p_ZL15pre_eligibility_2 {Type I LastRead 2 FirstWrite -1}
		p_ZL15pre_eligibility_3 {Type I LastRead 2 FirstWrite -1}
		p_ZL15pre_eligibility_4 {Type I LastRead 2 FirstWrite -1}
		p_ZL15pre_eligibility_5 {Type I LastRead 2 FirstWrite -1}
		p_ZL15pre_eligibility_6 {Type I LastRead 2 FirstWrite -1}
		p_ZL15pre_eligibility_7 {Type I LastRead 2 FirstWrite -1}
		p_ZL16post_eligibility_3 {Type I LastRead 0 FirstWrite -1}
		p_ZL16post_eligibility_7 {Type I LastRead 0 FirstWrite -1}
		p_ZL16post_eligibility_2 {Type I LastRead 0 FirstWrite -1}
		p_ZL16post_eligibility_6 {Type I LastRead 0 FirstWrite -1}
		p_ZL16post_eligibility_1 {Type I LastRead 0 FirstWrite -1}
		p_ZL16post_eligibility_5 {Type I LastRead 0 FirstWrite -1}
		p_ZL16post_eligibility_0 {Type I LastRead 0 FirstWrite -1}
		p_ZL16post_eligibility_4 {Type I LastRead 0 FirstWrite -1}
		p_ZL13weight_memory_0_0 {Type IO LastRead 1 FirstWrite 4}
		p_ZL13weight_memory_0_1 {Type IO LastRead 1 FirstWrite 4}
		p_ZL13weight_memory_1_0 {Type IO LastRead 1 FirstWrite 4}
		p_ZL13weight_memory_1_1 {Type IO LastRead 1 FirstWrite 4}
		p_ZL13weight_memory_2_0 {Type IO LastRead 3 FirstWrite 5}
		p_ZL13weight_memory_2_1 {Type IO LastRead 3 FirstWrite 5}
		p_ZL13weight_memory_3_0 {Type IO LastRead 3 FirstWrite 5}
		p_ZL13weight_memory_3_1 {Type IO LastRead 3 FirstWrite 5}}
	apply_rstdp_reward_Pipeline_RSTDP_INNER {
		i {Type I LastRead 0 FirstWrite -1}
		sext_ln239 {Type I LastRead 0 FirstWrite -1}
		icmp_ln243 {Type I LastRead 0 FirstWrite -1}
		icmp_ln243_1 {Type I LastRead 0 FirstWrite -1}
		icmp_ln243_2 {Type I LastRead 0 FirstWrite -1}
		empty_44 {Type I LastRead 0 FirstWrite -1}
		empty {Type I LastRead 0 FirstWrite -1}
		p_ZL16post_eligibility_3 {Type I LastRead 0 FirstWrite -1}
		p_ZL16post_eligibility_7 {Type I LastRead 0 FirstWrite -1}
		p_ZL16post_eligibility_2 {Type I LastRead 0 FirstWrite -1}
		p_ZL16post_eligibility_6 {Type I LastRead 0 FirstWrite -1}
		p_ZL16post_eligibility_1 {Type I LastRead 0 FirstWrite -1}
		p_ZL16post_eligibility_5 {Type I LastRead 0 FirstWrite -1}
		p_ZL16post_eligibility_0 {Type I LastRead 0 FirstWrite -1}
		p_ZL16post_eligibility_4 {Type I LastRead 0 FirstWrite -1}
		p_ZL13weight_memory_0_0 {Type IO LastRead 1 FirstWrite 4}
		p_ZL13weight_memory_0_1 {Type IO LastRead 1 FirstWrite 4}
		p_ZL13weight_memory_1_0 {Type IO LastRead 1 FirstWrite 4}
		p_ZL13weight_memory_1_1 {Type IO LastRead 1 FirstWrite 4}
		p_ZL13weight_memory_2_0 {Type IO LastRead 3 FirstWrite 5}
		p_ZL13weight_memory_2_1 {Type IO LastRead 3 FirstWrite 5}
		p_ZL13weight_memory_3_0 {Type IO LastRead 3 FirstWrite 5}
		p_ZL13weight_memory_3_1 {Type IO LastRead 3 FirstWrite 5}}
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
		p_ZL16post_eligibility_7 {Type IO LastRead 0 FirstWrite 2}}
	snn_top_hls_Pipeline_WEIGHT_SUM {
		weight_sum_out {Type O LastRead -1 FirstWrite 1}
		p_ZL13weight_memory_0_0 {Type I LastRead 1 FirstWrite -1}
		p_ZL13weight_memory_0_1 {Type I LastRead 1 FirstWrite -1}
		p_ZL13weight_memory_1_0 {Type I LastRead 1 FirstWrite -1}
		p_ZL13weight_memory_1_1 {Type I LastRead 1 FirstWrite -1}
		p_ZL13weight_memory_2_0 {Type I LastRead 1 FirstWrite -1}
		p_ZL13weight_memory_2_1 {Type I LastRead 1 FirstWrite -1}
		p_ZL13weight_memory_3_0 {Type I LastRead 1 FirstWrite -1}
		p_ZL13weight_memory_3_1 {Type I LastRead 1 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "18", "Max" : "6924"}
	, {"Name" : "Interval", "Min" : "19", "Max" : "6925"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	s_axis_spikes_V_data_V { axis {  { s_axis_spikes_TDATA in_data 0 32 } } }
	s_axis_spikes_V_keep_V { axis {  { s_axis_spikes_TKEEP in_data 0 4 } } }
	s_axis_spikes_V_strb_V { axis {  { s_axis_spikes_TSTRB in_data 0 4 } } }
	s_axis_spikes_V_user_V { axis {  { s_axis_spikes_TUSER in_data 0 1 } } }
	s_axis_spikes_V_last_V { axis {  { s_axis_spikes_TLAST in_data 0 1 } } }
	s_axis_spikes_V_id_V { axis {  { s_axis_spikes_TID in_data 0 1 } } }
	s_axis_spikes_V_dest_V { axis {  { s_axis_spikes_TVALID in_vld 0 1 }  { s_axis_spikes_TREADY in_acc 1 1 }  { s_axis_spikes_TDEST in_data 0 1 } } }
	m_axis_spikes_V_data_V { axis {  { m_axis_spikes_TDATA out_data 1 32 } } }
	m_axis_spikes_V_keep_V { axis {  { m_axis_spikes_TKEEP out_data 1 4 } } }
	m_axis_spikes_V_strb_V { axis {  { m_axis_spikes_TSTRB out_data 1 4 } } }
	m_axis_spikes_V_user_V { axis {  { m_axis_spikes_TUSER out_data 1 1 } } }
	m_axis_spikes_V_last_V { axis {  { m_axis_spikes_TLAST out_data 1 1 } } }
	m_axis_spikes_V_id_V { axis {  { m_axis_spikes_TID out_data 1 1 } } }
	m_axis_spikes_V_dest_V { axis {  { m_axis_spikes_TVALID out_vld 1 1 }  { m_axis_spikes_TREADY out_acc 0 1 }  { m_axis_spikes_TDEST out_data 1 1 } } }
	m_axis_weights_V_data_V { axis {  { m_axis_weights_TDATA out_data 1 32 } } }
	m_axis_weights_V_keep_V { axis {  { m_axis_weights_TKEEP out_data 1 4 } } }
	m_axis_weights_V_strb_V { axis {  { m_axis_weights_TSTRB out_data 1 4 } } }
	m_axis_weights_V_user_V { axis {  { m_axis_weights_TUSER out_data 1 1 } } }
	m_axis_weights_V_last_V { axis {  { m_axis_weights_TLAST out_data 1 1 } } }
	m_axis_weights_V_id_V { axis {  { m_axis_weights_TID out_data 1 1 } } }
	m_axis_weights_V_dest_V { axis {  { m_axis_weights_TVALID out_vld 1 1 }  { m_axis_weights_TREADY out_acc 0 1 }  { m_axis_weights_TDEST out_data 1 1 } } }
	spike_in_valid { ap_none {  { spike_in_valid out_data 1 1 } } }
	spike_in_neuron_id { ap_none {  { spike_in_neuron_id out_data 1 8 } } }
	spike_in_weight { ap_none {  { spike_in_weight out_data 1 8 } } }
	spike_in_ready { ap_none {  { spike_in_ready in_data 0 1 } } }
	spike_out_valid { ap_none {  { spike_out_valid in_data 0 1 } } }
	spike_out_neuron_id { ap_none {  { spike_out_neuron_id in_data 0 8 } } }
	spike_out_weight { ap_none {  { spike_out_weight in_data 0 8 } } }
	spike_out_ready { ap_none {  { spike_out_ready out_data 1 1 } } }
	snn_enable { ap_none {  { snn_enable out_data 1 1 } } }
	snn_reset { ap_none {  { snn_reset out_data 1 1 } } }
	threshold_out { ap_none {  { threshold_out out_data 1 16 } } }
	leak_rate_out { ap_none {  { leak_rate_out out_data 1 16 } } }
	snn_ready { ap_none {  { snn_ready in_data 0 1 } } }
	snn_busy { ap_none {  { snn_busy in_data 0 1 } } }
}

set maxi_interface_dict [dict create]

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
