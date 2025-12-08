set ModuleHierarchy {[{
"Name" : "snn_top_hls", "RefName" : "snn_top_hls","ID" : "0","Type" : "sequential",
"SubInsts" : [
	{"Name" : "grp_snn_top_hls_Pipeline_RESET_PRE_fu_994", "RefName" : "snn_top_hls_Pipeline_RESET_PRE","ID" : "1","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "RESET_PRE","RefName" : "RESET_PRE","ID" : "2","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_RESET_TRACE_OUTER_RESET_TRACE_INNER_fu_1030", "RefName" : "snn_top_hls_Pipeline_RESET_TRACE_OUTER_RESET_TRACE_INNER","ID" : "3","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "RESET_TRACE_OUTER_RESET_TRACE_INNER","RefName" : "RESET_TRACE_OUTER_RESET_TRACE_INNER","ID" : "4","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_INIT_WEIGHT_OUTER_INIT_WEIGHT_INNER_fu_1042", "RefName" : "snn_top_hls_Pipeline_INIT_WEIGHT_OUTER_INIT_WEIGHT_INNER","ID" : "5","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "INIT_WEIGHT_OUTER_INIT_WEIGHT_INNER","RefName" : "INIT_WEIGHT_OUTER_INIT_WEIGHT_INNER","ID" : "6","Type" : "pipeline"},]},
	{"Name" : "grp_process_pre_spike_fu_1062", "RefName" : "process_pre_spike","ID" : "7","Type" : "sequential",
		"SubInsts" : [
		{"Name" : "grp_process_pre_spike_Pipeline_STDP_LTD_LOOP_fu_256", "RefName" : "process_pre_spike_Pipeline_STDP_LTD_LOOP","ID" : "8","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "STDP_LTD_LOOP","RefName" : "STDP_LTD_LOOP","ID" : "9","Type" : "pipeline"},]},]},
	{"Name" : "grp_process_post_spike_fu_1108", "RefName" : "process_post_spike","ID" : "10","Type" : "sequential",
		"SubInsts" : [
		{"Name" : "grp_process_post_spike_Pipeline_STDP_LTP_LOOP_fu_256", "RefName" : "process_post_spike_Pipeline_STDP_LTP_LOOP","ID" : "11","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "STDP_LTP_LOOP","RefName" : "STDP_LTP_LOOP","ID" : "12","Type" : "pipeline"},]},]},
	{"Name" : "grp_apply_weight_updates_fu_1154", "RefName" : "apply_weight_updates","ID" : "13","Type" : "sequential",
		"SubInsts" : [
		{"Name" : "grp_apply_weight_updates_Pipeline_VITIS_LOOP_167_1_fu_108", "RefName" : "apply_weight_updates_Pipeline_VITIS_LOOP_167_1","ID" : "14","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "VITIS_LOOP_167_1","RefName" : "VITIS_LOOP_167_1","ID" : "15","Type" : "pipeline"},]},]},
	{"Name" : "grp_apply_reward_signal_fu_1188", "RefName" : "apply_reward_signal","ID" : "16","Type" : "sequential",
		"SubInsts" : [
		{"Name" : "grp_apply_reward_signal_Pipeline_REWARD_OUTER_REWARD_INNER_fu_78", "RefName" : "apply_reward_signal_Pipeline_REWARD_OUTER_REWARD_INNER","ID" : "17","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "REWARD_OUTER_REWARD_INNER","RefName" : "REWARD_OUTER_REWARD_INNER","ID" : "18","Type" : "pipeline"},]},]},
	{"Name" : "grp_decay_eligibility_traces_fu_1218", "RefName" : "decay_eligibility_traces","ID" : "19","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "DECAY_OUTER_DECAY_INNER","RefName" : "DECAY_OUTER_DECAY_INNER","ID" : "20","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_VITIS_LOOP_529_1_fu_1231", "RefName" : "snn_top_hls_Pipeline_VITIS_LOOP_529_1","ID" : "21","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_529_1","RefName" : "VITIS_LOOP_529_1","ID" : "22","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_VITIS_LOOP_529_11_fu_1252", "RefName" : "snn_top_hls_Pipeline_VITIS_LOOP_529_11","ID" : "23","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_529_1","RefName" : "VITIS_LOOP_529_1","ID" : "24","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_VITIS_LOOP_529_12_fu_1274", "RefName" : "snn_top_hls_Pipeline_VITIS_LOOP_529_12","ID" : "25","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_529_1","RefName" : "VITIS_LOOP_529_1","ID" : "26","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_VITIS_LOOP_529_13_fu_1296", "RefName" : "snn_top_hls_Pipeline_VITIS_LOOP_529_13","ID" : "27","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_529_1","RefName" : "VITIS_LOOP_529_1","ID" : "28","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_VITIS_LOOP_529_14_fu_1318", "RefName" : "snn_top_hls_Pipeline_VITIS_LOOP_529_14","ID" : "29","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_529_1","RefName" : "VITIS_LOOP_529_1","ID" : "30","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_VITIS_LOOP_529_15_fu_1340", "RefName" : "snn_top_hls_Pipeline_VITIS_LOOP_529_15","ID" : "31","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_529_1","RefName" : "VITIS_LOOP_529_1","ID" : "32","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_VITIS_LOOP_529_16_fu_1362", "RefName" : "snn_top_hls_Pipeline_VITIS_LOOP_529_16","ID" : "33","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_529_1","RefName" : "VITIS_LOOP_529_1","ID" : "34","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_VITIS_LOOP_529_17_fu_1384", "RefName" : "snn_top_hls_Pipeline_VITIS_LOOP_529_17","ID" : "35","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_529_1","RefName" : "VITIS_LOOP_529_1","ID" : "36","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_VITIS_LOOP_529_18_fu_1406", "RefName" : "snn_top_hls_Pipeline_VITIS_LOOP_529_18","ID" : "37","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_529_1","RefName" : "VITIS_LOOP_529_1","ID" : "38","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_VITIS_LOOP_529_19_fu_1428", "RefName" : "snn_top_hls_Pipeline_VITIS_LOOP_529_19","ID" : "39","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_529_1","RefName" : "VITIS_LOOP_529_1","ID" : "40","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_VITIS_LOOP_529_110_fu_1450", "RefName" : "snn_top_hls_Pipeline_VITIS_LOOP_529_110","ID" : "41","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_529_1","RefName" : "VITIS_LOOP_529_1","ID" : "42","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_VITIS_LOOP_529_111_fu_1472", "RefName" : "snn_top_hls_Pipeline_VITIS_LOOP_529_111","ID" : "43","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_529_1","RefName" : "VITIS_LOOP_529_1","ID" : "44","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_VITIS_LOOP_529_112_fu_1494", "RefName" : "snn_top_hls_Pipeline_VITIS_LOOP_529_112","ID" : "45","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_529_1","RefName" : "VITIS_LOOP_529_1","ID" : "46","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_VITIS_LOOP_529_113_fu_1516", "RefName" : "snn_top_hls_Pipeline_VITIS_LOOP_529_113","ID" : "47","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_529_1","RefName" : "VITIS_LOOP_529_1","ID" : "48","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_VITIS_LOOP_529_114_fu_1538", "RefName" : "snn_top_hls_Pipeline_VITIS_LOOP_529_114","ID" : "49","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_529_1","RefName" : "VITIS_LOOP_529_1","ID" : "50","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_VITIS_LOOP_529_115_fu_1560", "RefName" : "snn_top_hls_Pipeline_VITIS_LOOP_529_115","ID" : "51","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_529_1","RefName" : "VITIS_LOOP_529_1","ID" : "52","Type" : "pipeline"},]},]
}]}