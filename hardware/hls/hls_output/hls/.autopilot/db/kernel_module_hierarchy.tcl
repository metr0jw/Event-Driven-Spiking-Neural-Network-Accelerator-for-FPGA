set ModuleHierarchy {[{
"Name" : "snn_top_hls", "RefName" : "snn_top_hls","ID" : "0","Type" : "sequential",
"SubInsts" : [
	{"Name" : "grp_snn_top_hls_Pipeline_RESET_ELIG_fu_1209", "RefName" : "snn_top_hls_Pipeline_RESET_ELIG","ID" : "1","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "RESET_ELIG","RefName" : "RESET_ELIG","ID" : "2","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_RESET_TRACES_fu_1245", "RefName" : "snn_top_hls_Pipeline_RESET_TRACES","ID" : "3","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "RESET_TRACES","RefName" : "RESET_TRACES","ID" : "4","Type" : "pipeline"},]},
	{"Name" : "grp_snn_top_hls_Pipeline_INIT_WEIGHT_OUTER_INIT_WEIGHT_INNER_fu_1313", "RefName" : "snn_top_hls_Pipeline_INIT_WEIGHT_OUTER_INIT_WEIGHT_INNER","ID" : "5","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "INIT_WEIGHT_OUTER_INIT_WEIGHT_INNER","RefName" : "INIT_WEIGHT_OUTER_INIT_WEIGHT_INNER","ID" : "6","Type" : "pipeline"},]},
	{"Name" : "grp_process_pre_spike_aer_fu_1333", "RefName" : "process_pre_spike_aer","ID" : "7","Type" : "sequential",
		"SubInsts" : [
		{"Name" : "grp_process_pre_spike_aer_Pipeline_LTD_LOOP_fu_443", "RefName" : "process_pre_spike_aer_Pipeline_LTD_LOOP","ID" : "8","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "LTD_LOOP","RefName" : "LTD_LOOP","ID" : "9","Type" : "pipeline"},]},]},
	{"Name" : "grp_process_post_spike_aer_fu_1421", "RefName" : "process_post_spike_aer","ID" : "10","Type" : "sequential",
		"SubInsts" : [
		{"Name" : "grp_process_post_spike_aer_Pipeline_LTP_LOOP_fu_449", "RefName" : "process_post_spike_aer_Pipeline_LTP_LOOP","ID" : "11","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "LTP_LOOP","RefName" : "LTP_LOOP","ID" : "12","Type" : "pipeline"},]},]},
	{"Name" : "grp_apply_rstdp_reward_fu_1509", "RefName" : "apply_rstdp_reward","ID" : "13","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "RSTDP_OUTER","RefName" : "RSTDP_OUTER","ID" : "14","Type" : "no",
		"SubInsts" : [
		{"Name" : "grp_apply_rstdp_reward_Pipeline_RSTDP_INNER_fu_256", "RefName" : "apply_rstdp_reward_Pipeline_RSTDP_INNER","ID" : "15","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "RSTDP_INNER","RefName" : "RSTDP_INNER","ID" : "16","Type" : "pipeline"},]},]},]},
	{"Name" : "grp_decay_eligibility_traces_fu_1562", "RefName" : "decay_eligibility_traces","ID" : "17","Type" : "sequential",
		"SubInsts" : [
		{"Name" : "grp_decay_eligibility_traces_Pipeline_DECAY_PRE_fu_46", "RefName" : "decay_eligibility_traces_Pipeline_DECAY_PRE","ID" : "18","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "DECAY_PRE","RefName" : "DECAY_PRE","ID" : "19","Type" : "pipeline"},]},
		{"Name" : "grp_decay_eligibility_traces_Pipeline_DECAY_POST_fu_66", "RefName" : "decay_eligibility_traces_Pipeline_DECAY_POST","ID" : "20","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "DECAY_POST","RefName" : "DECAY_POST","ID" : "21","Type" : "pipeline"},]},]},
	{"Name" : "grp_snn_top_hls_Pipeline_WEIGHT_SUM_fu_1598", "RefName" : "snn_top_hls_Pipeline_WEIGHT_SUM","ID" : "22","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "WEIGHT_SUM","RefName" : "WEIGHT_SUM","ID" : "23","Type" : "pipeline"},]},]
}]}