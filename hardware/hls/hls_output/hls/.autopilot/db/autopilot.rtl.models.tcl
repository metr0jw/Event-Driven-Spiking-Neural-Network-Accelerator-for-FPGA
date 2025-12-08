set SynModuleInfo {
  {SRCNAME snn_top_hls_Pipeline_RESET_ELIG MODELNAME snn_top_hls_Pipeline_RESET_ELIG RTLNAME snn_top_hls_snn_top_hls_Pipeline_RESET_ELIG
    SUBMODULES {
      {MODELNAME snn_top_hls_flow_control_loop_pipe_sequential_init RTLNAME snn_top_hls_flow_control_loop_pipe_sequential_init BINDTYPE interface TYPE internal_upc_flow_control INSTNAME snn_top_hls_flow_control_loop_pipe_sequential_init_U}
    }
  }
  {SRCNAME snn_top_hls_Pipeline_RESET_TRACES MODELNAME snn_top_hls_Pipeline_RESET_TRACES RTLNAME snn_top_hls_snn_top_hls_Pipeline_RESET_TRACES}
  {SRCNAME snn_top_hls_Pipeline_INIT_WEIGHT_OUTER_INIT_WEIGHT_INNER MODELNAME snn_top_hls_Pipeline_INIT_WEIGHT_OUTER_INIT_WEIGHT_INNER RTLNAME snn_top_hls_snn_top_hls_Pipeline_INIT_WEIGHT_OUTER_INIT_WEIGHT_INNER}
  {SRCNAME process_pre_spike_aer_Pipeline_LTD_LOOP MODELNAME process_pre_spike_aer_Pipeline_LTD_LOOP RTLNAME snn_top_hls_process_pre_spike_aer_Pipeline_LTD_LOOP
    SUBMODULES {
      {MODELNAME snn_top_hls_mul_8ns_8ns_16_1_1 RTLNAME snn_top_hls_mul_8ns_8ns_16_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME snn_top_hls_process_pre_spike_aer_Pipeline_LTD_LOOP_EXP_DECAY_LUT_ROM_AUTO_1R RTLNAME snn_top_hls_process_pre_spike_aer_Pipeline_LTD_LOOP_EXP_DECAY_LUT_ROM_AUTO_1R BINDTYPE storage TYPE rom IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME process_pre_spike_aer MODELNAME process_pre_spike_aer RTLNAME snn_top_hls_process_pre_spike_aer
    SUBMODULES {
      {MODELNAME snn_top_hls_sparsemux_17_3_16_1_1 RTLNAME snn_top_hls_sparsemux_17_3_16_1_1 BINDTYPE op TYPE sparsemux IMPL compactencoding_dontcare}
      {MODELNAME snn_top_hls_sparsemux_17_3_8_1_1 RTLNAME snn_top_hls_sparsemux_17_3_8_1_1 BINDTYPE op TYPE sparsemux IMPL compactencoding_dontcare}
    }
  }
  {SRCNAME process_post_spike_aer_Pipeline_LTP_LOOP MODELNAME process_post_spike_aer_Pipeline_LTP_LOOP RTLNAME snn_top_hls_process_post_spike_aer_Pipeline_LTP_LOOP
    SUBMODULES {
      {MODELNAME snn_top_hls_sparsemux_9_2_8_1_1 RTLNAME snn_top_hls_sparsemux_9_2_8_1_1 BINDTYPE op TYPE sparsemux IMPL compactencoding_dontcare}
    }
  }
  {SRCNAME process_post_spike_aer MODELNAME process_post_spike_aer RTLNAME snn_top_hls_process_post_spike_aer}
  {SRCNAME apply_rstdp_reward_Pipeline_RSTDP_INNER MODELNAME apply_rstdp_reward_Pipeline_RSTDP_INNER RTLNAME snn_top_hls_apply_rstdp_reward_Pipeline_RSTDP_INNER
    SUBMODULES {
      {MODELNAME snn_top_hls_mul_8s_8s_16_1_1 RTLNAME snn_top_hls_mul_8s_8s_16_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME snn_top_hls_sparsemux_9_3_7_1_1 RTLNAME snn_top_hls_sparsemux_9_3_7_1_1 BINDTYPE op TYPE sparsemux IMPL onehotencoding_realdef}
    }
  }
  {SRCNAME apply_rstdp_reward MODELNAME apply_rstdp_reward RTLNAME snn_top_hls_apply_rstdp_reward
    SUBMODULES {
      {MODELNAME snn_top_hls_sparsemux_7_2_2_1_1 RTLNAME snn_top_hls_sparsemux_7_2_2_1_1 BINDTYPE op TYPE sparsemux IMPL onehotencoding_realdef}
    }
  }
  {SRCNAME decay_eligibility_traces_Pipeline_DECAY_PRE MODELNAME decay_eligibility_traces_Pipeline_DECAY_PRE RTLNAME snn_top_hls_decay_eligibility_traces_Pipeline_DECAY_PRE}
  {SRCNAME decay_eligibility_traces_Pipeline_DECAY_POST MODELNAME decay_eligibility_traces_Pipeline_DECAY_POST RTLNAME snn_top_hls_decay_eligibility_traces_Pipeline_DECAY_POST}
  {SRCNAME decay_eligibility_traces MODELNAME decay_eligibility_traces RTLNAME snn_top_hls_decay_eligibility_traces}
  {SRCNAME snn_top_hls_Pipeline_WEIGHT_SUM MODELNAME snn_top_hls_Pipeline_WEIGHT_SUM RTLNAME snn_top_hls_snn_top_hls_Pipeline_WEIGHT_SUM}
  {SRCNAME snn_top_hls MODELNAME snn_top_hls RTLNAME snn_top_hls IS_TOP 1
    SUBMODULES {
      {MODELNAME snn_top_hls_p_ZL15pre_eligibility_0_RAM_2P_BRAM_1R1W RTLNAME snn_top_hls_p_ZL15pre_eligibility_0_RAM_2P_BRAM_1R1W BINDTYPE storage TYPE ram_2p IMPL bram LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME snn_top_hls_pre_traces_last_spike_time_0_RAM_2P_BRAM_1R1W RTLNAME snn_top_hls_pre_traces_last_spike_time_0_RAM_2P_BRAM_1R1W BINDTYPE storage TYPE ram_2p IMPL bram LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME snn_top_hls_p_ZL13weight_memory_0_0_RAM_2P_BRAM_1R1W RTLNAME snn_top_hls_p_ZL13weight_memory_0_0_RAM_2P_BRAM_1R1W BINDTYPE storage TYPE ram_2p IMPL bram LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME snn_top_hls_ctrl_s_axi RTLNAME snn_top_hls_ctrl_s_axi BINDTYPE interface TYPE interface_s_axilite}
      {MODELNAME snn_top_hls_regslice_both RTLNAME snn_top_hls_regslice_both BINDTYPE interface TYPE adapter IMPL reg_slice}
    }
  }
}
