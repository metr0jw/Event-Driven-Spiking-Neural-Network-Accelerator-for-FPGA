dict set slaves ctrl {ports {ctrl_reg {type i_ap_none width 32} config_reg {type i_ap_none width 32} learning_params {type i_ap_none width 144} status_reg {type o_ap_vld width 32} spike_count_reg {type o_ap_vld width 32} weight_sum_reg {type o_ap_vld width 32} version_reg {type o_ap_vld width 32} reward_signal {type i_ap_none width 8} ap_start {type ap_ctrl width 1} ap_done {type ap_ctrl width 1} ap_ready {type ap_ctrl width 1} ap_idle {type ap_ctrl width 1}} mems {} has_ctrl 1}
set datawidth 32
set addrwidth 64
set intr_clr_mode TOW
