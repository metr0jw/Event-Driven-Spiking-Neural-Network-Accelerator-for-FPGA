# This script segment is generated automatically by AutoPilot

if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_p_ZL15pre_eligibility_0_RAM_2P_BRAM_1R1W BINDTYPE {storage} TYPE {ram_2p} IMPL {bram} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_pre_traces_last_spike_time_0_RAM_2P_BRAM_1R1W BINDTYPE {storage} TYPE {ram_2p} IMPL {bram} LATENCY 2 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_p_ZL13weight_memory_0_0_RAM_2P_BRAM_1R1W BINDTYPE {storage} TYPE {ram_2p} IMPL {bram} LATENCY 2 ALLOW_PRAGMA 1
}


# clear list
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_begin
    cg_default_interface_gen_bundle_begin
    AESL_LIB_XILADAPTER::native_axis_begin
}

set axilite_register_dict [dict create]
set port_ctrl {
ctrl_reg { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 16
	offset_end 23
}
config_reg { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 24
	offset_end 31
}
learning_params { 
	dir I
	width 144
	depth 1
	mode ap_none
	offset 32
	offset_end 55
}
status_reg { 
	dir O
	width 32
	depth 1
	mode ap_vld
	offset 56
	offset_end 63
}
spike_count_reg { 
	dir O
	width 32
	depth 1
	mode ap_vld
	offset 72
	offset_end 79
}
weight_sum_reg { 
	dir O
	width 32
	depth 1
	mode ap_vld
	offset 88
	offset_end 95
}
version_reg { 
	dir O
	width 32
	depth 1
	mode ap_vld
	offset 104
	offset_end 111
}
reward_signal { 
	dir I
	width 8
	depth 1
	mode ap_none
	offset 120
	offset_end 127
}
ap_start { }
ap_done { }
ap_ready { }
ap_idle { }
interrupt {
}
}
dict set axilite_register_dict ctrl $port_ctrl


# Native S_AXILite:
if {${::AESL::PGuard_simmodel_gen}} {
	if {[info proc ::AESL_LIB_XILADAPTER::s_axilite_gen] == "::AESL_LIB_XILADAPTER::s_axilite_gen"} {
		eval "::AESL_LIB_XILADAPTER::s_axilite_gen { \
			id 328 \
			corename snn_top_hls_ctrl_axilite \
			name snn_top_hls_ctrl_s_axi \
			ports {$port_ctrl} \
			op interface \
			interrupt_clear_mode TOW \
			interrupt_trigger_type default \
			is_flushable 0 \
			is_datawidth64 0 \
			is_addrwidth64 1 \
			enable_mem_auto_widen 1 \
		} "
	} else {
		puts "@W \[IMPL-110\] Cannot find AXI Lite interface model in the library. Ignored generation of AXI Lite  interface for 'ctrl'"
	}
}

if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_ctrl_s_axi BINDTYPE interface TYPE interface_s_axilite
}

# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 329 \
    name s_axis_spikes_V_data_V \
    reset_level 0 \
    sync_rst true \
    corename {s_axis_spikes} \
    metadata {  } \
    op interface \
    ports { s_axis_spikes_TDATA { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 's_axis_spikes_V_data_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 330 \
    name s_axis_spikes_V_keep_V \
    reset_level 0 \
    sync_rst true \
    corename {s_axis_spikes} \
    metadata {  } \
    op interface \
    ports { s_axis_spikes_TKEEP { I 4 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 's_axis_spikes_V_keep_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 331 \
    name s_axis_spikes_V_strb_V \
    reset_level 0 \
    sync_rst true \
    corename {s_axis_spikes} \
    metadata {  } \
    op interface \
    ports { s_axis_spikes_TSTRB { I 4 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 's_axis_spikes_V_strb_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 332 \
    name s_axis_spikes_V_user_V \
    reset_level 0 \
    sync_rst true \
    corename {s_axis_spikes} \
    metadata {  } \
    op interface \
    ports { s_axis_spikes_TUSER { I 1 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 's_axis_spikes_V_user_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 333 \
    name s_axis_spikes_V_last_V \
    reset_level 0 \
    sync_rst true \
    corename {s_axis_spikes} \
    metadata {  } \
    op interface \
    ports { s_axis_spikes_TLAST { I 1 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 's_axis_spikes_V_last_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 334 \
    name s_axis_spikes_V_id_V \
    reset_level 0 \
    sync_rst true \
    corename {s_axis_spikes} \
    metadata {  } \
    op interface \
    ports { s_axis_spikes_TID { I 1 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 's_axis_spikes_V_id_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 335 \
    name s_axis_spikes_V_dest_V \
    reset_level 0 \
    sync_rst true \
    corename {s_axis_spikes} \
    metadata {  } \
    op interface \
    ports { s_axis_spikes_TVALID { I 1 bit } s_axis_spikes_TREADY { O 1 bit } s_axis_spikes_TDEST { I 1 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 's_axis_spikes_V_dest_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 336 \
    name m_axis_spikes_V_data_V \
    reset_level 0 \
    sync_rst true \
    corename {m_axis_spikes} \
    metadata {  } \
    op interface \
    ports { m_axis_spikes_TDATA { O 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'm_axis_spikes_V_data_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 337 \
    name m_axis_spikes_V_keep_V \
    reset_level 0 \
    sync_rst true \
    corename {m_axis_spikes} \
    metadata {  } \
    op interface \
    ports { m_axis_spikes_TKEEP { O 4 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'm_axis_spikes_V_keep_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 338 \
    name m_axis_spikes_V_strb_V \
    reset_level 0 \
    sync_rst true \
    corename {m_axis_spikes} \
    metadata {  } \
    op interface \
    ports { m_axis_spikes_TSTRB { O 4 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'm_axis_spikes_V_strb_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 339 \
    name m_axis_spikes_V_user_V \
    reset_level 0 \
    sync_rst true \
    corename {m_axis_spikes} \
    metadata {  } \
    op interface \
    ports { m_axis_spikes_TUSER { O 1 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'm_axis_spikes_V_user_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 340 \
    name m_axis_spikes_V_last_V \
    reset_level 0 \
    sync_rst true \
    corename {m_axis_spikes} \
    metadata {  } \
    op interface \
    ports { m_axis_spikes_TLAST { O 1 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'm_axis_spikes_V_last_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 341 \
    name m_axis_spikes_V_id_V \
    reset_level 0 \
    sync_rst true \
    corename {m_axis_spikes} \
    metadata {  } \
    op interface \
    ports { m_axis_spikes_TID { O 1 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'm_axis_spikes_V_id_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 342 \
    name m_axis_spikes_V_dest_V \
    reset_level 0 \
    sync_rst true \
    corename {m_axis_spikes} \
    metadata {  } \
    op interface \
    ports { m_axis_spikes_TVALID { O 1 bit } m_axis_spikes_TREADY { I 1 bit } m_axis_spikes_TDEST { O 1 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'm_axis_spikes_V_dest_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 343 \
    name m_axis_weights_V_data_V \
    reset_level 0 \
    sync_rst true \
    corename {m_axis_weights} \
    metadata {  } \
    op interface \
    ports { m_axis_weights_TDATA { O 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'm_axis_weights_V_data_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 344 \
    name m_axis_weights_V_keep_V \
    reset_level 0 \
    sync_rst true \
    corename {m_axis_weights} \
    metadata {  } \
    op interface \
    ports { m_axis_weights_TKEEP { O 4 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'm_axis_weights_V_keep_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 345 \
    name m_axis_weights_V_strb_V \
    reset_level 0 \
    sync_rst true \
    corename {m_axis_weights} \
    metadata {  } \
    op interface \
    ports { m_axis_weights_TSTRB { O 4 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'm_axis_weights_V_strb_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 346 \
    name m_axis_weights_V_user_V \
    reset_level 0 \
    sync_rst true \
    corename {m_axis_weights} \
    metadata {  } \
    op interface \
    ports { m_axis_weights_TUSER { O 1 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'm_axis_weights_V_user_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 347 \
    name m_axis_weights_V_last_V \
    reset_level 0 \
    sync_rst true \
    corename {m_axis_weights} \
    metadata {  } \
    op interface \
    ports { m_axis_weights_TLAST { O 1 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'm_axis_weights_V_last_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 348 \
    name m_axis_weights_V_id_V \
    reset_level 0 \
    sync_rst true \
    corename {m_axis_weights} \
    metadata {  } \
    op interface \
    ports { m_axis_weights_TID { O 1 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'm_axis_weights_V_id_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 349 \
    name m_axis_weights_V_dest_V \
    reset_level 0 \
    sync_rst true \
    corename {m_axis_weights} \
    metadata {  } \
    op interface \
    ports { m_axis_weights_TVALID { O 1 bit } m_axis_weights_TREADY { I 1 bit } m_axis_weights_TDEST { O 1 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'm_axis_weights_V_dest_V'"
}
}


# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 350 \
    name spike_in_valid \
    type other \
    dir O \
    reset_level 0 \
    sync_rst true \
    corename dc_spike_in_valid \
    op interface \
    ports { spike_in_valid { O 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 351 \
    name spike_in_neuron_id \
    type other \
    dir O \
    reset_level 0 \
    sync_rst true \
    corename dc_spike_in_neuron_id \
    op interface \
    ports { spike_in_neuron_id { O 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 352 \
    name spike_in_weight \
    type other \
    dir O \
    reset_level 0 \
    sync_rst true \
    corename dc_spike_in_weight \
    op interface \
    ports { spike_in_weight { O 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 353 \
    name spike_in_ready \
    type other \
    dir I \
    reset_level 0 \
    sync_rst true \
    corename dc_spike_in_ready \
    op interface \
    ports { spike_in_ready { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 354 \
    name spike_out_valid \
    type other \
    dir I \
    reset_level 0 \
    sync_rst true \
    corename dc_spike_out_valid \
    op interface \
    ports { spike_out_valid { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 355 \
    name spike_out_neuron_id \
    type other \
    dir I \
    reset_level 0 \
    sync_rst true \
    corename dc_spike_out_neuron_id \
    op interface \
    ports { spike_out_neuron_id { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 356 \
    name spike_out_weight \
    type other \
    dir I \
    reset_level 0 \
    sync_rst true \
    corename dc_spike_out_weight \
    op interface \
    ports { spike_out_weight { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 357 \
    name spike_out_ready \
    type other \
    dir O \
    reset_level 0 \
    sync_rst true \
    corename dc_spike_out_ready \
    op interface \
    ports { spike_out_ready { O 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 358 \
    name snn_enable \
    type other \
    dir O \
    reset_level 0 \
    sync_rst true \
    corename dc_snn_enable \
    op interface \
    ports { snn_enable { O 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 359 \
    name snn_reset \
    type other \
    dir O \
    reset_level 0 \
    sync_rst true \
    corename dc_snn_reset \
    op interface \
    ports { snn_reset { O 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 360 \
    name threshold_out \
    type other \
    dir O \
    reset_level 0 \
    sync_rst true \
    corename dc_threshold_out \
    op interface \
    ports { threshold_out { O 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 361 \
    name leak_rate_out \
    type other \
    dir O \
    reset_level 0 \
    sync_rst true \
    corename dc_leak_rate_out \
    op interface \
    ports { leak_rate_out { O 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 362 \
    name snn_ready \
    type other \
    dir I \
    reset_level 0 \
    sync_rst true \
    corename dc_snn_ready \
    op interface \
    ports { snn_ready { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 363 \
    name snn_busy \
    type other \
    dir I \
    reset_level 0 \
    sync_rst true \
    corename dc_snn_busy \
    op interface \
    ports { snn_busy { I 1 vector } } \
} "
}


# Adapter definition:
set PortName ap_clk
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_clock] == "cg_default_interface_gen_clock"} {
eval "cg_default_interface_gen_clock { \
    id -1 \
    name ${PortName} \
    reset_level 0 \
    sync_rst true \
    corename apif_ap_clk \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-113\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}


# Adapter definition:
set PortName ap_rst_n
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_reset] == "cg_default_interface_gen_reset"} {
eval "cg_default_interface_gen_reset { \
    id -2 \
    name ${PortName} \
    reset_level 0 \
    sync_rst true \
    corename apif_ap_rst_n \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-114\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}



# merge
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_end
    cg_default_interface_gen_bundle_end
    AESL_LIB_XILADAPTER::native_axis_end
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_regslice_both BINDTYPE {interface} TYPE {adapter} IMPL {reg_slice}
}


