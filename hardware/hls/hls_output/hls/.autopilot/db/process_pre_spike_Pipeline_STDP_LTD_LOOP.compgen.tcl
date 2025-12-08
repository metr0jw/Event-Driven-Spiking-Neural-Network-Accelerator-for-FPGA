# This script segment is generated automatically by AutoPilot

set name snn_top_hls_sdiv_24ns_16s_16_28_1
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {sdiv} IMPL {auto} LATENCY 27 ALLOW_PRAGMA 1
}


set name snn_top_hls_mul_16ns_16s_31_1_1
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {mul} IMPL {auto} LATENCY 0 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_mac_mulsub_16s_16ns_1ns_31_4_1 BINDTYPE {op} TYPE {all} IMPL {dsp_slice} LATENCY 3
}


# clear list
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_begin
    cg_default_interface_gen_bundle_begin
    AESL_LIB_XILADAPTER::native_axis_begin
}

# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 72 \
    name p_ZL16post_spike_times_7 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename p_ZL16post_spike_times_7 \
    op interface \
    ports { p_ZL16post_spike_times_7_address0 { O 3 vector } p_ZL16post_spike_times_7_ce0 { O 1 bit } p_ZL16post_spike_times_7_q0 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL16post_spike_times_7'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 73 \
    name p_ZL16post_spike_times_6 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename p_ZL16post_spike_times_6 \
    op interface \
    ports { p_ZL16post_spike_times_6_address0 { O 3 vector } p_ZL16post_spike_times_6_ce0 { O 1 bit } p_ZL16post_spike_times_6_q0 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL16post_spike_times_6'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 74 \
    name p_ZL16post_spike_times_5 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename p_ZL16post_spike_times_5 \
    op interface \
    ports { p_ZL16post_spike_times_5_address0 { O 3 vector } p_ZL16post_spike_times_5_ce0 { O 1 bit } p_ZL16post_spike_times_5_q0 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL16post_spike_times_5'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 75 \
    name p_ZL16post_spike_times_4 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename p_ZL16post_spike_times_4 \
    op interface \
    ports { p_ZL16post_spike_times_4_address0 { O 3 vector } p_ZL16post_spike_times_4_ce0 { O 1 bit } p_ZL16post_spike_times_4_q0 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL16post_spike_times_4'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 76 \
    name p_ZL16post_spike_times_3 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename p_ZL16post_spike_times_3 \
    op interface \
    ports { p_ZL16post_spike_times_3_address0 { O 3 vector } p_ZL16post_spike_times_3_ce0 { O 1 bit } p_ZL16post_spike_times_3_q0 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL16post_spike_times_3'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 77 \
    name p_ZL16post_spike_times_2 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename p_ZL16post_spike_times_2 \
    op interface \
    ports { p_ZL16post_spike_times_2_address0 { O 3 vector } p_ZL16post_spike_times_2_ce0 { O 1 bit } p_ZL16post_spike_times_2_q0 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL16post_spike_times_2'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 78 \
    name p_ZL16post_spike_times_1 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename p_ZL16post_spike_times_1 \
    op interface \
    ports { p_ZL16post_spike_times_1_address0 { O 3 vector } p_ZL16post_spike_times_1_ce0 { O 1 bit } p_ZL16post_spike_times_1_q0 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL16post_spike_times_1'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 79 \
    name p_ZL16post_spike_times_0 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename p_ZL16post_spike_times_0 \
    op interface \
    ports { p_ZL16post_spike_times_0_address0 { O 3 vector } p_ZL16post_spike_times_0_ce0 { O 1 bit } p_ZL16post_spike_times_0_q0 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL16post_spike_times_0'"
}
}


# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 64 \
    name params_stdp_window_val \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_params_stdp_window_val \
    op interface \
    ports { params_stdp_window_val { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 65 \
    name current_time \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_current_time \
    op interface \
    ports { current_time { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 66 \
    name sext_ln55 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln55 \
    op interface \
    ports { sext_ln55 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 67 \
    name zext_ln46 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_zext_ln46 \
    op interface \
    ports { zext_ln46 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 68 \
    name sext_ln48 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln48 \
    op interface \
    ports { sext_ln48 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 69 \
    name sext_ln58 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln58 \
    op interface \
    ports { sext_ln58 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 70 \
    name sext_ln93 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln93 \
    op interface \
    ports { sext_ln93 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 71 \
    name pre_id \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_pre_id \
    op interface \
    ports { pre_id { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 80 \
    name weight_update_fifo \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_weight_update_fifo \
    op interface \
    ports { weight_update_fifo_din { O 64 vector } weight_update_fifo_full_n { I 1 bit } weight_update_fifo_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id -1 \
    name ap_ctrl \
    type ap_ctrl \
    reset_level 1 \
    sync_rst true \
    corename ap_ctrl \
    op interface \
    ports { ap_start { I 1 bit } ap_ready { O 1 bit } ap_done { O 1 bit } ap_idle { O 1 bit } } \
} "
}


# Adapter definition:
set PortName ap_clk
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_clock] == "cg_default_interface_gen_clock"} {
eval "cg_default_interface_gen_clock { \
    id -2 \
    name ${PortName} \
    reset_level 1 \
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
set PortName ap_rst
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_reset] == "cg_default_interface_gen_reset"} {
eval "cg_default_interface_gen_reset { \
    id -3 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_rst \
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


# flow_control definition:
set InstName snn_top_hls_flow_control_loop_pipe_sequential_init_U
set CompName snn_top_hls_flow_control_loop_pipe_sequential_init
set name flow_control_loop_pipe_sequential_init
if {${::AESL::PGuard_autocg_gen} && ${::AESL::PGuard_autocg_ipmgen}} {
if {[info proc ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control] == "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control"} {
eval "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control { \
    name ${name} \
    prefix snn_top_hls_ \
}"
} else {
puts "@W \[IMPL-107\] Cannot find ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control, check your platform lib"
}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $CompName BINDTYPE interface TYPE internal_upc_flow_control INSTNAME $InstName
}


