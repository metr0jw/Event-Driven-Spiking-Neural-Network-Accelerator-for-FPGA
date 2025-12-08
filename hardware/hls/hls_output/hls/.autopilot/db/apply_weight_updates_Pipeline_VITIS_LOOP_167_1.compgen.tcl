# This script segment is generated automatically by AutoPilot

set name snn_top_hls_mul_16s_10s_26_1_1
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {mul} IMPL {auto} LATENCY 0 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_sparsemux_9_2_16_1_1 BINDTYPE {op} TYPE {sparsemux} IMPL {compactencoding_dontcare}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_sparsemux_17_3_8_1_1 BINDTYPE {op} TYPE {sparsemux} IMPL {compactencoding_dontcare}
}


set name snn_top_hls_mul_16s_16s_32_1_1
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {mul} IMPL {auto} LATENCY 0 ALLOW_PRAGMA 1
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler snn_top_hls_mac_muladd_16s_16s_24s_32_4_1 BINDTYPE {op} TYPE {all} IMPL {dsp_slice} LATENCY 3
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
    id 193 \
    name p_ZL13weight_memory_0 \
    reset_level 1 \
    sync_rst true \
    dir IO \
    corename p_ZL13weight_memory_0 \
    op interface \
    ports { p_ZL13weight_memory_0_address0 { O 9 vector } p_ZL13weight_memory_0_ce0 { O 1 bit } p_ZL13weight_memory_0_q0 { I 8 vector } p_ZL13weight_memory_0_address1 { O 9 vector } p_ZL13weight_memory_0_ce1 { O 1 bit } p_ZL13weight_memory_0_we1 { O 1 bit } p_ZL13weight_memory_0_d1 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL13weight_memory_0'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 194 \
    name p_ZL13weight_memory_1 \
    reset_level 1 \
    sync_rst true \
    dir IO \
    corename p_ZL13weight_memory_1 \
    op interface \
    ports { p_ZL13weight_memory_1_address0 { O 9 vector } p_ZL13weight_memory_1_ce0 { O 1 bit } p_ZL13weight_memory_1_q0 { I 8 vector } p_ZL13weight_memory_1_address1 { O 9 vector } p_ZL13weight_memory_1_ce1 { O 1 bit } p_ZL13weight_memory_1_we1 { O 1 bit } p_ZL13weight_memory_1_d1 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL13weight_memory_1'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 195 \
    name p_ZL13weight_memory_2 \
    reset_level 1 \
    sync_rst true \
    dir IO \
    corename p_ZL13weight_memory_2 \
    op interface \
    ports { p_ZL13weight_memory_2_address0 { O 9 vector } p_ZL13weight_memory_2_ce0 { O 1 bit } p_ZL13weight_memory_2_q0 { I 8 vector } p_ZL13weight_memory_2_address1 { O 9 vector } p_ZL13weight_memory_2_ce1 { O 1 bit } p_ZL13weight_memory_2_we1 { O 1 bit } p_ZL13weight_memory_2_d1 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL13weight_memory_2'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 196 \
    name p_ZL13weight_memory_3 \
    reset_level 1 \
    sync_rst true \
    dir IO \
    corename p_ZL13weight_memory_3 \
    op interface \
    ports { p_ZL13weight_memory_3_address0 { O 9 vector } p_ZL13weight_memory_3_ce0 { O 1 bit } p_ZL13weight_memory_3_q0 { I 8 vector } p_ZL13weight_memory_3_address1 { O 9 vector } p_ZL13weight_memory_3_ce1 { O 1 bit } p_ZL13weight_memory_3_we1 { O 1 bit } p_ZL13weight_memory_3_d1 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL13weight_memory_3'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 197 \
    name p_ZL13weight_memory_4 \
    reset_level 1 \
    sync_rst true \
    dir IO \
    corename p_ZL13weight_memory_4 \
    op interface \
    ports { p_ZL13weight_memory_4_address0 { O 9 vector } p_ZL13weight_memory_4_ce0 { O 1 bit } p_ZL13weight_memory_4_q0 { I 8 vector } p_ZL13weight_memory_4_address1 { O 9 vector } p_ZL13weight_memory_4_ce1 { O 1 bit } p_ZL13weight_memory_4_we1 { O 1 bit } p_ZL13weight_memory_4_d1 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL13weight_memory_4'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 198 \
    name p_ZL13weight_memory_5 \
    reset_level 1 \
    sync_rst true \
    dir IO \
    corename p_ZL13weight_memory_5 \
    op interface \
    ports { p_ZL13weight_memory_5_address0 { O 9 vector } p_ZL13weight_memory_5_ce0 { O 1 bit } p_ZL13weight_memory_5_q0 { I 8 vector } p_ZL13weight_memory_5_address1 { O 9 vector } p_ZL13weight_memory_5_ce1 { O 1 bit } p_ZL13weight_memory_5_we1 { O 1 bit } p_ZL13weight_memory_5_d1 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL13weight_memory_5'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 199 \
    name p_ZL13weight_memory_6 \
    reset_level 1 \
    sync_rst true \
    dir IO \
    corename p_ZL13weight_memory_6 \
    op interface \
    ports { p_ZL13weight_memory_6_address0 { O 9 vector } p_ZL13weight_memory_6_ce0 { O 1 bit } p_ZL13weight_memory_6_q0 { I 8 vector } p_ZL13weight_memory_6_address1 { O 9 vector } p_ZL13weight_memory_6_ce1 { O 1 bit } p_ZL13weight_memory_6_we1 { O 1 bit } p_ZL13weight_memory_6_d1 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL13weight_memory_6'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 200 \
    name p_ZL13weight_memory_7 \
    reset_level 1 \
    sync_rst true \
    dir IO \
    corename p_ZL13weight_memory_7 \
    op interface \
    ports { p_ZL13weight_memory_7_address0 { O 9 vector } p_ZL13weight_memory_7_ce0 { O 1 bit } p_ZL13weight_memory_7_q0 { I 8 vector } p_ZL13weight_memory_7_address1 { O 9 vector } p_ZL13weight_memory_7_ce1 { O 1 bit } p_ZL13weight_memory_7_we1 { O 1 bit } p_ZL13weight_memory_7_d1 { O 8 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL13weight_memory_7'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 201 \
    name p_ZL18eligibility_traces_0 \
    reset_level 1 \
    sync_rst true \
    dir IO \
    corename p_ZL18eligibility_traces_0 \
    op interface \
    ports { p_ZL18eligibility_traces_0_address0 { O 10 vector } p_ZL18eligibility_traces_0_ce0 { O 1 bit } p_ZL18eligibility_traces_0_q0 { I 16 vector } p_ZL18eligibility_traces_0_address1 { O 10 vector } p_ZL18eligibility_traces_0_ce1 { O 1 bit } p_ZL18eligibility_traces_0_we1 { O 1 bit } p_ZL18eligibility_traces_0_d1 { O 16 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL18eligibility_traces_0'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 202 \
    name p_ZL18eligibility_traces_1 \
    reset_level 1 \
    sync_rst true \
    dir IO \
    corename p_ZL18eligibility_traces_1 \
    op interface \
    ports { p_ZL18eligibility_traces_1_address0 { O 10 vector } p_ZL18eligibility_traces_1_ce0 { O 1 bit } p_ZL18eligibility_traces_1_q0 { I 16 vector } p_ZL18eligibility_traces_1_address1 { O 10 vector } p_ZL18eligibility_traces_1_ce1 { O 1 bit } p_ZL18eligibility_traces_1_we1 { O 1 bit } p_ZL18eligibility_traces_1_d1 { O 16 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL18eligibility_traces_1'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 203 \
    name p_ZL18eligibility_traces_2 \
    reset_level 1 \
    sync_rst true \
    dir IO \
    corename p_ZL18eligibility_traces_2 \
    op interface \
    ports { p_ZL18eligibility_traces_2_address0 { O 10 vector } p_ZL18eligibility_traces_2_ce0 { O 1 bit } p_ZL18eligibility_traces_2_q0 { I 16 vector } p_ZL18eligibility_traces_2_address1 { O 10 vector } p_ZL18eligibility_traces_2_ce1 { O 1 bit } p_ZL18eligibility_traces_2_we1 { O 1 bit } p_ZL18eligibility_traces_2_d1 { O 16 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL18eligibility_traces_2'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 204 \
    name p_ZL18eligibility_traces_3 \
    reset_level 1 \
    sync_rst true \
    dir IO \
    corename p_ZL18eligibility_traces_3 \
    op interface \
    ports { p_ZL18eligibility_traces_3_address0 { O 10 vector } p_ZL18eligibility_traces_3_ce0 { O 1 bit } p_ZL18eligibility_traces_3_q0 { I 16 vector } p_ZL18eligibility_traces_3_address1 { O 10 vector } p_ZL18eligibility_traces_3_ce1 { O 1 bit } p_ZL18eligibility_traces_3_we1 { O 1 bit } p_ZL18eligibility_traces_3_d1 { O 16 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'p_ZL18eligibility_traces_3'"
}
}


# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 188 \
    name sext_ln189 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln189 \
    op interface \
    ports { sext_ln189 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 189 \
    name and_ln177 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_and_ln177 \
    op interface \
    ports { and_ln177 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 190 \
    name sext_ln184_1 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln184_1 \
    op interface \
    ports { sext_ln184_1 { I 10 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 191 \
    name sext_ln184 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln184 \
    op interface \
    ports { sext_ln184 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 192 \
    name weight_update_fifo \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_weight_update_fifo \
    op interface \
    ports { weight_update_fifo_dout { I 64 vector } weight_update_fifo_empty_n { I 1 bit } weight_update_fifo_read { O 1 bit } } \
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


