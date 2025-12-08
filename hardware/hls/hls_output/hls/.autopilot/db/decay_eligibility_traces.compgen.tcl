# This script segment is generated automatically by AutoPilot

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
    id 258 \
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
    id 259 \
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
    id 260 \
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
    id 261 \
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
    id 257 \
    name params_trace_decay_val \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_params_trace_decay_val \
    op interface \
    ports { params_trace_decay_val { I 16 vector } } \
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


