#include "hls_signal_handler.h"
#include <algorithm>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include "ap_fixed.h"
#include "ap_int.h"
#include "autopilot_cbe.h"
#include "hls_half.h"
#include "hls_directio.h"
#include "hls_stream.h"

using namespace std;

// wrapc file define:
#define AUTOTB_TVIN_ctrl_reg "../tv/cdatafile/c.snn_top_hls.autotvin_ctrl_reg.dat"
#define AUTOTB_TVOUT_ctrl_reg "../tv/cdatafile/c.snn_top_hls.autotvout_ctrl_reg.dat"
#define AUTOTB_TVIN_config_reg "../tv/cdatafile/c.snn_top_hls.autotvin_config_reg.dat"
#define AUTOTB_TVOUT_config_reg "../tv/cdatafile/c.snn_top_hls.autotvout_config_reg.dat"
#define AUTOTB_TVIN_learning_params "../tv/cdatafile/c.snn_top_hls.autotvin_learning_params.dat"
#define AUTOTB_TVOUT_learning_params "../tv/cdatafile/c.snn_top_hls.autotvout_learning_params.dat"
#define AUTOTB_TVIN_status_reg "../tv/cdatafile/c.snn_top_hls.autotvin_status_reg.dat"
#define AUTOTB_TVOUT_status_reg "../tv/cdatafile/c.snn_top_hls.autotvout_status_reg.dat"
#define AUTOTB_TVIN_spike_count_reg "../tv/cdatafile/c.snn_top_hls.autotvin_spike_count_reg.dat"
#define AUTOTB_TVOUT_spike_count_reg "../tv/cdatafile/c.snn_top_hls.autotvout_spike_count_reg.dat"
#define AUTOTB_TVIN_weight_sum_reg "../tv/cdatafile/c.snn_top_hls.autotvin_weight_sum_reg.dat"
#define AUTOTB_TVOUT_weight_sum_reg "../tv/cdatafile/c.snn_top_hls.autotvout_weight_sum_reg.dat"
#define AUTOTB_TVIN_version_reg "../tv/cdatafile/c.snn_top_hls.autotvin_version_reg.dat"
#define AUTOTB_TVOUT_version_reg "../tv/cdatafile/c.snn_top_hls.autotvout_version_reg.dat"
#define AUTOTB_TVIN_s_axis_spikes_V_data_V "../tv/cdatafile/c.snn_top_hls.autotvin_s_axis_spikes_V_data_V.dat"
#define AUTOTB_TVOUT_s_axis_spikes_V_data_V "../tv/cdatafile/c.snn_top_hls.autotvout_s_axis_spikes_V_data_V.dat"
#define AUTOTB_TVIN_s_axis_spikes_V_keep_V "../tv/cdatafile/c.snn_top_hls.autotvin_s_axis_spikes_V_keep_V.dat"
#define AUTOTB_TVOUT_s_axis_spikes_V_keep_V "../tv/cdatafile/c.snn_top_hls.autotvout_s_axis_spikes_V_keep_V.dat"
#define AUTOTB_TVIN_s_axis_spikes_V_strb_V "../tv/cdatafile/c.snn_top_hls.autotvin_s_axis_spikes_V_strb_V.dat"
#define AUTOTB_TVOUT_s_axis_spikes_V_strb_V "../tv/cdatafile/c.snn_top_hls.autotvout_s_axis_spikes_V_strb_V.dat"
#define AUTOTB_TVIN_s_axis_spikes_V_user_V "../tv/cdatafile/c.snn_top_hls.autotvin_s_axis_spikes_V_user_V.dat"
#define AUTOTB_TVOUT_s_axis_spikes_V_user_V "../tv/cdatafile/c.snn_top_hls.autotvout_s_axis_spikes_V_user_V.dat"
#define AUTOTB_TVIN_s_axis_spikes_V_last_V "../tv/cdatafile/c.snn_top_hls.autotvin_s_axis_spikes_V_last_V.dat"
#define AUTOTB_TVOUT_s_axis_spikes_V_last_V "../tv/cdatafile/c.snn_top_hls.autotvout_s_axis_spikes_V_last_V.dat"
#define AUTOTB_TVIN_s_axis_spikes_V_id_V "../tv/cdatafile/c.snn_top_hls.autotvin_s_axis_spikes_V_id_V.dat"
#define AUTOTB_TVOUT_s_axis_spikes_V_id_V "../tv/cdatafile/c.snn_top_hls.autotvout_s_axis_spikes_V_id_V.dat"
#define AUTOTB_TVIN_s_axis_spikes_V_dest_V "../tv/cdatafile/c.snn_top_hls.autotvin_s_axis_spikes_V_dest_V.dat"
#define AUTOTB_TVOUT_s_axis_spikes_V_dest_V "../tv/cdatafile/c.snn_top_hls.autotvout_s_axis_spikes_V_dest_V.dat"
#define WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_data_V "../tv/stream_size/stream_size_in_s_axis_spikes_V_data_V.dat"
#define WRAPC_STREAM_INGRESS_STATUS_s_axis_spikes_V_data_V "../tv/stream_size/stream_ingress_status_s_axis_spikes_V_data_V.dat"
#define WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_keep_V "../tv/stream_size/stream_size_in_s_axis_spikes_V_keep_V.dat"
#define WRAPC_STREAM_INGRESS_STATUS_s_axis_spikes_V_keep_V "../tv/stream_size/stream_ingress_status_s_axis_spikes_V_keep_V.dat"
#define WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_strb_V "../tv/stream_size/stream_size_in_s_axis_spikes_V_strb_V.dat"
#define WRAPC_STREAM_INGRESS_STATUS_s_axis_spikes_V_strb_V "../tv/stream_size/stream_ingress_status_s_axis_spikes_V_strb_V.dat"
#define WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_user_V "../tv/stream_size/stream_size_in_s_axis_spikes_V_user_V.dat"
#define WRAPC_STREAM_INGRESS_STATUS_s_axis_spikes_V_user_V "../tv/stream_size/stream_ingress_status_s_axis_spikes_V_user_V.dat"
#define WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_last_V "../tv/stream_size/stream_size_in_s_axis_spikes_V_last_V.dat"
#define WRAPC_STREAM_INGRESS_STATUS_s_axis_spikes_V_last_V "../tv/stream_size/stream_ingress_status_s_axis_spikes_V_last_V.dat"
#define WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_id_V "../tv/stream_size/stream_size_in_s_axis_spikes_V_id_V.dat"
#define WRAPC_STREAM_INGRESS_STATUS_s_axis_spikes_V_id_V "../tv/stream_size/stream_ingress_status_s_axis_spikes_V_id_V.dat"
#define WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_dest_V "../tv/stream_size/stream_size_in_s_axis_spikes_V_dest_V.dat"
#define WRAPC_STREAM_INGRESS_STATUS_s_axis_spikes_V_dest_V "../tv/stream_size/stream_ingress_status_s_axis_spikes_V_dest_V.dat"
#define AUTOTB_TVIN_m_axis_spikes_V_data_V "../tv/cdatafile/c.snn_top_hls.autotvin_m_axis_spikes_V_data_V.dat"
#define AUTOTB_TVOUT_m_axis_spikes_V_data_V "../tv/cdatafile/c.snn_top_hls.autotvout_m_axis_spikes_V_data_V.dat"
#define AUTOTB_TVIN_m_axis_spikes_V_keep_V "../tv/cdatafile/c.snn_top_hls.autotvin_m_axis_spikes_V_keep_V.dat"
#define AUTOTB_TVOUT_m_axis_spikes_V_keep_V "../tv/cdatafile/c.snn_top_hls.autotvout_m_axis_spikes_V_keep_V.dat"
#define AUTOTB_TVIN_m_axis_spikes_V_strb_V "../tv/cdatafile/c.snn_top_hls.autotvin_m_axis_spikes_V_strb_V.dat"
#define AUTOTB_TVOUT_m_axis_spikes_V_strb_V "../tv/cdatafile/c.snn_top_hls.autotvout_m_axis_spikes_V_strb_V.dat"
#define AUTOTB_TVIN_m_axis_spikes_V_user_V "../tv/cdatafile/c.snn_top_hls.autotvin_m_axis_spikes_V_user_V.dat"
#define AUTOTB_TVOUT_m_axis_spikes_V_user_V "../tv/cdatafile/c.snn_top_hls.autotvout_m_axis_spikes_V_user_V.dat"
#define AUTOTB_TVIN_m_axis_spikes_V_last_V "../tv/cdatafile/c.snn_top_hls.autotvin_m_axis_spikes_V_last_V.dat"
#define AUTOTB_TVOUT_m_axis_spikes_V_last_V "../tv/cdatafile/c.snn_top_hls.autotvout_m_axis_spikes_V_last_V.dat"
#define AUTOTB_TVIN_m_axis_spikes_V_id_V "../tv/cdatafile/c.snn_top_hls.autotvin_m_axis_spikes_V_id_V.dat"
#define AUTOTB_TVOUT_m_axis_spikes_V_id_V "../tv/cdatafile/c.snn_top_hls.autotvout_m_axis_spikes_V_id_V.dat"
#define AUTOTB_TVIN_m_axis_spikes_V_dest_V "../tv/cdatafile/c.snn_top_hls.autotvin_m_axis_spikes_V_dest_V.dat"
#define AUTOTB_TVOUT_m_axis_spikes_V_dest_V "../tv/cdatafile/c.snn_top_hls.autotvout_m_axis_spikes_V_dest_V.dat"
#define WRAPC_STREAM_SIZE_OUT_m_axis_spikes_V_data_V "../tv/stream_size/stream_size_out_m_axis_spikes_V_data_V.dat"
#define WRAPC_STREAM_EGRESS_STATUS_m_axis_spikes_V_data_V "../tv/stream_size/stream_egress_status_m_axis_spikes_V_data_V.dat"
#define WRAPC_STREAM_SIZE_OUT_m_axis_spikes_V_keep_V "../tv/stream_size/stream_size_out_m_axis_spikes_V_keep_V.dat"
#define WRAPC_STREAM_EGRESS_STATUS_m_axis_spikes_V_keep_V "../tv/stream_size/stream_egress_status_m_axis_spikes_V_keep_V.dat"
#define WRAPC_STREAM_SIZE_OUT_m_axis_spikes_V_strb_V "../tv/stream_size/stream_size_out_m_axis_spikes_V_strb_V.dat"
#define WRAPC_STREAM_EGRESS_STATUS_m_axis_spikes_V_strb_V "../tv/stream_size/stream_egress_status_m_axis_spikes_V_strb_V.dat"
#define WRAPC_STREAM_SIZE_OUT_m_axis_spikes_V_user_V "../tv/stream_size/stream_size_out_m_axis_spikes_V_user_V.dat"
#define WRAPC_STREAM_EGRESS_STATUS_m_axis_spikes_V_user_V "../tv/stream_size/stream_egress_status_m_axis_spikes_V_user_V.dat"
#define WRAPC_STREAM_SIZE_OUT_m_axis_spikes_V_last_V "../tv/stream_size/stream_size_out_m_axis_spikes_V_last_V.dat"
#define WRAPC_STREAM_EGRESS_STATUS_m_axis_spikes_V_last_V "../tv/stream_size/stream_egress_status_m_axis_spikes_V_last_V.dat"
#define WRAPC_STREAM_SIZE_OUT_m_axis_spikes_V_id_V "../tv/stream_size/stream_size_out_m_axis_spikes_V_id_V.dat"
#define WRAPC_STREAM_EGRESS_STATUS_m_axis_spikes_V_id_V "../tv/stream_size/stream_egress_status_m_axis_spikes_V_id_V.dat"
#define WRAPC_STREAM_SIZE_OUT_m_axis_spikes_V_dest_V "../tv/stream_size/stream_size_out_m_axis_spikes_V_dest_V.dat"
#define WRAPC_STREAM_EGRESS_STATUS_m_axis_spikes_V_dest_V "../tv/stream_size/stream_egress_status_m_axis_spikes_V_dest_V.dat"
#define AUTOTB_TVIN_m_axis_weights_V_data_V "../tv/cdatafile/c.snn_top_hls.autotvin_m_axis_weights_V_data_V.dat"
#define AUTOTB_TVOUT_m_axis_weights_V_data_V "../tv/cdatafile/c.snn_top_hls.autotvout_m_axis_weights_V_data_V.dat"
#define AUTOTB_TVIN_m_axis_weights_V_keep_V "../tv/cdatafile/c.snn_top_hls.autotvin_m_axis_weights_V_keep_V.dat"
#define AUTOTB_TVOUT_m_axis_weights_V_keep_V "../tv/cdatafile/c.snn_top_hls.autotvout_m_axis_weights_V_keep_V.dat"
#define AUTOTB_TVIN_m_axis_weights_V_strb_V "../tv/cdatafile/c.snn_top_hls.autotvin_m_axis_weights_V_strb_V.dat"
#define AUTOTB_TVOUT_m_axis_weights_V_strb_V "../tv/cdatafile/c.snn_top_hls.autotvout_m_axis_weights_V_strb_V.dat"
#define AUTOTB_TVIN_m_axis_weights_V_user_V "../tv/cdatafile/c.snn_top_hls.autotvin_m_axis_weights_V_user_V.dat"
#define AUTOTB_TVOUT_m_axis_weights_V_user_V "../tv/cdatafile/c.snn_top_hls.autotvout_m_axis_weights_V_user_V.dat"
#define AUTOTB_TVIN_m_axis_weights_V_last_V "../tv/cdatafile/c.snn_top_hls.autotvin_m_axis_weights_V_last_V.dat"
#define AUTOTB_TVOUT_m_axis_weights_V_last_V "../tv/cdatafile/c.snn_top_hls.autotvout_m_axis_weights_V_last_V.dat"
#define AUTOTB_TVIN_m_axis_weights_V_id_V "../tv/cdatafile/c.snn_top_hls.autotvin_m_axis_weights_V_id_V.dat"
#define AUTOTB_TVOUT_m_axis_weights_V_id_V "../tv/cdatafile/c.snn_top_hls.autotvout_m_axis_weights_V_id_V.dat"
#define AUTOTB_TVIN_m_axis_weights_V_dest_V "../tv/cdatafile/c.snn_top_hls.autotvin_m_axis_weights_V_dest_V.dat"
#define AUTOTB_TVOUT_m_axis_weights_V_dest_V "../tv/cdatafile/c.snn_top_hls.autotvout_m_axis_weights_V_dest_V.dat"
#define WRAPC_STREAM_SIZE_OUT_m_axis_weights_V_data_V "../tv/stream_size/stream_size_out_m_axis_weights_V_data_V.dat"
#define WRAPC_STREAM_EGRESS_STATUS_m_axis_weights_V_data_V "../tv/stream_size/stream_egress_status_m_axis_weights_V_data_V.dat"
#define WRAPC_STREAM_SIZE_OUT_m_axis_weights_V_keep_V "../tv/stream_size/stream_size_out_m_axis_weights_V_keep_V.dat"
#define WRAPC_STREAM_EGRESS_STATUS_m_axis_weights_V_keep_V "../tv/stream_size/stream_egress_status_m_axis_weights_V_keep_V.dat"
#define WRAPC_STREAM_SIZE_OUT_m_axis_weights_V_strb_V "../tv/stream_size/stream_size_out_m_axis_weights_V_strb_V.dat"
#define WRAPC_STREAM_EGRESS_STATUS_m_axis_weights_V_strb_V "../tv/stream_size/stream_egress_status_m_axis_weights_V_strb_V.dat"
#define WRAPC_STREAM_SIZE_OUT_m_axis_weights_V_user_V "../tv/stream_size/stream_size_out_m_axis_weights_V_user_V.dat"
#define WRAPC_STREAM_EGRESS_STATUS_m_axis_weights_V_user_V "../tv/stream_size/stream_egress_status_m_axis_weights_V_user_V.dat"
#define WRAPC_STREAM_SIZE_OUT_m_axis_weights_V_last_V "../tv/stream_size/stream_size_out_m_axis_weights_V_last_V.dat"
#define WRAPC_STREAM_EGRESS_STATUS_m_axis_weights_V_last_V "../tv/stream_size/stream_egress_status_m_axis_weights_V_last_V.dat"
#define WRAPC_STREAM_SIZE_OUT_m_axis_weights_V_id_V "../tv/stream_size/stream_size_out_m_axis_weights_V_id_V.dat"
#define WRAPC_STREAM_EGRESS_STATUS_m_axis_weights_V_id_V "../tv/stream_size/stream_egress_status_m_axis_weights_V_id_V.dat"
#define WRAPC_STREAM_SIZE_OUT_m_axis_weights_V_dest_V "../tv/stream_size/stream_size_out_m_axis_weights_V_dest_V.dat"
#define WRAPC_STREAM_EGRESS_STATUS_m_axis_weights_V_dest_V "../tv/stream_size/stream_egress_status_m_axis_weights_V_dest_V.dat"
#define AUTOTB_TVIN_reward_signal "../tv/cdatafile/c.snn_top_hls.autotvin_reward_signal.dat"
#define AUTOTB_TVOUT_reward_signal "../tv/cdatafile/c.snn_top_hls.autotvout_reward_signal.dat"
#define AUTOTB_TVIN_spike_in_valid "../tv/cdatafile/c.snn_top_hls.autotvin_spike_in_valid.dat"
#define AUTOTB_TVOUT_spike_in_valid "../tv/cdatafile/c.snn_top_hls.autotvout_spike_in_valid.dat"
#define AUTOTB_TVIN_spike_in_neuron_id "../tv/cdatafile/c.snn_top_hls.autotvin_spike_in_neuron_id.dat"
#define AUTOTB_TVOUT_spike_in_neuron_id "../tv/cdatafile/c.snn_top_hls.autotvout_spike_in_neuron_id.dat"
#define AUTOTB_TVIN_spike_in_weight "../tv/cdatafile/c.snn_top_hls.autotvin_spike_in_weight.dat"
#define AUTOTB_TVOUT_spike_in_weight "../tv/cdatafile/c.snn_top_hls.autotvout_spike_in_weight.dat"
#define AUTOTB_TVIN_spike_in_ready "../tv/cdatafile/c.snn_top_hls.autotvin_spike_in_ready.dat"
#define AUTOTB_TVOUT_spike_in_ready "../tv/cdatafile/c.snn_top_hls.autotvout_spike_in_ready.dat"
#define AUTOTB_TVIN_spike_out_valid "../tv/cdatafile/c.snn_top_hls.autotvin_spike_out_valid.dat"
#define AUTOTB_TVOUT_spike_out_valid "../tv/cdatafile/c.snn_top_hls.autotvout_spike_out_valid.dat"
#define AUTOTB_TVIN_spike_out_neuron_id "../tv/cdatafile/c.snn_top_hls.autotvin_spike_out_neuron_id.dat"
#define AUTOTB_TVOUT_spike_out_neuron_id "../tv/cdatafile/c.snn_top_hls.autotvout_spike_out_neuron_id.dat"
#define AUTOTB_TVIN_spike_out_weight "../tv/cdatafile/c.snn_top_hls.autotvin_spike_out_weight.dat"
#define AUTOTB_TVOUT_spike_out_weight "../tv/cdatafile/c.snn_top_hls.autotvout_spike_out_weight.dat"
#define AUTOTB_TVIN_spike_out_ready "../tv/cdatafile/c.snn_top_hls.autotvin_spike_out_ready.dat"
#define AUTOTB_TVOUT_spike_out_ready "../tv/cdatafile/c.snn_top_hls.autotvout_spike_out_ready.dat"
#define AUTOTB_TVIN_snn_enable "../tv/cdatafile/c.snn_top_hls.autotvin_snn_enable.dat"
#define AUTOTB_TVOUT_snn_enable "../tv/cdatafile/c.snn_top_hls.autotvout_snn_enable.dat"
#define AUTOTB_TVIN_snn_reset "../tv/cdatafile/c.snn_top_hls.autotvin_snn_reset.dat"
#define AUTOTB_TVOUT_snn_reset "../tv/cdatafile/c.snn_top_hls.autotvout_snn_reset.dat"
#define AUTOTB_TVIN_threshold_out "../tv/cdatafile/c.snn_top_hls.autotvin_threshold_out.dat"
#define AUTOTB_TVOUT_threshold_out "../tv/cdatafile/c.snn_top_hls.autotvout_threshold_out.dat"
#define AUTOTB_TVIN_leak_rate_out "../tv/cdatafile/c.snn_top_hls.autotvin_leak_rate_out.dat"
#define AUTOTB_TVOUT_leak_rate_out "../tv/cdatafile/c.snn_top_hls.autotvout_leak_rate_out.dat"
#define AUTOTB_TVIN_snn_ready "../tv/cdatafile/c.snn_top_hls.autotvin_snn_ready.dat"
#define AUTOTB_TVOUT_snn_ready "../tv/cdatafile/c.snn_top_hls.autotvout_snn_ready.dat"
#define AUTOTB_TVIN_snn_busy "../tv/cdatafile/c.snn_top_hls.autotvin_snn_busy.dat"
#define AUTOTB_TVOUT_snn_busy "../tv/cdatafile/c.snn_top_hls.autotvout_snn_busy.dat"


// tvout file define:
#define AUTOTB_TVOUT_PC_status_reg "../tv/rtldatafile/rtl.snn_top_hls.autotvout_status_reg.dat"
#define AUTOTB_TVOUT_PC_spike_count_reg "../tv/rtldatafile/rtl.snn_top_hls.autotvout_spike_count_reg.dat"
#define AUTOTB_TVOUT_PC_weight_sum_reg "../tv/rtldatafile/rtl.snn_top_hls.autotvout_weight_sum_reg.dat"
#define AUTOTB_TVOUT_PC_version_reg "../tv/rtldatafile/rtl.snn_top_hls.autotvout_version_reg.dat"
#define AUTOTB_TVOUT_PC_s_axis_spikes_V_data_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_s_axis_spikes_V_data_V.dat"
#define AUTOTB_TVOUT_PC_s_axis_spikes_V_keep_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_s_axis_spikes_V_keep_V.dat"
#define AUTOTB_TVOUT_PC_s_axis_spikes_V_strb_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_s_axis_spikes_V_strb_V.dat"
#define AUTOTB_TVOUT_PC_s_axis_spikes_V_user_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_s_axis_spikes_V_user_V.dat"
#define AUTOTB_TVOUT_PC_s_axis_spikes_V_last_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_s_axis_spikes_V_last_V.dat"
#define AUTOTB_TVOUT_PC_s_axis_spikes_V_id_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_s_axis_spikes_V_id_V.dat"
#define AUTOTB_TVOUT_PC_s_axis_spikes_V_dest_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_s_axis_spikes_V_dest_V.dat"
#define AUTOTB_TVOUT_PC_m_axis_spikes_V_data_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_m_axis_spikes_V_data_V.dat"
#define AUTOTB_TVOUT_PC_m_axis_spikes_V_keep_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_m_axis_spikes_V_keep_V.dat"
#define AUTOTB_TVOUT_PC_m_axis_spikes_V_strb_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_m_axis_spikes_V_strb_V.dat"
#define AUTOTB_TVOUT_PC_m_axis_spikes_V_user_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_m_axis_spikes_V_user_V.dat"
#define AUTOTB_TVOUT_PC_m_axis_spikes_V_last_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_m_axis_spikes_V_last_V.dat"
#define AUTOTB_TVOUT_PC_m_axis_spikes_V_id_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_m_axis_spikes_V_id_V.dat"
#define AUTOTB_TVOUT_PC_m_axis_spikes_V_dest_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_m_axis_spikes_V_dest_V.dat"
#define AUTOTB_TVOUT_PC_m_axis_weights_V_data_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_m_axis_weights_V_data_V.dat"
#define AUTOTB_TVOUT_PC_m_axis_weights_V_keep_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_m_axis_weights_V_keep_V.dat"
#define AUTOTB_TVOUT_PC_m_axis_weights_V_strb_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_m_axis_weights_V_strb_V.dat"
#define AUTOTB_TVOUT_PC_m_axis_weights_V_user_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_m_axis_weights_V_user_V.dat"
#define AUTOTB_TVOUT_PC_m_axis_weights_V_last_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_m_axis_weights_V_last_V.dat"
#define AUTOTB_TVOUT_PC_m_axis_weights_V_id_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_m_axis_weights_V_id_V.dat"
#define AUTOTB_TVOUT_PC_m_axis_weights_V_dest_V "../tv/rtldatafile/rtl.snn_top_hls.autotvout_m_axis_weights_V_dest_V.dat"
#define AUTOTB_TVOUT_PC_spike_in_valid "../tv/rtldatafile/rtl.snn_top_hls.autotvout_spike_in_valid.dat"
#define AUTOTB_TVOUT_PC_spike_in_neuron_id "../tv/rtldatafile/rtl.snn_top_hls.autotvout_spike_in_neuron_id.dat"
#define AUTOTB_TVOUT_PC_spike_in_weight "../tv/rtldatafile/rtl.snn_top_hls.autotvout_spike_in_weight.dat"
#define AUTOTB_TVOUT_PC_spike_out_ready "../tv/rtldatafile/rtl.snn_top_hls.autotvout_spike_out_ready.dat"
#define AUTOTB_TVOUT_PC_snn_enable "../tv/rtldatafile/rtl.snn_top_hls.autotvout_snn_enable.dat"
#define AUTOTB_TVOUT_PC_snn_reset "../tv/rtldatafile/rtl.snn_top_hls.autotvout_snn_reset.dat"
#define AUTOTB_TVOUT_PC_threshold_out "../tv/rtldatafile/rtl.snn_top_hls.autotvout_threshold_out.dat"
#define AUTOTB_TVOUT_PC_leak_rate_out "../tv/rtldatafile/rtl.snn_top_hls.autotvout_leak_rate_out.dat"


namespace hls::sim
{
  template<size_t n>
  struct Byte {
    unsigned char a[n];

    Byte()
    {
      for (size_t i = 0; i < n; ++i) {
        a[i] = 0;
      }
    }

    template<typename T>
    Byte<n>& operator= (const T &val)
    {
      std::memcpy(a, &val, n);
      return *this;
    }
  };

  struct SimException : public std::exception {
    const std::string msg;
    const size_t line;
    SimException(const std::string &msg, const size_t line)
      : msg(msg), line(line)
    {
    }
  };

  void errExit(const size_t line, const std::string &msg)
  {
    std::string s;
    s += "ERROR";
//  s += '(';
//  s += __FILE__;
//  s += ":";
//  s += std::to_string(line);
//  s += ')';
    s += ": ";
    s += msg;
    s += "\n";
    fputs(s.c_str(), stderr);
    exit(1);
  }
}


static std::vector<unsigned> autorestart_seq;
extern "C" {
  void __hls_sim_static_autorestart_seq_push(int value);
}

void __hls_sim_static_autorestart_seq_push(int value) {
  autorestart_seq.push_back(value);
}
namespace hls::sim
{
  size_t divide_ceil(size_t a, size_t b)
  {
    return (a + b - 1) / b;
  }

  const bool little_endian()
  {
    int a = 1;
    return *(char*)&a == 1;
  }

  inline void rev_endian(unsigned char *p, size_t nbytes)
  {
    std::reverse(p, p+nbytes);
  }

  const bool LE = little_endian();

  inline size_t least_nbyte(size_t width)
  {
    return (width+7)>>3;
  }

  std::string formatData(unsigned char *pos, size_t wbits)
  {
    size_t wbytes = least_nbyte(wbits);
    size_t i = LE ? wbytes-1 : 0;
    auto next = [&] () {
      auto c = pos[i];
      LE ? --i : ++i;
      return c;
    };
    std::ostringstream ss;
    ss << "0x";
    if (int t = (wbits & 0x7)) {
      if (t <= 4) {
        unsigned char mask = (1<<t)-1;
        ss << std::hex << std::setfill('0') << std::setw(1)
           << (int) (next() & mask);
        wbytes -= 1;
      }
    }
    for (size_t i = 0; i < wbytes; ++i) {
      ss << std::hex << std::setfill('0') << std::setw(2) << (int)next();
    }
    return ss.str();
  }

  char ord(char c)
  {
    if (c >= 'a' && c <= 'f') {
      return c-'a'+10;
    } else if (c >= 'A' && c <= 'F') {
      return c-'A'+10;
    } else if (c >= '0' && c <= '9') {
      return c-'0';
    } else {
      throw SimException("Not Hexdecimal Digit", __LINE__);
    }
  }

  void unformatData(const char *data, unsigned char *put, size_t pbytes = 0)
  {
    size_t nchars = strlen(data+2);
    size_t nbytes = (nchars+1)>>1;
    if (pbytes == 0) {
      pbytes = nbytes;
    } else if (pbytes > nbytes) {
      throw SimException("Wrong size specified", __LINE__);
    }
    put = LE ? put : put+pbytes-1;
    auto nextp = [&] () {
      return LE ? put++ : put--;
    };
    const char *c = data + (nchars + 2) - 1;
    auto next = [&] () {
      char res { *c == 'x' ? (char)0 : ord(*c) };
      --c;
      return res;
    };
    for (size_t i = 0; i < pbytes; ++i) {
      char l = next();
      char h = next();
      *nextp() = (h<<4)+l;
    }
  }

  char* strip(char *s)
  {
    while (isspace(*s)) {
      ++s;
    }
    for (char *p = s+strlen(s)-1; p >= s; --p) {
      if (isspace(*p)) {
        *p = 0;
      } else {
        return s;
      }
    }
    return s;
  }

  size_t sum(const std::vector<size_t> &v)
  {
    size_t res = 0;
    for (const auto &e : v) {
      res += e;
    }
    return res;
  }

  const char* bad = "Bad TV file";
  const char* err = "Error on TV file";

  const unsigned char bmark[] = {
    0x5a, 0x5a, 0xa5, 0xa5, 0x0f, 0x0f, 0xf0, 0xf0
  };

  class Input {
    FILE *fp;
    long pos;

    void read(unsigned char *buf, size_t size)
    {
      if (fread(buf, size, 1, fp) != 1) {
        throw SimException(bad, __LINE__);
      }
      if (LE) {
        rev_endian(buf, size);
      }
    }

  public:
    void advance(size_t nbytes)
    {
      if (fseek(fp, nbytes, SEEK_CUR) == -1) {
        throw SimException(bad, __LINE__);
      }
    }

    Input(const char *path) : fp(nullptr)
    {
      fp = fopen(path, "rb");
      if (fp == nullptr) {
        errExit(__LINE__, err);
      }
    }

    void begin()
    {
      advance(8);
      pos = ftell(fp);
    }

    void reset()
    {
      fseek(fp, pos, SEEK_SET);
    }

    void into(unsigned char *param, size_t wbytes, size_t asize, size_t nbytes)
    {
      size_t n = nbytes / asize;
      size_t r = nbytes % asize;
      for (size_t i = 0; i < n; ++i) {
        read(param, wbytes);
        param += asize;
      }
      if (r > 0) {
        advance(asize-r);
        read(param, r);
      }
    }

    ~Input()
    {
      long curPos = ftell(fp);
      unsigned char buf[8];
      size_t res = fread(buf, 8, 1, fp);
      fclose(fp);
      if (res != 1) {
        errExit(__LINE__, bad);
      }
      // curPos == 0 -> the file is only opened but not read
      if (curPos != 0 && std::memcmp(buf, bmark, 8) != 0) {
        errExit(__LINE__, bad);
      }
    }
  };

  class Output {
    FILE *fp;

    void write(unsigned char *buf, size_t size)
    {
      if (LE) {
        rev_endian(buf, size);
      }
      if (fwrite(buf, size, 1, fp) != 1) {
        throw SimException(err, __LINE__);
      }
      if (LE) {
        rev_endian(buf, size);
      }
    }

  public:
    Output(const char *path) : fp(nullptr)
    {
      fp = fopen(path, "wb");
      if (fp == nullptr) {
        errExit(__LINE__, err);
      }
    }

    void begin(size_t total)
    {
      unsigned char buf[8] = {0};
      std::memcpy(buf, &total, sizeof(buf));
      write(buf, sizeof(buf));
    }

    void from(unsigned char *param, size_t wbytes, size_t asize, size_t nbytes, size_t skip)
    {
      param -= asize*skip;
      size_t n = divide_ceil(nbytes, asize);
      for (size_t i = 0; i < n; ++i) {
        write(param, wbytes);
        param += asize;
      }
    }

    ~Output()
    {
      size_t res = fwrite(bmark, 8, 1, fp);
      fclose(fp);
      if (res != 1) {
        errExit(__LINE__, err);
      }
    }
  };

  class Reader {
    FILE *fp;
    long pos;
    int size;
    char *s;

    void readline()
    {
      s = fgets(s, size, fp);
      if (s == nullptr) {
        throw SimException(bad, __LINE__);
      }
    }

  public:
    Reader(const char *path) : fp(nullptr), size(1<<12), s(new char[size])
    {
      try {
        fp = fopen(path, "r");
        if (fp == nullptr) {
          throw SimException(err, __LINE__);
        } else {
          readline();
          static const char mark[] = "[[[runtime]]]\n";
          if (strcmp(s, mark) != 0) {
            throw SimException(bad, __LINE__);
          }
        }
      } catch (const hls::sim::SimException &e) {
        errExit(e.line, e.msg);
      }
    }

    ~Reader()
    {
      fclose(fp);
      delete[] s;
    }

    void begin()
    {
      readline();
      static const char mark[] = "[[transaction]]";
      if (strncmp(s, mark, strlen(mark)) != 0) {
        throw SimException(bad, __LINE__);
      }
      pos = ftell(fp);
    }

    void reset()
    {
      fseek(fp, pos, SEEK_SET);
    }

    void skip(size_t n)
    {
      for (size_t i = 0; i < n; ++i) {
        readline();
      }
    }

    char* next()
    {
      long pos = ftell(fp);
      readline();
      if (*s == '[') {
        fseek(fp, pos, SEEK_SET);
        return nullptr;
      }
      return strip(s);
    }

    void end()
    {
      do {
        readline();
      } while (strcmp(s, "[[/transaction]]\n") != 0);
    }
  };

  class Writer {
    FILE *fp;

    void write(const char *s)
    {
      if (fputs(s, fp) == EOF) {
        throw SimException(err, __LINE__);
      }
    }

  public:
    Writer(const char *path) : fp(nullptr)
    {
      try {
        fp = fopen(path, "w");
        if (fp == nullptr) {
          throw SimException(err, __LINE__);
        } else {
          static const char mark[] = "[[[runtime]]]\n";
          write(mark);
        }
      } catch (const hls::sim::SimException &e) {
        errExit(e.line, e.msg);
      }
    }

    virtual ~Writer()
    {
      try {
        static const char mark[] = "[[[/runtime]]]\n";
        write(mark);
      } catch (const hls::sim::SimException &e) {
        errExit(e.line, e.msg);
      }
      fclose(fp);
    }

    void begin(size_t AESL_transaction)
    {
      static const char mark[] = "[[transaction]]           ";
      write(mark);
      auto buf = std::to_string(AESL_transaction);
      buf.push_back('\n');
      buf.push_back('\0');
      write(buf.data());
    }

    void next(const char *s)
    {
      write(s);
      write("\n");
    }

    void end()
    {
      static const char mark[] = "[[/transaction]]\n";
      write(mark);
    }
  };

  bool RTLOutputCheckAndReplacement(char *data)
  {
    bool changed = false;
    for (size_t i = 2; i < strlen(data); ++i) {
      if (data[i] == 'X' || data[i] == 'x') {
        data[i] = '0';
        changed = true;
      }
    }
    return changed;
  }

  void warnOnX()
  {
    static const char msg[] =
      "WARNING: [SIM 212-201] RTL produces unknown value "
      "'x' or 'X' on some port, possible cause: "
      "There are uninitialized variables in the design.\n";
    fprintf(stderr, msg);
  }

#ifndef POST_CHECK
  class RefTCL {
    FILE *fp;
    std::ostringstream ss;

    void fmt(std::vector<size_t> &vec)
    {
      ss << "{";
      for (auto &x : vec) {
        ss << " " << x;
      }
      ss << " }";
    }

    void formatDepth()
    {
      ss << "set depth_list {\n";
      for (auto &p : depth) {
        ss << "  {" << p.first << " " << p.second << "}\n";
      }
      if (nameHBM != "") {
        ss << "  {" << nameHBM << " " << depthHBM << "}\n";
      }
      ss << "}\n";
    }

    void formatTransDepth()
    {
      ss << "set trans_depth {\n";
      for (auto &p : transDepth) {
        ss << "  {" << p.first << " ";
        fmt(p.second);
        ss << " " << bundleNameFor[p.first] << "}\n";
      }
      ss << "}\n";
    }

    void formatTransNum()
    {
      ss << "set trans_num " << AESL_transaction << "\n";
    }

    void formatContainsVLA()
    {
      ss << "set containsVLA " << containsVLA << "\n";
    }

    void formatHBM()
    {
      ss << "set HBM_ArgDict {\n"
         << "  Name " << nameHBM << "\n"
         << "  Port " << portHBM << "\n"
         << "  BitWidth " << widthHBM << "\n"
         << "}\n";
    }
    
    void formatAutorestartSeq()
    {
      if (!autorestart_seq.empty()) {
        ss << "set Autorestart_seq {\n";
        for (const auto &val : autorestart_seq) {
          ss << "  " << val << "\n";
        }
        ss << "}\n";
      }
    }

    void close()
    {
      formatDepth();
      formatTransDepth();
      formatContainsVLA();
      formatTransNum();
      formatAutorestartSeq();
      if (nameHBM != "") {
        formatHBM();
      }
      std::string &&s { ss.str() };
      size_t res = fwrite(s.data(), s.size(), 1, fp);
      fclose(fp);
      if (res != 1) {
        errExit(__LINE__, err);
      }
    }

  public:
    std::map<const std::string, size_t> depth;
    typedef const std::string PortName;
    typedef const char *BundleName;
    std::map<PortName, std::vector<size_t>> transDepth;
    std::map<PortName, BundleName> bundleNameFor;
    std::string nameHBM;
    size_t depthHBM;
    std::string portHBM;
    unsigned widthHBM;
    size_t AESL_transaction;
    bool containsVLA;
    std::mutex mut;

    RefTCL(const char *path)
    {
      fp = fopen(path, "w");
      if (fp == nullptr) {
        errExit(__LINE__, err);
      }
    }

    void set(const char* name, size_t dep)
    {
      std::lock_guard<std::mutex> guard(mut);
      if (depth[name] < dep) {
        depth[name] = dep;
      }
    }

    void append(const char* portName, size_t dep, const char* bundleName)
    {
      std::lock_guard<std::mutex> guard(mut);
      transDepth[portName].push_back(dep);
      bundleNameFor[portName] = bundleName;
    }

    ~RefTCL()
    {
      close();
    }
  };

#endif

  struct Register {
    const char* name;
    unsigned width;
#ifdef POST_CHECK
    Reader* reader;
#else
    Writer* owriter;
    Writer* iwriter;
#endif
    void* param;
    std::vector<std::function<void()>> delayed;

#ifndef POST_CHECK
    void doTCL(RefTCL &tcl)
    {
      if (strcmp(name, "return") == 0) {
        tcl.set("ap_return", 1);
      } else {
        tcl.set(name, 1);
      }
    }
#endif
    ~Register()
    {
      for (auto &f : delayed) {
        f();
      }
      delayed.clear();
#ifdef POST_CHECK
      delete reader;
#else
      delete owriter;
      delete iwriter;
#endif
    }
  };

  template<typename E>
  struct DirectIO {
    unsigned width;
    const char* name;
#ifdef POST_CHECK
    Reader* reader;
#else
    Writer* writer;
    Writer* swriter;
    Writer* gwriter;
#endif
    hls::directio<E>* param;
    std::vector<E> buf;
    size_t initSize;
    size_t depth;
    bool hasWrite;

    void markSize()
    {
      initSize = param->size();
    }

    void buffer()
    {
      buf.clear();
      while (param->valid()) {
        buf.push_back(param->read());
      }
      for (auto &e : buf) {
        param->write(e);
      }
    }

#ifndef POST_CHECK
    void doTCL(RefTCL &tcl)
    {
      tcl.set(name, depth);
    }
#endif

    ~DirectIO()
    {
#ifdef POST_CHECK
      delete reader;
#else
      delete writer;
      delete swriter;
      delete gwriter;
#endif
    }
  };

  template<typename Reader, typename Writer>
  struct Memory {
    unsigned width;
    unsigned asize;
    bool hbm;
    std::vector<const char*> name;
#ifdef POST_CHECK
    Reader* reader;
#else
    Writer* owriter;
    Writer* iwriter;
#endif
    std::vector<void*> param;
    std::vector<const char*> mname;
    std::vector<size_t> offset;
    std::vector<bool> hasWrite;
    std::vector<size_t> nbytes;
    std::vector<size_t> max_nbytes;

    size_t depth()
    {
      if (hbm) {
        return divide_ceil(nbytes[0], asize);
      }
      else {
        size_t depth = 0;
        for (size_t n : nbytes) {
          depth += divide_ceil(n, asize);
        }
        return depth;
      }
    }

#ifndef POST_CHECK
    void doTCL(RefTCL &tcl)
    {
      if (hbm) {
        tcl.nameHBM.clear();
        tcl.portHBM.clear();
        tcl.nameHBM.append(name[0]);
        tcl.portHBM.append("{").append(name[0]);
        for (size_t i = 1; i < name.size(); ++i) {
          tcl.nameHBM.append("_").append(name[i]);
          tcl.portHBM.append(" ").append(name[i]);
        }
        tcl.nameHBM.append("_HBM");
        tcl.portHBM.append("}");
        tcl.widthHBM = width;
        size_t depthHBM = divide_ceil(nbytes[0], asize);
        tcl.append(tcl.nameHBM.c_str(), depthHBM, tcl.nameHBM.c_str());
        if (depthHBM > tcl.depthHBM) {
          tcl.depthHBM = depthHBM;
        }
      } else {
        tcl.set(name[0], depth());
        for (size_t i = 0; i < mname.size(); ++i) {
          tcl.append(mname[i], divide_ceil(nbytes[i], asize), name[0]);
        }
      }
    }
#endif

    ~Memory()
    {
#ifdef POST_CHECK
      delete reader;
#else
      delete owriter;
      delete iwriter;
#endif
    }
  };

  struct A2Stream {
    unsigned width;
    unsigned asize;
    const char* name;
#ifdef POST_CHECK
    Reader* reader;
#else
    Writer* owriter;
    Writer* iwriter;
#endif
    void* param;
    size_t nbytes;
    bool hasWrite;

#ifndef POST_CHECK
    void doTCL(RefTCL &tcl)
    {
      tcl.set(name, divide_ceil(nbytes, asize));
    }
#endif

    ~A2Stream()
    {
#ifdef POST_CHECK
      delete reader;
#else
      delete owriter;
      delete iwriter;
#endif
    }
  };

  template<typename E>
  struct Stream {
    unsigned width;
    const char* name;
#ifdef POST_CHECK
    Reader* reader;
#else
    Writer* writer;
    Writer* swriter;
    Writer* gwriter;
#endif
    hls::stream<E>* param;
    std::vector<E> buf;
    size_t initSize;
    size_t depth;
    bool hasWrite;

    void markSize()
    {
      initSize = param->size();
    }

    void buffer()
    {
      buf.clear();
      while (!param->empty()) {
        buf.push_back(param->read());
      }
      for (auto &e : buf) {
        param->write(e);
      }
    }

#ifndef POST_CHECK
    void doTCL(RefTCL &tcl)
    {
      tcl.set(name, depth);
    }
#endif

    ~Stream()
    {
#ifdef POST_CHECK
      delete reader;
#else
      delete writer;
      delete swriter;
      delete gwriter;
#endif
    }
  };

#ifdef POST_CHECK
  void check(Register &port)
  {
    port.reader->begin();
    bool foundX = false;
    if (char *s = port.reader->next()) {
      foundX |= RTLOutputCheckAndReplacement(s);
      unformatData(s, (unsigned char*)port.param);
    }
    port.reader->end();
    if (foundX) {
      warnOnX();
    }
  }

  template<typename E>
  void check(DirectIO<E> &port)
  {
    if (port.hasWrite) {
      port.reader->begin();
      bool foundX = false;
      E *p = new E;
      while (char *s = port.reader->next()) {
        foundX |= RTLOutputCheckAndReplacement(s);
        unformatData(s, (unsigned char*)p);
        port.param->write(*p);
      }
      delete p;
      port.reader->end();
      if (foundX) {
        warnOnX();
      }
    } else {
      port.reader->begin();
      size_t n = 0;
      if (char *s = port.reader->next()) {
        std::istringstream ss(s);
        ss >> n;
      } else {
        throw SimException(bad, __LINE__);
      }
      port.reader->end();
      for (size_t j = 0; j < n; ++j) {
        port.param->read();
      }
    }
  }

  void checkHBM(Memory<Input, Output> &port)
  {
    port.reader->begin();
    size_t wbytes = least_nbyte(port.width);
    for (size_t i = 0; i < port.param.size(); ++i) {
      if (port.hasWrite[i]) {
        port.reader->reset();
        size_t skip = wbytes * port.offset[i];
        port.reader->advance(skip);
        port.reader->into((unsigned char*)port.param[i], wbytes,
                           port.asize, port.nbytes[i] - skip);
      }
    }
  }

  void check(Memory<Input, Output> &port)
  {
    if (port.hbm) {
      return checkHBM(port);
    } else {
      port.reader->begin();
      size_t wbytes = least_nbyte(port.width);
      for (size_t i = 0; i < port.param.size(); ++i) {
        if (port.hasWrite[i]) {
          port.reader->into((unsigned char*)port.param[i], wbytes,
                             port.asize, port.nbytes[i]);
        } else {
          size_t n = divide_ceil(port.nbytes[i], port.asize);
          port.reader->advance(port.asize*n);
        }
      }
    }
  }

  void transfer(Reader *reader, size_t nbytes, unsigned char *put, bool &foundX)
  {
    if (char *s = reader->next()) {
      foundX |= RTLOutputCheckAndReplacement(s);
      unformatData(s, put, nbytes);
    } else {
      throw SimException("No more data", __LINE__);
    }
  }

  void checkHBM(Memory<Reader, Writer> &port)
  {
    port.reader->begin();
    bool foundX = false;
    size_t wbytes = least_nbyte(port.width);
    for (size_t i = 0, last = port.param.size()-1; i <= last; ++i) {
      if (port.hasWrite[i]) {
        port.reader->skip(port.offset[i]);
        size_t n = port.nbytes[i] / port.asize - port.offset[i];
        unsigned char *put = (unsigned char*)port.param[i];
        for (size_t j = 0; j < n; ++j) {
          transfer(port.reader, wbytes, put, foundX);
          put += port.asize;
        }
        if (i < last) {
          port.reader->reset();
        }
      }
    }
    port.reader->end();
    if (foundX) {
      warnOnX();
    }
  }

  void check(Memory<Reader, Writer> &port)
  {
    if (port.hbm) {
      return checkHBM(port);
    } else {
      port.reader->begin();
      bool foundX = false;
      size_t wbytes = least_nbyte(port.width);
      for (size_t i = 0; i < port.param.size(); ++i) {
        if (port.hasWrite[i]) {
          size_t n = port.nbytes[i] / port.asize;
          size_t r = port.nbytes[i] % port.asize;
          unsigned char *put = (unsigned char*)port.param[i];
          for (size_t j = 0; j < n; ++j) {
            transfer(port.reader, wbytes, put, foundX);
            put += port.asize;
          }
          if (r > 0) {
            transfer(port.reader, r, put, foundX);
          }
        } else {
          size_t n = divide_ceil(port.nbytes[i], port.asize);
          port.reader->skip(n);
        }
      }
      port.reader->end();
      if (foundX) {
        warnOnX();
      }
    }
  }

  void check(A2Stream &port)
  {
    port.reader->begin();
    bool foundX = false;
    if (port.hasWrite) {
      size_t wbytes = least_nbyte(port.width);
      size_t n = port.nbytes / port.asize;
      size_t r = port.nbytes % port.asize;
      unsigned char *put = (unsigned char*)port.param;
      for (size_t j = 0; j < n; ++j) {
        if (char *s = port.reader->next()) {
          foundX |= RTLOutputCheckAndReplacement(s);
          unformatData(s, put, wbytes);
        }
        put += port.asize;
      }
      if (r > 0) {
        if (char *s = port.reader->next()) {
          foundX |= RTLOutputCheckAndReplacement(s);
          unformatData(s, put, r);
        }
      }
    }
    port.reader->end();
    if (foundX) {
      warnOnX();
    }
  }

  template<typename E>
  void check(Stream<E> &port)
  {
    if (port.hasWrite) {
      port.reader->begin();
      bool foundX = false;
      E *p = new E;
      while (char *s = port.reader->next()) {
        foundX |= RTLOutputCheckAndReplacement(s);
        unformatData(s, (unsigned char*)p);
        port.param->write(*p);
      }
      delete p;
      port.reader->end();
      if (foundX) {
        warnOnX();
      }
    } else {
      port.reader->begin();
      size_t n = 0;
      if (char *s = port.reader->next()) {
        std::istringstream ss(s);
        ss >> n;
      } else {
        throw SimException(bad, __LINE__);
      }
      port.reader->end();
      for (size_t j = 0; j < n; ++j) {
        port.param->read();
      }
    }
  }
#else
  void dump(Register &port, Writer *writer, size_t AESL_transaction)
  {
    writer->begin(AESL_transaction);
    std::string &&s { formatData((unsigned char*)port.param, port.width) };
    writer->next(s.data());
    writer->end();
  }

  void delay_dump(Register &port, Writer *writer, size_t AESL_transaction)
  {
    port.delayed.push_back(std::bind(dump, std::ref(port), writer, AESL_transaction));
  }

  template<typename E>
  void dump(DirectIO<E> &port, size_t AESL_transaction)
  {
    if (port.hasWrite) {
      port.writer->begin(AESL_transaction);
      port.depth = port.param->size()-port.initSize;
      for (size_t j = 0; j < port.depth; ++j) {
        std::string &&s {
          formatData((unsigned char*)&port.buf[port.initSize+j], port.width)
        };
        port.writer->next(s.c_str());
      }
      port.writer->end();

      port.swriter->begin(AESL_transaction);
      port.swriter->next(std::to_string(port.depth).c_str());
      port.swriter->end();
    } else {
      port.writer->begin(AESL_transaction);
      port.depth = port.initSize-port.param->size();
      for (size_t j = 0; j < port.depth; ++j) {
        std::string &&s {
          formatData((unsigned char*)&port.buf[j], port.width)
        };
        port.writer->next(s.c_str());
      }
      port.writer->end();

      port.swriter->begin(AESL_transaction);
      port.swriter->next(std::to_string(port.depth).c_str());
      port.swriter->end();

      port.gwriter->begin(AESL_transaction);
      size_t n = (port.depth ? port.initSize : port.depth);
      size_t d = port.depth;
      do {
        port.gwriter->next(std::to_string(n--).c_str());
      } while (d--);
      port.gwriter->end();
    }
  }

  void error_on_depth_unspecified(const char *portName)
  {
    std::string msg {"A depth specification is required for interface port "};
    msg.append("'");
    msg.append(portName);
    msg.append("'");
    msg.append(" for cosimulation.");
    throw SimException(msg, __LINE__);
  }

  void dump(Memory<Input, Output> &port, Output *writer, size_t AESL_transaction)
  {
    for (size_t i = 0; i < port.param.size(); ++i) {
      if (port.nbytes[i] == 0) {
        error_on_depth_unspecified(port.mname[i]);
      }
    }

    writer->begin(port.depth());
    size_t wbytes = least_nbyte(port.width);
    if (port.hbm) {
      writer->from((unsigned char*)port.param[0], wbytes, port.asize,
                   port.nbytes[0], 0);
    }
    else {
      for (size_t i = 0; i < port.param.size(); ++i) {
        writer->from((unsigned char*)port.param[i], wbytes, port.asize,
                     port.nbytes[i], 0);
      }
    }
  }

  void dump(Memory<Reader, Writer> &port, Writer *writer, size_t AESL_transaction)
  {
    for (size_t i = 0; i < port.param.size(); ++i) {
      if (port.nbytes[i] == 0) {
        error_on_depth_unspecified(port.mname[i]);
      }
    }
    writer->begin(AESL_transaction);
    for (size_t i = 0; i < port.param.size(); ++i) {
      size_t n = divide_ceil(port.nbytes[i], port.asize);
      unsigned char *put = (unsigned char*)port.param[i];
      for (size_t j = 0; j < n; ++j) {
        std::string &&s {
          formatData(put, port.width)
        };
        writer->next(s.data());
        put += port.asize;
      }
      if (port.hbm) {
        break;
      }
    }
    writer->end();
  }

  void dump(A2Stream &port, Writer *writer, size_t AESL_transaction)
  {
    if (port.nbytes == 0) {
      error_on_depth_unspecified(port.name);
    }
    writer->begin(AESL_transaction);
    size_t n = divide_ceil(port.nbytes, port.asize);
    unsigned char *put = (unsigned char*)port.param;
    for (size_t j = 0; j < n; ++j) {
      std::string &&s { formatData(put, port.width) };
      writer->next(s.data());
      put += port.asize;
    }
    writer->end();
  }

  template<typename E>
  void dump(Stream<E> &port, size_t AESL_transaction)
  {
    if (port.hasWrite) {
      port.writer->begin(AESL_transaction);
      port.depth = port.param->size()-port.initSize;
      for (size_t j = 0; j < port.depth; ++j) {
        std::string &&s {
          formatData((unsigned char*)&port.buf[port.initSize+j], port.width)
        };
        port.writer->next(s.c_str());
      }
      port.writer->end();

      port.swriter->begin(AESL_transaction);
      port.swriter->next(std::to_string(port.depth).c_str());
      port.swriter->end();
    } else {
      port.writer->begin(AESL_transaction);
      port.depth = port.initSize-port.param->size();
      for (size_t j = 0; j < port.depth; ++j) {
        std::string &&s {
          formatData((unsigned char*)&port.buf[j], port.width)
        };
        port.writer->next(s.c_str());
      }
      port.writer->end();

      port.swriter->begin(AESL_transaction);
      port.swriter->next(std::to_string(port.depth).c_str());
      port.swriter->end();

      port.gwriter->begin(AESL_transaction);
      size_t n = (port.depth ? port.initSize : port.depth);
      size_t d = port.depth;
      do {
        port.gwriter->next(std::to_string(n--).c_str());
      } while (d--);
      port.gwriter->end();
    }
  }
#endif
}



extern "C"
void snn_top_hls_hw_stub_wrapper(hls::sim::Byte<4>*, hls::sim::Byte<4>*, hls::sim::Byte<18>*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, hls::sim::Byte<1>*, void*, void*, void*, hls::sim::Byte<1>*, hls::sim::Byte<1>*, hls::sim::Byte<1>*, hls::sim::Byte<1>*, void*, void*, void*, void*, void*, hls::sim::Byte<1>*, hls::sim::Byte<1>*);

extern "C"
void apatb_snn_top_hls_hw(hls::sim::Byte<4>* __xlx_apatb_param_ctrl_reg, hls::sim::Byte<4>* __xlx_apatb_param_config_reg, hls::sim::Byte<18>* __xlx_apatb_param_learning_params, void* __xlx_apatb_param_status_reg, void* __xlx_apatb_param_spike_count_reg, void* __xlx_apatb_param_weight_sum_reg, void* __xlx_apatb_param_version_reg, void* __xlx_apatb_param_s_axis_spikes_V_data_V, void* __xlx_apatb_param_s_axis_spikes_V_keep_V, void* __xlx_apatb_param_s_axis_spikes_V_strb_V, void* __xlx_apatb_param_s_axis_spikes_V_user_V, void* __xlx_apatb_param_s_axis_spikes_V_last_V, void* __xlx_apatb_param_s_axis_spikes_V_id_V, void* __xlx_apatb_param_s_axis_spikes_V_dest_V, void* __xlx_apatb_param_m_axis_spikes_V_data_V, void* __xlx_apatb_param_m_axis_spikes_V_keep_V, void* __xlx_apatb_param_m_axis_spikes_V_strb_V, void* __xlx_apatb_param_m_axis_spikes_V_user_V, void* __xlx_apatb_param_m_axis_spikes_V_last_V, void* __xlx_apatb_param_m_axis_spikes_V_id_V, void* __xlx_apatb_param_m_axis_spikes_V_dest_V, void* __xlx_apatb_param_m_axis_weights_V_data_V, void* __xlx_apatb_param_m_axis_weights_V_keep_V, void* __xlx_apatb_param_m_axis_weights_V_strb_V, void* __xlx_apatb_param_m_axis_weights_V_user_V, void* __xlx_apatb_param_m_axis_weights_V_last_V, void* __xlx_apatb_param_m_axis_weights_V_id_V, void* __xlx_apatb_param_m_axis_weights_V_dest_V, hls::sim::Byte<1>* __xlx_apatb_param_reward_signal, void* __xlx_apatb_param_spike_in_valid, void* __xlx_apatb_param_spike_in_neuron_id, void* __xlx_apatb_param_spike_in_weight, hls::sim::Byte<1>* __xlx_apatb_param_spike_in_ready, hls::sim::Byte<1>* __xlx_apatb_param_spike_out_valid, hls::sim::Byte<1>* __xlx_apatb_param_spike_out_neuron_id, hls::sim::Byte<1>* __xlx_apatb_param_spike_out_weight, void* __xlx_apatb_param_spike_out_ready, void* __xlx_apatb_param_snn_enable, void* __xlx_apatb_param_snn_reset, void* __xlx_apatb_param_threshold_out, void* __xlx_apatb_param_leak_rate_out, hls::sim::Byte<1>* __xlx_apatb_param_snn_ready, hls::sim::Byte<1>* __xlx_apatb_param_snn_busy)
{
  static hls::sim::Register port0 {
    .name = "ctrl_reg",
    .width = 32,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_ctrl_reg),
#endif
  };
  port0.param = __xlx_apatb_param_ctrl_reg;

  static hls::sim::Register port1 {
    .name = "config_reg",
    .width = 32,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_config_reg),
#endif
  };
  port1.param = __xlx_apatb_param_config_reg;

  static hls::sim::Register port2 {
    .name = "learning_params",
    .width = 144,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_learning_params),
#endif
  };
  port2.param = __xlx_apatb_param_learning_params;

  static hls::sim::Register port3 {
    .name = "status_reg",
    .width = 32,
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_status_reg),
#else
    .owriter = new hls::sim::Writer(AUTOTB_TVOUT_status_reg),
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_status_reg),
#endif
  };
  port3.param = __xlx_apatb_param_status_reg;

  static hls::sim::Register port4 {
    .name = "spike_count_reg",
    .width = 32,
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_spike_count_reg),
#else
    .owriter = new hls::sim::Writer(AUTOTB_TVOUT_spike_count_reg),
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_spike_count_reg),
#endif
  };
  port4.param = __xlx_apatb_param_spike_count_reg;

  static hls::sim::Register port5 {
    .name = "weight_sum_reg",
    .width = 32,
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_weight_sum_reg),
#else
    .owriter = new hls::sim::Writer(AUTOTB_TVOUT_weight_sum_reg),
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_weight_sum_reg),
#endif
  };
  port5.param = __xlx_apatb_param_weight_sum_reg;

  static hls::sim::Register port6 {
    .name = "version_reg",
    .width = 32,
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_version_reg),
#else
    .owriter = new hls::sim::Writer(AUTOTB_TVOUT_version_reg),
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_version_reg),
#endif
  };
  port6.param = __xlx_apatb_param_version_reg;

  static hls::sim::Stream<hls::sim::Byte<4>> port7 {
    .width = 32,
    .name = "s_axis_spikes_V_data_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_data_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVIN_s_axis_spikes_V_data_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_data_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_INGRESS_STATUS_s_axis_spikes_V_data_V),
#endif
  };
  port7.param = (hls::stream<hls::sim::Byte<4>>*)__xlx_apatb_param_s_axis_spikes_V_data_V;
  port7.hasWrite = false;

  static hls::sim::Stream<hls::sim::Byte<1>> port8 {
    .width = 4,
    .name = "s_axis_spikes_V_keep_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_keep_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVIN_s_axis_spikes_V_keep_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_keep_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_INGRESS_STATUS_s_axis_spikes_V_keep_V),
#endif
  };
  port8.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_s_axis_spikes_V_keep_V;
  port8.hasWrite = false;

  static hls::sim::Stream<hls::sim::Byte<1>> port9 {
    .width = 4,
    .name = "s_axis_spikes_V_strb_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_strb_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVIN_s_axis_spikes_V_strb_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_strb_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_INGRESS_STATUS_s_axis_spikes_V_strb_V),
#endif
  };
  port9.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_s_axis_spikes_V_strb_V;
  port9.hasWrite = false;

  static hls::sim::Stream<hls::sim::Byte<1>> port10 {
    .width = 1,
    .name = "s_axis_spikes_V_user_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_user_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVIN_s_axis_spikes_V_user_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_user_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_INGRESS_STATUS_s_axis_spikes_V_user_V),
#endif
  };
  port10.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_s_axis_spikes_V_user_V;
  port10.hasWrite = false;

  static hls::sim::Stream<hls::sim::Byte<1>> port11 {
    .width = 1,
    .name = "s_axis_spikes_V_last_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_last_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVIN_s_axis_spikes_V_last_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_last_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_INGRESS_STATUS_s_axis_spikes_V_last_V),
#endif
  };
  port11.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_s_axis_spikes_V_last_V;
  port11.hasWrite = false;

  static hls::sim::Stream<hls::sim::Byte<1>> port12 {
    .width = 1,
    .name = "s_axis_spikes_V_id_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_id_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVIN_s_axis_spikes_V_id_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_id_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_INGRESS_STATUS_s_axis_spikes_V_id_V),
#endif
  };
  port12.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_s_axis_spikes_V_id_V;
  port12.hasWrite = false;

  static hls::sim::Stream<hls::sim::Byte<1>> port13 {
    .width = 1,
    .name = "s_axis_spikes_V_dest_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_dest_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVIN_s_axis_spikes_V_dest_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_IN_s_axis_spikes_V_dest_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_INGRESS_STATUS_s_axis_spikes_V_dest_V),
#endif
  };
  port13.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_s_axis_spikes_V_dest_V;
  port13.hasWrite = false;

  static hls::sim::Stream<hls::sim::Byte<4>> port14 {
    .width = 32,
    .name = "m_axis_spikes_V_data_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_m_axis_spikes_V_data_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVOUT_m_axis_spikes_V_data_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_OUT_m_axis_spikes_V_data_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_EGRESS_STATUS_m_axis_spikes_V_data_V),
#endif
  };
  port14.param = (hls::stream<hls::sim::Byte<4>>*)__xlx_apatb_param_m_axis_spikes_V_data_V;
  port14.hasWrite = true;

  static hls::sim::Stream<hls::sim::Byte<1>> port15 {
    .width = 4,
    .name = "m_axis_spikes_V_keep_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_m_axis_spikes_V_keep_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVOUT_m_axis_spikes_V_keep_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_OUT_m_axis_spikes_V_keep_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_EGRESS_STATUS_m_axis_spikes_V_keep_V),
#endif
  };
  port15.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_m_axis_spikes_V_keep_V;
  port15.hasWrite = true;

  static hls::sim::Stream<hls::sim::Byte<1>> port16 {
    .width = 4,
    .name = "m_axis_spikes_V_strb_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_m_axis_spikes_V_strb_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVOUT_m_axis_spikes_V_strb_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_OUT_m_axis_spikes_V_strb_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_EGRESS_STATUS_m_axis_spikes_V_strb_V),
#endif
  };
  port16.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_m_axis_spikes_V_strb_V;
  port16.hasWrite = true;

  static hls::sim::Stream<hls::sim::Byte<1>> port17 {
    .width = 1,
    .name = "m_axis_spikes_V_user_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_m_axis_spikes_V_user_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVOUT_m_axis_spikes_V_user_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_OUT_m_axis_spikes_V_user_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_EGRESS_STATUS_m_axis_spikes_V_user_V),
#endif
  };
  port17.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_m_axis_spikes_V_user_V;
  port17.hasWrite = true;

  static hls::sim::Stream<hls::sim::Byte<1>> port18 {
    .width = 1,
    .name = "m_axis_spikes_V_last_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_m_axis_spikes_V_last_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVOUT_m_axis_spikes_V_last_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_OUT_m_axis_spikes_V_last_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_EGRESS_STATUS_m_axis_spikes_V_last_V),
#endif
  };
  port18.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_m_axis_spikes_V_last_V;
  port18.hasWrite = true;

  static hls::sim::Stream<hls::sim::Byte<1>> port19 {
    .width = 1,
    .name = "m_axis_spikes_V_id_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_m_axis_spikes_V_id_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVOUT_m_axis_spikes_V_id_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_OUT_m_axis_spikes_V_id_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_EGRESS_STATUS_m_axis_spikes_V_id_V),
#endif
  };
  port19.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_m_axis_spikes_V_id_V;
  port19.hasWrite = true;

  static hls::sim::Stream<hls::sim::Byte<1>> port20 {
    .width = 1,
    .name = "m_axis_spikes_V_dest_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_m_axis_spikes_V_dest_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVOUT_m_axis_spikes_V_dest_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_OUT_m_axis_spikes_V_dest_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_EGRESS_STATUS_m_axis_spikes_V_dest_V),
#endif
  };
  port20.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_m_axis_spikes_V_dest_V;
  port20.hasWrite = true;

  static hls::sim::Stream<hls::sim::Byte<4>> port21 {
    .width = 32,
    .name = "m_axis_weights_V_data_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_m_axis_weights_V_data_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVOUT_m_axis_weights_V_data_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_OUT_m_axis_weights_V_data_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_EGRESS_STATUS_m_axis_weights_V_data_V),
#endif
  };
  port21.param = (hls::stream<hls::sim::Byte<4>>*)__xlx_apatb_param_m_axis_weights_V_data_V;
  port21.hasWrite = true;

  static hls::sim::Stream<hls::sim::Byte<1>> port22 {
    .width = 4,
    .name = "m_axis_weights_V_keep_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_m_axis_weights_V_keep_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVOUT_m_axis_weights_V_keep_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_OUT_m_axis_weights_V_keep_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_EGRESS_STATUS_m_axis_weights_V_keep_V),
#endif
  };
  port22.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_m_axis_weights_V_keep_V;
  port22.hasWrite = true;

  static hls::sim::Stream<hls::sim::Byte<1>> port23 {
    .width = 4,
    .name = "m_axis_weights_V_strb_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_m_axis_weights_V_strb_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVOUT_m_axis_weights_V_strb_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_OUT_m_axis_weights_V_strb_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_EGRESS_STATUS_m_axis_weights_V_strb_V),
#endif
  };
  port23.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_m_axis_weights_V_strb_V;
  port23.hasWrite = true;

  static hls::sim::Stream<hls::sim::Byte<1>> port24 {
    .width = 1,
    .name = "m_axis_weights_V_user_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_m_axis_weights_V_user_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVOUT_m_axis_weights_V_user_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_OUT_m_axis_weights_V_user_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_EGRESS_STATUS_m_axis_weights_V_user_V),
#endif
  };
  port24.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_m_axis_weights_V_user_V;
  port24.hasWrite = true;

  static hls::sim::Stream<hls::sim::Byte<1>> port25 {
    .width = 1,
    .name = "m_axis_weights_V_last_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_m_axis_weights_V_last_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVOUT_m_axis_weights_V_last_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_OUT_m_axis_weights_V_last_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_EGRESS_STATUS_m_axis_weights_V_last_V),
#endif
  };
  port25.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_m_axis_weights_V_last_V;
  port25.hasWrite = true;

  static hls::sim::Stream<hls::sim::Byte<1>> port26 {
    .width = 1,
    .name = "m_axis_weights_V_id_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_m_axis_weights_V_id_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVOUT_m_axis_weights_V_id_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_OUT_m_axis_weights_V_id_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_EGRESS_STATUS_m_axis_weights_V_id_V),
#endif
  };
  port26.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_m_axis_weights_V_id_V;
  port26.hasWrite = true;

  static hls::sim::Stream<hls::sim::Byte<1>> port27 {
    .width = 1,
    .name = "m_axis_weights_V_dest_V",
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_m_axis_weights_V_dest_V),
#else
    .writer = new hls::sim::Writer(AUTOTB_TVOUT_m_axis_weights_V_dest_V),
    .swriter = new hls::sim::Writer(WRAPC_STREAM_SIZE_OUT_m_axis_weights_V_dest_V),
    .gwriter = new hls::sim::Writer(WRAPC_STREAM_EGRESS_STATUS_m_axis_weights_V_dest_V),
#endif
  };
  port27.param = (hls::stream<hls::sim::Byte<1>>*)__xlx_apatb_param_m_axis_weights_V_dest_V;
  port27.hasWrite = true;

  static hls::sim::Register port28 {
    .name = "reward_signal",
    .width = 8,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_reward_signal),
#endif
  };
  port28.param = __xlx_apatb_param_reward_signal;

  static hls::sim::Register port29 {
    .name = "spike_in_valid",
    .width = 1,
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_spike_in_valid),
#else
    .owriter = new hls::sim::Writer(AUTOTB_TVOUT_spike_in_valid),
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_spike_in_valid),
#endif
  };
  port29.param = __xlx_apatb_param_spike_in_valid;

  static hls::sim::Register port30 {
    .name = "spike_in_neuron_id",
    .width = 8,
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_spike_in_neuron_id),
#else
    .owriter = new hls::sim::Writer(AUTOTB_TVOUT_spike_in_neuron_id),
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_spike_in_neuron_id),
#endif
  };
  port30.param = __xlx_apatb_param_spike_in_neuron_id;

  static hls::sim::Register port31 {
    .name = "spike_in_weight",
    .width = 8,
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_spike_in_weight),
#else
    .owriter = new hls::sim::Writer(AUTOTB_TVOUT_spike_in_weight),
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_spike_in_weight),
#endif
  };
  port31.param = __xlx_apatb_param_spike_in_weight;

  static hls::sim::Register port32 {
    .name = "spike_in_ready",
    .width = 1,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_spike_in_ready),
#endif
  };
  port32.param = __xlx_apatb_param_spike_in_ready;

  static hls::sim::Register port33 {
    .name = "spike_out_valid",
    .width = 1,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_spike_out_valid),
#endif
  };
  port33.param = __xlx_apatb_param_spike_out_valid;

  static hls::sim::Register port34 {
    .name = "spike_out_neuron_id",
    .width = 8,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_spike_out_neuron_id),
#endif
  };
  port34.param = __xlx_apatb_param_spike_out_neuron_id;

  static hls::sim::Register port35 {
    .name = "spike_out_weight",
    .width = 8,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_spike_out_weight),
#endif
  };
  port35.param = __xlx_apatb_param_spike_out_weight;

  static hls::sim::Register port36 {
    .name = "spike_out_ready",
    .width = 1,
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_spike_out_ready),
#else
    .owriter = new hls::sim::Writer(AUTOTB_TVOUT_spike_out_ready),
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_spike_out_ready),
#endif
  };
  port36.param = __xlx_apatb_param_spike_out_ready;

  static hls::sim::Register port37 {
    .name = "snn_enable",
    .width = 1,
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_snn_enable),
#else
    .owriter = new hls::sim::Writer(AUTOTB_TVOUT_snn_enable),
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_snn_enable),
#endif
  };
  port37.param = __xlx_apatb_param_snn_enable;

  static hls::sim::Register port38 {
    .name = "snn_reset",
    .width = 1,
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_snn_reset),
#else
    .owriter = new hls::sim::Writer(AUTOTB_TVOUT_snn_reset),
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_snn_reset),
#endif
  };
  port38.param = __xlx_apatb_param_snn_reset;

  static hls::sim::Register port39 {
    .name = "threshold_out",
    .width = 16,
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_threshold_out),
#else
    .owriter = new hls::sim::Writer(AUTOTB_TVOUT_threshold_out),
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_threshold_out),
#endif
  };
  port39.param = __xlx_apatb_param_threshold_out;

  static hls::sim::Register port40 {
    .name = "leak_rate_out",
    .width = 16,
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_leak_rate_out),
#else
    .owriter = new hls::sim::Writer(AUTOTB_TVOUT_leak_rate_out),
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_leak_rate_out),
#endif
  };
  port40.param = __xlx_apatb_param_leak_rate_out;

  static hls::sim::Register port41 {
    .name = "snn_ready",
    .width = 1,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_snn_ready),
#endif
  };
  port41.param = __xlx_apatb_param_snn_ready;

  static hls::sim::Register port42 {
    .name = "snn_busy",
    .width = 1,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_snn_busy),
#endif
  };
  port42.param = __xlx_apatb_param_snn_busy;

  try {
#ifdef POST_CHECK
    CodeState = ENTER_WRAPC_PC;
    check(port3);
    check(port4);
    check(port5);
    check(port6);
    check(port29);
    check(port30);
    check(port31);
    check(port36);
    check(port37);
    check(port38);
    check(port39);
    check(port40);
    check(port7);
    check(port8);
    check(port9);
    check(port10);
    check(port11);
    check(port12);
    check(port13);
    check(port14);
    check(port15);
    check(port16);
    check(port17);
    check(port18);
    check(port19);
    check(port20);
    check(port21);
    check(port22);
    check(port23);
    check(port24);
    check(port25);
    check(port26);
    check(port27);
#else
    static hls::sim::RefTCL tcl("../tv/cdatafile/ref.tcl");
    tcl.containsVLA = 0;
    CodeState = DUMP_INPUTS;
    dump(port0, port0.iwriter, tcl.AESL_transaction);
    dump(port1, port1.iwriter, tcl.AESL_transaction);
    dump(port2, port2.iwriter, tcl.AESL_transaction);
    dump(port3, port3.iwriter, tcl.AESL_transaction);
    dump(port4, port4.iwriter, tcl.AESL_transaction);
    dump(port5, port5.iwriter, tcl.AESL_transaction);
    dump(port6, port6.iwriter, tcl.AESL_transaction);
    dump(port28, port28.iwriter, tcl.AESL_transaction);
    dump(port29, port29.iwriter, tcl.AESL_transaction);
    dump(port30, port30.iwriter, tcl.AESL_transaction);
    dump(port31, port31.iwriter, tcl.AESL_transaction);
    dump(port32, port32.iwriter, tcl.AESL_transaction);
    dump(port33, port33.iwriter, tcl.AESL_transaction);
    dump(port34, port34.iwriter, tcl.AESL_transaction);
    dump(port35, port35.iwriter, tcl.AESL_transaction);
    dump(port36, port36.iwriter, tcl.AESL_transaction);
    dump(port37, port37.iwriter, tcl.AESL_transaction);
    dump(port38, port38.iwriter, tcl.AESL_transaction);
    dump(port39, port39.iwriter, tcl.AESL_transaction);
    dump(port40, port40.iwriter, tcl.AESL_transaction);
    dump(port41, port41.iwriter, tcl.AESL_transaction);
    dump(port42, port42.iwriter, tcl.AESL_transaction);
    port0.doTCL(tcl);
    port1.doTCL(tcl);
    port2.doTCL(tcl);
    port3.doTCL(tcl);
    port4.doTCL(tcl);
    port5.doTCL(tcl);
    port6.doTCL(tcl);
    port28.doTCL(tcl);
    port29.doTCL(tcl);
    port30.doTCL(tcl);
    port31.doTCL(tcl);
    port32.doTCL(tcl);
    port33.doTCL(tcl);
    port34.doTCL(tcl);
    port35.doTCL(tcl);
    port36.doTCL(tcl);
    port37.doTCL(tcl);
    port38.doTCL(tcl);
    port39.doTCL(tcl);
    port40.doTCL(tcl);
    port41.doTCL(tcl);
    port42.doTCL(tcl);
    port7.markSize();
    port8.markSize();
    port9.markSize();
    port10.markSize();
    port11.markSize();
    port12.markSize();
    port13.markSize();
    port7.buffer();
    port8.buffer();
    port9.buffer();
    port10.buffer();
    port11.buffer();
    port12.buffer();
    port13.buffer();
    port14.markSize();
    port15.markSize();
    port16.markSize();
    port17.markSize();
    port18.markSize();
    port19.markSize();
    port20.markSize();
    port21.markSize();
    port22.markSize();
    port23.markSize();
    port24.markSize();
    port25.markSize();
    port26.markSize();
    port27.markSize();
    CodeState = CALL_C_DUT;
    snn_top_hls_hw_stub_wrapper(__xlx_apatb_param_ctrl_reg, __xlx_apatb_param_config_reg, __xlx_apatb_param_learning_params, __xlx_apatb_param_status_reg, __xlx_apatb_param_spike_count_reg, __xlx_apatb_param_weight_sum_reg, __xlx_apatb_param_version_reg, __xlx_apatb_param_s_axis_spikes_V_data_V, __xlx_apatb_param_s_axis_spikes_V_keep_V, __xlx_apatb_param_s_axis_spikes_V_strb_V, __xlx_apatb_param_s_axis_spikes_V_user_V, __xlx_apatb_param_s_axis_spikes_V_last_V, __xlx_apatb_param_s_axis_spikes_V_id_V, __xlx_apatb_param_s_axis_spikes_V_dest_V, __xlx_apatb_param_m_axis_spikes_V_data_V, __xlx_apatb_param_m_axis_spikes_V_keep_V, __xlx_apatb_param_m_axis_spikes_V_strb_V, __xlx_apatb_param_m_axis_spikes_V_user_V, __xlx_apatb_param_m_axis_spikes_V_last_V, __xlx_apatb_param_m_axis_spikes_V_id_V, __xlx_apatb_param_m_axis_spikes_V_dest_V, __xlx_apatb_param_m_axis_weights_V_data_V, __xlx_apatb_param_m_axis_weights_V_keep_V, __xlx_apatb_param_m_axis_weights_V_strb_V, __xlx_apatb_param_m_axis_weights_V_user_V, __xlx_apatb_param_m_axis_weights_V_last_V, __xlx_apatb_param_m_axis_weights_V_id_V, __xlx_apatb_param_m_axis_weights_V_dest_V, __xlx_apatb_param_reward_signal, __xlx_apatb_param_spike_in_valid, __xlx_apatb_param_spike_in_neuron_id, __xlx_apatb_param_spike_in_weight, __xlx_apatb_param_spike_in_ready, __xlx_apatb_param_spike_out_valid, __xlx_apatb_param_spike_out_neuron_id, __xlx_apatb_param_spike_out_weight, __xlx_apatb_param_spike_out_ready, __xlx_apatb_param_snn_enable, __xlx_apatb_param_snn_reset, __xlx_apatb_param_threshold_out, __xlx_apatb_param_leak_rate_out, __xlx_apatb_param_snn_ready, __xlx_apatb_param_snn_busy);
    port14.buffer();
    port15.buffer();
    port16.buffer();
    port17.buffer();
    port18.buffer();
    port19.buffer();
    port20.buffer();
    port21.buffer();
    port22.buffer();
    port23.buffer();
    port24.buffer();
    port25.buffer();
    port26.buffer();
    port27.buffer();
    dump(port7, tcl.AESL_transaction);
    dump(port8, tcl.AESL_transaction);
    dump(port9, tcl.AESL_transaction);
    dump(port10, tcl.AESL_transaction);
    dump(port11, tcl.AESL_transaction);
    dump(port12, tcl.AESL_transaction);
    dump(port13, tcl.AESL_transaction);
    port7.doTCL(tcl);
    port8.doTCL(tcl);
    port9.doTCL(tcl);
    port10.doTCL(tcl);
    port11.doTCL(tcl);
    port12.doTCL(tcl);
    port13.doTCL(tcl);
    CodeState = DUMP_OUTPUTS;
    dump(port3, port3.owriter, tcl.AESL_transaction);
    dump(port4, port4.owriter, tcl.AESL_transaction);
    dump(port5, port5.owriter, tcl.AESL_transaction);
    dump(port6, port6.owriter, tcl.AESL_transaction);
    dump(port29, port29.owriter, tcl.AESL_transaction);
    dump(port30, port30.owriter, tcl.AESL_transaction);
    dump(port31, port31.owriter, tcl.AESL_transaction);
    dump(port36, port36.owriter, tcl.AESL_transaction);
    dump(port37, port37.owriter, tcl.AESL_transaction);
    dump(port38, port38.owriter, tcl.AESL_transaction);
    dump(port39, port39.owriter, tcl.AESL_transaction);
    dump(port40, port40.owriter, tcl.AESL_transaction);
    dump(port14, tcl.AESL_transaction);
    dump(port15, tcl.AESL_transaction);
    dump(port16, tcl.AESL_transaction);
    dump(port17, tcl.AESL_transaction);
    dump(port18, tcl.AESL_transaction);
    dump(port19, tcl.AESL_transaction);
    dump(port20, tcl.AESL_transaction);
    dump(port21, tcl.AESL_transaction);
    dump(port22, tcl.AESL_transaction);
    dump(port23, tcl.AESL_transaction);
    dump(port24, tcl.AESL_transaction);
    dump(port25, tcl.AESL_transaction);
    dump(port26, tcl.AESL_transaction);
    dump(port27, tcl.AESL_transaction);
    port14.doTCL(tcl);
    port15.doTCL(tcl);
    port16.doTCL(tcl);
    port17.doTCL(tcl);
    port18.doTCL(tcl);
    port19.doTCL(tcl);
    port20.doTCL(tcl);
    port21.doTCL(tcl);
    port22.doTCL(tcl);
    port23.doTCL(tcl);
    port24.doTCL(tcl);
    port25.doTCL(tcl);
    port26.doTCL(tcl);
    port27.doTCL(tcl);
    tcl.AESL_transaction++;
#endif
  } catch (const hls::sim::SimException &e) {
    hls::sim::errExit(e.line, e.msg);
  }
}