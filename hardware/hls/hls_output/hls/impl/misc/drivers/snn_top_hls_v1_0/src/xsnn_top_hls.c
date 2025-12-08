// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xsnn_top_hls.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XSnn_top_hls_CfgInitialize(XSnn_top_hls *InstancePtr, XSnn_top_hls_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Ctrl_BaseAddress = ConfigPtr->Ctrl_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XSnn_top_hls_Start(XSnn_top_hls *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_AP_CTRL) & 0x80;
    XSnn_top_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XSnn_top_hls_IsDone(XSnn_top_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XSnn_top_hls_IsIdle(XSnn_top_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XSnn_top_hls_IsReady(XSnn_top_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XSnn_top_hls_EnableAutoRestart(XSnn_top_hls *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_top_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_AP_CTRL, 0x80);
}

void XSnn_top_hls_DisableAutoRestart(XSnn_top_hls *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_top_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_AP_CTRL, 0);
}

void XSnn_top_hls_Set_ctrl_reg(XSnn_top_hls *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_top_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_CTRL_REG_DATA, Data);
}

u32 XSnn_top_hls_Get_ctrl_reg(XSnn_top_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_CTRL_REG_DATA);
    return Data;
}

void XSnn_top_hls_Set_config_reg(XSnn_top_hls *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_top_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_CONFIG_REG_DATA, Data);
}

u32 XSnn_top_hls_Get_config_reg(XSnn_top_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_CONFIG_REG_DATA);
    return Data;
}

void XSnn_top_hls_Set_learning_params(XSnn_top_hls *InstancePtr, XSnn_top_hls_Learning_params Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_top_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 0, Data.word_0);
    XSnn_top_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 4, Data.word_1);
    XSnn_top_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 8, Data.word_2);
    XSnn_top_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 12, Data.word_3);
    XSnn_top_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 16, Data.word_4);
}

XSnn_top_hls_Learning_params XSnn_top_hls_Get_learning_params(XSnn_top_hls *InstancePtr) {
    XSnn_top_hls_Learning_params Data;

    Data.word_0 = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 0);
    Data.word_1 = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 4);
    Data.word_2 = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 8);
    Data.word_3 = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 12);
    Data.word_4 = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_LEARNING_PARAMS_DATA + 16);
    return Data;
}

u32 XSnn_top_hls_Get_status_reg(XSnn_top_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_STATUS_REG_DATA);
    return Data;
}

u32 XSnn_top_hls_Get_status_reg_vld(XSnn_top_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_STATUS_REG_CTRL);
    return Data & 0x1;
}

u32 XSnn_top_hls_Get_spike_count_reg(XSnn_top_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_SPIKE_COUNT_REG_DATA);
    return Data;
}

u32 XSnn_top_hls_Get_spike_count_reg_vld(XSnn_top_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_SPIKE_COUNT_REG_CTRL);
    return Data & 0x1;
}

u32 XSnn_top_hls_Get_weight_sum_reg(XSnn_top_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_WEIGHT_SUM_REG_DATA);
    return Data;
}

u32 XSnn_top_hls_Get_weight_sum_reg_vld(XSnn_top_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_WEIGHT_SUM_REG_CTRL);
    return Data & 0x1;
}

u32 XSnn_top_hls_Get_version_reg(XSnn_top_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_VERSION_REG_DATA);
    return Data;
}

u32 XSnn_top_hls_Get_version_reg_vld(XSnn_top_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_VERSION_REG_CTRL);
    return Data & 0x1;
}

void XSnn_top_hls_Set_reward_signal(XSnn_top_hls *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_top_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_REWARD_SIGNAL_DATA, Data);
}

u32 XSnn_top_hls_Get_reward_signal(XSnn_top_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_REWARD_SIGNAL_DATA);
    return Data;
}

void XSnn_top_hls_InterruptGlobalEnable(XSnn_top_hls *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_top_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_GIE, 1);
}

void XSnn_top_hls_InterruptGlobalDisable(XSnn_top_hls *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_top_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_GIE, 0);
}

void XSnn_top_hls_InterruptEnable(XSnn_top_hls *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_IER);
    XSnn_top_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_IER, Register | Mask);
}

void XSnn_top_hls_InterruptDisable(XSnn_top_hls *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_IER);
    XSnn_top_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_IER, Register & (~Mask));
}

void XSnn_top_hls_InterruptClear(XSnn_top_hls *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XSnn_top_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_ISR, Mask);
}

u32 XSnn_top_hls_InterruptGetEnabled(XSnn_top_hls *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_IER);
}

u32 XSnn_top_hls_InterruptGetStatus(XSnn_top_hls *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XSnn_top_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XSNN_TOP_HLS_CTRL_ADDR_ISR);
}

