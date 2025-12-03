//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2025.1 (lin64) Build 6140274 Wed May 21 22:58:25 MDT 2025
//Date        : Wed Nov  5 15:39:04 2025
//Host        : jwlee-BCML-workstation running 64-bit Ubuntu 24.04.3 LTS
//Command     : generate_target axi_bfm_1_wrapper.bd
//Design      : axi_bfm_1_wrapper
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module axi_bfm_1_wrapper
   (ACLK,
    ARESETN);
  input ACLK;
  input ARESETN;

  wire ACLK;
  wire ARESETN;

  axi_bfm_1 axi_bfm_1_i
       (.ACLK(ACLK),
        .ARESETN(ARESETN));
endmodule
