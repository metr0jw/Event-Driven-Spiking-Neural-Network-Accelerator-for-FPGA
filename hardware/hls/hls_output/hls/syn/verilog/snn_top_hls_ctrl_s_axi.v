// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
`timescale 1ns/1ps
(* DowngradeIPIdentifiedWarnings="yes" *) module snn_top_hls_ctrl_s_axi
#(parameter
    C_S_AXI_ADDR_WIDTH = 8,
    C_S_AXI_DATA_WIDTH = 32
)(
    input  wire                          ACLK,
    input  wire                          ARESET,
    input  wire                          ACLK_EN,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] AWADDR,
    input  wire                          AWVALID,
    output wire                          AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1:0] WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0] WSTRB,
    input  wire                          WVALID,
    output wire                          WREADY,
    output wire [1:0]                    BRESP,
    output wire                          BVALID,
    input  wire                          BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] ARADDR,
    input  wire                          ARVALID,
    output wire                          ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0] RDATA,
    output wire [1:0]                    RRESP,
    output wire                          RVALID,
    input  wire                          RREADY,
    output wire                          interrupt,
    output wire [31:0]                   ctrl_reg,
    output wire [31:0]                   config_reg,
    output wire [31:0]                   mode_reg,
    output wire [31:0]                   time_steps_reg,
    output wire [143:0]                  learning_params,
    output wire [143:0]                  encoder_config,
    input  wire [31:0]                   status_reg,
    input  wire                          status_reg_ap_vld,
    input  wire [31:0]                   spike_count_reg,
    input  wire                          spike_count_reg_ap_vld,
    input  wire [31:0]                   weight_sum_reg,
    input  wire                          weight_sum_reg_ap_vld,
    input  wire [31:0]                   version_reg,
    input  wire                          version_reg_ap_vld,
    output wire [7:0]                    reward_signal,
    output wire                          ap_start,
    input  wire                          ap_done,
    input  wire                          ap_ready,
    input  wire                          ap_idle
);
//------------------------Address Info-------------------
// Protocol Used: ap_ctrl_hs
//
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read/COR)
//        bit 7  - auto_restart (Read/Write)
//        bit 9  - interrupt (Read)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0 - enable ap_done interrupt (Read/Write)
//        bit 1 - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0 - ap_done (Read/TOW)
//        bit 1 - ap_ready (Read/TOW)
//        others - reserved
// 0x10 : Data signal of ctrl_reg
//        bit 31~0 - ctrl_reg[31:0] (Read/Write)
// 0x14 : reserved
// 0x18 : Data signal of config_reg
//        bit 31~0 - config_reg[31:0] (Read/Write)
// 0x1c : reserved
// 0x20 : Data signal of mode_reg
//        bit 31~0 - mode_reg[31:0] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of time_steps_reg
//        bit 31~0 - time_steps_reg[31:0] (Read/Write)
// 0x2c : reserved
// 0x30 : Data signal of learning_params
//        bit 31~0 - learning_params[31:0] (Read/Write)
// 0x34 : Data signal of learning_params
//        bit 31~0 - learning_params[63:32] (Read/Write)
// 0x38 : Data signal of learning_params
//        bit 31~0 - learning_params[95:64] (Read/Write)
// 0x3c : Data signal of learning_params
//        bit 31~0 - learning_params[127:96] (Read/Write)
// 0x40 : Data signal of learning_params
//        bit 15~0 - learning_params[143:128] (Read/Write)
//        others   - reserved
// 0x44 : reserved
// 0x48 : Data signal of encoder_config
//        bit 31~0 - encoder_config[31:0] (Read/Write)
// 0x4c : Data signal of encoder_config
//        bit 31~0 - encoder_config[63:32] (Read/Write)
// 0x50 : Data signal of encoder_config
//        bit 31~0 - encoder_config[95:64] (Read/Write)
// 0x54 : Data signal of encoder_config
//        bit 31~0 - encoder_config[127:96] (Read/Write)
// 0x58 : Data signal of encoder_config
//        bit 15~0 - encoder_config[143:128] (Read/Write)
//        others   - reserved
// 0x5c : reserved
// 0x60 : Data signal of status_reg
//        bit 31~0 - status_reg[31:0] (Read)
// 0x64 : Control signal of status_reg
//        bit 0  - status_reg_ap_vld (Read/COR)
//        others - reserved
// 0x70 : Data signal of spike_count_reg
//        bit 31~0 - spike_count_reg[31:0] (Read)
// 0x74 : Control signal of spike_count_reg
//        bit 0  - spike_count_reg_ap_vld (Read/COR)
//        others - reserved
// 0x80 : Data signal of weight_sum_reg
//        bit 31~0 - weight_sum_reg[31:0] (Read)
// 0x84 : Control signal of weight_sum_reg
//        bit 0  - weight_sum_reg_ap_vld (Read/COR)
//        others - reserved
// 0x90 : Data signal of version_reg
//        bit 31~0 - version_reg[31:0] (Read)
// 0x94 : Control signal of version_reg
//        bit 0  - version_reg_ap_vld (Read/COR)
//        others - reserved
// 0xa0 : Data signal of reward_signal
//        bit 7~0 - reward_signal[7:0] (Read/Write)
//        others  - reserved
// 0xa4 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

//------------------------Parameter----------------------
localparam
    ADDR_AP_CTRL                = 8'h00,
    ADDR_GIE                    = 8'h04,
    ADDR_IER                    = 8'h08,
    ADDR_ISR                    = 8'h0c,
    ADDR_CTRL_REG_DATA_0        = 8'h10,
    ADDR_CTRL_REG_CTRL          = 8'h14,
    ADDR_CONFIG_REG_DATA_0      = 8'h18,
    ADDR_CONFIG_REG_CTRL        = 8'h1c,
    ADDR_MODE_REG_DATA_0        = 8'h20,
    ADDR_MODE_REG_CTRL          = 8'h24,
    ADDR_TIME_STEPS_REG_DATA_0  = 8'h28,
    ADDR_TIME_STEPS_REG_CTRL    = 8'h2c,
    ADDR_LEARNING_PARAMS_DATA_0 = 8'h30,
    ADDR_LEARNING_PARAMS_DATA_1 = 8'h34,
    ADDR_LEARNING_PARAMS_DATA_2 = 8'h38,
    ADDR_LEARNING_PARAMS_DATA_3 = 8'h3c,
    ADDR_LEARNING_PARAMS_DATA_4 = 8'h40,
    ADDR_LEARNING_PARAMS_CTRL   = 8'h44,
    ADDR_ENCODER_CONFIG_DATA_0  = 8'h48,
    ADDR_ENCODER_CONFIG_DATA_1  = 8'h4c,
    ADDR_ENCODER_CONFIG_DATA_2  = 8'h50,
    ADDR_ENCODER_CONFIG_DATA_3  = 8'h54,
    ADDR_ENCODER_CONFIG_DATA_4  = 8'h58,
    ADDR_ENCODER_CONFIG_CTRL    = 8'h5c,
    ADDR_STATUS_REG_DATA_0      = 8'h60,
    ADDR_STATUS_REG_CTRL        = 8'h64,
    ADDR_SPIKE_COUNT_REG_DATA_0 = 8'h70,
    ADDR_SPIKE_COUNT_REG_CTRL   = 8'h74,
    ADDR_WEIGHT_SUM_REG_DATA_0  = 8'h80,
    ADDR_WEIGHT_SUM_REG_CTRL    = 8'h84,
    ADDR_VERSION_REG_DATA_0     = 8'h90,
    ADDR_VERSION_REG_CTRL       = 8'h94,
    ADDR_REWARD_SIGNAL_DATA_0   = 8'ha0,
    ADDR_REWARD_SIGNAL_CTRL     = 8'ha4,
    WRIDLE                      = 2'd0,
    WRDATA                      = 2'd1,
    WRRESP                      = 2'd2,
    WRRESET                     = 2'd3,
    RDIDLE                      = 2'd0,
    RDDATA                      = 2'd1,
    RDRESET                     = 2'd2,
    ADDR_BITS                = 8;

//------------------------Local signal-------------------
    reg  [1:0]                    wstate = WRRESET;
    reg  [1:0]                    wnext;
    reg  [ADDR_BITS-1:0]          waddr;
    wire [C_S_AXI_DATA_WIDTH-1:0] wmask;
    wire                          aw_hs;
    wire                          w_hs;
    reg  [1:0]                    rstate = RDRESET;
    reg  [1:0]                    rnext;
    reg  [C_S_AXI_DATA_WIDTH-1:0] rdata;
    wire                          ar_hs;
    wire [ADDR_BITS-1:0]          raddr;
    // internal registers
    reg                           int_ap_idle = 1'b0;
    reg                           int_ap_ready = 1'b0;
    wire                          task_ap_ready;
    reg                           int_ap_done = 1'b0;
    wire                          task_ap_done;
    reg                           int_task_ap_done = 1'b0;
    reg                           int_ap_start = 1'b0;
    reg                           int_interrupt = 1'b0;
    reg                           int_auto_restart = 1'b0;
    reg                           auto_restart_status = 1'b0;
    wire                          auto_restart_done;
    reg                           int_gie = 1'b0;
    reg  [1:0]                    int_ier = 2'b0;
    reg  [1:0]                    int_isr = 2'b0;
    reg  [31:0]                   int_ctrl_reg = 'b0;
    reg  [31:0]                   int_config_reg = 'b0;
    reg  [31:0]                   int_mode_reg = 'b0;
    reg  [31:0]                   int_time_steps_reg = 'b0;
    reg  [143:0]                  int_learning_params = 'b0;
    reg  [143:0]                  int_encoder_config = 'b0;
    reg                           int_status_reg_ap_vld;
    reg  [31:0]                   int_status_reg = 'b0;
    reg                           int_spike_count_reg_ap_vld;
    reg  [31:0]                   int_spike_count_reg = 'b0;
    reg                           int_weight_sum_reg_ap_vld;
    reg  [31:0]                   int_weight_sum_reg = 'b0;
    reg                           int_version_reg_ap_vld;
    reg  [31:0]                   int_version_reg = 'b0;
    reg  [7:0]                    int_reward_signal = 'b0;

//------------------------Instantiation------------------


//------------------------AXI write fsm------------------
assign AWREADY = (wstate == WRIDLE);
assign WREADY  = (wstate == WRDATA);
assign BVALID  = (wstate == WRRESP);
assign BRESP   = 2'b00;  // OKAY
assign wmask   = { {8{WSTRB[3]}}, {8{WSTRB[2]}}, {8{WSTRB[1]}}, {8{WSTRB[0]}} };
assign aw_hs   = AWVALID & AWREADY;
assign w_hs    = WVALID & WREADY;

// wstate
always @(posedge ACLK) begin
    if (ARESET)
        wstate <= WRRESET;
    else if (ACLK_EN)
        wstate <= wnext;
end

// wnext
always @(*) begin
    case (wstate)
        WRIDLE:
            if (AWVALID)
                wnext = WRDATA;
            else
                wnext = WRIDLE;
        WRDATA:
            if (WVALID)
                wnext = WRRESP;
            else
                wnext = WRDATA;
        WRRESP:
            if (BREADY & BVALID)
                wnext = WRIDLE;
            else
                wnext = WRRESP;
        default:
            wnext = WRIDLE;
    endcase
end

// waddr
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (aw_hs)
            waddr <= {AWADDR[ADDR_BITS-1:2], {2{1'b0}}};
    end
end

//------------------------AXI read fsm-------------------
assign ARREADY = (rstate == RDIDLE);
assign RDATA   = rdata;
assign RRESP   = 2'b00;  // OKAY
assign RVALID  = (rstate == RDDATA);
assign ar_hs   = ARVALID & ARREADY;
assign raddr   = ARADDR[ADDR_BITS-1:0];

// rstate
always @(posedge ACLK) begin
    if (ARESET)
        rstate <= RDRESET;
    else if (ACLK_EN)
        rstate <= rnext;
end

// rnext
always @(*) begin
    case (rstate)
        RDIDLE:
            if (ARVALID)
                rnext = RDDATA;
            else
                rnext = RDIDLE;
        RDDATA:
            if (RREADY & RVALID)
                rnext = RDIDLE;
            else
                rnext = RDDATA;
        default:
            rnext = RDIDLE;
    endcase
end

// rdata
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (ar_hs) begin
            rdata <= 'b0;
            case (raddr)
                ADDR_AP_CTRL: begin
                    rdata[0] <= int_ap_start;
                    rdata[1] <= int_task_ap_done;
                    rdata[2] <= int_ap_idle;
                    rdata[3] <= int_ap_ready;
                    rdata[7] <= int_auto_restart;
                    rdata[9] <= int_interrupt;
                end
                ADDR_GIE: begin
                    rdata <= int_gie;
                end
                ADDR_IER: begin
                    rdata <= int_ier;
                end
                ADDR_ISR: begin
                    rdata <= int_isr;
                end
                ADDR_CTRL_REG_DATA_0: begin
                    rdata <= int_ctrl_reg[31:0];
                end
                ADDR_CONFIG_REG_DATA_0: begin
                    rdata <= int_config_reg[31:0];
                end
                ADDR_MODE_REG_DATA_0: begin
                    rdata <= int_mode_reg[31:0];
                end
                ADDR_TIME_STEPS_REG_DATA_0: begin
                    rdata <= int_time_steps_reg[31:0];
                end
                ADDR_LEARNING_PARAMS_DATA_0: begin
                    rdata <= int_learning_params[31:0];
                end
                ADDR_LEARNING_PARAMS_DATA_1: begin
                    rdata <= int_learning_params[63:32];
                end
                ADDR_LEARNING_PARAMS_DATA_2: begin
                    rdata <= int_learning_params[95:64];
                end
                ADDR_LEARNING_PARAMS_DATA_3: begin
                    rdata <= int_learning_params[127:96];
                end
                ADDR_LEARNING_PARAMS_DATA_4: begin
                    rdata <= int_learning_params[143:128];
                end
                ADDR_ENCODER_CONFIG_DATA_0: begin
                    rdata <= int_encoder_config[31:0];
                end
                ADDR_ENCODER_CONFIG_DATA_1: begin
                    rdata <= int_encoder_config[63:32];
                end
                ADDR_ENCODER_CONFIG_DATA_2: begin
                    rdata <= int_encoder_config[95:64];
                end
                ADDR_ENCODER_CONFIG_DATA_3: begin
                    rdata <= int_encoder_config[127:96];
                end
                ADDR_ENCODER_CONFIG_DATA_4: begin
                    rdata <= int_encoder_config[143:128];
                end
                ADDR_STATUS_REG_DATA_0: begin
                    rdata <= int_status_reg[31:0];
                end
                ADDR_STATUS_REG_CTRL: begin
                    rdata[0] <= int_status_reg_ap_vld;
                end
                ADDR_SPIKE_COUNT_REG_DATA_0: begin
                    rdata <= int_spike_count_reg[31:0];
                end
                ADDR_SPIKE_COUNT_REG_CTRL: begin
                    rdata[0] <= int_spike_count_reg_ap_vld;
                end
                ADDR_WEIGHT_SUM_REG_DATA_0: begin
                    rdata <= int_weight_sum_reg[31:0];
                end
                ADDR_WEIGHT_SUM_REG_CTRL: begin
                    rdata[0] <= int_weight_sum_reg_ap_vld;
                end
                ADDR_VERSION_REG_DATA_0: begin
                    rdata <= int_version_reg[31:0];
                end
                ADDR_VERSION_REG_CTRL: begin
                    rdata[0] <= int_version_reg_ap_vld;
                end
                ADDR_REWARD_SIGNAL_DATA_0: begin
                    rdata <= int_reward_signal[7:0];
                end
            endcase
        end
    end
end


//------------------------Register logic-----------------
assign interrupt         = int_interrupt;
assign ap_start          = int_ap_start;
assign task_ap_done      = (ap_done && !auto_restart_status) || auto_restart_done;
assign task_ap_ready     = ap_ready && !int_auto_restart;
assign auto_restart_done = auto_restart_status && (ap_idle && !int_ap_idle);
assign ctrl_reg          = int_ctrl_reg;
assign config_reg        = int_config_reg;
assign mode_reg          = int_mode_reg;
assign time_steps_reg    = int_time_steps_reg;
assign learning_params   = int_learning_params;
assign encoder_config    = int_encoder_config;
assign reward_signal     = int_reward_signal;
// int_interrupt
always @(posedge ACLK) begin
    if (ARESET)
        int_interrupt <= 1'b0;
    else if (ACLK_EN) begin
        if (int_gie && (|int_isr))
            int_interrupt <= 1'b1;
        else
            int_interrupt <= 1'b0;
    end
end

// int_ap_start
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_start <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AP_CTRL && WSTRB[0] && WDATA[0])
            int_ap_start <= 1'b1;
        else if (ap_ready)
            int_ap_start <= int_auto_restart; // clear on handshake/auto restart
    end
end

// int_ap_done
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_done <= 1'b0;
    else if (ACLK_EN) begin
            int_ap_done <= ap_done;
    end
end

// int_task_ap_done
always @(posedge ACLK) begin
    if (ARESET)
        int_task_ap_done <= 1'b0;
    else if (ACLK_EN) begin
        if (task_ap_done)
            int_task_ap_done <= 1'b1;
        else if (ar_hs && raddr == ADDR_AP_CTRL)
            int_task_ap_done <= 1'b0; // clear on read
    end
end

// int_ap_idle
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_idle <= 1'b0;
    else if (ACLK_EN) begin
            int_ap_idle <= ap_idle;
    end
end

// int_ap_ready
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_ready <= 1'b0;
    else if (ACLK_EN) begin
        if (task_ap_ready)
            int_ap_ready <= 1'b1;
        else if (ar_hs && raddr == ADDR_AP_CTRL)
            int_ap_ready <= 1'b0;
    end
end

// int_auto_restart
always @(posedge ACLK) begin
    if (ARESET)
        int_auto_restart <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AP_CTRL && WSTRB[0])
            int_auto_restart <= WDATA[7];
    end
end

// auto_restart_status
always @(posedge ACLK) begin
    if (ARESET)
        auto_restart_status <= 1'b0;
    else if (ACLK_EN) begin
        if (int_auto_restart)
            auto_restart_status <= 1'b1;
        else if (ap_idle)
            auto_restart_status <= 1'b0;
    end
end

// int_gie
always @(posedge ACLK) begin
    if (ARESET)
        int_gie <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_GIE && WSTRB[0])
            int_gie <= WDATA[0];
    end
end

// int_ier
always @(posedge ACLK) begin
    if (ARESET)
        int_ier <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IER && WSTRB[0])
            int_ier <= WDATA[1:0];
    end
end

// int_isr[0]
always @(posedge ACLK) begin
    if (ARESET)
        int_isr[0] <= 1'b0;
    else if (ACLK_EN) begin
        if (int_ier[0] & ap_done)
            int_isr[0] <= 1'b1;
        else if (w_hs && waddr == ADDR_ISR && WSTRB[0])
            int_isr[0] <= int_isr[0] ^ WDATA[0]; // toggle on write
    end
end

// int_isr[1]
always @(posedge ACLK) begin
    if (ARESET)
        int_isr[1] <= 1'b0;
    else if (ACLK_EN) begin
        if (int_ier[1] & ap_ready)
            int_isr[1] <= 1'b1;
        else if (w_hs && waddr == ADDR_ISR && WSTRB[0])
            int_isr[1] <= int_isr[1] ^ WDATA[1]; // toggle on write
    end
end

// int_ctrl_reg[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_ctrl_reg[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_CTRL_REG_DATA_0)
            int_ctrl_reg[31:0] <= (WDATA[31:0] & wmask) | (int_ctrl_reg[31:0] & ~wmask);
    end
end

// int_config_reg[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_config_reg[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_CONFIG_REG_DATA_0)
            int_config_reg[31:0] <= (WDATA[31:0] & wmask) | (int_config_reg[31:0] & ~wmask);
    end
end

// int_mode_reg[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_mode_reg[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_MODE_REG_DATA_0)
            int_mode_reg[31:0] <= (WDATA[31:0] & wmask) | (int_mode_reg[31:0] & ~wmask);
    end
end

// int_time_steps_reg[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_time_steps_reg[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_TIME_STEPS_REG_DATA_0)
            int_time_steps_reg[31:0] <= (WDATA[31:0] & wmask) | (int_time_steps_reg[31:0] & ~wmask);
    end
end

// int_learning_params[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_learning_params[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LEARNING_PARAMS_DATA_0)
            int_learning_params[31:0] <= (WDATA[31:0] & wmask) | (int_learning_params[31:0] & ~wmask);
    end
end

// int_learning_params[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_learning_params[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LEARNING_PARAMS_DATA_1)
            int_learning_params[63:32] <= (WDATA[31:0] & wmask) | (int_learning_params[63:32] & ~wmask);
    end
end

// int_learning_params[95:64]
always @(posedge ACLK) begin
    if (ARESET)
        int_learning_params[95:64] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LEARNING_PARAMS_DATA_2)
            int_learning_params[95:64] <= (WDATA[31:0] & wmask) | (int_learning_params[95:64] & ~wmask);
    end
end

// int_learning_params[127:96]
always @(posedge ACLK) begin
    if (ARESET)
        int_learning_params[127:96] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LEARNING_PARAMS_DATA_3)
            int_learning_params[127:96] <= (WDATA[31:0] & wmask) | (int_learning_params[127:96] & ~wmask);
    end
end

// int_learning_params[143:128]
always @(posedge ACLK) begin
    if (ARESET)
        int_learning_params[143:128] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LEARNING_PARAMS_DATA_4)
            int_learning_params[143:128] <= (WDATA[31:0] & wmask) | (int_learning_params[143:128] & ~wmask);
    end
end

// int_encoder_config[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_encoder_config[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_ENCODER_CONFIG_DATA_0)
            int_encoder_config[31:0] <= (WDATA[31:0] & wmask) | (int_encoder_config[31:0] & ~wmask);
    end
end

// int_encoder_config[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_encoder_config[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_ENCODER_CONFIG_DATA_1)
            int_encoder_config[63:32] <= (WDATA[31:0] & wmask) | (int_encoder_config[63:32] & ~wmask);
    end
end

// int_encoder_config[95:64]
always @(posedge ACLK) begin
    if (ARESET)
        int_encoder_config[95:64] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_ENCODER_CONFIG_DATA_2)
            int_encoder_config[95:64] <= (WDATA[31:0] & wmask) | (int_encoder_config[95:64] & ~wmask);
    end
end

// int_encoder_config[127:96]
always @(posedge ACLK) begin
    if (ARESET)
        int_encoder_config[127:96] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_ENCODER_CONFIG_DATA_3)
            int_encoder_config[127:96] <= (WDATA[31:0] & wmask) | (int_encoder_config[127:96] & ~wmask);
    end
end

// int_encoder_config[143:128]
always @(posedge ACLK) begin
    if (ARESET)
        int_encoder_config[143:128] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_ENCODER_CONFIG_DATA_4)
            int_encoder_config[143:128] <= (WDATA[31:0] & wmask) | (int_encoder_config[143:128] & ~wmask);
    end
end

// int_status_reg
always @(posedge ACLK) begin
    if (ARESET)
        int_status_reg <= 0;
    else if (ACLK_EN) begin
        if (status_reg_ap_vld)
            int_status_reg <= status_reg;
    end
end

// int_status_reg_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_reg_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_reg_ap_vld)
            int_status_reg_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_REG_CTRL)
            int_status_reg_ap_vld <= 1'b0; // clear on read
    end
end

// int_spike_count_reg
always @(posedge ACLK) begin
    if (ARESET)
        int_spike_count_reg <= 0;
    else if (ACLK_EN) begin
        if (spike_count_reg_ap_vld)
            int_spike_count_reg <= spike_count_reg;
    end
end

// int_spike_count_reg_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_spike_count_reg_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (spike_count_reg_ap_vld)
            int_spike_count_reg_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_SPIKE_COUNT_REG_CTRL)
            int_spike_count_reg_ap_vld <= 1'b0; // clear on read
    end
end

// int_weight_sum_reg
always @(posedge ACLK) begin
    if (ARESET)
        int_weight_sum_reg <= 0;
    else if (ACLK_EN) begin
        if (weight_sum_reg_ap_vld)
            int_weight_sum_reg <= weight_sum_reg;
    end
end

// int_weight_sum_reg_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_weight_sum_reg_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (weight_sum_reg_ap_vld)
            int_weight_sum_reg_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_WEIGHT_SUM_REG_CTRL)
            int_weight_sum_reg_ap_vld <= 1'b0; // clear on read
    end
end

// int_version_reg
always @(posedge ACLK) begin
    if (ARESET)
        int_version_reg <= 0;
    else if (ACLK_EN) begin
        if (version_reg_ap_vld)
            int_version_reg <= version_reg;
    end
end

// int_version_reg_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_version_reg_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (version_reg_ap_vld)
            int_version_reg_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_VERSION_REG_CTRL)
            int_version_reg_ap_vld <= 1'b0; // clear on read
    end
end

// int_reward_signal[7:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_reward_signal[7:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_REWARD_SIGNAL_DATA_0)
            int_reward_signal[7:0] <= (WDATA[31:0] & wmask) | (int_reward_signal[7:0] & ~wmask);
    end
end

//synthesis translate_off
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (int_gie & ~int_isr[0] & int_ier[0] & ap_done)
            $display ("// Interrupt Monitor : interrupt for ap_done detected @ \"%0t\"", $time);
        if (int_gie & ~int_isr[1] & int_ier[1] & ap_ready)
            $display ("// Interrupt Monitor : interrupt for ap_ready detected @ \"%0t\"", $time);
    end
end
//synthesis translate_on

//------------------------Memory logic-------------------

endmodule
