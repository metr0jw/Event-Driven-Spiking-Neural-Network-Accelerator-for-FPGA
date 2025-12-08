-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
-- Tool Version Limit: 2025.11
-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
-- 
-- ==============================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity snn_top_hls_flow_control_loop_pipe_sequential_init is
port (
    ap_clk             : in  std_logic;
    ap_rst             : in  std_logic;

    --Block level handshake with outside loop
    ap_start           : in  std_logic;
    ap_ready           : out std_logic;
    ap_done            : out std_logic;

    --Block level handshake with loop body
    ap_start_int       : out std_logic;
    ap_ready_int       : in  std_logic;
    ap_done_int        : in  std_logic;
    ap_continue_int    : out std_logic;

    --Init live in variables
    --power-on initialization to 1
    ap_loop_init       : out std_logic := '1';

    --Exit signal from loop body
    ap_loop_exit_ready : in  std_logic;
    ap_loop_exit_done  : in  std_logic
);
end entity snn_top_hls_flow_control_loop_pipe_sequential_init;

architecture behav of snn_top_hls_flow_control_loop_pipe_sequential_init is
attribute DowngradeIPIdentifiedWarnings : STRING;
attribute DowngradeIPIdentifiedWarnings of behav : architecture is "yes";

    signal ap_done_cache : std_logic := '0';
    signal ap_loop_init_int : std_logic := '1';
begin
    ap_start_int    <= ap_start;   
    ap_continue_int <= '1';
    ap_ready        <= ap_loop_exit_ready;

    --ap_loop_init is valid for the first II
    --of the first loop run so as to enable
    --the init block ops which are pushed into
    --the first state of the pipeline region
    genLoopInit: process(ap_clk)
    begin
        if rising_edge(ap_clk) then
            if (ap_rst = '1') then
                ap_loop_init_int <= '1';
            elsif (ap_loop_exit_done = '1') then
                ap_loop_init_int <= '1';
            elsif (ap_ready_int = '1') then
                ap_loop_init_int <= '0';
            end if;
        end if;
    end process genLoopInit;

    ap_loop_init <= ap_loop_init_int and ap_start;

    -- if no ap_continue port and current module is not top module, 
    -- ap_done handshakes with ap_start. Internally, flow control sends out 
    -- ap_conintue_int = 1'b1 to DUT so the ap_loop_exit_done is asserted high for 1 clock cycle.
    -- ap_done_cache is used to record ap_loop_exit_done, and de-assert if ap_start_int
    -- is asserted, so DUT can start the next run
    genApDoneCache: process(ap_clk)
    begin
        if rising_edge(ap_clk) then
            if (ap_rst = '1') then
                ap_done_cache <= '0';
            elsif (ap_loop_exit_done = '1') then
                ap_done_cache <= '1';
            elsif (ap_start = '1') then
                ap_done_cache <= '0';
            end if;
        end if;
    end process genApDoneCache;

    -- if no ap_continue port and current module is not top module, ap_done handshakes with ap_start
    genApDone: process(ap_loop_exit_done, ap_done_cache, ap_start)
    begin
        if ((ap_loop_exit_done = '1') or ((ap_done_cache = '1') and (ap_start = '0'))) then
            ap_done <= '1';
        else
            ap_done <= '0';
        end if;
    end process genApDone;

end architecture;
        
