//-----------------------------------------------------------------------------
// Title         : LIF Neuron Array with BRAM/DSP Optimization
// Project       : PYNQ-Z2 SNN Accelerator
// File          : lif_neuron_array.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Optimized array of LIF neurons using:
//                 - Block RAM for state storage
//                 - DSP blocks for arithmetic operations
//                 - Pipelined parallel processing
//                 - Hardware-accurate shift-based exponential leak
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module lif_neuron_array #(
    // Scalable parameters
    parameter NUM_NEURONS           = 256,    // Scaled up for better utilization
    parameter NUM_AXONS             = 256,
    parameter DATA_WIDTH            = 16,
    parameter WEIGHT_WIDTH          = 8,
    parameter THRESHOLD_WIDTH       = 16,
    parameter LEAK_WIDTH            = 8,
    parameter REFRAC_WIDTH          = 8,
    
    // Parallelism control
    parameter NUM_PARALLEL_UNITS    = 8,      // Parallel processing units
    parameter SPIKE_BUFFER_DEPTH    = 64,     // Input spike buffer
    
    // Resource hints
    parameter USE_BRAM              = 1,      // Use BRAM for state storage
    parameter USE_DSP               = 1,      // Use DSP for arithmetic
    
    // Derived parameters
    parameter NEURON_ID_WIDTH       = $clog2(NUM_NEURONS)
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         enable,
    
    // Input spike interface (AXI-Stream compatible)
    input  wire                         s_axis_spike_valid,
    input  wire [NEURON_ID_WIDTH-1:0]   s_axis_spike_dest_id,
    input  wire [WEIGHT_WIDTH-1:0]      s_axis_spike_weight,
    input  wire                         s_axis_spike_exc_inh,  // 1: exc, 0: inh
    output wire                         s_axis_spike_ready,
    
    // Output spike interface
    output reg                          m_axis_spike_valid,
    output reg  [NEURON_ID_WIDTH-1:0]   m_axis_spike_neuron_id,
    input  wire                         m_axis_spike_ready,
    
    // Configuration interface
    input  wire                         config_we,
    input  wire [NEURON_ID_WIDTH-1:0]   config_addr,
    input  wire [31:0]                  config_data,
    
    // Global neuron parameters
    input  wire [THRESHOLD_WIDTH-1:0]   global_threshold,
    input  wire [LEAK_WIDTH-1:0]        global_leak_rate,  // [7:4]=shift2, [3:0]=shift1
    input  wire [REFRAC_WIDTH-1:0]      global_refrac_period,
    
    // Status outputs
    output wire [31:0]                  spike_count,
    output wire                         array_busy,
    output wire [31:0]                  throughput_counter,
    output wire [7:0]                   active_neurons
);

    //=========================================================================
    // BRAM-based State Memory
    //=========================================================================
    // Xilinx BRAM inference with registered outputs for better timing
    (* ram_style = "block" *)
    reg [DATA_WIDTH-1:0] membrane_bram [0:NUM_NEURONS-1];
    
    (* ram_style = "block" *)
    reg [REFRAC_WIDTH-1:0] refrac_bram [0:NUM_NEURONS-1];
    
    // Spike flags (distributed RAM - small width)
    reg [NUM_NEURONS-1:0] spike_flags;
    reg [NUM_NEURONS-1:0] spike_processed;

    //=========================================================================
    // Pipeline Registers for BRAM Access (improves timing)
    //=========================================================================
    // Stage 1: Address
    reg [NEURON_ID_WIDTH-1:0] read_addr_s1 [0:NUM_PARALLEL_UNITS-1];
    reg [NUM_PARALLEL_UNITS-1:0] read_valid_s1;
    
    // Stage 2: Data (BRAM output)
    reg [DATA_WIDTH-1:0] membrane_s2 [0:NUM_PARALLEL_UNITS-1];
    reg [REFRAC_WIDTH-1:0] refrac_s2 [0:NUM_PARALLEL_UNITS-1];
    reg [NEURON_ID_WIDTH-1:0] addr_s2 [0:NUM_PARALLEL_UNITS-1];
    reg [NUM_PARALLEL_UNITS-1:0] valid_s2;
    
    // Stage 3: Compute result
    reg [DATA_WIDTH-1:0] new_membrane_s3 [0:NUM_PARALLEL_UNITS-1];
    reg [REFRAC_WIDTH-1:0] new_refrac_s3 [0:NUM_PARALLEL_UNITS-1];
    reg [NEURON_ID_WIDTH-1:0] addr_s3 [0:NUM_PARALLEL_UNITS-1];
    reg [NUM_PARALLEL_UNITS-1:0] valid_s3;
    reg [NUM_PARALLEL_UNITS-1:0] spike_s3;

    //=========================================================================
    // Input Spike FIFO (using BRAM for deeper buffer)
    //=========================================================================
    (* ram_style = "block" *)
    reg [NEURON_ID_WIDTH+WEIGHT_WIDTH:0] spike_fifo [0:SPIKE_BUFFER_DEPTH-1];
    
    reg [$clog2(SPIKE_BUFFER_DEPTH)-1:0] fifo_wr_ptr;
    reg [$clog2(SPIKE_BUFFER_DEPTH)-1:0] fifo_rd_ptr;
    reg [$clog2(SPIKE_BUFFER_DEPTH):0] fifo_count;
    
    wire fifo_empty = (fifo_count == 0);
    wire fifo_full = (fifo_count == SPIKE_BUFFER_DEPTH);
    
    assign s_axis_spike_ready = !fifo_full;

    //=========================================================================
    // State Machine
    //=========================================================================
    localparam [2:0] 
        ST_IDLE     = 3'd0,
        ST_LEAK     = 3'd1,
        ST_PROCESS  = 3'd2,
        ST_OUTPUT   = 3'd3;
    
    reg [2:0] state;
    reg [NEURON_ID_WIDTH-1:0] leak_idx;
    reg leak_cycle_done;
    
    assign array_busy = (state != ST_IDLE) || !fifo_empty;

    //=========================================================================
    // Shift-based Leak Parameters (Hardware-accurate)
    //=========================================================================
    // leak_rate encoding: [7:4] = shift2, [3:0] = shift1
    // tau = 1 - 2^(-shift1) - 2^(-shift2)
    wire [3:0] shift1 = global_leak_rate[3:0];
    wire [3:0] shift2 = global_leak_rate[7:4];

    //=========================================================================
    // FIFO Write (Input Spikes)
    //=========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            fifo_wr_ptr <= 0;
            fifo_count <= 0;
        end else begin
            // Write to FIFO
            if (s_axis_spike_valid && !fifo_full) begin
                spike_fifo[fifo_wr_ptr] <= {s_axis_spike_exc_inh, s_axis_spike_weight, s_axis_spike_dest_id};
                fifo_wr_ptr <= fifo_wr_ptr + 1;
                if (!(fifo_rd_ptr != fifo_wr_ptr && state == ST_PROCESS))
                    fifo_count <= fifo_count + 1;
            end
            
            // Decrement count when reading
            if (state == ST_PROCESS && !fifo_empty && fifo_count > 0) begin
                if (!(s_axis_spike_valid && !fifo_full))
                    fifo_count <= fifo_count - 1;
            end
        end
    end

    //=========================================================================
    // Main State Machine
    //=========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            leak_idx <= 0;
            leak_cycle_done <= 0;
            fifo_rd_ptr <= 0;
        end else if (enable) begin
            case (state)
                ST_IDLE: begin
                    // Priority: process spikes first, then leak
                    if (!fifo_empty) begin
                        state <= ST_PROCESS;
                    end else if (!leak_cycle_done) begin
                        state <= ST_LEAK;
                    end
                end
                
                ST_LEAK: begin
                    // Interrupt leak if spike arrives
                    if (!fifo_empty) begin
                        state <= ST_PROCESS;
                    end else begin
                        leak_idx <= leak_idx + NUM_PARALLEL_UNITS;
                        if (leak_idx + NUM_PARALLEL_UNITS >= NUM_NEURONS) begin
                            leak_idx <= 0;
                            leak_cycle_done <= 1;
                            state <= ST_IDLE;
                        end
                    end
                end
                
                ST_PROCESS: begin
                    if (!fifo_empty) begin
                        fifo_rd_ptr <= fifo_rd_ptr + 1;
                    end else begin
                        state <= ST_IDLE;
                    end
                end
                
                default: state <= ST_IDLE;
            endcase
            
            // Reset leak_cycle_done periodically for continuous leak
            if (leak_cycle_done && fifo_empty) begin
                // Allow leak cycle to restart after some idle time
            end
        end
    end

    //=========================================================================
    // Parallel Leak Processing Units with DSP
    //=========================================================================
    genvar pu;
    generate
        for (pu = 0; pu < NUM_PARALLEL_UNITS; pu = pu + 1) begin : gen_parallel_units
            
            // Stage 1: Read from BRAM
            always @(posedge clk) begin
                if (!rst_n) begin
                    read_valid_s1[pu] <= 0;
                    read_addr_s1[pu] <= 0;
                end else begin
                    read_valid_s1[pu] <= 0;
                    
                    if (state == ST_LEAK && enable) begin
                        if (leak_idx + pu < NUM_NEURONS) begin
                            read_addr_s1[pu] <= leak_idx + pu;
                            read_valid_s1[pu] <= 1;
                        end
                    end
                end
            end
            
            // Stage 2: BRAM read data arrives
            always @(posedge clk) begin
                if (!rst_n) begin
                    valid_s2[pu] <= 0;
                    membrane_s2[pu] <= 0;
                    refrac_s2[pu] <= 0;
                    addr_s2[pu] <= 0;
                end else begin
                    valid_s2[pu] <= read_valid_s1[pu];
                    addr_s2[pu] <= read_addr_s1[pu];
                    
                    if (read_valid_s1[pu]) begin
                        membrane_s2[pu] <= membrane_bram[read_addr_s1[pu]];
                        refrac_s2[pu] <= refrac_bram[read_addr_s1[pu]];
                    end
                end
            end
            
            // Stage 3: Compute new values with shift-based leak
            // DSP inference for leak calculation
            reg [DATA_WIDTH-1:0] leak_primary;
            reg [DATA_WIDTH-1:0] leak_secondary;
            reg [DATA_WIDTH-1:0] leak_total;
            
            always @(posedge clk) begin
                if (!rst_n) begin
                    valid_s3[pu] <= 0;
                    new_membrane_s3[pu] <= 0;
                    new_refrac_s3[pu] <= 0;
                    addr_s3[pu] <= 0;
                    spike_s3[pu] <= 0;
                    leak_primary <= 0;
                    leak_secondary <= 0;
                    leak_total <= 0;
                end else begin
                    valid_s3[pu] <= valid_s2[pu];
                    addr_s3[pu] <= addr_s2[pu];
                    spike_s3[pu] <= 0;
                    
                    if (valid_s2[pu]) begin
                        if (refrac_s2[pu] > 0) begin
                            // In refractory period
                            new_membrane_s3[pu] <= 0;
                            new_refrac_s3[pu] <= refrac_s2[pu] - 1;
                        end else begin
                            // Shift-based exponential leak (hardware-accurate)
                            // leak = v >> shift1 + v >> shift2
                            leak_primary = membrane_s2[pu] >> shift1;
                            leak_secondary = membrane_s2[pu] >> shift2;
                            leak_total = leak_primary + leak_secondary;
                            
                            if (membrane_s2[pu] > leak_total)
                                new_membrane_s3[pu] <= membrane_s2[pu] - leak_total;
                            else
                                new_membrane_s3[pu] <= 0;
                                
                            new_refrac_s3[pu] <= 0;
                        end
                    end
                end
            end
        end
    endgenerate

    //=========================================================================
    // Spike Processing Pipeline (separate from leak)
    //=========================================================================
    wire [NEURON_ID_WIDTH-1:0] spike_dest_id = spike_fifo[fifo_rd_ptr][NEURON_ID_WIDTH-1:0];
    wire [WEIGHT_WIDTH-1:0] spike_weight = spike_fifo[fifo_rd_ptr][NEURON_ID_WIDTH+WEIGHT_WIDTH-1:NEURON_ID_WIDTH];
    wire spike_exc = spike_fifo[fifo_rd_ptr][NEURON_ID_WIDTH+WEIGHT_WIDTH];
    
    // Spike processing pipeline stages
    reg spike_proc_valid_s1, spike_proc_valid_s2, spike_proc_valid_s3;
    reg [NEURON_ID_WIDTH-1:0] spike_proc_addr_s1, spike_proc_addr_s2, spike_proc_addr_s3;
    reg [WEIGHT_WIDTH-1:0] spike_proc_weight_s1, spike_proc_weight_s2;
    reg spike_proc_exc_s1, spike_proc_exc_s2;
    reg [DATA_WIDTH-1:0] spike_proc_membrane_s2;
    reg [REFRAC_WIDTH-1:0] spike_proc_refrac_s2;
    reg [DATA_WIDTH-1:0] spike_proc_new_mem_s3;
    reg spike_proc_fired_s3;
    
    // DSP-friendly addition/subtraction
    (* use_dsp = "yes" *)
    reg [DATA_WIDTH:0] synaptic_sum;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            spike_proc_valid_s1 <= 0;
            spike_proc_valid_s2 <= 0;
            spike_proc_valid_s3 <= 0;
            spike_proc_addr_s1 <= 0;
            spike_proc_addr_s2 <= 0;
            spike_proc_addr_s3 <= 0;
            spike_proc_weight_s1 <= 0;
            spike_proc_weight_s2 <= 0;
            spike_proc_exc_s1 <= 0;
            spike_proc_exc_s2 <= 0;
            spike_proc_membrane_s2 <= 0;
            spike_proc_refrac_s2 <= 0;
            spike_proc_new_mem_s3 <= 0;
            spike_proc_fired_s3 <= 0;
            synaptic_sum <= 0;
        end else begin
            // Stage 1: Capture spike info
            spike_proc_valid_s1 <= (state == ST_PROCESS) && !fifo_empty;
            spike_proc_addr_s1 <= spike_dest_id;
            spike_proc_weight_s1 <= spike_weight;
            spike_proc_exc_s1 <= spike_exc;
            
            // Stage 2: Read BRAM
            spike_proc_valid_s2 <= spike_proc_valid_s1;
            spike_proc_addr_s2 <= spike_proc_addr_s1;
            spike_proc_weight_s2 <= spike_proc_weight_s1;
            spike_proc_exc_s2 <= spike_proc_exc_s1;
            if (spike_proc_valid_s1) begin
                spike_proc_membrane_s2 <= membrane_bram[spike_proc_addr_s1];
                spike_proc_refrac_s2 <= refrac_bram[spike_proc_addr_s1];
            end
            
            // Stage 3: Compute using DSP
            spike_proc_valid_s3 <= spike_proc_valid_s2;
            spike_proc_addr_s3 <= spike_proc_addr_s2;
            spike_proc_fired_s3 <= 0;
            
            if (spike_proc_valid_s2 && spike_proc_refrac_s2 == 0) begin
                if (spike_proc_exc_s2) begin
                    // Excitatory - add weight (DSP)
                    synaptic_sum = spike_proc_membrane_s2 + spike_proc_weight_s2;
                    
                    if (synaptic_sum >= global_threshold) begin
                        spike_proc_new_mem_s3 <= 0;
                        spike_proc_fired_s3 <= 1;
                    end else if (synaptic_sum[DATA_WIDTH]) begin
                        // Overflow - saturate
                        spike_proc_new_mem_s3 <= {DATA_WIDTH{1'b1}};
                    end else begin
                        spike_proc_new_mem_s3 <= synaptic_sum[DATA_WIDTH-1:0];
                    end
                end else begin
                    // Inhibitory - subtract weight
                    if (spike_proc_membrane_s2 <= spike_proc_weight_s2) begin
                        spike_proc_new_mem_s3 <= 0;
                    end else begin
                        spike_proc_new_mem_s3 <= spike_proc_membrane_s2 - spike_proc_weight_s2;
                    end
                end
            end else begin
                spike_proc_new_mem_s3 <= spike_proc_membrane_s2;
            end
        end
    end

    //=========================================================================
    // BRAM Write-back (unified write port)
    //=========================================================================
    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < NUM_NEURONS; i = i + 1) begin
                membrane_bram[i] <= 0;
                refrac_bram[i] <= 0;
            end
            spike_flags <= 0;
        end else begin
            // Configuration write (highest priority)
            if (config_we && config_addr < NUM_NEURONS) begin
                case (config_data[31:30])
                    2'b00: membrane_bram[config_addr] <= config_data[DATA_WIDTH-1:0];
                    2'b01: refrac_bram[config_addr] <= config_data[REFRAC_WIDTH-1:0];
                endcase
            end
            
            // Spike processing write-back (high priority)
            if (spike_proc_valid_s3) begin
                membrane_bram[spike_proc_addr_s3] <= spike_proc_new_mem_s3;
                if (spike_proc_fired_s3) begin
                    refrac_bram[spike_proc_addr_s3] <= global_refrac_period;
                    spike_flags[spike_proc_addr_s3] <= 1;
                end
            end
            
            // Leak write-back (parallel units)
            for (i = 0; i < NUM_PARALLEL_UNITS; i = i + 1) begin
                if (valid_s3[i] && !spike_proc_valid_s3) begin
                    membrane_bram[addr_s3[i]] <= new_membrane_s3[i];
                    refrac_bram[addr_s3[i]] <= new_refrac_s3[i];
                end
            end
            
            // Clear processed spike flags
            for (i = 0; i < NUM_NEURONS; i = i + 1) begin
                if (spike_processed[i])
                    spike_flags[i] <= 0;
            end
        end
    end

    //=========================================================================
    // Output Spike Generation
    //=========================================================================
    reg [NEURON_ID_WIDTH-1:0] output_scan_idx;
    reg [31:0] total_spikes;
    reg [31:0] throughput_cnt;
    reg [7:0] active_neuron_cnt;
    
    assign spike_count = total_spikes;
    assign throughput_counter = throughput_cnt;
    assign active_neurons = active_neuron_cnt;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            m_axis_spike_valid <= 0;
            m_axis_spike_neuron_id <= 0;
            output_scan_idx <= 0;
            total_spikes <= 0;
            throughput_cnt <= 0;
            active_neuron_cnt <= 0;
            spike_processed <= 0;
        end else begin
            spike_processed <= 0;
            throughput_cnt <= throughput_cnt + 1;
            
            // Scan for output spikes
            if (m_axis_spike_ready || !m_axis_spike_valid) begin
                m_axis_spike_valid <= 0;
                
                // Find next spike flag
                if (spike_flags[output_scan_idx] && !spike_processed[output_scan_idx]) begin
                    m_axis_spike_valid <= 1;
                    m_axis_spike_neuron_id <= output_scan_idx;
                    spike_processed[output_scan_idx] <= 1;
                    total_spikes <= total_spikes + 1;
                    active_neuron_cnt <= active_neuron_cnt + 1;
                end
                
                output_scan_idx <= output_scan_idx + 1;
            end
            
            // Reset active count periodically
            if (throughput_cnt[15:0] == 0) begin
                active_neuron_cnt <= 0;
            end
        end
    end

endmodule
