//-----------------------------------------------------------------------------
// Title         : LIF Neuron Array with Time-Multiplexing
// Project       : PYNQ-Z2 SNN Accelerator
// File          : lif_neuron_array.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Array of LIF neurons with shared processing units
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module lif_neuron_array #(
    parameter NUM_NEURONS       = 64,     // Total number of neurons
    parameter NUM_AXONS         = 64,     // Number of input axons
    parameter DATA_WIDTH        = 16,
    parameter WEIGHT_WIDTH      = 8,
    parameter THRESHOLD_WIDTH   = 16,
    parameter LEAK_WIDTH        = 8,
    parameter REFRAC_WIDTH      = 8,
    parameter NEURON_ID_WIDTH   = $clog2(NUM_NEURONS)
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         enable,
    
    // Input spike interface (AXI-Stream compatible)
    input  wire                         s_axis_spike_valid,
    input  wire [NEURON_ID_WIDTH-1:0]  s_axis_spike_dest_id,
    input  wire [WEIGHT_WIDTH-1:0]     s_axis_spike_weight,
    input  wire                         s_axis_spike_exc_inh,  // 1: exc, 0: inh
    output wire                         s_axis_spike_ready,
    
    // Output spike interface
    output reg                          m_axis_spike_valid,
    output reg  [NEURON_ID_WIDTH-1:0]  m_axis_spike_neuron_id,
    input  wire                         m_axis_spike_ready,
    
    // Configuration interface (AXI-Lite would connect here)
    input  wire                         config_we,
    input  wire [NEURON_ID_WIDTH-1:0]  config_addr,
    input  wire [31:0]                 config_data,
    
    // Global neuron parameters
    input  wire [THRESHOLD_WIDTH-1:0]  global_threshold,
    input  wire [LEAK_WIDTH-1:0]       global_leak_rate,
    input  wire [REFRAC_WIDTH-1:0]     global_refrac_period,
    
    // Status outputs
    output wire [31:0]                 spike_count,
    output wire                        array_busy
);

    // State machine states
    localparam IDLE         = 3'd0;
    localparam RECEIVE      = 3'd1;
    localparam PROCESS      = 3'd2;
    localparam UPDATE       = 3'd3;
    localparam OUTPUT       = 3'd4;
    
    reg [2:0] state, next_state;
    
    // Neuron state memory (using BRAM)
    reg [DATA_WIDTH-1:0]    membrane_potential [0:NUM_NEURONS-1];
    reg [REFRAC_WIDTH-1:0]  refractory_counter [0:NUM_NEURONS-1];
    reg [NUM_NEURONS-1:0]   spike_flags;
    
    // Input spike buffer
    reg                         spike_pending;
    reg [NEURON_ID_WIDTH-1:0]  spike_dest;
    reg [WEIGHT_WIDTH-1:0]     spike_weight;
    reg                         spike_exc_inh;
    
    // Processing variables
    reg [NEURON_ID_WIDTH-1:0]  process_idx;
    reg [DATA_WIDTH-1:0]       current_potential;
    reg [REFRAC_WIDTH-1:0]     current_refrac;
    reg                         update_en;
    
    // Output spike queue
    reg [NEURON_ID_WIDTH-1:0]  spike_queue [0:15];  // 16-entry queue
    reg [3:0]                  queue_wr_ptr;
    reg [3:0]                  queue_rd_ptr;
    reg [4:0]                  queue_count;
    
    // Statistics
    reg [31:0] total_spikes;
    assign spike_count = total_spikes;
    
    // Control signals
    assign s_axis_spike_ready = (state == IDLE) && !spike_pending;
    assign array_busy = (state != IDLE);
    
    // State machine
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    always @(*) begin
        next_state = state;
        case (state)
            IDLE: begin
                if (spike_pending) begin
                    next_state = PROCESS;
                end else if (s_axis_spike_valid) begin
                    next_state = RECEIVE;
                end else if (queue_count > 0 && m_axis_spike_ready) begin
                    next_state = OUTPUT;
                end
            end
            
            RECEIVE: begin
                next_state = IDLE;
            end
            
            PROCESS: begin
                next_state = UPDATE;
            end
            
            UPDATE: begin
                next_state = IDLE;
            end
            
            OUTPUT: begin
                if (m_axis_spike_ready) begin
                    next_state = IDLE;
                end
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    // Input spike handling
    always @(posedge clk) begin
        if (!rst_n) begin
            spike_pending <= 1'b0;
            spike_dest <= 0;
            spike_weight <= 0;
            spike_exc_inh <= 1'b1;
        end else begin
            if (state == RECEIVE && s_axis_spike_valid) begin
                spike_pending <= 1'b1;
                spike_dest <= s_axis_spike_dest_id;
                spike_weight <= s_axis_spike_weight;
                spike_exc_inh <= s_axis_spike_exc_inh;
            end else if (state == UPDATE) begin
                spike_pending <= 1'b0;
            end
        end
    end
    
    // Neuron processing
    always @(posedge clk) begin
        if (!rst_n) begin
            update_en <= 1'b0;
            current_potential <= 0;
            current_refrac <= 0;
        end else begin
            update_en <= 1'b0;
            
            if (state == PROCESS && spike_pending) begin
                // Read neuron state
                current_potential <= membrane_potential[spike_dest];
                current_refrac <= refractory_counter[spike_dest];
                update_en <= 1'b1;
            end
        end
    end
    
    // Neuron state update
    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < NUM_NEURONS; i = i + 1) begin
                membrane_potential[i] <= 0;
                refractory_counter[i] <= 0;
            end
            spike_flags <= 0;
        end else if (enable) begin
            // Update specific neuron
            if (state == UPDATE && update_en) begin
                if (current_refrac > 0) begin
                    // In refractory period
                    refractory_counter[spike_dest] <= current_refrac - 1'b1;
                end else begin
                    // Update membrane potential
                    if (spike_exc_inh) begin
                        // Excitatory
                        if (current_potential + spike_weight < current_potential) begin
                            // Overflow - saturate
                            membrane_potential[spike_dest] <= {DATA_WIDTH{1'b1}};
                        end else begin
                            membrane_potential[spike_dest] <= current_potential + spike_weight;
                        end
                    end else begin
                        // Inhibitory
                        if (current_potential < spike_weight) begin
                            membrane_potential[spike_dest] <= 0;
                        end else begin
                            membrane_potential[spike_dest] <= current_potential - spike_weight;
                        end
                    end
                    
                    // Check threshold
                    if (membrane_potential[spike_dest] >= global_threshold) begin
                        spike_flags[spike_dest] <= 1'b1;
                        membrane_potential[spike_dest] <= 0;
                        refractory_counter[spike_dest] <= global_refrac_period;
                    end
                end
            end
            
            // Global leak update (time-multiplexed)
            for (i = 0; i < 4; i = i + 1) begin  // Process 4 neurons per cycle
                if (refractory_counter[process_idx + i] == 0) begin
                    if (membrane_potential[process_idx + i] > global_leak_rate) begin
                        membrane_potential[process_idx + i] <= 
                            membrane_potential[process_idx + i] - global_leak_rate;
                    end else begin
                        membrane_potential[process_idx + i] <= 0;
                    end
                end
            end
            
            // Clear spike flags that have been queued
            if (spike_flags[queue_wr_ptr[3:0]] && queue_count < 16) begin
                spike_flags[queue_wr_ptr[3:0]] <= 1'b0;
            end
        end
    end
    
    // Process index counter for leak updates
    always @(posedge clk) begin
        if (!rst_n) begin
            process_idx <= 0;
        end else begin
            process_idx <= process_idx + 4;
            if (process_idx >= NUM_NEURONS - 4) begin
                process_idx <= 0;
            end
        end
    end
    
    // Output spike queue management
    always @(posedge clk) begin
        if (!rst_n) begin
            queue_wr_ptr <= 0;
            queue_rd_ptr <= 0;
            queue_count <= 0;
            m_axis_spike_valid <= 1'b0;
            m_axis_spike_neuron_id <= 0;
            total_spikes <= 0;
        end else begin
            // Add spikes to queue
            if (|spike_flags && queue_count < 16) begin
                for (i = 0; i < NUM_NEURONS; i = i + 1) begin
                    if (spike_flags[i] && queue_count < 16) begin
                        spike_queue[queue_wr_ptr] <= i;
                        queue_wr_ptr <= queue_wr_ptr + 1'b1;
                        queue_count <= queue_count + 1'b1;
                        total_spikes <= total_spikes + 1'b1;
                        spike_flags[i] <= 1'b0;
                    end
                end
            end
            
            // Output spikes from queue
            if (state == OUTPUT && m_axis_spike_ready && queue_count > 0) begin
                m_axis_spike_valid <= 1'b1;
                m_axis_spike_neuron_id <= spike_queue[queue_rd_ptr];
                queue_rd_ptr <= queue_rd_ptr + 1'b1;
                queue_count <= queue_count - 1'b1;
            end else if (m_axis_spike_ready) begin
                m_axis_spike_valid <= 1'b0;
            end
        end
    end
    
    // Configuration write
    always @(posedge clk) begin
        if (!rst_n) begin
            // Reset handled above
        end else if (config_we && config_addr < NUM_NEURONS) begin
            case (config_data[31:30])
                2'b00: membrane_potential[config_addr] <= config_data[DATA_WIDTH-1:0];
                2'b01: refractory_counter[config_addr] <= config_data[REFRAC_WIDTH-1:0];
                default: ; // Reserved
            endcase
        end
    end

endmodule
