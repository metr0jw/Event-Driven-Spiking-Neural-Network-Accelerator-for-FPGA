//-----------------------------------------------------------------------------
// Title         : PC Communication Interface
// Project       : PYNQ-Z2 SNN Accelerator
// File          : pc_interface.cpp
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Email         : jwlee@linux.com
// Description   : HLS implementation of PC communication interface
//-----------------------------------------------------------------------------

#include "pc_interface.h"
#include <hls_stream.h>

// Main PC interface function
void pc_interface(
    // Control interface
    bool enable,
    bool reset,
    ap_uint<32> control_reg,
    ap_uint<32> &status_reg,
    
    // Spike input from PC
    hls::stream<spike_packet_t> &pc_spike_in,
    bool &pc_spike_in_ready,
    
    // Spike output to PC
    hls::stream<spike_packet_t> &pc_spike_out,
    bool pc_spike_out_ready,
    
    // SNN core interface
    hls::stream<spike_event_t> &snn_spike_in,
    hls::stream<spike_event_t> &snn_spike_out,
    
    // Weight update interface
    hls::stream<weight_update_t> &weight_updates,
    
    // Configuration interface
    ap_uint<32> config_addr,
    ap_uint<32> config_data,
    bool config_write,
    ap_uint<32> &config_read_data,
    
    // Performance counters
    ap_uint<32> &input_spike_count,
    ap_uint<32> &output_spike_count,
    ap_uint<32> &cycle_count
) {
    
    #pragma HLS INTERFACE s_axilite port=enable
    #pragma HLS INTERFACE s_axilite port=reset
    #pragma HLS INTERFACE s_axilite port=control_reg
    #pragma HLS INTERFACE s_axilite port=status_reg
    #pragma HLS INTERFACE s_axilite port=config_addr
    #pragma HLS INTERFACE s_axilite port=config_data
    #pragma HLS INTERFACE s_axilite port=config_write
    #pragma HLS INTERFACE s_axilite port=config_read_data
    #pragma HLS INTERFACE s_axilite port=input_spike_count
    #pragma HLS INTERFACE s_axilite port=output_spike_count
    #pragma HLS INTERFACE s_axilite port=cycle_count
    #pragma HLS INTERFACE s_axilite port=return
    
    #pragma HLS INTERFACE axis port=pc_spike_in
    #pragma HLS INTERFACE axis port=pc_spike_out
    #pragma HLS INTERFACE axis port=snn_spike_in
    #pragma HLS INTERFACE axis port=snn_spike_out
    #pragma HLS INTERFACE axis port=weight_updates
    
    // Static variables for state retention
    static ap_uint<32> local_input_count = 0;
    static ap_uint<32> local_output_count = 0;
    static ap_uint<32> local_cycle_count = 0;
    static ap_uint<32> error_count = 0;
    
    static spike_buffer_t input_buffer[SPIKE_BUFFER_SIZE];
    static spike_buffer_t output_buffer[SPIKE_BUFFER_SIZE];
    static ap_uint<8> input_buffer_head = 0;
    static ap_uint<8> input_buffer_tail = 0;
    static ap_uint<8> output_buffer_head = 0;
    static ap_uint<8> output_buffer_tail = 0;
    
    #pragma HLS ARRAY_PARTITION variable=input_buffer cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=output_buffer cyclic factor=4
    
    // Reset logic
    if (reset) {
        local_input_count = 0;
        local_output_count = 0;
        local_cycle_count = 0;
        error_count = 0;
        input_buffer_head = 0;
        input_buffer_tail = 0;
        output_buffer_head = 0;
        output_buffer_tail = 0;
        
        // Clear buffers
        RESET_LOOP: for (int i = 0; i < SPIKE_BUFFER_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            input_buffer[i].valid = false;
            output_buffer[i].valid = false;
        }
        
        status_reg = 0;
        input_spike_count = 0;
        output_spike_count = 0;
        cycle_count = 0;
        pc_spike_in_ready = true;
        return;
    }
    
    if (!enable) {
        status_reg = STATUS_DISABLED;
        return;
    }
    
    // Increment cycle counter
    local_cycle_count++;
    
    // Process input spikes from PC
    if (!pc_spike_in.empty()) {
        spike_packet_t pc_packet = pc_spike_in.read();
        
        // Validate packet
        if (validate_spike_packet(pc_packet)) {
            // Convert PC packet to SNN spike event
            spike_event_t snn_spike;
            snn_spike.neuron_id = pc_packet.neuron_id;
            snn_spike.timestamp = pc_packet.timestamp;
            snn_spike.weight = pc_packet.weight;
            
            // Buffer the spike if SNN core is not ready
            if (snn_spike_in.full()) {
                // Add to input buffer
                if (!is_buffer_full(input_buffer_head, input_buffer_tail)) {
                    input_buffer[input_buffer_tail].spike = snn_spike;
                    input_buffer[input_buffer_tail].valid = true;
                    input_buffer_tail = (input_buffer_tail + 1) % SPIKE_BUFFER_SIZE;
                } else {
                    error_count++;
                }
            } else {
                // Send directly to SNN core
                snn_spike_in.write(snn_spike);
            }
            
            local_input_count++;
        } else {
            error_count++;
        }
    }
    
    // Drain input buffer to SNN core
    if (!snn_spike_in.full() && !is_buffer_empty(input_buffer_head, input_buffer_tail)) {
        if (input_buffer[input_buffer_head].valid) {
            snn_spike_in.write(input_buffer[input_buffer_head].spike);
            input_buffer[input_buffer_head].valid = false;
            input_buffer_head = (input_buffer_head + 1) % SPIKE_BUFFER_SIZE;
        }
    }
    
    // Process output spikes from SNN core
    if (!snn_spike_out.empty()) {
        spike_event_t snn_spike = snn_spike_out.read();
        
        // Convert SNN spike to PC packet
        spike_packet_t pc_packet;
        pc_packet.neuron_id = snn_spike.neuron_id;
        pc_packet.timestamp = snn_spike.timestamp;
        pc_packet.weight = snn_spike.weight;
        pc_packet.packet_type = PACKET_TYPE_SPIKE;
        pc_packet.sequence_num = local_output_count;
        
        // Buffer the spike if PC interface is not ready
        if (pc_spike_out.full() || !pc_spike_out_ready) {
            if (!is_buffer_full(output_buffer_head, output_buffer_tail)) {
                output_buffer[output_buffer_tail].packet = pc_packet;
                output_buffer[output_buffer_tail].valid = true;
                output_buffer_tail = (output_buffer_tail + 1) % SPIKE_BUFFER_SIZE;
            } else {
                error_count++;
            }
        } else {
            // Send directly to PC
            pc_spike_out.write(pc_packet);
        }
        
        local_output_count++;
    }
    
    // Drain output buffer to PC
    if (!pc_spike_out.full() && pc_spike_out_ready && 
        !is_buffer_empty(output_buffer_head, output_buffer_tail)) {
        if (output_buffer[output_buffer_head].valid) {
            pc_spike_out.write(output_buffer[output_buffer_head].packet);
            output_buffer[output_buffer_head].valid = false;
            output_buffer_head = (output_buffer_head + 1) % SPIKE_BUFFER_SIZE;
        }
    }
    
    // Process weight updates
    if (!weight_updates.empty() && pc_spike_out_ready) {
        weight_update_t weight_update = weight_updates.read();
        
        // Convert to PC packet format
        spike_packet_t update_packet;
        update_packet.packet_type = PACKET_TYPE_WEIGHT_UPDATE;
        update_packet.neuron_id = weight_update.pre_neuron_id;
        update_packet.timestamp = weight_update.post_neuron_id;
        update_packet.weight = weight_update.new_weight;
        update_packet.sequence_num = local_output_count;
        
        if (!pc_spike_out.full()) {
            pc_spike_out.write(update_packet);
        }
    }
    
    // Handle configuration reads/writes
    if (config_write) {
        handle_config_write(config_addr, config_data);
    } else {
        config_read_data = handle_config_read(config_addr);
    }
    
    // Update status register
    ap_uint<32> status = 0;
    status |= (enable ? STATUS_ENABLED : 0);
    status |= (error_count > 0 ? STATUS_ERROR : 0);
    status |= (is_buffer_full(input_buffer_head, input_buffer_tail) ? STATUS_INPUT_BUFFER_FULL : 0);
    status |= (is_buffer_full(output_buffer_head, output_buffer_tail) ? STATUS_OUTPUT_BUFFER_FULL : 0);
    status |= ((error_count & 0xFF) << 8);  // Error count in upper bits
    
    status_reg = status;
    
    // Update performance counters
    input_spike_count = local_input_count;
    output_spike_count = local_output_count;
    cycle_count = local_cycle_count;
    
    // Set PC input ready status
    pc_spike_in_ready = !is_buffer_full(input_buffer_head, input_buffer_tail) && 
                       !snn_spike_in.full();
}

// Validate incoming spike packet
bool validate_spike_packet(const spike_packet_t &packet) {
    #pragma HLS INLINE
    
    // Check packet type
    if (packet.packet_type != PACKET_TYPE_SPIKE && 
        packet.packet_type != PACKET_TYPE_CONFIG) {
        return false;
    }
    
    // Check neuron ID range
    if (packet.neuron_id >= MAX_NEURONS) {
        return false;
    }
    
    // Check weight range
    if (packet.weight < MIN_WEIGHT || packet.weight > MAX_WEIGHT) {
        return false;
    }
    
    // Check timestamp validity (basic sanity check)
    if (packet.timestamp > MAX_TIMESTAMP) {
        return false;
    }
    
    return true;
}

// Check if buffer is full
bool is_buffer_full(ap_uint<8> head, ap_uint<8> tail) {
    #pragma HLS INLINE
    return ((tail + 1) % SPIKE_BUFFER_SIZE) == head;
}

// Check if buffer is empty
bool is_buffer_empty(ap_uint<8> head, ap_uint<8> tail) {
    #pragma HLS INLINE
    return head == tail;
}

// Handle configuration write
void handle_config_write(ap_uint<32> addr, ap_uint<32> data) {
    #pragma HLS INLINE
    
    static ap_uint<32> config_memory[CONFIG_MEMORY_SIZE];
    #pragma HLS ARRAY_PARTITION variable=config_memory cyclic factor=8
    
    if (addr < CONFIG_MEMORY_SIZE) {
        config_memory[addr] = data;
        
        // Handle special configuration addresses
        switch (addr) {
            case CONFIG_ADDR_NEURON_THRESHOLD:
                // Update neuron threshold
                break;
            case CONFIG_ADDR_LEARNING_RATE:
                // Update learning rate
                break;
            case CONFIG_ADDR_STDP_WINDOW:
                // Update STDP time window
                break;
            default:
                // Regular configuration register
                break;
        }
    }
}

// Handle configuration read
ap_uint<32> handle_config_read(ap_uint<32> addr) {
    #pragma HLS INLINE
    
    static ap_uint<32> config_memory[CONFIG_MEMORY_SIZE];
    
    if (addr < CONFIG_MEMORY_SIZE) {
        return config_memory[addr];
    } else {
        // Return status information for special addresses
        switch (addr) {
            case CONFIG_ADDR_VERSION:
                return INTERFACE_VERSION;
            case CONFIG_ADDR_FEATURES:
                return INTERFACE_FEATURES;
            case CONFIG_ADDR_MAX_NEURONS:
                return MAX_NEURONS;
            default:
                return 0;
        }
    }
}

// Ethernet packet processing (for future use)
void process_ethernet_packet(
    hls::stream<ethernet_packet_t> &eth_in,
    hls::stream<ethernet_packet_t> &eth_out,
    hls::stream<spike_packet_t> &spike_out
) {
    #pragma HLS INTERFACE axis port=eth_in
    #pragma HLS INTERFACE axis port=eth_out
    #pragma HLS INTERFACE axis port=spike_out
    
    if (!eth_in.empty()) {
        ethernet_packet_t eth_packet = eth_in.read();
        
        // Parse Ethernet header
        if (eth_packet.eth_type == ETH_TYPE_SNN) {
            // Extract spike data from payload
            spike_packet_t spike_packet;
            
            // Simple parsing (would be more complex in practice)
            spike_packet.neuron_id = eth_packet.payload.range(31, 24);
            spike_packet.timestamp = eth_packet.payload.range(23, 8);
            spike_packet.weight = eth_packet.payload.range(7, 0);
            spike_packet.packet_type = PACKET_TYPE_SPIKE;
            
            if (validate_spike_packet(spike_packet)) {
                spike_out.write(spike_packet);
                
                // Send acknowledgment
                ethernet_packet_t ack_packet;
                ack_packet.eth_type = ETH_TYPE_SNN_ACK;
                ack_packet.payload = spike_packet.sequence_num;
                eth_out.write(ack_packet);
            }
        }
    }
}

// Real-time spike stream processor
void spike_stream_processor(
    hls::stream<spike_event_t> &input_stream,
    hls::stream<spike_event_t> &output_stream,
    ap_uint<32> processing_mode,
    ap_uint<32> &processed_count
) {
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE s_axilite port=processing_mode
    #pragma HLS INTERFACE s_axilite port=processed_count
    #pragma HLS INTERFACE s_axilite port=return
    
    static ap_uint<32> local_count = 0;
    
    if (!input_stream.empty()) {
        spike_event_t spike = input_stream.read();
        
        // Process spike based on mode
        switch (processing_mode) {
            case PROCESS_MODE_PASSTHROUGH:
                // Direct passthrough
                output_stream.write(spike);
                break;
                
            case PROCESS_MODE_FILTER:
                // Filter spikes based on criteria
                if (spike.weight > FILTER_THRESHOLD) {
                    output_stream.write(spike);
                }
                break;
                
            case PROCESS_MODE_TRANSFORM:
                // Transform spike properties
                spike.weight = spike.weight * TRANSFORM_SCALE;
                output_stream.write(spike);
                break;
                
            default:
                // Unknown mode, passthrough
                output_stream.write(spike);
                break;
        }
        
        local_count++;
    }
    
    processed_count = local_count;
}
