//-----------------------------------------------------------------------------
// Title         : Spike Decoder
// Project       : PYNQ-Z2 SNN Accelerator
// File          : spike_decoder.cpp
// Author        : Jiwoon Lee (@metr0jw)
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Email         : jwlee@linux.com
// Description   : Decodes spike trains to output values
//-----------------------------------------------------------------------------

#include "spike_decoder.h"

void spike_decoder(
    // Control
    bool enable,
    decoder_config_t config,
    
    // Input spike stream
    hls::stream<spike_event_t> &spikes_in,
    
    // Output data stream
    hls::stream<output_data_t> &data_out,
    
    // Status
    ap_uint<32> &status
) {
    #pragma HLS INTERFACE s_axilite port=enable
    #pragma HLS INTERFACE s_axilite port=config
    #pragma HLS INTERFACE s_axilite port=status
    #pragma HLS INTERFACE axis port=spikes_in
    #pragma HLS INTERFACE axis port=data_out
    #pragma HLS INTERFACE s_axilite port=return
    
    static ap_uint<16> spike_counts[MAX_OUTPUT_NEURONS];
    static ap_fixed<16,8> spike_rates[MAX_OUTPUT_NEURONS];
    static ap_uint<32> window_counter = 0;
    
    #pragma HLS ARRAY_PARTITION variable=spike_counts cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=spike_rates cyclic factor=8
    
    if (!enable) {
        status = 0x80000000; // Disabled
        return;
    }
    
    // Process incoming spikes
    if (!spikes_in.empty()) {
        spike_event_t spike = spikes_in.read();
        
        if (spike.neuron_id < MAX_OUTPUT_NEURONS) {
            spike_counts[spike.neuron_id]++;
            
            // Update spike rate using exponential moving average
            ap_fixed<16,8> alpha = config.rate_alpha;
            spike_rates[spike.neuron_id] = 
                alpha * spike_counts[spike.neuron_id] + 
                (1.0 - alpha) * spike_rates[spike.neuron_id];
        }
    }
    
    // Check if decoding window has elapsed
    window_counter++;
    if (window_counter >= config.window_size) {
        window_counter = 0;
        
        output_data_t output;
        
        switch (config.decoding_type) {
            case SPIKE_COUNT:
                decode_spike_count(spike_counts, config, output);
                break;
                
            case SPIKE_RATE:
                decode_spike_rate(spike_rates, config, output);
                break;
                
            case FIRST_SPIKE:
                decode_first_spike(spike_counts, config, output);
                break;
                
            default:
                break;
        }
        
        data_out.write(output);
        
        // Reset counters for next window
        RESET_LOOP: for (int i = 0; i < MAX_OUTPUT_NEURONS; i++) {
            #pragma HLS UNROLL factor=8
            spike_counts[i] = 0;
        }
    }
    
    status = window_counter;
}

// Decode based on spike count
void decode_spike_count(
    ap_uint<16> counts[MAX_OUTPUT_NEURONS],
    decoder_config_t config,
    output_data_t &output
) {
    #pragma HLS INLINE
    
    // Find neuron with maximum spike count
    ap_uint<16> max_count = 0;
    ap_uint<8> max_idx = 0;
    
    MAX_LOOP: for (int i = 0; i < config.num_outputs; i++) {
        #pragma HLS PIPELINE II=1
        if (counts[i] > max_count) {
            max_count = counts[i];
            max_idx = i;
        }
    }
    
    // Set output
    output.class_id = max_idx;
    output.confidence = (max_count * 255) / config.window_size;
    
    // Copy all counts
    COUNT_COPY: for (int i = 0; i < MAX_OUTPUT_NEURONS; i++) {
        #pragma HLS UNROLL factor=8
        output.values[i] = counts[i];
    }
}

// Decode based on spike rate
void decode_spike_rate(
    ap_fixed<16,8> rates[MAX_OUTPUT_NEURONS],
    decoder_config_t config,
    output_data_t &output
) {
    #pragma HLS INLINE
    
    // Normalize rates and find maximum
    ap_fixed<16,8> max_rate = 0;
    ap_uint<8> max_idx = 0;
    ap_fixed<16,8> sum_rates = 0;
    
    // Calculate sum for normalization
    SUM_LOOP: for (int i = 0; i < config.num_outputs; i++) {
        #pragma HLS PIPELINE II=1
        sum_rates += rates[i];
    }
    
    // Find maximum and normalize
    NORM_LOOP: for (int i = 0; i < config.num_outputs; i++) {
        #pragma HLS PIPELINE II=1
        ap_fixed<16,8> norm_rate = (sum_rates > 0) ? rates[i] / sum_rates : 0;
        
        if (norm_rate > max_rate) {
            max_rate = norm_rate;
            max_idx = i;
        }
        
        output.values[i] = norm_rate * 255;
    }
    
    output.class_id = max_idx;
    output.confidence = max_rate * 255;
}

// Decode based on first spike timing
void decode_first_spike(
    ap_uint<16> counts[MAX_OUTPUT_NEURONS],
    decoder_config_t config,
    output_data_t &output
) {
    #pragma HLS INLINE
    
    // Implementation for first-spike decoding
    // (simplified version using counts as proxy)
    decode_spike_count(counts, config, output);
}
