//-----------------------------------------------------------------------------
// Title         : Spike Encoder
// Project       : PYNQ-Z2 SNN Accelerator
// File          : spike_encoder.cpp
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Description   : Converts input data to spike trains
//-----------------------------------------------------------------------------

#include "spike_encoder.h"

void spike_encoder(
    // Control
    bool enable,
    encoder_config_t config,
    
    // Input data stream
    hls::stream<input_data_t> &data_in,
    
    // Output spike stream
    hls::stream<spike_event_t> &spikes_out,
    
    // Status
    ap_uint<32> &spike_count
) {
    #pragma HLS INTERFACE s_axilite port=enable
    #pragma HLS INTERFACE s_axilite port=config
    #pragma HLS INTERFACE s_axilite port=spike_count
    #pragma HLS INTERFACE axis port=data_in
    #pragma HLS INTERFACE axis port=spikes_out
    #pragma HLS INTERFACE s_axilite port=return
    
    static ap_uint<32> time_counter = 0;
    static ap_uint<32> total_spikes = 0;
    static ap_uint<16> phase_accumulator[MAX_INPUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=phase_accumulator cyclic factor=16
    
    if (!enable) {
        spike_count = total_spikes;
        return;
    }
    
    // Increment time
    time_counter++;
    
    // Process input data
    if (!data_in.empty()) {
        input_data_t data = data_in.read();
        
        // Encode each channel
        ENCODE_LOOP: for (int ch = 0; ch < MAX_INPUT_CHANNELS; ch++) {
            #pragma HLS UNROLL factor=8
            
            if (ch < config.num_channels) {
                pixel_t pixel_value = data.pixels[ch];
                
                switch (config.encoding_type) {
                    case RATE_CODING:
                        encode_rate(ch, pixel_value, time_counter, config, spikes_out, total_spikes);
                        break;
                        
                    case TEMPORAL_CODING:
                        encode_temporal(ch, pixel_value, time_counter, config, spikes_out, total_spikes);
                        break;
                        
                    case PHASE_CODING:
                        encode_phase(ch, pixel_value, time_counter, config, 
                                   phase_accumulator[ch], spikes_out, total_spikes);
                        break;
                        
                    default:
                        break;
                }
            }
        }
    }
    
    spike_count = total_spikes;
}

// Rate coding: spike probability proportional to input value
void encode_rate(
    int channel,
    pixel_t value,
    ap_uint<32> time,
    encoder_config_t config,
    hls::stream<spike_event_t> &spikes_out,
    ap_uint<32> &spike_counter
) {
    #pragma HLS INLINE
    
    // Calculate spike probability
    ap_uint<16> threshold = ((255 - value) * config.rate_scale) >> 8;
    
    // Generate random number for stochastic spiking
    ap_uint<16> random = lfsr_random();
    
    if (random < threshold) {
        spike_event_t spike;
        spike.neuron_id = channel;
        spike.timestamp = time;
        spike.weight = config.default_weight;
        
        spikes_out.write(spike);
        spike_counter++;
    }
}

// Temporal coding: time to first spike encodes value
void encode_temporal(
    int channel,
    pixel_t value,
    ap_uint<32> time,
    encoder_config_t config,
    hls::stream<spike_event_t> &spikes_out,
    ap_uint<32> &spike_counter
) {
    #pragma HLS INLINE
    
    static bool fired[MAX_INPUT_CHANNELS] = {false};
    static ap_uint<32> start_time[MAX_INPUT_CHANNELS] = {0};
    
    // Reset at window boundaries
    if (time % config.time_window == 0) {
        fired[channel] = false;
        start_time[channel] = time;
    }
    
    if (!fired[channel]) {
        // Calculate spike time based on value (higher value = earlier spike)
        ap_uint<32> spike_delay = ((255 - value) * config.time_window) >> 8;
        
        if (time >= start_time[channel] + spike_delay) {
            spike_event_t spike;
            spike.neuron_id = channel;
            spike.timestamp = time;
            spike.weight = config.default_weight;
            
            spikes_out.write(spike);
            spike_counter++;
            fired[channel] = true;
        }
    }
}

// Phase coding: spike phase encodes value
void encode_phase(
    int channel,
    pixel_t value,
    ap_uint<32> time,
    encoder_config_t config,
    ap_uint<16> &phase_acc,
    hls::stream<spike_event_t> &spikes_out,
    ap_uint<32> &spike_counter
) {
    #pragma HLS INLINE
    
    // Update phase accumulator
    ap_uint<16> phase_increment = (value * config.phase_scale) >> 4;
    phase_acc += phase_increment;
    
    // Check for phase wrap (spike generation)
    if (phase_acc >= config.phase_threshold) {
        spike_event_t spike;
        spike.neuron_id = channel;
        spike.timestamp = time;
        spike.weight = config.default_weight;
        
        spikes_out.write(spike);
        spike_counter++;
        
        phase_acc -= config.phase_threshold;
    }
}

// Linear feedback shift register for pseudo-random numbers
ap_uint<16> lfsr_random() {
    #pragma HLS INLINE
    
    static ap_uint<16> lfsr = 0xACE1;
    bool bit = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1;
    lfsr = (lfsr >> 1) | (bit << 15);
    
    return lfsr;
}
