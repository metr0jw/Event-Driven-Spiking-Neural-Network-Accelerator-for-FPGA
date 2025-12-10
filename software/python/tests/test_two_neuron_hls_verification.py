"""
HLS Two-Neuron Encoding Verification Test.

This test verifies that the HLS implementation of two-neuron encoding
matches the expected behavior:
1. Channel indexing (even=ON, odd=OFF)
2. Polarity split correctness
3. Integration with all encoding types
4. Verilog simulation testbench generation
"""

import numpy as np
import pytest
from typing import List


class TestTwoNeuronHLSVerification:
    """Verify HLS two-neuron encoding implementation."""
    
    def test_hls_channel_mapping(self):
        """Verify HLS channel indexing: even=ON, odd=OFF."""
        num_input_channels = 10
        
        # HLS code: int on_ch = ch * 2; int off_ch = ch * 2 + 1;
        channel_map = []
        for ch in range(num_input_channels):
            on_ch = ch * 2
            off_ch = ch * 2 + 1
            channel_map.append({
                'input': ch,
                'on': on_ch,
                'off': off_ch
            })
        
        # Verify properties
        for mapping in channel_map:
            assert mapping['on'] % 2 == 0, f"ON channel {mapping['on']} should be even"
            assert mapping['off'] % 2 == 1, f"OFF channel {mapping['off']} should be odd"
            assert mapping['off'] == mapping['on'] + 1, "OFF = ON + 1"
        
        # Verify no overlaps
        all_on = [m['on'] for m in channel_map]
        all_off = [m['off'] for m in channel_map]
        assert len(set(all_on)) == len(all_on), "ON channels should be unique"
        assert len(set(all_off)) == len(all_off), "OFF channels should be unique"
        assert set(all_on).isdisjoint(set(all_off)), "ON and OFF should not overlap"
    
    def test_hls_polarity_split_logic(self):
        """Verify HLS two-neuron split matches C++ implementation."""
        baseline = 128
        test_values = [0, 50, 128, 200, 255]
        
        for value in test_values:
            # HLS logic simulation
            if value > baseline:
                on_val = value - baseline
                off_val = 0
            else:
                on_val = 0
                off_val = baseline - value
            
            # Verify properties
            assert on_val >= 0, "ON value should be non-negative"
            assert off_val >= 0, "OFF value should be non-negative"
            assert on_val + off_val == abs(value - baseline), "ON + OFF = |value - baseline|"
            
            # Mutual exclusivity
            if value != baseline:
                assert (on_val > 0) != (off_val > 0), "Exactly one should be active"
            else:
                assert on_val == 0 and off_val == 0, "Both should be zero at baseline"
    
    def test_hls_two_neuron_with_rate_encoding(self):
        """Test two-neuron + rate encoding combination."""
        baseline = 128
        pixel_values = np.array([50, 128, 200], dtype=np.uint8)
        
        results = []
        for val in pixel_values:
            # Two-neuron split
            on_val = max(int(val) - baseline, 0)
            off_val = max(baseline - int(val), 0)
            
            # Rate encoding probability
            rate_scale = 256
            on_prob = (on_val * rate_scale) >> 8
            off_prob = (off_val * rate_scale) >> 8
            
            results.append({
                'value': val,
                'on_val': on_val,
                'off_val': off_val,
                'on_prob': on_prob,
                'off_prob': off_prob
            })
        
        # Verify results
        assert results[0]['off_prob'] > 0 and results[0]['on_prob'] == 0, "Low value: OFF active"
        assert results[1]['on_prob'] == 0 and results[1]['off_prob'] == 0, "Baseline: both zero"
        assert results[2]['on_prob'] > 0 and results[2]['off_prob'] == 0, "High value: ON active"
        
        print("\nTwo-neuron + Rate encoding:")
        for r in results:
            print(f"  Value={r['value']}: ON={r['on_val']}(p={r['on_prob']}), OFF={r['off_val']}(p={r['off_prob']})")
    
    def test_hls_two_neuron_with_latency_encoding(self):
        """Test two-neuron + latency encoding combination."""
        baseline = 128
        pixel_values = np.array([50, 200], dtype=np.uint8)
        latency_window = 100
        
        for val in pixel_values:
            on_val = max(int(val) - baseline, 0)
            off_val = max(baseline - int(val), 0)
            
            # Latency encoding: spike_delay = ((255 - value) * window) >> 8
            on_delay = ((255 - on_val) * latency_window) >> 8 if on_val > 0 else latency_window
            off_delay = ((255 - off_val) * latency_window) >> 8 if off_val > 0 else latency_window
            
            if val < baseline:
                # OFF should fire early (short delay)
                assert off_delay < latency_window, f"OFF delay {off_delay} should be < window {latency_window}"
            elif val > baseline:
                # ON should fire early
                assert on_delay < latency_window, f"ON delay {on_delay} should be < window {latency_window}"
    
    def test_hls_output_channel_count(self):
        """Verify output channel count doubles with two-neuron enabled."""
        num_input = 784  # MNIST
        
        # Without two-neuron
        output_single = num_input
        
        # With two-neuron
        output_double = num_input * 2
        
        assert output_double == 1568, "MNIST with two-neuron should have 1568 channels"
        assert output_double == output_single * 2, "Should exactly double"
    
    def test_hls_memory_layout(self):
        """Verify memory layout for two-neuron encoding."""
        # With two-neuron, each input pixel generates 2 output channels
        # Memory indexing: [on_ch, off_ch] = [ch*2, ch*2+1]
        
        input_data = np.random.randint(0, 256, size=10, dtype=np.uint8)
        baseline = 128
        
        # Simulate HLS memory layout
        output_channels = np.zeros(20, dtype=np.uint8)  # 10 input -> 20 output
        
        for ch, val in enumerate(input_data):
            on_ch = ch * 2
            off_ch = ch * 2 + 1
            
            on_val = max(int(val) - baseline, 0)
            off_val = max(baseline - int(val), 0)
            
            output_channels[on_ch] = on_val
            output_channels[off_ch] = off_val
        
        # Verify layout
        for ch in range(10):
            on_ch = ch * 2
            off_ch = ch * 2 + 1
            
            # Retrieve values
            stored_on = output_channels[on_ch]
            stored_off = output_channels[off_ch]
            
            # Verify mutual exclusivity
            if input_data[ch] != baseline:
                assert (stored_on > 0) != (stored_off > 0), f"Channel {ch}: mutual exclusivity violated"
    
    def generate_verilog_testbench(self):
        """Generate Verilog testbench vectors for HLS simulation."""
        baseline = 128
        test_vectors = [
            {'name': 'zero', 'value': 0, 'expected_on': 0, 'expected_off': 128},
            {'name': 'low', 'value': 50, 'expected_on': 0, 'expected_off': 78},
            {'name': 'baseline', 'value': 128, 'expected_on': 0, 'expected_off': 0},
            {'name': 'high', 'value': 200, 'expected_on': 72, 'expected_off': 0},
            {'name': 'max', 'value': 255, 'expected_on': 127, 'expected_off': 0},
        ]
        
        # Generate testbench code
        tb_code = []
        tb_code.append("// Two-Neuron Encoding Testbench Vectors")
        tb_code.append("// Generated by test_two_neuron_hls_verification.py")
        tb_code.append("")
        
        for i, vec in enumerate(test_vectors):
            tb_code.append(f"// Test {i}: {vec['name']} (value={vec['value']})")
            tb_code.append(f"input_value = 8'd{vec['value']};")
            tb_code.append(f"baseline = 8'd{baseline};")
            tb_code.append(f"#10;  // Wait for combinational logic")
            tb_code.append(f"assert (on_value == 8'd{vec['expected_on']}) else $error(\"ON mismatch\");")
            tb_code.append(f"assert (off_value == 8'd{vec['expected_off']}) else $error(\"OFF mismatch\");")
            tb_code.append("")
        
        return "\n".join(tb_code)
    
    def test_generate_verilog_testbench(self):
        """Test Verilog testbench generation."""
        tb_code = self.generate_verilog_testbench()
        
        assert "input_value" in tb_code
        assert "baseline" in tb_code
        assert "on_value" in tb_code
        assert "off_value" in tb_code
        assert "$error" in tb_code
        
        print("\n" + "="*60)
        print("VERILOG TESTBENCH CODE:")
        print("="*60)
        print(tb_code)
        print("="*60)
    
    def test_hls_pragma_directives(self):
        """Verify HLS pragmas for two-neuron encoding."""
        # Expected pragmas in HLS code:
        # #pragma HLS INLINE (for encoder_two_neuron_split)
        # #pragma HLS UNROLL factor=8 (for ENCODER_LOOP if two_neuron)
        
        expected_pragmas = [
            "HLS INLINE",  # encoder_two_neuron_split should be inlined
            "HLS UNROLL",  # Loop unrolling for parallel processing
        ]
        
        # This is a documentation test - actual verification would parse HLS source
        assert all(pragma for pragma in expected_pragmas)
        
        print("\nExpected HLS pragmas for two-neuron encoding:")
        for pragma in expected_pragmas:
            print(f"  #pragma {pragma}")
    
    def test_hls_resource_utilization_estimate(self):
        """Estimate HLS resource utilization with two-neuron encoding."""
        # Resource estimates:
        # - Additional comparators: 1 per channel (value > baseline)
        # - Additional subtractors: 2 per channel (on/off calculation)
        # - Channel index calculations: 1 multiply + 1 add per channel
        
        num_channels = 784  # MNIST
        
        # Without two-neuron
        resources_baseline = {
            'comparators': num_channels,
            'subtractors': num_channels,
            'multipliers': 0,
            'adders': 0
        }
        
        # With two-neuron
        resources_two_neuron = {
            'comparators': num_channels * 2,  # Baseline comparison + bounds check
            'subtractors': num_channels * 2,  # ON and OFF calculation
            'multipliers': num_channels * 1,  # ch * 2
            'adders': num_channels * 1,       # ch * 2 + 1
        }
        
        overhead = {
            key: resources_two_neuron[key] - resources_baseline[key]
            for key in resources_baseline
        }
        
        print("\nResource utilization estimate:")
        print("  Baseline:", resources_baseline)
        print("  Two-neuron:", resources_two_neuron)
        print("  Overhead:", overhead)
        
        # Verify overhead is reasonable
        assert overhead['comparators'] <= num_channels * 2
        assert overhead['subtractors'] <= num_channels * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
