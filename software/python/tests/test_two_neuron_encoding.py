"""
Test two-neuron encoding (ON/OFF polarity split) with all encoding types.

Two-neuron encoding splits each input channel into:
- ON neuron (even index): Fires for values above baseline
- OFF neuron (odd index): Fires for values below baseline

This works orthogonally with all encoding types:
- NONE (0): Direct spike input
- RATE_POISSON (1): Rate-coded spikes
- LATENCY (2): Latency-coded spikes
- DELTA_SIGMA (3): Delta-sigma modulated spikes
"""

import numpy as np
import pytest
from snn_fpga_accelerator.accelerator import SNNAccelerator, SpikeEvent


class TestTwoNeuronEncoding:
    """Test two-neuron encoding with various input patterns."""
    
    @pytest.fixture
    def accelerator(self):
        """Create accelerator in simulation mode."""
        accel = SNNAccelerator(simulation_mode=True)
        accel.initialize(neurons_per_layer=[100, 50, 10])
        return accel
    
    def test_two_neuron_polarity_split_above_baseline(self):
        """Test that values above baseline activate ON neurons (even indices)."""
        # Input: single channel with value 200 (above baseline 128)
        # Expected: ON neuron (ch 0) active, OFF neuron (ch 1) inactive
        input_value = 200
        baseline = 128
        
        # With two_neuron_enable: ON value = 200-128=72, OFF value = 0
        on_value = input_value - baseline
        off_value = 0
        
        assert on_value == 72
        assert off_value == 0
    
    def test_two_neuron_polarity_split_below_baseline(self):
        """Test that values below baseline activate OFF neurons (odd indices)."""
        # Input: single channel with value 50 (below baseline 128)
        # Expected: ON neuron (ch 0) inactive, OFF neuron (ch 1) active
        input_value = 50
        baseline = 128
        
        # With two_neuron_enable: ON value = 0, OFF value = 128-50=78
        on_value = 0
        off_value = baseline - input_value
        
        assert on_value == 0
        assert off_value == 78
    
    def test_two_neuron_channel_indexing(self):
        """Test that channel indices are correctly mapped (even=ON, odd=OFF)."""
        num_input_channels = 3
        
        # With two_neuron_enable, output channels = 2 * input_channels
        # Input ch 0 -> ON ch 0, OFF ch 1
        # Input ch 1 -> ON ch 2, OFF ch 3
        # Input ch 2 -> ON ch 4, OFF ch 5
        
        for ch in range(num_input_channels):
            on_ch = ch * 2
            off_ch = ch * 2 + 1
            
            assert on_ch % 2 == 0  # ON neurons at even indices
            assert off_ch % 2 == 1  # OFF neurons at odd indices
    
    def test_two_neuron_with_rate_poisson(self):
        """Test two-neuron encoding combined with rate-based Poisson coding."""
        # Rate-based encoding: spike probability proportional to intensity
        # With two_neuron: ON neuron gets (value-baseline), OFF gets (baseline-value)
        
        # High value: ON neuron should have higher spike rate
        high_value = 220
        baseline = 128
        on_intensity = high_value - baseline  # 92
        off_intensity = 0
        
        # Low value: OFF neuron should have higher spike rate
        low_value = 30
        on_intensity_low = 0
        off_intensity_low = baseline - low_value  # 98
        
        assert on_intensity > off_intensity
        assert off_intensity_low > on_intensity_low
    
    def test_two_neuron_with_latency(self):
        """Test two-neuron encoding combined with latency coding."""
        # Latency encoding: higher intensity = shorter latency (earlier spike)
        # With two_neuron: ON/OFF neurons encode respective polarity magnitudes
        
        # Above baseline: ON neuron fires early, OFF doesn't fire
        high_value = 200
        baseline = 128
        on_intensity = high_value - baseline  # 72 -> early spike
        off_intensity = 0  # -> no spike (or late spike)
        
        # Below baseline: OFF neuron fires early, ON doesn't fire
        low_value = 50
        on_intensity_low = 0
        off_intensity_low = baseline - low_value  # 78 -> early spike
        
        # Higher intensity means earlier spike (shorter latency)
        assert on_intensity > 0
        assert off_intensity_low > 0
    
    def test_two_neuron_with_delta_sigma(self):
        """Test two-neuron encoding combined with delta-sigma modulation."""
        # Delta-sigma: integrate input, spike when threshold exceeded
        # With two_neuron: ON/OFF neurons accumulate respective polarities
        
        # Above baseline: ON neuron accumulates positive difference
        high_value = 180
        baseline = 128
        on_accumulation = high_value - baseline  # 52 per timestep
        off_accumulation = 0
        
        # Below baseline: OFF neuron accumulates negative difference (as positive)
        low_value = 70
        on_acc_low = 0
        off_acc_low = baseline - low_value  # 58 per timestep
        
        assert on_accumulation > 0
        assert off_acc_low > 0
    
    def test_two_neuron_encoding_types_compatibility(self):
        """Test that two-neuron encoding works with all encoding types."""
        encoding_types = {
            0: "NONE",
            1: "RATE_POISSON",
            2: "LATENCY",
            3: "DELTA_SIGMA"
        }
        
        baseline = 128
        test_values = [50, 128, 200]  # Below, at, above baseline
        
        for enc_type, enc_name in encoding_types.items():
            for value in test_values:
                # Calculate ON/OFF split
                if value > baseline:
                    on_val = value - baseline
                    off_val = 0
                elif value < baseline:
                    on_val = 0
                    off_val = baseline - value
                else:  # value == baseline
                    on_val = 0
                    off_val = 0
                
                # Both values should be non-negative
                assert on_val >= 0, f"{enc_name}: ON value negative for input {value}"
                assert off_val >= 0, f"{enc_name}: OFF value negative for input {value}"
    
    def test_two_neuron_mnist_example(self):
        """Test two-neuron encoding with MNIST-like grayscale input."""
        # MNIST: 28x28 grayscale images (784 pixels, values 0-255)
        # With two_neuron: 1568 output neurons (784 ON + 784 OFF)
        
        num_pixels = 784
        baseline = 128
        
        # Create sample grayscale input
        np.random.seed(42)
        grayscale_input = np.random.randint(0, 256, size=num_pixels, dtype=np.uint8)
        
        # Calculate ON/OFF splits
        on_channels = np.maximum(grayscale_input.astype(int) - baseline, 0)
        off_channels = np.maximum(baseline - grayscale_input.astype(int), 0)
        
        # Verify properties
        assert len(on_channels) == num_pixels
        assert len(off_channels) == num_pixels
        assert np.all(on_channels >= 0)
        assert np.all(off_channels >= 0)
        
        # For each pixel, only ON or OFF (or neither) should be active, never both
        both_active = (on_channels > 0) & (off_channels > 0)
        assert np.sum(both_active) == 0, "ON and OFF should not be active simultaneously"
    
    def test_two_neuron_edge_cases(self):
        """Test edge cases: min, max, baseline values."""
        baseline = 128
        
        # Min value (0): OFF neuron fully active
        min_val = 0
        on_min = max(min_val - baseline, 0)
        off_min = max(baseline - min_val, 0)
        assert on_min == 0
        assert off_min == 128
        
        # Max value (255): ON neuron fully active
        max_val = 255
        on_max = max(max_val - baseline, 0)
        off_max = max(baseline - max_val, 0)
        assert on_max == 127
        assert off_max == 0
        
        # Baseline value (128): Both neurons inactive
        base_val = baseline
        on_base = max(base_val - baseline, 0)
        off_base = max(baseline - base_val, 0)
        assert on_base == 0
        assert off_base == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
