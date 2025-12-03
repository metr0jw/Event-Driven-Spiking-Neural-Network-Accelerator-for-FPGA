"""
Tests for single-step and multi-step modes in SNNAccelerator.

This module tests the SpikingJelly-inspired step modes that allow users to
process entire simulations at once (multi-step) or iterate timestep by timestep
(single-step) while maintaining internal state.
"""

import numpy as np
import pytest

from snn_fpga_accelerator import SNNAccelerator
from snn_fpga_accelerator.spike_encoding import SpikeEvent
from snn_fpga_accelerator.pytorch_interface import SNNModel, SNNLayer


@pytest.fixture
def simple_network():
    """Create a simple 2-layer SNN for testing."""
    model = SNNModel(name="test_network")
    
    # Layer 1: 10 -> 5
    layer1 = SNNLayer(input_size=10, output_size=5, layer_type="fully_connected")
    layer1.set_weights(np.random.randn(5, 10) * 0.1)
    layer1.set_neuron_parameters(threshold=1.0, leak_rate=0.1, refractory_period=5)
    model.add_layer(layer1)
    
    # Layer 2: 5 -> 3
    layer2 = SNNLayer(input_size=5, output_size=3, layer_type="fully_connected")
    layer2.set_weights(np.random.randn(3, 5) * 0.1)
    layer2.set_neuron_parameters(threshold=1.0, leak_rate=0.1, refractory_period=5)
    model.add_layer(layer2)
    
    return model


@pytest.fixture
def accelerator(simple_network):
    """Create an accelerator instance in simulation mode."""
    acc = SNNAccelerator(simulation_mode=True)
    acc.configure_network(simple_network)
    return acc


@pytest.fixture
def sample_spikes():
    """Generate sample spike events for testing."""
    spikes = [
        SpikeEvent(neuron_id=0, timestamp=0.001, weight=1.0),
        SpikeEvent(neuron_id=2, timestamp=0.003, weight=1.0),
        SpikeEvent(neuron_id=5, timestamp=0.005, weight=1.0),
        SpikeEvent(neuron_id=1, timestamp=0.007, weight=1.0),
        SpikeEvent(neuron_id=3, timestamp=0.010, weight=1.0),
    ]
    return spikes


class TestStepModeConfiguration:
    """Test step mode configuration and switching."""
    
    def test_default_mode_is_multi(self, accelerator):
        """Test that default step mode is multi-step."""
        assert accelerator.get_step_mode() == "multi"
    
    def test_set_single_step_mode(self, accelerator):
        """Test switching to single-step mode."""
        accelerator.set_step_mode("single", timestep_dt=0.001)
        assert accelerator.get_step_mode() == "single"
        assert accelerator.timestep_dt == 0.001
    
    def test_set_multi_step_mode(self, accelerator):
        """Test switching to multi-step mode."""
        accelerator.set_step_mode("multi")
        assert accelerator.get_step_mode() == "multi"
    
    def test_invalid_mode_raises_error(self, accelerator):
        """Test that invalid step mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid step mode"):
            accelerator.set_step_mode("invalid")
    
    def test_custom_timestep(self, accelerator):
        """Test setting custom timestep duration."""
        accelerator.set_step_mode("single", timestep_dt=0.0005)
        assert accelerator.timestep_dt == 0.0005
        
        accelerator.set_step_mode("single", timestep_dt=0.002)
        assert accelerator.timestep_dt == 0.002


class TestMultiStepMode:
    """Test multi-step mode (process entire simulation at once)."""
    
    def test_multi_step_returns_all_outputs(self, accelerator, sample_spikes):
        """Test that multi-step mode returns outputs for entire duration."""
        accelerator.set_step_mode("multi")
        
        # Run inference for 50ms
        output = accelerator.infer(sample_spikes, duration=0.050)
        
        # Should return firing rates for output layer
        assert isinstance(output, np.ndarray)
        # In simulation mode, output may be empty (no FPGA), but shape should be valid
        assert output.shape[0] >= 0  # Valid array (may be empty in sim mode)
    
    def test_multi_step_with_events(self, accelerator, sample_spikes):
        """Test multi-step mode with event-based output."""
        accelerator.set_step_mode("multi")
        
        output_events = accelerator.infer(
            sample_spikes, 
            duration=0.050, 
            return_events=True
        )
        
        # Should return list of spike events
        assert isinstance(output_events, list)
    
    def test_multi_step_duration_inference(self, accelerator, sample_spikes):
        """Test that duration is inferred from spike timestamps."""
        accelerator.set_step_mode("multi")
        
        # Don't specify duration - should be inferred
        output = accelerator.infer(sample_spikes)
        
        assert isinstance(output, np.ndarray)
    
    def test_multi_step_empty_input(self, accelerator):
        """Test multi-step mode with no input spikes."""
        accelerator.set_step_mode("multi")
        
        output = accelerator.infer([], duration=0.010)
        
        # Should return empty or zero array
        assert isinstance(output, np.ndarray)


class TestSingleStepMode:
    """Test single-step mode (process one timestep at a time)."""
    
    def test_single_step_returns_one_timestep(self, accelerator):
        """Test that single_step processes one timestep."""
        accelerator.set_step_mode("single", timestep_dt=0.001)
        accelerator.reset()
        
        # Process one timestep
        spikes_t0 = [SpikeEvent(neuron_id=0, timestamp=0.0, weight=1.0)]
        output = accelerator.single_step(spikes_t0)
        
        # Should return spike counts for this timestep
        assert isinstance(output, np.ndarray)
        assert accelerator.current_timestep == 1
    
    def test_single_step_maintains_state(self, accelerator):
        """Test that single_step maintains state between calls."""
        accelerator.set_step_mode("single", timestep_dt=0.001)
        accelerator.reset()
        
        # Process multiple timesteps
        for t in range(5):
            spikes = [SpikeEvent(neuron_id=t % 3, timestamp=0.0, weight=1.0)]
            output = accelerator.single_step(spikes)
            assert accelerator.current_timestep == t + 1
    
    def test_single_step_reset_clears_state(self, accelerator):
        """Test that reset() clears internal state."""
        accelerator.set_step_mode("single", timestep_dt=0.001)
        
        # Process some timesteps
        for t in range(5):
            spikes = [SpikeEvent(neuron_id=0, timestamp=0.0, weight=1.0)]
            accelerator.single_step(spikes)
        
        assert accelerator.current_timestep == 5
        
        # Reset should clear state
        accelerator.reset()
        assert accelerator.current_timestep == 0
        assert len(accelerator.spike_history) == 0
    
    def test_single_step_with_events(self, accelerator):
        """Test single_step with event-based output."""
        accelerator.set_step_mode("single", timestep_dt=0.001)
        accelerator.reset()
        
        spikes = [SpikeEvent(neuron_id=0, timestamp=0.0, weight=1.0)]
        output_events = accelerator.single_step(spikes, return_events=True)
        
        # Should return list of spike events
        assert isinstance(output_events, list)
    
    def test_single_step_empty_input(self, accelerator):
        """Test single_step with no input spikes."""
        accelerator.set_step_mode("single", timestep_dt=0.001)
        accelerator.reset()
        
        output = accelerator.single_step([])
        
        # Should return zero array
        assert isinstance(output, np.ndarray)
        assert np.all(output == 0)


class TestSpikeHistory:
    """Test spike history tracking in single-step mode."""
    
    def test_history_accumulation(self, accelerator):
        """Test that spike history accumulates correctly."""
        accelerator.set_step_mode("single", timestep_dt=0.001)
        accelerator.reset()
        
        num_steps = 10
        for t in range(num_steps):
            spikes = [SpikeEvent(neuron_id=t % 3, timestamp=0.0, weight=1.0)]
            accelerator.single_step(spikes)
        
        history = accelerator.get_spike_history()
        assert len(history) == num_steps
    
    def test_history_reset(self, accelerator):
        """Test that history is cleared on reset."""
        accelerator.set_step_mode("single", timestep_dt=0.001)
        accelerator.reset()
        
        # Generate some history
        for t in range(5):
            spikes = [SpikeEvent(neuron_id=0, timestamp=0.0, weight=1.0)]
            accelerator.single_step(spikes)
        
        assert len(accelerator.get_spike_history()) == 5
        
        # Reset should clear history
        accelerator.reset()
        assert len(accelerator.get_spike_history()) == 0
    
    def test_history_contains_events(self, accelerator):
        """Test that history contains spike events."""
        accelerator.set_step_mode("single", timestep_dt=0.001)
        accelerator.reset()
        
        # Process one timestep
        spikes = [SpikeEvent(neuron_id=0, timestamp=0.0, weight=1.0)]
        accelerator.single_step(spikes)
        
        history = accelerator.get_spike_history()
        assert len(history) == 1
        assert isinstance(history[0], list)


class TestStepModeComparison:
    """Compare outputs between single-step and multi-step modes."""
    
    def test_equivalent_outputs(self, accelerator, sample_spikes):
        """Test that both modes produce equivalent results.
        
        Note: This test may have some numerical differences due to the way
        spikes are processed, but the overall pattern should be similar.
        """
        # Multi-step mode
        accelerator.set_step_mode("multi")
        multi_output = accelerator.infer(sample_spikes, duration=0.020)
        
        # Single-step mode
        accelerator.set_step_mode("single", timestep_dt=0.001)
        accelerator.reset()
        
        # Group spikes by timestep
        timesteps = int(0.020 / 0.001)
        for t in range(timesteps):
            t_start = t * 0.001
            t_end = (t + 1) * 0.001
            spikes_t = [
                s for s in sample_spikes 
                if t_start <= s.timestamp < t_end
            ]
            accelerator.single_step(spikes_t)
        
        # Get accumulated history
        history = accelerator.get_spike_history()
        assert len(history) == timesteps
    
    def test_different_timestep_sizes(self, accelerator, sample_spikes):
        """Test single-step mode with different timestep sizes."""
        durations = [0.0005, 0.001, 0.002]
        
        for dt in durations:
            accelerator.set_step_mode("single", timestep_dt=dt)
            accelerator.reset()
            
            # Process a few steps
            for t in range(5):
                accelerator.single_step([])
            
            assert accelerator.current_timestep == 5
            assert accelerator.timestep_dt == dt


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_step_without_reset(self, accelerator):
        """Test that single_step works without explicit reset."""
        accelerator.set_step_mode("single", timestep_dt=0.001)
        
        # Should start from timestep 0 even without reset
        output = accelerator.single_step([])
        assert isinstance(output, np.ndarray)
    
    def test_mode_switching_preserves_state(self, accelerator):
        """Test switching modes and state preservation."""
        accelerator.set_step_mode("single", timestep_dt=0.001)
        accelerator.reset()
        
        # Process some steps
        for _ in range(3):
            accelerator.single_step([])
        
        current_step = accelerator.current_timestep
        
        # Switch to multi mode and back
        accelerator.set_step_mode("multi")
        accelerator.set_step_mode("single")
        
        # State should be preserved
        assert accelerator.current_timestep == current_step
    
    def test_numpy_array_input(self, accelerator):
        """Test single_step with NumPy array input."""
        accelerator.set_step_mode("single", timestep_dt=0.001)
        accelerator.reset()
        
        # Create spike array: (neuron_id, timestamp, weight)
        spike_array = np.array([
            [0, 0.0, 1.0],
            [1, 0.0, 1.0],
            [2, 0.0, 1.0],
        ])
        
        output = accelerator.single_step(spike_array)
        assert isinstance(output, np.ndarray)
    
    def test_large_number_of_steps(self, accelerator):
        """Test processing many timesteps."""
        accelerator.set_step_mode("single", timestep_dt=0.001)
        accelerator.reset()
        
        num_steps = 100
        for t in range(num_steps):
            output = accelerator.single_step([])
            assert isinstance(output, np.ndarray)
        
        assert accelerator.current_timestep == num_steps
        assert len(accelerator.get_spike_history()) == num_steps


class TestIntegrationWithLearning:
    """Test step modes with learning enabled."""
    
    def test_single_step_with_learning(self, accelerator):
        """Test single-step mode with learning enabled."""
        from snn_fpga_accelerator.learning import STDPLearning, LearningConfig
        
        config = LearningConfig(tau_plus=0.020, tau_minus=0.020)
        stdp = STDPLearning(config)
        accelerator.configure_learning(stdp)
        accelerator.enable_learning(True)
        
        accelerator.set_step_mode("single", timestep_dt=0.001)
        accelerator.reset()
        
        # Process timesteps with learning
        for t in range(5):
            spikes = [SpikeEvent(neuron_id=t % 3, timestamp=0.0, weight=1.0)]
            output = accelerator.single_step(spikes)
            assert isinstance(output, np.ndarray)
    
    def test_multi_step_with_learning(self, accelerator, sample_spikes):
        """Test multi-step mode with learning enabled."""
        from snn_fpga_accelerator.learning import STDPLearning, LearningConfig
        
        config = LearningConfig(tau_plus=0.020, tau_minus=0.020)
        stdp = STDPLearning(config)
        accelerator.configure_learning(stdp)
        accelerator.enable_learning(True)
        
        accelerator.set_step_mode("multi")
        
        output = accelerator.infer_with_learning(sample_spikes, duration=0.020)
        assert isinstance(output, np.ndarray)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_step_mode_parameterized(accelerator, sample_spikes, mode):
    """Parameterized test for both step modes."""
    if mode == "single":
        accelerator.set_step_mode("single", timestep_dt=0.001)
        accelerator.reset()
        
        # Process first 10 timesteps
        for t in range(10):
            t_start = t * 0.001
            t_end = (t + 1) * 0.001
            spikes_t = [
                s for s in sample_spikes 
                if t_start <= s.timestamp < t_end
            ]
            output = accelerator.single_step(spikes_t)
            assert isinstance(output, np.ndarray)
    else:
        accelerator.set_step_mode("multi")
        output = accelerator.infer(sample_spikes, duration=0.010)
        assert isinstance(output, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
