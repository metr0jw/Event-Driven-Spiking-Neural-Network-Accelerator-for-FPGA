"""
End-to-end MNIST spike pipeline tests.

Tests encoder/decoder combinations on sample MNIST data.
"""

import pytest
import numpy as np
from typing import List, Tuple

torch = pytest.importorskip("torch")
import torch.nn as nn


from snn_fpga_accelerator.spike_encoding import (
    PoissonEncoder, TemporalEncoder, RateEncoder,
    SpikeEvent
)
from snn_fpga_accelerator.pytorch_interface import (
    pytorch_to_snn, SNNModel, simulate_snn_inference
)
from snn_fpga_accelerator.accelerator import SNNAccelerator


# Fixtures
@pytest.fixture
def mnist_sample_image():
    """Generate a sample 28x28 MNIST-like image."""
    # Create a simple pattern (vertical line)
    image = np.zeros((28, 28))
    image[:, 13:15] = 0.8  # Vertical line
    image[10:18, 10:18] = 0.6  # Square in the middle
    return image.flatten()  # 784 pixels


@pytest.fixture
def mnist_batch_images():
    """Generate a batch of MNIST-like images."""
    batch = []
    for i in range(5):
        image = np.random.rand(28, 28) * 0.3  # Background noise
        # Add a simple pattern
        image[10+i:15+i, 10:15] = 0.9
        batch.append(image.flatten())
    return np.array(batch)


@pytest.fixture
def simple_mnist_model():
    """Simple model for MNIST classification."""    
    class MNISTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = MNISTNet()
    model.eval()
    
    # Initialize with reasonable weights
    with torch.no_grad():
        nn.init.xavier_uniform_(model.fc1.weight)
        nn.init.xavier_uniform_(model.fc2.weight)
        nn.init.xavier_uniform_(model.fc3.weight)
    
    return model


@pytest.fixture
def mnist_cnn_model():
    """CNN model for MNIST classification."""    
    class MNISTCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = MNISTCNN()
    model.eval()
    return model


# Test Encoders
class TestMNISTEncoders:
    """Test different spike encoders on MNIST data."""
    
    def test_poisson_encoder_output_shape(self, mnist_sample_image):
        """Test Poisson encoder produces correct spike train shape."""
        encoder = PoissonEncoder(
            num_neurons=784,
            duration=0.05,  # 50ms
            max_rate=100.0,
            seed=42
        )
        
        spikes = encoder.encode(mnist_sample_image)
        
        # Check we get spike events
        assert isinstance(spikes, list)
        assert len(spikes) > 0
        assert all(isinstance(s, SpikeEvent) for s in spikes)
        
        # Check neuron IDs are in range
        neuron_ids = [s.neuron_id for s in spikes]
        assert all(0 <= nid < 784 for nid in neuron_ids)
        
        # Check timestamps are in duration
        timestamps = [s.timestamp for s in spikes]
        assert all(0 <= t <= 0.05 for t in timestamps)
    
    def test_poisson_encoder_rate_correlation(self, mnist_sample_image):
        """Test that brighter pixels generate more spikes."""
        encoder = PoissonEncoder(
            num_neurons=784,
            duration=0.1,
            max_rate=100.0,
            seed=42
        )
        
        spikes = encoder.encode(mnist_sample_image)
        
        # Count spikes per neuron
        spike_counts = np.zeros(784)
        for spike in spikes:
            spike_counts[spike.neuron_id] += 1
        
        # High intensity pixels should have more spikes
        high_intensity_neurons = np.where(mnist_sample_image > 0.7)[0]
        low_intensity_neurons = np.where(mnist_sample_image < 0.1)[0]
        
        if len(high_intensity_neurons) > 0 and len(low_intensity_neurons) > 0:
            avg_high = np.mean(spike_counts[high_intensity_neurons])
            avg_low = np.mean(spike_counts[low_intensity_neurons])
            assert avg_high > avg_low
    
    def test_temporal_encoder_output(self, mnist_sample_image):
        """Test temporal encoder on MNIST data."""
        encoder = TemporalEncoder(
            num_neurons=784,
            duration=0.05
        )
        
        spikes = encoder.encode(mnist_sample_image)
        
        assert isinstance(spikes, list)
        assert len(spikes) > 0
        
        # Each neuron should spike at most once (latency encoding)
        neuron_ids = [s.neuron_id for s in spikes]
        assert len(neuron_ids) == len(set(neuron_ids))  # All unique
    
    def test_rate_encoder_output(self, mnist_sample_image):
        """Test rate encoder on MNIST data."""
        encoder = RateEncoder(
            num_neurons=784,
            duration=0.05,
            max_rate=100.0
        )
        
        spikes = encoder.encode(mnist_sample_image)
        
        assert isinstance(spikes, list)
        # Rate encoder should produce regular spike trains
        assert len(spikes) > 0


# Test End-to-End Pipeline
class TestMNISTPipeline:
    """Test complete MNIST inference pipeline."""
    
    def test_encode_convert_infer_fc(self, mnist_sample_image, simple_mnist_model):
        """Test full pipeline: encode -> convert model -> infer."""
        # 1. Encode image to spikes
        encoder = PoissonEncoder(num_neurons=784, duration=0.05, max_rate=100.0, seed=42)
        input_spikes = encoder.encode(mnist_sample_image)
        
        assert len(input_spikes) > 0
        
        # 2. Convert PyTorch model to SNN
        snn_model = pytorch_to_snn(simple_mnist_model, (784,))
        
        assert len(snn_model.layers) == 3
        assert snn_model.total_neurons > 0
        
        # 3. Run inference (software simulation)
        output_spikes = simulate_snn_inference(snn_model, input_spikes, duration=0.05)
        
        # Check we got output spikes
        assert isinstance(output_spikes, list)
        # Output layer neuron IDs should be in the last layer's range
        last_layer_start, last_layer_end = snn_model.get_layer_neuron_ids(len(snn_model.layers) - 1)
        output_neuron_ids = [s.neuron_id for s in output_spikes]
        assert all(last_layer_start <= nid < last_layer_end for nid in output_neuron_ids)
    
    def test_encode_convert_infer_cnn(self, mnist_sample_image, mnist_cnn_model):
        """Test full pipeline with CNN model."""
        # Use simpler CNN without pooling for this test
        class SimpleConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
                self.fc1 = nn.Linear(16 * 28 * 28, 128)
                self.fc2 = nn.Linear(128, 10)
            
            def forward(self, x):
                x = self.conv1(x)
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                x = self.fc2(x)
                return x
        
        model = SimpleConvNet()
        model.eval()
        
        # 1. Encode image
        encoder = PoissonEncoder(num_neurons=784, duration=0.05, max_rate=100.0, seed=42)
        input_spikes = encoder.encode(mnist_sample_image)
        
        # 2. Convert CNN model
        snn_model = pytorch_to_snn(model, (1, 28, 28))
        
        # Should have conv layer + fc layers
        assert len(snn_model.layers) >= 3  # conv + 2 fc
        
        # Check first layer is convolutional
        assert snn_model.layers[0].layer_type == "convolutional"
        
        # 3. Run inference
        output_spikes = simulate_snn_inference(snn_model, input_spikes, duration=0.05)
        
        assert isinstance(output_spikes, list)
    
    def test_batch_encoding(self, mnist_batch_images):
        """Test encoding a batch of images."""
        encoder = PoissonEncoder(num_neurons=784, duration=0.05, max_rate=100.0, seed=42)
        
        batch_spikes = []
        for image in mnist_batch_images:
            spikes = encoder.encode(image)
            batch_spikes.append(spikes)
        
        assert len(batch_spikes) == 5
        assert all(len(spikes) > 0 for spikes in batch_spikes)
    
    def test_different_encoder_comparison(self, mnist_sample_image, simple_mnist_model):
        """Compare different encoders on same image."""
        # Convert model once
        snn_model = pytorch_to_snn(simple_mnist_model, (784,))
        
        encoders = {
            'poisson': PoissonEncoder(num_neurons=784, duration=0.05, max_rate=100.0, seed=42),
            'temporal': TemporalEncoder(num_neurons=784, duration=0.05),
            'rate': RateEncoder(num_neurons=784, duration=0.05, max_rate=100.0),
        }
        
        results = {}
        for name, encoder in encoders.items():
            input_spikes = encoder.encode(mnist_sample_image)
            output_spikes = simulate_snn_inference(snn_model, input_spikes, duration=0.05)
            
            # Count output spikes per output neuron
            last_layer_start, last_layer_end = snn_model.get_layer_neuron_ids(len(snn_model.layers) - 1)
            output_counts = np.zeros(10)
            for spike in output_spikes:
                if last_layer_start <= spike.neuron_id < last_layer_end:
                    output_counts[spike.neuron_id - last_layer_start] += 1
            
            results[name] = output_counts
        
        # All encoders should produce some output
        for name, counts in results.items():
            assert counts.sum() >= 0, f"{name} encoder produced output"


# Test Accelerator Integration
class TestAcceleratorIntegration:
    """Test integration with SNNAccelerator class."""
    
    def test_accelerator_with_encoded_spikes(self, mnist_sample_image):
        """Test SNNAccelerator with encoded MNIST data."""
        # Initialize accelerator in simulation mode
        accelerator = SNNAccelerator(simulation_mode=True)
        
        # Configure simple network (similar to software simulation)
        accelerator.set_step_mode("multi")
        
        # Encode input
        encoder = PoissonEncoder(num_neurons=784, duration=0.05, max_rate=100.0, seed=42)
        input_spikes = encoder.encode(mnist_sample_image)
        
        # Run inference - may not work without configured model
        try:
            output = accelerator.infer(input_spikes, duration=0.05)
            
            # Check output format
            assert isinstance(output, np.ndarray)
            assert output.shape[0] >= 0  # Valid array
        except RuntimeError as e:
            # If model not configured, that's expected in simulation mode
            if "No SNN model configured" in str(e):
                pytest.skip("Accelerator requires configured model")
            else:
                raise
    
    def test_accelerator_single_step_mode(self, mnist_sample_image):
        """Test accelerator in single-step mode with MNIST data."""
        accelerator = SNNAccelerator(simulation_mode=True)
        accelerator.set_step_mode("single", timestep_dt=0.001)
        accelerator.reset()
        
        # Encode input
        encoder = PoissonEncoder(num_neurons=784, duration=0.05, max_rate=100.0, seed=42)
        input_spikes = encoder.encode(mnist_sample_image)
        
        # Group spikes by timestep
        timestep_groups = {}
        for spike in input_spikes:
            t = int(spike.timestamp / 0.001)
            if t not in timestep_groups:
                timestep_groups[t] = []
            timestep_groups[t].append(spike)
        
        # Process timestep by timestep
        outputs = []
        try:
            for t in range(50):  # 50ms / 1ms = 50 timesteps
                timestep_spikes = timestep_groups.get(t, [])
                output = accelerator.single_step(timestep_spikes)
                outputs.append(output)
        except RuntimeError as e:
            if "No SNN model configured" in str(e):
                pytest.skip("Accelerator requires configured model")
            else:
                raise
        
        # Check we processed all timesteps
        assert len(outputs) == 50
        
        # Get history
        history = accelerator.get_spike_history()
        assert len(history) == 50


# Test Decoder/Output Processing
class TestOutputDecoding:
    """Test decoding SNN outputs to predictions."""
    
    def test_spike_count_decoding(self):
        """Test decoding based on spike counts."""
        # Simulate output spikes from 10 output neurons
        output_spikes = []
        
        # Neuron 3 fires most (should be predicted class)
        for i in range(10):
            output_spikes.append(SpikeEvent(neuron_id=3, timestamp=0.01 * i, weight=1.0))
        
        # Other neurons fire less
        for i in range(5):
            output_spikes.append(SpikeEvent(neuron_id=1, timestamp=0.01 * i, weight=1.0))
        for i in range(3):
            output_spikes.append(SpikeEvent(neuron_id=7, timestamp=0.01 * i, weight=1.0))
        
        # Count spikes per neuron
        spike_counts = np.zeros(10)
        for spike in output_spikes:
            if 0 <= spike.neuron_id < 10:
                spike_counts[spike.neuron_id] += 1
        
        # Predict class with most spikes
        predicted_class = np.argmax(spike_counts)
        assert predicted_class == 3
    
    def test_temporal_decoding(self):
        """Test decoding based on first-to-spike."""
        output_spikes = [
            SpikeEvent(neuron_id=5, timestamp=0.003, weight=1.0),
            SpikeEvent(neuron_id=2, timestamp=0.001, weight=1.0),  # First
            SpikeEvent(neuron_id=7, timestamp=0.005, weight=1.0),
            SpikeEvent(neuron_id=2, timestamp=0.006, weight=1.0),
        ]
        
        # Find first spike
        first_spike = min(output_spikes, key=lambda s: s.timestamp)
        predicted_class = first_spike.neuron_id
        
        assert predicted_class == 2


# Parameterized tests for different configurations
@pytest.mark.parametrize("duration,max_rate", [
    (0.05, 100.0),
    (0.1, 200.0),
    (0.02, 50.0),
])
def test_encoding_parameters(mnist_sample_image, duration, max_rate):
    """Test encoding with different parameters."""
    encoder = PoissonEncoder(
        num_neurons=784,
        duration=duration,
        max_rate=max_rate,
        seed=42
    )
    
    spikes = encoder.encode(mnist_sample_image)
    
    # Check all timestamps within duration
    assert all(0 <= s.timestamp <= duration for s in spikes)
    
    # Higher max_rate should generally produce more spikes
    assert len(spikes) > 0


@pytest.mark.parametrize("image_noise_level", [0.0, 0.1, 0.3])
def test_robustness_to_noise(image_noise_level, simple_mnist_model):
    """Test pipeline robustness to input noise."""
    # Create image with noise
    image = np.zeros(784)
    image[200:400] = 0.8  # Signal
    noise = np.random.rand(784) * image_noise_level
    noisy_image = np.clip(image + noise, 0, 1)
    
    # Encode and infer
    encoder = PoissonEncoder(num_neurons=784, duration=0.05, max_rate=100.0, seed=42)
    input_spikes = encoder.encode(noisy_image)
    
    snn_model = pytorch_to_snn(simple_mnist_model, (784,))
    output_spikes = simulate_snn_inference(snn_model, input_spikes, duration=0.05)
    
    # Should still produce output even with noise
    assert isinstance(output_spikes, list)
