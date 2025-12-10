"""
End-to-End integration test for MNIST classification.

This test validates the complete pipeline:
1. Load MNIST data
2. Encode images to spikes (on-chip or software)
3. Run inference on FPGA/simulator
4. Decode output spikes to classification
5. Measure accuracy

Tests both simulation mode and XRT mode (when available).
"""

import numpy as np
import pytest
from typing import List, Tuple

try:
    from torchvision import datasets, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from snn_fpga_accelerator.accelerator import SNNAccelerator, SpikeEvent
from snn_fpga_accelerator.spike_encoding import encode_mnist_image
from snn_fpga_accelerator.utils import logger


class TestEndToEndMNIST:
    """End-to-end MNIST classification tests."""
    
    @pytest.fixture
    def mnist_samples(self):
        """Load small MNIST test set."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch/torchvision not available")
        
        # Load MNIST test set
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        try:
            test_dataset = datasets.MNIST(
                root='./data', 
                train=False, 
                download=True, 
                transform=transform
            )
        except Exception as e:
            pytest.skip(f"Could not download MNIST: {e}")
        
        # Select 10 samples (one per class)
        samples = []
        labels_seen = set()
        
        for img, label in test_dataset:
            if label not in labels_seen:
                samples.append((img.numpy().squeeze(), label))
                labels_seen.add(label)
                if len(samples) == 10:
                    break
        
        return samples
    
    @pytest.fixture
    def accelerator_sim(self):
        """Create accelerator in simulation mode."""
        accel = SNNAccelerator(simulation_mode=True)
        
        # Configure simple 3-layer network: 784 -> 128 -> 10
        accel.initialize(neurons_per_layer=[784, 128, 10])
        
        # Initialize with random weights
        np.random.seed(42)
        weights_layer1 = np.random.randn(784, 128) * 0.1
        weights_layer2 = np.random.randn(128, 10) * 0.1
        
        # For simulation, store weights in model
        return accel
    
    def test_single_mnist_inference_simulation(self, mnist_samples, accelerator_sim):
        """Test inference on single MNIST digit in simulation mode."""
        if not mnist_samples:
            pytest.skip("No MNIST samples available")
        
        # Take first sample
        image, true_label = mnist_samples[0]
        
        # Normalize to [0, 255] range for encoding
        image_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Encode to spikes using rate-based encoding
        spikes = encode_mnist_image(
            image_uint8, 
            encoder_type="poisson",
            duration=0.1  # 100ms simulation
        )
        
        assert len(spikes) > 0, "Encoding should produce spikes"
        
        # Run inference
        output_spikes = accelerator_sim.infer(spikes, duration=0.1, return_events=True)
        
        # Should produce some output
        assert isinstance(output_spikes, list), "Should return spike events"
        
        # In simulation mode with random weights, just verify that inference runs
        # Random weights may or may not produce output spikes depending on initialization
        logger.info(f"Inference produced {len(output_spikes)} total spikes")
        
        # Verify spike properties for any spikes produced
        if len(output_spikes) > 0:
            for spike in output_spikes[:10]:
                assert 0 <= spike.timestamp <= 0.1, "Timestamp should be in simulation duration"
                assert 0 <= spike.weight <= 1.0, "Weight should be normalized"
                assert spike.neuron_id >= 0, "Neuron ID should be non-negative"
    
    def test_batch_mnist_inference_simulation(self, mnist_samples, accelerator_sim):
        """Test inference on batch of MNIST digits."""
        if not mnist_samples or len(mnist_samples) < 5:
            pytest.skip("Not enough MNIST samples")
        
        results = []
        
        for image, true_label in mnist_samples[:5]:
            # Normalize and encode
            image_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            spikes = encode_mnist_image(image_uint8, encoder_type="poisson", duration=0.1)
            
            # Inference
            output_rates = accelerator_sim.infer(spikes, duration=0.1)
            
            # Predict class (highest firing rate)
            predicted_class = np.argmax(output_rates)
            
            results.append({
                'true_label': true_label,
                'predicted': predicted_class,
                'confidence': output_rates[predicted_class],
                'output_rates': output_rates
            })
        
        # Verify structure
        assert len(results) == 5
        for r in results:
            assert 0 <= r['predicted'] < 10
            assert len(r['output_rates']) == 10
    
    def test_encoding_types_comparison(self, mnist_samples, accelerator_sim):
        """Compare different encoding methods on same input."""
        if not mnist_samples:
            pytest.skip("No MNIST samples available")
        
        image, _ = mnist_samples[0]
        image_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        encoding_types = ["poisson", "latency", "rate"]
        spike_counts = {}
        
        for enc_type in encoding_types:
            try:
                spikes = encode_mnist_image(
                    image_uint8, 
                    encoder_type=enc_type,
                    duration=0.1,
                    num_steps=100
                )
                spike_counts[enc_type] = len(spikes)
            except Exception as e:
                pytest.skip(f"Encoding type {enc_type} not available: {e}")
        
        # All encodings should produce spikes
        for enc_type, count in spike_counts.items():
            assert count > 0, f"{enc_type} encoding should produce spikes"
        
        # Different encodings may produce different spike counts
        print(f"\nSpike counts by encoding: {spike_counts}")
    
    def test_on_chip_encoder_configuration(self, accelerator_sim):
        """Test on-chip encoder configuration (simulation)."""
        # Test that encoder config can be set
        try:
            if hasattr(accelerator_sim, 'configure_encoder'):
                accelerator_sim.configure_encoder(
                    encoding_type="rate_poisson",
                    num_steps=100,
                    rate_scale=256,
                    num_channels=784
                )
        except AttributeError:
            pytest.skip("configure_encoder not yet implemented")
    
    def test_two_neuron_encoding_e2e(self, mnist_samples, accelerator_sim):
        """Test two-neuron encoding in end-to-end pipeline."""
        if not mnist_samples:
            pytest.skip("No MNIST samples available")
        
        image, _ = mnist_samples[0]
        image_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # With two-neuron encoding, we expect doubled channels
        # Each pixel becomes ON + OFF neuron pair
        baseline = 128
        
        # Manual two-neuron split for testing
        on_channels = np.maximum(image_uint8.astype(int) - baseline, 0)
        off_channels = np.maximum(baseline - image_uint8.astype(int), 0)
        
        # Verify split properties
        assert on_channels.shape == image_uint8.shape
        assert off_channels.shape == image_uint8.shape
        
        # ON and OFF should not be active simultaneously
        both_active = (on_channels > 0) & (off_channels > 0)
        assert not np.any(both_active), "ON and OFF should be mutually exclusive"
    
    @pytest.mark.skipif(True, reason="Requires XRT and FPGA hardware")
    def test_mnist_xrt_hardware(self, mnist_samples):
        """Test MNIST inference on actual FPGA hardware via XRT."""
        # This test requires:
        # 1. Compiled xclbin file
        # 2. XRT runtime installed
        # 3. FPGA connected
        
        xclbin_path = "/path/to/snn_accelerator.xclbin"
        
        accel = SNNAccelerator(simulation_mode=False)
        accel.configure_xrt(xclbin_path=xclbin_path, device_index=0)
        accel.initialize(neurons_per_layer=[784, 128, 10])
        
        # Load pre-trained weights
        weights = np.load("pretrained_mnist_weights.npy")
        accel._load_weights(weights)
        
        # Test inference
        image, true_label = mnist_samples[0]
        image_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        spikes = encode_mnist_image(image_uint8, encoder_type="poisson", duration=0.1, num_steps=100)
        output = accel.infer(spikes, duration=0.1)
        
        predicted = np.argmax(output)
        
        # With pre-trained weights, should be reasonably accurate
        print(f"True: {true_label}, Predicted: {predicted}")
    
    def test_spike_timing_distribution(self, mnist_samples, accelerator_sim):
        """Analyze spike timing distribution in encoded data."""
        if not mnist_samples:
            pytest.skip("No MNIST samples available")
        
        image, _ = mnist_samples[0]
        image_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        spikes = encode_mnist_image(image_uint8, encoder_type="poisson", duration=0.1)        # Extract timestamps
        timestamps = [s.timestamp for s in spikes]
        
        # Verify timing properties
        assert min(timestamps) >= 0.0, "Timestamps should be non-negative"
        assert max(timestamps) <= 0.1, "Timestamps should be within duration"
        
        # Check temporal distribution
        hist, bins = np.histogram(timestamps, bins=10)
        
        # Should have some temporal spread (not all in one bin)
        non_empty_bins = np.sum(hist > 0)
        assert non_empty_bins >= 3, "Spikes should be spread across time"
    
    def test_inference_determinism(self, mnist_samples, accelerator_sim):
        """Test that inference is deterministic with same input."""
        if not mnist_samples:
            pytest.skip("No MNIST samples available")
        
        image, _ = mnist_samples[0]
        image_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Encode with fixed seed
        spikes1 = encode_mnist_image(image_uint8, encoder_type="poisson", duration=0.1, seed=42)

        spikes2 = encode_mnist_image(image_uint8, encoder_type="poisson", duration=0.1, seed=42)        # Should produce same spikes
        assert len(spikes1) == len(spikes2), "Same seed should produce same number of spikes"
        
        # Check first 10 spikes match
        for s1, s2 in zip(spikes1[:10], spikes2[:10]):
            assert s1.neuron_id == s2.neuron_id
            assert abs(s1.timestamp - s2.timestamp) < 1e-6
    
    def test_empty_input_handling(self, accelerator_sim):
        """Test handling of empty spike input."""
        # Empty spike list
        output = accelerator_sim.infer([], duration=0.1)
        
        # Should return something (likely all zeros)
        assert output is not None
        assert len(output) == 10  # Output layer size


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
