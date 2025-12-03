"""
Tests for PyTorch to SNN conversion and weight validation.

Tests convolutional layer support and validates converted weights.
"""

import pytest
import numpy as np
from typing import Tuple

torch = pytest.importorskip("torch")
import torch.nn as nn

from snn_fpga_accelerator.pytorch_interface import (
    pytorch_to_snn, SNNLayer, SNNModel
)


# Fixtures for test models
@pytest.fixture
def simple_fc_model():
    """Simple fully connected model."""  
    class SimpleFC(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            return x
    
    model = SimpleFC()
    model.eval()
    return model


@pytest.fixture
def simple_cnn_model():
    """Simple CNN model with convolution and fully connected layers."""
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
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
    
    model = SimpleCNN()
    model.eval()
    return model


@pytest.fixture
def conversion_params():
    """Standard conversion parameters."""
    return {
        'weight_scale': 128.0,
        'threshold_scale': 1.0,
        'leak_rate': 0.1,
        'refractory_period': 5
    }


# Test Weight Conversion
class TestWeightConversion:
    """Test weight conversion accuracy and dimensions."""
    
    def test_fc_weight_shape(self, simple_fc_model, conversion_params):
        """Test that FC weights are converted with correct shape."""
        snn_model = pytorch_to_snn(simple_fc_model, (784,), conversion_params)
        
        # Check we have 2 layers
        assert len(snn_model.layers) == 2
        
        # Check first layer shape
        layer1 = snn_model.layers[0]
        assert layer1.input_size == 784
        assert layer1.output_size == 128
        assert layer1.weights.shape == (128, 784)
        
        # Check second layer shape
        layer2 = snn_model.layers[1]
        assert layer2.input_size == 128
        assert layer2.output_size == 10
        assert layer2.weights.shape == (10, 128)
    
    def test_fc_weight_scaling(self, simple_fc_model, conversion_params):
        """Test that weights are properly scaled and clipped."""
        snn_model = pytorch_to_snn(simple_fc_model, (784,), conversion_params)
        
        for layer in snn_model.layers:
            # Check weights are within fixed-point range
            assert np.all(layer.weights >= -128)
            assert np.all(layer.weights <= 127)
            
            # Check bias is also scaled
            if layer.bias is not None:
                assert np.all(layer.bias >= -128)
                assert np.all(layer.bias <= 127)
    
    def test_fc_weight_preservation(self, simple_fc_model, conversion_params):
        """Test that weight values are approximately preserved after scaling."""
        # Get original PyTorch weights
        fc1_weight_orig = simple_fc_model.fc1.weight.detach().numpy()
        
        # Convert model
        snn_model = pytorch_to_snn(simple_fc_model, (784,), conversion_params)
        
        # Check weights are scaled versions of originals
        scale = conversion_params['weight_scale']
        fc1_weight_snn = snn_model.layers[0].weights
        
        # Before clipping, they should be approximately equal
        fc1_weight_expected = fc1_weight_orig * scale
        
        # Most values should match (some may be clipped)
        matching = np.abs(fc1_weight_snn - fc1_weight_expected) < 1e-3
        assert np.mean(matching) > 0.95  # At least 95% should match
    
    def test_conv_weight_shape(self, simple_cnn_model, conversion_params):
        """Test that Conv2d weights are converted with correct shape."""
        # Note: This test uses a model without pooling layers in between
        # to test pure convolution conversion
        class SimpleConvOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x
        
        model = SimpleConvOnly()
        model.eval()
        
        snn_model = pytorch_to_snn(model, (1, 28, 28), conversion_params)
        
        # Should have 2 conv layers
        assert len(snn_model.layers) == 2
        
        # Check first conv layer
        conv1 = snn_model.layers[0]
        assert conv1.layer_type == "convolutional"
        assert conv1.weights.shape == (16, 1, 3, 3)  # out_ch, in_ch, kH, kW
        assert conv1.layer_config['kernel_size'] == (3, 3)
        assert conv1.layer_config['in_channels'] == 1
        assert conv1.layer_config['out_channels'] == 16
        
        # Check second conv layer
        conv2 = snn_model.layers[1]
        assert conv2.layer_type == "convolutional"
        assert conv2.weights.shape == (32, 16, 3, 3)
        assert conv2.layer_config['in_channels'] == 16
        assert conv2.layer_config['out_channels'] == 32
    
    def test_conv_output_size_calculation(self, simple_cnn_model, conversion_params):
        """Test that convolutional output sizes are calculated correctly."""
        # Use simple model without pooling
        class SimpleConvOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
            
            def forward(self, x):
                return self.conv1(x)
        
        model = SimpleConvOnly()
        model.eval()
        
        snn_model = pytorch_to_snn(model, (1, 28, 28), conversion_params)
        
        # First conv: 28x28 -> (28+2*1-3)/1+1 = 28x28 with 16 channels
        conv1 = snn_model.layers[0]
        expected_output = 16 * 28 * 28
        assert conv1.output_size == expected_output
        
        # Check output shape in config
        assert conv1.layer_config['output_shape'] == (16, 28, 28)
    
    def test_neuron_parameters_set(self, simple_fc_model, conversion_params):
        """Test that neuron parameters are properly set."""
        snn_model = pytorch_to_snn(simple_fc_model, (784,), conversion_params)
        
        for layer in snn_model.layers:
            assert 'threshold' in layer.neuron_params
            assert 'leak_rate' in layer.neuron_params
            assert 'refractory_period' in layer.neuron_params
            
            assert layer.neuron_params['threshold'] == conversion_params['threshold_scale']
            assert layer.neuron_params['leak_rate'] == conversion_params['leak_rate']
            assert layer.neuron_params['refractory_period'] == conversion_params['refractory_period']


# Test Layer Validation
class TestLayerValidation:
    """Test SNNLayer weight validation."""
    
    def test_fc_layer_rejects_wrong_shape(self):
        """Test that FC layer rejects incorrectly shaped weights."""
        layer = SNNLayer(784, 128, layer_type="fully_connected")
        
        # Correct shape should work
        correct_weights = np.random.randn(128, 784)
        layer.set_weights(correct_weights)
        
        # Wrong shape should raise error
        wrong_weights = np.random.randn(784, 128)  # Transposed
        with pytest.raises(ValueError, match="doesn't match expected"):
            layer.set_weights(wrong_weights)
    
    def test_conv_layer_accepts_conv_weights(self):
        """Test that conv layer accepts convolutional weights."""
        layer = SNNLayer(
            784, 16 * 28 * 28,
            layer_type="convolutional",
            in_channels=1,
            out_channels=16,
            kernel_size=(3, 3)
        )
        
        # Correct shape: (out_ch, in_ch, kH, kW)
        correct_weights = np.random.randn(16, 1, 3, 3)
        layer.set_weights(correct_weights)
        
        assert layer.weights.shape == (16, 1, 3, 3)
    
    def test_conv_layer_rejects_wrong_channels(self):
        """Test that conv layer rejects wrong channel dimensions."""
        layer = SNNLayer(
            784, 16 * 28 * 28,
            layer_type="convolutional",
            in_channels=1,
            out_channels=16,
            kernel_size=(3, 3)
        )
        
        # Wrong channels
        wrong_weights = np.random.randn(32, 1, 3, 3)  # Wrong out_channels
        with pytest.raises(ValueError, match="doesn't match expected"):
            layer.set_weights(wrong_weights)


# Test Model Save/Load
class TestModelSerialization:
    """Test saving and loading converted models."""
    def test_save_load_fc_model(self, simple_fc_model, conversion_params, tmp_path):
        """Test saving and loading FC model preserves weights."""
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not available")
        
        # Convert model
        snn_model = pytorch_to_snn(simple_fc_model, (784,), conversion_params)
        
        # Save weights
        save_path = tmp_path / "test_model.h5"
        snn_model.save_weights(str(save_path))
        
        # Load weights into new model
        loaded_model = SNNModel()
        loaded_model.load_weights(str(save_path))
        
        # Check layers match
        assert len(loaded_model.layers) == len(snn_model.layers)
        
        for orig, loaded in zip(snn_model.layers, loaded_model.layers):
            assert orig.input_size == loaded.input_size
            assert orig.output_size == loaded.output_size
            assert orig.layer_type == loaded.layer_type
            assert np.allclose(orig.weights, loaded.weights)
            if orig.bias is not None:
                assert np.allclose(orig.bias, loaded.bias)
    
    def test_save_load_conv_model(self, simple_cnn_model, conversion_params, tmp_path):
        """Test saving and loading CNN model preserves conv weights."""
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not available")
        
        # Use simple conv model without pooling
        class SimpleConvOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x
        
        model = SimpleConvOnly()
        model.eval()
        
        # Convert model
        snn_model = pytorch_to_snn(model, (1, 28, 28), conversion_params)
        
        # Save weights
        save_path = tmp_path / "test_cnn.h5"
        snn_model.save_weights(str(save_path))
        
        # Load weights
        loaded_model = SNNModel()
        loaded_model.load_weights(str(save_path))
        
        # Check convolutional layers
        for orig, loaded in zip(snn_model.layers, loaded_model.layers):
            if orig.layer_type == "convolutional":
                assert loaded.layer_type == "convolutional"
                assert orig.weights.shape == loaded.weights.shape
                assert np.allclose(orig.weights, loaded.weights)


# Parameterized tests
@pytest.mark.parametrize("input_size,hidden_sizes,output_size", [
    (784, [128], 10),
    (784, [256, 128], 10),
    (1024, [512, 256, 128], 10),
])
def test_various_fc_architectures(input_size, hidden_sizes, output_size):
    """Test conversion with various FC architectures."""
    # Build model dynamically
    layers = []
    in_size = input_size
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(in_size, hidden_size))
        layers.append(nn.ReLU())
        in_size = hidden_size
    layers.append(nn.Linear(in_size, output_size))
    
    model = nn.Sequential(*layers)
    model.eval()
    
    # Convert
    snn_model = pytorch_to_snn(model, (input_size,))
    
    # Check architecture
    assert len(snn_model.layers) == len(hidden_sizes) + 1
    assert snn_model.layers[0].input_size == input_size
    assert snn_model.layers[-1].output_size == output_size


@pytest.mark.parametrize("kernel_size,stride,padding", [
    (3, 1, 1),
    (5, 1, 2),
    (3, 2, 1),
])
def test_various_conv_parameters(kernel_size, stride, padding):
    """Test conversion with various convolution parameters."""
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=kernel_size, stride=stride, padding=padding)
    )
    model.eval()
    
    # Convert
    snn_model = pytorch_to_snn(model, (1, 28, 28))
    
    # Check layer
    conv_layer = snn_model.layers[0]
    assert conv_layer.layer_type == "convolutional"
    assert conv_layer.layer_config['kernel_size'] == (kernel_size, kernel_size)
    assert conv_layer.layer_config['stride'] == (stride, stride)
    assert conv_layer.layer_config['padding'] == (padding, padding)
