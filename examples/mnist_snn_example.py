"""
MNIST SNN Classification Example with PyTorch-Compatible Layers

This example demonstrates how to use PyTorch-compatible SNN layers
for MNIST digit classification on the FPGA accelerator.
Now supports convolution, pooling, and area-efficient STDP learning.

Author: Jiwoon Lee (@metr0jw)
Organization: Kwangwoon University, Seoul, South Korea
Contact: jwlee@linux.com
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'software', 'python'))

import numpy as np
import matplotlib.pyplot as plt
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom modules
from snn_fpga_accelerator.pytorch_snn_layers import (
    SNNConv2d, SNNAvgPool2d, SNNMaxPool2d, SNNSequential,
    create_spike_train, convert_pytorch_to_snn
)
from snn_fpga_accelerator.fpga_controller import SNNFPGAController, PyTorchFPGABridge

# Try to import torch for comparison (optional)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    logger.info("PyTorch available for comparison")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using simulation only")

def load_mnist_data():
    """Load MNIST dataset (simplified version)"""
    # For this example, create synthetic MNIST-like data
    # In practice, you would load real MNIST data
    np.random.seed(42)
    
    # Generate synthetic data
    train_images = np.random.rand(1000, 28, 28).astype(np.float32)
    train_labels = np.random.randint(0, 10, 1000)
    
    test_images = np.random.rand(200, 28, 28).astype(np.float32)
    test_labels = np.random.randint(0, 10, 200)
    
    logger.info(f"Loaded {len(train_images)} training samples, {len(test_images)} test samples")
    return train_images, train_labels, test_images, test_labels

def create_snn_cnn_model():
    """Create a CNN-like SNN model using our PyTorch-compatible layers"""
    model = SNNSequential(
        # First convolutional block
        SNNConv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, 
                  threshold=0.25, decay_factor=0.9),
        SNNMaxPool2d(kernel_size=2, stride=2, winner_take_all=True),
        
        # Second convolutional block  
        SNNConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,
                  threshold=0.3, decay_factor=0.9),
        SNNAvgPool2d(kernel_size=2, stride=2, threshold=0.15),
        
        # Third convolutional block
        SNNConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
                  threshold=0.35, decay_factor=0.85),
        SNNMaxPool2d(kernel_size=2, stride=2, winner_take_all=False)
    )
    
    logger.info("Created SNN CNN model with Conv2d and Pooling layers")
    return model

def create_spike_train_numpy(image, time_steps):
    """Numpy version of spike train creation"""
    # Rate encoding
    normalized = image / np.max(image) if np.max(image) > 0 else image
    spike_train = np.zeros((28, 28, time_steps))
    
    for t in range(time_steps):
        random_vals = np.random.rand(28, 28)
        spike_train[:, :, t] = (random_vals < normalized).astype(np.float32)
    
    return spike_train

def main():
    print("PyTorch-Compatible SNN MNIST Example")
    print("====================================")
    print("Features: Conv2d, Pooling, Area-Efficient STDP Learning")
    print()
    
    # Load data
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    
    # Create SNN CNN model
    snn_model = create_snn_cnn_model()
    print(f"SNN Model Configuration:")
    fpga_config = snn_model.get_fpga_config()
    print(f"  Number of layers: {fpga_config['num_layers']}")
    for layer in fpga_config['layers']:
        print(f"  Layer {layer['layer_id']}: {layer['layer_type']}")
    print()
    
    # Initialize FPGA controller
    fpga_controller = SNNFPGAController()
    if fpga_controller.initialize_hardware():
        print("FPGA hardware initialized successfully")
        
        # Deploy model to FPGA
        bridge = PyTorchFPGABridge(fpga_controller)
        sample_input = np.random.rand(1, 1, 28, 28, 50).astype(np.float32)
        success = bridge.deploy_model(snn_model, sample_input)
        
        if success:
            print("Model deployed to FPGA successfully")
        else:
            print("Failed to deploy model to FPGA")
    else:
        print("FPGA hardware not available, running in simulation mode")
    
    # Test with sample data
    test_image = test_images[0]
    
    # Convert to spike train
    if TORCH_AVAILABLE:
        spike_input = create_spike_train(
            torch.tensor(test_image).unsqueeze(0).unsqueeze(0),
            time_steps=100, encoding='rate'
        )[0].numpy()
    else:
        spike_input_2d = create_spike_train_numpy(test_image, 100)
        spike_input = np.expand_dims(spike_input_2d, axis=(0, 1))
    
    # Process through model
    if TORCH_AVAILABLE and hasattr(snn_model, 'forward'):
        output_spikes = snn_model(torch.tensor(spike_input)).detach().numpy()
    else:
        # Simulation fallback
        output_spikes = spike_input
    
    print(f"Input shape: {spike_input.shape}")
    print(f"Output shape: {output_spikes.shape}")
    print(f"Total input spikes: {np.sum(spike_input)}")
    print(f"Total output spikes: {np.sum(output_spikes)}")
    
    # Cleanup
    if fpga_controller.is_initialized:
        fpga_controller.shutdown()
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
