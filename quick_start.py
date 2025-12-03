"""
Quick Start Example for PYNQ-Z2 SNN Accelerator

Demonstrates:
1. Creating an SNN model
2. Running inference on the SNN accelerator (simulation mode)
3. Comparing CPU (ANN) vs SNN inference results
"""

import numpy as np
from snn_fpga_accelerator import SNNAccelerator
from snn_fpga_accelerator.spike_encoding import RateEncoder
from snn_fpga_accelerator.pytorch_interface import (
    SNNModel, SNNLayer, CPUvsSNNComparator
)

# Check if PyTorch is available for CPU comparison
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def create_simple_pytorch_model(input_size=784, hidden_size=128, output_size=10):
    """Create a simple PyTorch model for comparison."""
    if not TORCH_AVAILABLE:
        return None
    
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )
    return model


def create_snn_model_from_pytorch(torch_model, input_size=784, hidden_size=128, output_size=10):
    """Create an SNN model with weights from PyTorch model."""
    snn_model = SNNModel(name="converted_model")
    
    # Get PyTorch weights
    with torch.no_grad():
        fc1_weight = torch_model[0].weight.numpy()
        fc1_bias = torch_model[0].bias.numpy()
        fc2_weight = torch_model[2].weight.numpy()
        fc2_bias = torch_model[2].bias.numpy()
    
    # Create SNN layers with same weights
    layer1 = SNNLayer(input_size=input_size, output_size=hidden_size, layer_type="fully_connected")
    layer1.set_weights(fc1_weight, fc1_bias)
    layer1.set_neuron_parameters(threshold=0.5, leak_rate=0.05, refractory_period=3)
    snn_model.add_layer(layer1)
    
    layer2 = SNNLayer(input_size=hidden_size, output_size=output_size, layer_type="fully_connected")
    layer2.set_weights(fc2_weight, fc2_bias)
    layer2.set_neuron_parameters(threshold=0.5, leak_rate=0.05, refractory_period=3)
    snn_model.add_layer(layer2)
    
    return snn_model


def main():
    print("PYNQ-Z2 SNN Accelerator Quick Start")
    print("=" * 60)
    
    # Create input data (simulated MNIST-like input)
    np.random.seed(42)
    input_data = np.random.rand(784) * 0.8 + 0.2  # Range [0.2, 1.0]
    
    if TORCH_AVAILABLE:
        print("\n[Mode: CPU vs SNN Comparison]")
        print("-" * 60)
        
        # Create PyTorch model
        torch.manual_seed(42)
        torch_model = create_simple_pytorch_model()
        
        # Create matching SNN model
        snn_model = create_snn_model_from_pytorch(torch_model)
        
        # Create comparator
        comparator = CPUvsSNNComparator(
            torch_model=torch_model,
            snn_model=snn_model
        )
        
        # Run comparison
        results = comparator.compare(
            input_data,
            duration=0.1,
            max_rate=100.0,
            num_repeats=3
        )
        
        # Print report
        CPUvsSNNComparator.print_report(results)
        
    else:
        print("\n[Mode: SNN-only (PyTorch not available)]")
        print("-" * 60)
        
        # Create standalone SNN model
        snn_model = SNNModel(name="quick_start_model")
        
        layer1 = SNNLayer(input_size=784, output_size=128, layer_type="fully_connected")
        layer1.set_weights(np.random.randn(128, 784).astype(np.float32) * 0.5)
        layer1.set_neuron_parameters(threshold=0.5, leak_rate=0.05, refractory_period=3)
        snn_model.add_layer(layer1)
        
        layer2 = SNNLayer(input_size=128, output_size=10, layer_type="fully_connected")
        layer2.set_weights(np.random.randn(10, 128).astype(np.float32) * 0.5)
        layer2.set_neuron_parameters(threshold=0.5, leak_rate=0.05, refractory_period=3)
        snn_model.add_layer(layer2)
        
        # Initialize accelerator
        accelerator = SNNAccelerator(simulation_mode=True)
        accelerator.configure_network(snn_model)
        
        # Encode and run
        encoder = RateEncoder(num_neurons=784, duration=0.1, max_rate=100.0)
        spikes = encoder.encode(input_data)
        output = accelerator.infer(spikes)
        
        print(f"\nResults:")
        print(f"  Input shape: {input_data.shape}")
        print(f"  Number of input spikes: {len(spikes)}")
        print(f"  Output spikes: {len(output)}")
    
    print("\nQuick start completed successfully!")


if __name__ == '__main__':
    main()
