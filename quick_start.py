"""
Quick Start Example for PYNQ-Z2 SNN Accelerator
"""

import numpy as np
from snn_fpga_accelerator import SNNAccelerator
from snn_fpga_accelerator.spike_encoding import rate_encode

def main():
    print("🚀 PYNQ-Z2 SNN Accelerator Quick Start")
    
    # Initialize accelerator in simulation mode
    accelerator = SNNAccelerator(simulation_mode=True)
    
    # Create dummy input data
    input_data = np.random.rand(784)  # MNIST-like input
    
    # Encode to spikes
    spikes = rate_encode(input_data, num_steps=100, max_rate=50.0)
    
    # Run inference
    output = accelerator.infer(spikes)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Spike shape: {spikes.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Predicted class: {np.argmax(output)}")
    
    print("✅ Quick start completed successfully!")

if __name__ == '__main__':
    main()
