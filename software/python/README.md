# SNN FPGA Accelerator Python Interface

This Python package provides PyTorch integration for the FPGA-based Spiking Neural Network accelerator.

## Features

- Event-driven SNN simulation interface
- PyTorch model conversion and weight loading
- STDP and R-STDP learning algorithms
- Real-time spike encoding/decoding
- Performance monitoring and visualization

## Installation

```bash
cd software/python
pip install -e .
```

## Quick Start

```python
import torch
from snn_fpga_accelerator import SNNAccelerator, pytorch_to_snn

# Load your PyTorch model
torch_model = torch.load('my_model.pth')

# Convert to SNN format
snn_model = pytorch_to_snn(torch_model)

# Initialize FPGA accelerator
accelerator = SNNAccelerator()
accelerator.load_model(snn_model)

# Run inference
spikes = accelerator.forward(input_data)
```
