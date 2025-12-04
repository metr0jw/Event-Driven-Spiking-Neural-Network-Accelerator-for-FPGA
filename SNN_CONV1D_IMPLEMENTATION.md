# SNN 1D Convolution Implementation

## Overview
This document describes the implementation of 1D convolution layers for the PyTorch-compatible SNN accelerator. The 1D convolution is optimized for temporal and sequential data processing applications such as audio signals, time series analysis, ECG/EEG processing, and speech recognition.

## Key Features

### Hardware Implementation (`snn_conv1d.v`)
- **Temporal Processing**: Optimized for sequential data with variable-length inputs
- **Multi-Channel Support**: Handles multiple input/output channels simultaneously  
- **Configurable Architecture**: Parameterizable kernel size, stride, padding, and channels
- **Pipeline Processing**: Efficient weight loading and convolution computation pipeline
- **Memory Efficient**: Smart buffering and weight caching strategies
- **AXI-Stream Interface**: Standard streaming interface for FPGA integration

### Software Implementation (`SNNConv1d` class)
- **PyTorch Compatibility**: Drop-in replacement for `torch.nn.Conv1d`
- **Spike Train Processing**: Native support for temporal spike data
- **Multiple Encodings**: Rate, temporal, and latency encoding schemes
- **FPGA Deployment**: Seamless hardware acceleration integration
- **Quantization Support**: Automatic weight quantization for hardware efficiency

## Technical Specifications

### Hardware Parameters
```verilog
parameter INPUT_LENGTH   = 1000;  // Maximum input sequence length
parameter INPUT_CHANNELS = 16;    // Number of input channels
parameter OUTPUT_CHANNELS = 32;   // Number of output channels  
parameter KERNEL_SIZE    = 3;     // Convolution kernel size
parameter STRIDE         = 1;     // Convolution stride
parameter PADDING        = 1;     // Zero padding amount
parameter WEIGHT_WIDTH   = 8;     // Weight precision (signed)
parameter VMEM_WIDTH     = 16;    // Membrane potential precision
parameter THRESHOLD      = 16'h4000; // Spike threshold (Q8.8 format)
parameter DECAY_FACTOR   = 8'hE6;    // Membrane decay factor
```

### Data Formats
- **Input Spikes**: `{channel[15:0], position[15:0]}` - 32-bit AXI-Stream
- **Output Spikes**: `{channel[15:0], position[15:0]}` - 32-bit AXI-Stream
- **Weights**: 8-bit signed integers (-127 to +127)
- **Membrane Potentials**: Q8.8 fixed-point (16-bit total)
- **Thresholds**: Q8.8 fixed-point format

### Performance Characteristics
- **Throughput**: ~1000 spikes per 1000 clock cycles (typical)
- **Latency**: Variable based on sequence length and kernel size
- **Memory Usage**: O(channels × length) for membrane potentials
- **Power Efficiency**: Event-driven processing reduces unnecessary computation

## Architecture Details

### Processing Pipeline
1. **Input Spike Reception**: AXI-Stream spike data parsing
2. **Weight Loading**: Dynamic weight retrieval from memory
3. **Convolution Computation**: Sliding window operation across sequence
4. **Membrane Integration**: Accumulation with leak dynamics
5. **Spike Generation**: Threshold-based output spike creation
6. **Output Transmission**: AXI-Stream output spike formatting

### State Machine Design
- **Weight Loading States**: Multi-cycle weight fetching from external memory
- **Convolution States**: Systematic processing of all output positions
- **Output States**: Efficient spike generation and transmission
- **Cleanup States**: Memory reset and preparation for next time step

### Memory Organization
- **Input Spike Buffer**: Temporal storage for current time step
- **Membrane Potential Array**: Per-neuron state maintenance
- **Weight Cache**: Kernel weights for current computation
- **Output FIFO**: Spike buffering for smooth output flow

## Software Interface

### Basic Usage
```python
import torch
from snn_fpga_accelerator.pytorch_snn_layers import SNNConv1d

# Create 1D convolution layer
conv1d = SNNConv1d(
    in_channels=8,
    out_channels=32, 
    kernel_size=5,
    stride=1,
    padding=2,
    threshold=0.25,
    decay_factor=0.9
)

# Process spike data
# Input shape: (batch, channels, length, time_steps)
input_spikes = torch.rand(4, 8, 1000, 100)
output_spikes = conv1d(input_spikes)
print(f"Output shape: {output_spikes.shape}")
```

### Spike Encoding Options
```python
from snn_fpga_accelerator.pytorch_snn_layers import create_spike_train_1d

# Rate encoding (higher values = higher spike rate)
spike_data_rate = create_spike_train_1d(temporal_data, time_steps=100, encoding='rate')

# Temporal encoding (higher values = earlier spikes)  
spike_data_temporal = create_spike_train_1d(temporal_data, time_steps=100, encoding='temporal')

# Latency encoding (value determines spike timing pattern)
spike_data_latency = create_spike_train_1d(temporal_data, time_steps=100, encoding='latency')
```

### FPGA Deployment
```python
from snn_fpga_accelerator.fpga_controller import SNNFPGAController, PyTorchFPGABridge

# Initialize FPGA
controller = SNNFPGAController()
controller.initialize_hardware()

# Deploy model
bridge = PyTorchFPGABridge(controller)
success = bridge.deploy_model(conv1d_model, sample_input)

# Run inference
output = bridge.forward(input_spikes)
```

## Application Examples

### Audio Signal Processing
- **Speech Recognition**: Temporal pattern detection in audio spectrograms
- **Music Analysis**: Rhythm and melody pattern extraction
- **Environmental Sound**: Classification of audio events
- **Noise Filtering**: Spike-based signal enhancement

### Biomedical Signal Analysis
- **ECG Processing**: Heart rhythm analysis and anomaly detection
- **EEG Analysis**: Brain wave pattern recognition
- **EMG Signals**: Muscle activity pattern detection
- **Vital Sign Monitoring**: Real-time physiological signal processing

### Time Series Analysis
- **Financial Data**: Market trend pattern recognition
- **Sensor Networks**: Environmental monitoring and prediction
- **Industrial IoT**: Equipment health monitoring
- **Weather Forecasting**: Temporal weather pattern analysis

### Sequential Data Processing
- **Text Analysis**: Character-level or word-level sequence processing
- **DNA Sequencing**: Genetic pattern recognition
- **Log Analysis**: System event pattern detection
- **Network Traffic**: Communication pattern analysis

## Performance Optimization

### Hardware Optimizations
- **Pipeline Depth**: Configurable processing stages for throughput/latency trade-offs
- **Memory Bandwidth**: Optimized weight loading and spike buffering
- **Parallel Processing**: Multi-channel computation in parallel
- **Resource Sharing**: Efficient use of DSP slices and BRAM

### Software Optimizations
- **Batch Processing**: Multiple sequences processed simultaneously
- **Sparse Encoding**: Efficient representation of sparse spike data
- **Memory Management**: Smart caching and buffer reuse
- **Quantization**: Automatic precision optimization for hardware

## Comparison with 2D Convolution

| Aspect | 1D Convolution | 2D Convolution |
|--------|----------------|----------------|
| **Data Type** | Temporal/Sequential | Spatial/Image |
| **Kernel Shape** | Linear (1×K) | Rectangular (H×W) |
| **Memory Usage** | O(C×L) | O(C×H×W) |
| **Applications** | Audio, Time Series | Vision, Image |
| **Complexity** | Lower | Higher |
| **Parallelization** | Temporal | Spatial |

## Integration with Existing Layers

### Layer Stacking
```python
# Multi-scale temporal feature extraction
model = SNNSequential(
    SNNConv1d(8, 32, kernel_size=3, padding=1),   # Short patterns
    SNNConv1d(32, 64, kernel_size=11, stride=2),  # Medium patterns  
    SNNConv1d(64, 128, kernel_size=21, stride=4), # Long patterns
    SNNConv1d(128, 256, kernel_size=15, stride=2) # Feature extraction
)
```

### Mixed Architectures
```python
# Combine 1D and 2D processing
hybrid_model = SNNSequential(
    SNNConv1d(16, 32, kernel_size=5),     # Temporal processing
    # Reshape for 2D processing
    SNNConv2d(32, 64, kernel_size=3),     # Spatial processing
    SNNAvgPool2d(kernel_size=2)           # Spatial pooling
)
```

## Testing and Validation

### Hardware Testbench (`tb_snn_conv1d.v`)
- **Basic Functionality**: Spike processing and output generation
- **Performance Testing**: Throughput and latency measurements
- **Threshold Sensitivity**: Response to different spike thresholds
- **Architecture Validation**: Multi-channel and sequence processing

### Software Testing (`temporal_snn_example.py`)
- **Encoding Validation**: Multiple spike encoding schemes
- **Pattern Analysis**: Temporal pattern detection capabilities
- **Performance Benchmarking**: Latency and throughput measurements
- **Visualization**: Spike pattern and feature evolution analysis

## Future Enhancements

### Hardware Improvements
- **Dynamic Reconfiguration**: Runtime parameter adjustment
- **Multi-Core Processing**: Parallel 1D convolution units
- **Advanced Memory**: HBM integration for large sequences
- **Power Management**: Dynamic voltage and frequency scaling

### Software Features
- **Automatic Architecture Search**: Neural architecture search for 1D CNNs
- **Advanced Encodings**: Population coding and delta modulation
- **Training Integration**: On-device STDP learning
- **Model Compression**: Pruning and quantization for deployment

### Application Extensions
- **Multi-Modal Processing**: Audio-visual temporal integration
- **Real-Time Streaming**: Continuous sequence processing
- **Edge Computing**: Ultra-low-power deployment
- **Neuromorphic Integration**: DVS camera and cochlea interfacing

## Conclusion

The SNN 1D convolution implementation provides a comprehensive solution for temporal and sequential data processing on FPGA hardware. Key benefits include:

- **High Performance**: Optimized pipeline processing for low latency
- **Energy Efficiency**: Event-driven computation reduces power consumption
- **Flexibility**: Configurable architecture for diverse applications
- **Compatibility**: Seamless PyTorch integration for easy adoption
- **Scalability**: Supports small embedded systems to large server deployments

This implementation enables efficient processing of temporal patterns while maintaining the benefits of spiking neural networks: biological plausibility, power efficiency, and real-time processing capabilities.
