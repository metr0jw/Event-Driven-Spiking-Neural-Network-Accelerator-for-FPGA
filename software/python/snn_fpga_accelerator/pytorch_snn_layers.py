"""
PyTorch-Compatible SNN Layers for FPGA Accelerator

This module provides PyTorch-like interface for SNN layers implemented on FPGA.
No surrogate gradients - uses efficient STDP-based learning instead.

Author: Jiwoon Lee (@metr0jw)
Organization: Kwangwoon University, Seoul, South Korea
Contact: jwlee@linux.com
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SNNConv2d(nn.Module):
    """
    PyTorch-compatible SNN 2D Convolution Layer
    
    This layer mimics torch.nn.Conv2d but operates on spike trains.
    Uses integer arithmetic for area-efficient FPGA implementation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        threshold: float = 0.25,
        decay_factor: float = 0.9,
        device_id: Optional[int] = None
    ):
        super(SNNConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.threshold = threshold
        self.decay_factor = decay_factor
        self.device_id = device_id
        
        # Initialize weights (will be converted to fixed-point for FPGA)
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * 0.1)
        
        # Membrane potential tracking (for simulation)
        self.membrane_potential = None
        self.register_buffer('spike_count', torch.zeros(1))
        
        # FPGA-specific parameters
        self.weight_scale = 127  # For 8-bit signed weights
        self.threshold_int = int(threshold * 16384)  # Q8.8 format
        self.decay_factor_int = int(decay_factor * 256)  # 8-bit factor
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SNN convolution layer
        
        Args:
            x: Input spike tensor of shape (batch, channels, height, width, time_steps)
            
        Returns:
            Output spike tensor of shape (batch, out_channels, out_height, out_width, time_steps)
        """
        batch_size, in_channels, height, width, time_steps = x.shape
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize membrane potentials if needed
        if self.membrane_potential is None or self.membrane_potential.shape != (
            batch_size, self.out_channels, out_height, out_width
        ):
            self.membrane_potential = torch.zeros(
                batch_size, self.out_channels, out_height, out_width,
                device=x.device, dtype=torch.float32
            )
        
        # Output spike tensor
        output_spikes = torch.zeros(
            batch_size, self.out_channels, out_height, out_width, time_steps,
            device=x.device, dtype=torch.float32
        )
        
        # Process each time step
        for t in range(time_steps):
            # Apply convolution to current time step
            input_t = x[:, :, :, :, t]
            
            # Pad input if necessary
            if self.padding > 0:
                input_t = torch.nn.functional.pad(
                    input_t, (self.padding, self.padding, self.padding, self.padding)
                )
            
            # Convolution operation
            for b in range(batch_size):
                for oc in range(self.out_channels):
                    for oy in range(out_height):
                        for ox in range(out_width):
                            # Calculate convolution window
                            y_start = oy * self.stride
                            y_end = y_start + self.kernel_size
                            x_start = ox * self.stride
                            x_end = x_start + self.kernel_size
                            
                            # Extract window and apply weights
                            window = input_t[b, :, y_start:y_end, x_start:x_end]
                            weight_kernel = self.weight[oc, :, :, :]
                            
                            # Accumulate weighted spikes
                            activation = torch.sum(window * weight_kernel)
                            self.membrane_potential[b, oc, oy, ox] += activation
            
            # Apply membrane leak
            self.membrane_potential *= self.decay_factor
            
            # Generate spikes where membrane potential exceeds threshold
            spike_mask = self.membrane_potential >= self.threshold
            output_spikes[:, :, :, :, t] = spike_mask.float()
            
            # Reset membrane potential where spikes occurred
            self.membrane_potential[spike_mask] = 0.0
            
            # Update spike count
            self.spike_count += torch.sum(spike_mask)
        
        return output_spikes
    
    def get_fpga_config(self) -> Dict[str, Any]:
        """Get configuration for FPGA deployment"""
        return {
            'layer_type': 'conv2d',
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'threshold': self.threshold_int,
            'decay_factor': self.decay_factor_int,
            'weights': self._quantize_weights()
        }
    
    def _quantize_weights(self) -> np.ndarray:
        """Quantize weights to 8-bit integers for FPGA"""
        # Clamp weights to [-1, 1] range
        weights_clamped = torch.clamp(self.weight, -1.0, 1.0)
        
        # Scale to 8-bit signed integer range
        weights_int = (weights_clamped * self.weight_scale).round().int()
        
        return weights_int.detach().cpu().numpy()

class SNNAvgPool2d(nn.Module):
    """PyTorch-compatible SNN 2D Average Pooling Layer"""
    
    def __init__(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        threshold: float = 0.125,
        decay_factor: float = 0.9,
        pooling_window_time: int = 100
    ):
        super(SNNAvgPool2d, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.threshold = threshold
        self.decay_factor = decay_factor
        self.pooling_window_time = pooling_window_time
        
        # FPGA parameters
        self.threshold_int = int(threshold * 16384)  # Q8.8 format
        self.decay_factor_int = int(decay_factor * 256)
        
        self.register_buffer('spike_count', torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SNN average pooling layer"""
        batch_size, channels, height, width, time_steps = x.shape
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Output tensor
        output_spikes = torch.zeros(
            batch_size, channels, out_height, out_width, time_steps,
            device=x.device, dtype=torch.float32
        )
        
        # Accumulator for pooling windows
        pool_accumulator = torch.zeros(
            batch_size, channels, out_height, out_width,
            device=x.device, dtype=torch.float32
        )
        
        # Process pooling windows over time
        window_step = self.pooling_window_time
        for t_start in range(0, time_steps, window_step):
            t_end = min(t_start + window_step, time_steps)
            
            # Reset accumulator
            pool_accumulator.zero_()
            spike_counts = torch.zeros_like(pool_accumulator)
            
            # Accumulate spikes in time window
            for t in range(t_start, t_end):
                input_t = x[:, :, :, :, t]
                
                # Apply pooling
                for b in range(batch_size):
                    for c in range(channels):
                        for oy in range(out_height):
                            for ox in range(out_width):
                                # Define pooling window
                                y_start = oy * self.stride
                                y_end = min(y_start + self.kernel_size, height)
                                x_start = ox * self.stride
                                x_end = min(x_start + self.kernel_size, width)
                                
                                # Sum spikes in window
                                window_spikes = input_t[b, c, y_start:y_end, x_start:x_end]
                                spike_sum = torch.sum(window_spikes)
                                
                                pool_accumulator[b, c, oy, ox] += spike_sum
                                spike_counts[b, c, oy, ox] += torch.sum(window_spikes > 0)
            
            # Calculate averages and generate output spikes
            avg_mask = spike_counts > 0
            averages = torch.zeros_like(pool_accumulator)
            averages[avg_mask] = pool_accumulator[avg_mask] / spike_counts[avg_mask]
            
            # Generate output spikes based on averages
            output_mask = averages >= self.threshold
            
            # Place output spikes at middle of time window
            t_output = min(t_start + window_step // 2, time_steps - 1)
            output_spikes[:, :, :, :, t_output] = output_mask.float()
            
            self.spike_count += torch.sum(output_mask)
        
        return output_spikes
    
    def get_fpga_config(self) -> Dict[str, Any]:
        """Get configuration for FPGA deployment"""
        return {
            'layer_type': 'avgpool2d',
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'threshold': self.threshold_int,
            'decay_factor': self.decay_factor_int,
            'pooling_window_time': self.pooling_window_time
        }

class SNNMaxPool2d(nn.Module):
    """PyTorch-compatible SNN 2D Max Pooling Layer"""
    
    def __init__(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        winner_take_all: bool = True,
        pooling_window_time: int = 100
    ):
        super(SNNMaxPool2d, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.winner_take_all = winner_take_all
        self.pooling_window_time = pooling_window_time
        
        self.register_buffer('spike_count', torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SNN max pooling layer"""
        batch_size, channels, height, width, time_steps = x.shape
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Output tensor
        output_spikes = torch.zeros(
            batch_size, channels, out_height, out_width, time_steps,
            device=x.device, dtype=torch.float32
        )
        
        if self.winner_take_all:
            # Winner-take-all: earliest spike wins
            earliest_spike_time = torch.full(
                (batch_size, channels, out_height, out_width),
                time_steps, device=x.device, dtype=torch.long
            )
            spike_present = torch.zeros(
                batch_size, channels, out_height, out_width,
                device=x.device, dtype=torch.bool
            )
            
            # Find earliest spikes in each pooling window
            for t in range(time_steps):
                input_t = x[:, :, :, :, t]
                
                for b in range(batch_size):
                    for c in range(channels):
                        for oy in range(out_height):
                            for ox in range(out_width):
                                # Define pooling window
                                y_start = oy * self.stride
                                y_end = min(y_start + self.kernel_size, height)
                                x_start = ox * self.stride
                                x_end = min(x_start + self.kernel_size, width)
                                
                                # Check for spikes in window
                                window_spikes = input_t[b, c, y_start:y_end, x_start:x_end]
                                if torch.any(window_spikes > 0):
                                    if not spike_present[b, c, oy, ox] or t < earliest_spike_time[b, c, oy, ox]:
                                        earliest_spike_time[b, c, oy, ox] = t
                                        spike_present[b, c, oy, ox] = True
            
            # Generate output spikes at earliest times
            for b in range(batch_size):
                for c in range(channels):
                    for oy in range(out_height):
                        for ox in range(out_width):
                            if spike_present[b, c, oy, ox]:
                                t = earliest_spike_time[b, c, oy, ox]
                                output_spikes[b, c, oy, ox, t] = 1.0
                                self.spike_count += 1
        
        else:
            # Frequency-based max pooling
            window_step = self.pooling_window_time
            for t_start in range(0, time_steps, window_step):
                t_end = min(t_start + window_step, time_steps)
                
                # Count spikes in time window
                spike_counts = torch.zeros(
                    batch_size, channels, out_height, out_width,
                    device=x.device, dtype=torch.float32
                )
                
                for t in range(t_start, t_end):
                    input_t = x[:, :, :, :, t]
                    
                    for b in range(batch_size):
                        for c in range(channels):
                            for oy in range(out_height):
                                for ox in range(out_width):
                                    # Define pooling window
                                    y_start = oy * self.stride
                                    y_end = min(y_start + self.kernel_size, height)
                                    x_start = ox * self.stride
                                    x_end = min(x_start + self.kernel_size, width)
                                    
                                    # Sum spikes in window
                                    window_spikes = input_t[b, c, y_start:y_end, x_start:x_end]
                                    spike_counts[b, c, oy, ox] += torch.sum(window_spikes)
                
                # Generate output based on spike counts
                max_count = torch.max(spike_counts)
                if max_count > 0:
                    output_mask = spike_counts >= max_count * 0.8  # Top 20% threshold
                    t_output = min(t_start + window_step // 2, time_steps - 1)
                    output_spikes[:, :, :, :, t_output] = output_mask.float()
                    self.spike_count += torch.sum(output_mask)
        
        return output_spikes
    
    def get_fpga_config(self) -> Dict[str, Any]:
        """Get configuration for FPGA deployment"""
        return {
            'layer_type': 'maxpool2d',
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'winner_take_all': self.winner_take_all,
            'pooling_window_time': self.pooling_window_time
        }

class SNNSequential(nn.Module):
    """PyTorch-compatible sequential container for SNN layers"""
    
    def __init__(self, *layers):
        super(SNNSequential, self).__init__()
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers"""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_fpga_config(self) -> Dict[str, Any]:
        """Get configuration for FPGA deployment"""
        layer_configs = []
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_fpga_config'):
                config = layer.get_fpga_config()
                config['layer_id'] = i
                layer_configs.append(config)
        
        return {
            'num_layers': len(layer_configs),
            'layers': layer_configs
        }

class SNNConv1d(nn.Module):
    """
    PyTorch-compatible SNN 1D Convolution Layer
    
    This layer mimics torch.nn.Conv1d but operates on spike trains.
    Ideal for temporal/sequential data processing like audio or time series.
    Uses integer arithmetic for area-efficient FPGA implementation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        threshold: float = 0.25,
        decay_factor: float = 0.9,
        device_id: Optional[int] = None
    ):
        super(SNNConv1d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.threshold = threshold
        self.decay_factor = decay_factor
        self.device_id = device_id
        
        # Initialize weights (will be converted to fixed-point for FPGA)
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size
        ) * 0.1)
        
        # Membrane potential tracking (for simulation)
        self.membrane_potential = None
        self.register_buffer('spike_count', torch.zeros(1))
        
        # FPGA-specific parameters
        self.weight_scale = 127  # For 8-bit signed weights
        self.threshold_int = int(threshold * 16384)  # Q8.8 format
        self.decay_factor_int = int(decay_factor * 256)  # 8-bit factor
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SNN 1D convolution layer
        
        Args:
            x: Input spike tensor of shape (batch, channels, length, time_steps)
            
        Returns:
            Output spike tensor of shape (batch, out_channels, out_length, time_steps)
        """
        batch_size, in_channels, length, time_steps = x.shape
        
        # Calculate output dimensions
        out_length = (length + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize membrane potentials if needed
        if self.membrane_potential is None or self.membrane_potential.shape != (
            batch_size, self.out_channels, out_length
        ):
            self.membrane_potential = torch.zeros(
                batch_size, self.out_channels, out_length,
                device=x.device, dtype=torch.float32
            )
        
        # Output spike tensor
        output_spikes = torch.zeros(
            batch_size, self.out_channels, out_length, time_steps,
            device=x.device, dtype=torch.float32
        )
        
        # Process each time step
        for t in range(time_steps):
            # Apply convolution to current time step
            input_t = x[:, :, :, t]
            
            # Pad input if necessary
            if self.padding > 0:
                input_t = torch.nn.functional.pad(
                    input_t, (self.padding, self.padding)
                )
            
            # 1D Convolution operation
            for b in range(batch_size):
                for oc in range(self.out_channels):
                    for ol in range(out_length):
                        # Calculate convolution window
                        l_start = ol * self.stride
                        l_end = l_start + self.kernel_size
                        
                        # Extract window and apply weights
                        window = input_t[b, :, l_start:l_end]
                        weight_kernel = self.weight[oc, :, :]
                        
                        # Accumulate weighted spikes
                        activation = torch.sum(window * weight_kernel)
                        self.membrane_potential[b, oc, ol] += activation
            
            # Apply membrane leak
            self.membrane_potential *= self.decay_factor
            
            # Generate spikes where membrane potential exceeds threshold
            spike_mask = self.membrane_potential >= self.threshold
            output_spikes[:, :, :, t] = spike_mask.float()
            
            # Reset membrane potential where spikes occurred
            self.membrane_potential[spike_mask] = 0.0
            
            # Update spike count
            self.spike_count += torch.sum(spike_mask)
        
        return output_spikes
    
    def get_fpga_config(self) -> Dict[str, Any]:
        """Get configuration for FPGA deployment"""
        return {
            'layer_type': 'conv1d',
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'threshold': self.threshold_int,
            'decay_factor': self.decay_factor_int,
            'weights': self._quantize_weights()
        }
    
    def _quantize_weights(self) -> np.ndarray:
        """Quantize weights to 8-bit integers for FPGA"""
        # Clamp weights to [-1, 1] range
        weights_clamped = torch.clamp(self.weight, -1.0, 1.0)
        
        # Scale to 8-bit signed integer range
        weights_int = (weights_clamped * self.weight_scale).round().int()
        
        return weights_int.detach().cpu().numpy()

# Utility functions for PyTorch compatibility
def convert_pytorch_to_snn(pytorch_model: nn.Module, input_shape: Tuple[int, ...]) -> SNNSequential:
    """
    Convert a PyTorch CNN model to SNN layers
    
    Args:
        pytorch_model: PyTorch model with Conv1d, Conv2d and pooling layers
        input_shape: Input tensor shape (C, H, W) or (C, L) for 1D
        
    Returns:
        SNNSequential model with equivalent SNN layers
    """
    snn_layers = []
    
    for name, module in pytorch_model.named_modules():
        if isinstance(module, nn.Conv1d):
            snn_layer = SNNConv1d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size[0],
                stride=module.stride[0],
                padding=module.padding[0]
            )
            # Copy weights
            snn_layer.weight.data = module.weight.data.clone()
            snn_layers.append(snn_layer)
            
        elif isinstance(module, nn.Conv2d):
            snn_layer = SNNConv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size[0],
                stride=module.stride[0],
                padding=module.padding[0]
            )
            # Copy weights
            snn_layer.weight.data = module.weight.data.clone()
            snn_layers.append(snn_layer)
            
        elif isinstance(module, nn.AvgPool2d):
            snn_layer = SNNAvgPool2d(
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding
            )
            snn_layers.append(snn_layer)
            
        elif isinstance(module, nn.MaxPool2d):
            snn_layer = SNNMaxPool2d(
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding
            )
            snn_layers.append(snn_layer)
    
    return SNNSequential(*snn_layers)

def create_spike_train(data: torch.Tensor, num_time_steps: int = 100, 
                      encoding: str = 'rate') -> torch.Tensor:
    """
    Convert input data to spike trains
    
    Args:
        data: Input data tensor of shape (batch, channels, height, width)
        num_time_steps: Number of time steps
        encoding: Encoding method ('rate', 'temporal', 'poisson')
        
    Returns:
        Spike train tensor of shape (batch, channels, height, width, time_steps)
    """
    batch_size, channels, height, width = data.shape
    
    if encoding == 'rate':
        # Rate encoding: higher values = higher spike rate
        spike_prob = data.unsqueeze(-1).expand(-1, -1, -1, -1, num_time_steps)
        spikes = torch.rand_like(spike_prob) < spike_prob
        return spikes.float()
        
    elif encoding == 'temporal':
        # Temporal encoding: higher values = earlier spikes
        spike_times = ((1.0 - data.clamp(0, 1)) * (num_time_steps - 1)).long()
        spikes = torch.zeros(batch_size, channels, height, width, num_time_steps)
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        t = spike_times[b, c, h, w]
                        if data[b, c, h, w] > 0:
                            spikes[b, c, h, w, t] = 1.0
        
        return spikes
        
    elif encoding == 'poisson':
        # Poisson encoding
        rates = data.clamp(0, 1) * 100  # Max 100 Hz
        dt = 1.0 / num_time_steps
        spike_prob = rates * dt
        
        spikes = torch.zeros(batch_size, channels, height, width, num_time_steps)
        for t in range(num_time_steps):
            random_vals = torch.rand_like(data)
            spikes[:, :, :, :, t] = (random_vals < spike_prob).float()
        
        return spikes
    
    else:
        raise ValueError(f"Unknown encoding method: {encoding}")

# Example usage
if __name__ == "__main__":
    # Create a simple SNN CNN
    model = SNNSequential(
        SNNConv2d(1, 32, kernel_size=3, stride=1, padding=1),
        SNNMaxPool2d(kernel_size=2, stride=2),
        SNNConv2d(32, 64, kernel_size=3, stride=1, padding=1),
        SNNAvgPool2d(kernel_size=2, stride=2)
    )
    
    # Create sample input
    batch_size = 4
    input_data = torch.rand(batch_size, 1, 28, 28)
    
    # Convert to spike trains
    spike_input = create_spike_train(input_data, num_time_steps=100, encoding='rate')
    
    # Forward pass
    output = model(spike_input)
    
    print(f"Input shape: {spike_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Get FPGA configuration
    fpga_config = model.get_fpga_config()
    print(f"FPGA config: {fpga_config}")
