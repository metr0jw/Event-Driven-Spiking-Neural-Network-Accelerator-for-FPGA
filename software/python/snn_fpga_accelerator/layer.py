"""
Spiking Neural Network Layers

PyTorch-compatible SNN layers with clean API.
Designed for easy training and FPGA deployment.

Usage:
    import snn_fpga_accelerator as snn
    
    model = nn.Sequential(
        snn.Linear(784, 256),
        snn.LIF(),
        snn.Linear(256, 10),
        snn.LIF(),
    )
    
Author: Jiwoon Lee (@metr0jw)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, List, Union
import math

from .neuron import LIF, IF, SpikingNeuron, reset_neurons

__all__ = [
    # Linear layers
    'Linear', 'SLinear',
    # Convolutional layers
    'Conv2d', 'SConv2d',
    # Pooling layers
    'AvgPool2d', 'MaxPool2d',
    # Normalization
    'BatchNorm', 'LayerNorm',
    # Recurrent
    'SRNN', 'SLSTM',
    # Container
    'Sequential', 'SNN',
    # Dropout
    'Dropout',
]


# =============================================================================
# Linear Layers
# =============================================================================

class Linear(nn.Linear):
    """
    Linear layer with optional weight quantization for HW deployment.
    
    Same as torch.nn.Linear but supports:
    - Weight quantization (8-bit for HW)
    - Weight clamping
    - HW-compatible initialization
    
    Args:
        in_features: Input size
        out_features: Output size
        bias: Include bias (default: True)
        hw_mode: Enable HW constraints (default: False)
        
    Examples:
        >>> layer = Linear(784, 256)
        >>> out = layer(input)
        
        >>> # HW mode
        >>> layer = Linear(784, 256, hw_mode=True)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        hw_mode: bool = False,
    ):
        super().__init__(in_features, out_features, bias)
        self.hw_mode = hw_mode
        self.weight_bits = 8
        
        # HW-friendly initialization
        if hw_mode:
            self._hw_init()
    
    def _hw_init(self):
        """Initialize weights for HW compatibility."""
        # Smaller weights for stable fixed-point
        nn.init.uniform_(self.weight, -0.5, 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def quantize_weights(self) -> Tensor:
        """Quantize weights to 8-bit."""
        scale = 127.0 / max(self.weight.abs().max().item(), 1e-6)
        return torch.round(self.weight * scale) / scale
    
    def forward(self, x: Tensor) -> Tensor:
        if self.hw_mode and self.training:
            # Straight-through estimator for quantization
            w = self.quantize_weights()
        else:
            w = self.weight
        return F.linear(x, w, self.bias)


class SLinear(nn.Module):
    """
    Spiking Linear layer (Linear + Neuron combined).
    
    Convenience layer combining Linear and LIF neuron.
    
    Args:
        in_features: Input size
        out_features: Output size
        neuron: Neuron type (default: LIF)
        **kwargs: Passed to neuron
        
    Examples:
        >>> layer = SLinear(784, 256)
        >>> spk = layer(input)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        neuron: type = LIF,
        **kwargs
    ):
        super().__init__()
        self.fc = Linear(in_features, out_features, bias)
        self.lif = neuron(**kwargs)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.lif(self.fc(x))
    
    def reset(self):
        self.lif.reset_state()


# =============================================================================
# Convolutional Layers
# =============================================================================

class Conv2d(nn.Conv2d):
    """
    Conv2d with optional weight quantization for HW deployment.
    
    Same as torch.nn.Conv2d but supports HW constraints.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        hw_mode: bool = False,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.hw_mode = hw_mode
        
        if hw_mode:
            self._hw_init()
    
    def _hw_init(self):
        nn.init.uniform_(self.weight, -0.5, 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class SConv2d(nn.Module):
    """
    Spiking Conv2d layer (Conv2d + Neuron combined).
    
    Examples:
        >>> layer = SConv2d(1, 32, 3, padding=1)
        >>> spk = layer(input)  # input: (B, C, H, W) or (B, T, C, H, W)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        neuron: type = LIF,
        **kwargs
    ):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.lif = neuron(**kwargs)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.lif(self.conv(x))
    
    def reset(self):
        self.lif.reset_state()


# =============================================================================
# Pooling Layers
# =============================================================================

class AvgPool2d(nn.AvgPool2d):
    """Average pooling that works with spike tensors."""
    pass


class MaxPool2d(nn.MaxPool2d):
    """Max pooling that works with spike tensors."""
    pass


# =============================================================================
# Normalization Layers
# =============================================================================

class BatchNorm(nn.Module):
    """
    Batch normalization for SNNs.
    
    Applies BN before the spiking nonlinearity.
    Uses threshold-dependent normalization for stable training.
    """
    
    def __init__(self, num_features: int, thresh: float = 1.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.thresh = thresh
    
    def forward(self, x: Tensor) -> Tensor:
        # Handle temporal dimension
        if x.dim() == 3:  # (B, T, F)
            B, T, F = x.shape
            x = x.view(B * T, F)
            x = self.bn(x) * self.thresh
            x = x.view(B, T, F)
        else:
            x = self.bn(x) * self.thresh
        return x


class LayerNorm(nn.LayerNorm):
    """Layer normalization for SNNs."""
    
    def __init__(self, normalized_shape, thresh: float = 1.0):
        super().__init__(normalized_shape)
        self.thresh = thresh
    
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x) * self.thresh


# =============================================================================
# Recurrent Layers
# =============================================================================

class SRNN(nn.Module):
    """
    Spiking Recurrent Neural Network.
    
    Simple spiking RNN with recurrent connections.
    
    Args:
        input_size: Input feature size
        hidden_size: Hidden state size
        neuron: Neuron type
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        neuron: type = LIF,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.fc_in = Linear(input_size, hidden_size)
        self.fc_rec = Linear(hidden_size, hidden_size, bias=False)
        self.lif = neuron(**kwargs)
        
        self.spk: Optional[Tensor] = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input (B, F) for single step or (B, T, F) for sequence
        """
        if self.spk is None:
            batch_size = x.shape[0]
            self.spk = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Input + recurrent
        h = self.fc_in(x) + self.fc_rec(self.spk)
        
        # Spike
        self.spk = self.lif(h)
        
        return self.spk
    
    def reset(self):
        self.spk = None
        self.lif.reset_state()


class SLSTM(nn.Module):
    """
    Spiking LSTM-like layer.
    
    Uses spiking neurons in an LSTM-inspired architecture.
    """
    
    def __init__(self, input_size: int, hidden_size: int, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Gates
        self.fc_i = Linear(input_size + hidden_size, hidden_size)
        self.fc_f = Linear(input_size + hidden_size, hidden_size)
        self.fc_g = Linear(input_size + hidden_size, hidden_size)
        self.fc_o = Linear(input_size + hidden_size, hidden_size)
        
        self.lif_o = LIF(**kwargs)
        
        self.h: Optional[Tensor] = None
        self.c: Optional[Tensor] = None
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        
        if self.h is None:
            self.h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            self.c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        combined = torch.cat([x, self.h], dim=-1)
        
        # Gates with spike-compatible activations
        i = torch.sigmoid(self.fc_i(combined))
        f = torch.sigmoid(self.fc_f(combined))
        g = torch.tanh(self.fc_g(combined))
        
        # Cell state
        self.c = f * self.c + i * g
        
        # Output (spiking)
        o = torch.sigmoid(self.fc_o(combined))
        self.h = self.lif_o(o * self.c)
        
        return self.h
    
    def reset(self):
        self.h = None
        self.c = None
        self.lif_o.reset_state()


# =============================================================================
# Dropout
# =============================================================================

class Dropout(nn.Dropout):
    """
    Dropout for spiking networks.
    
    Applies dropout to spike tensors.
    """
    pass


# =============================================================================
# Container Modules
# =============================================================================

class Sequential(nn.Sequential):
    """
    Sequential container with automatic state reset.
    
    Examples:
        >>> model = Sequential(
        ...     Linear(784, 256),
        ...     LIF(),
        ...     Linear(256, 10),
        ...     LIF(),
        ... )
        >>> model.reset()  # Reset all neuron states
    """
    
    def reset(self):
        """Reset all spiking neurons in the container."""
        reset_neurons(self)


class SNN(nn.Module):
    """
    Base class for SNN models.
    
    Provides convenience methods for:
    - Time-stepped inference
    - State management
    - HW export
    
    Examples:
        >>> class MyModel(SNN):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fc1 = Linear(784, 256)
        ...         self.lif1 = LIF()
        ...         self.fc2 = Linear(256, 10)
        ...         self.lif2 = LIF()
        ...     
        ...     def forward(self, x):
        ...         x = self.lif1(self.fc1(x))
        ...         x = self.lif2(self.fc2(x))
        ...         return x
        >>> 
        >>> model = MyModel()
        >>> out = model.run(input, T=100)  # Run for 100 timesteps
    """
    
    def reset(self):
        """Reset all neuron states."""
        reset_neurons(self)
    
    def run(
        self,
        x: Tensor,
        T: int,
        return_all: bool = False,
    ) -> Union[Tensor, List[Tensor]]:
        """
        Run SNN for T timesteps.
        
        Args:
            x: Input tensor (B, F) - static input repeated each timestep
               or (B, T, F) - temporal input
            T: Number of timesteps
            return_all: Return all timesteps or just final
            
        Returns:
            Output spikes: (B, T, F) if return_all else (B, F) summed
        """
        self.reset()
        
        outputs = []
        
        # Handle temporal vs static input
        if x.dim() == 3 and x.shape[1] == T:
            # Temporal input
            for t in range(T):
                out = self(x[:, t])
                outputs.append(out)
        else:
            # Static input repeated
            for t in range(T):
                out = self(x)
                outputs.append(out)
        
        if return_all:
            return torch.stack(outputs, dim=1)  # (B, T, F)
        else:
            return torch.stack(outputs, dim=1).sum(dim=1)  # (B, F) - spike counts
    
    def set_hw_mode(self, enabled: bool = True):
        """Enable/disable HW mode for all layers."""
        from .neuron import set_hw_mode
        set_hw_mode(self, enabled)
    
    def export_weights(self, path: str = None) -> dict:
        """Export weights in HW-compatible format."""
        weights = {}
        for name, param in self.named_parameters():
            if 'weight' in name:
                # Quantize to 8-bit
                w = param.detach().cpu().numpy()
                scale = 127.0 / max(abs(w.max()), abs(w.min()), 1e-6)
                w_int8 = (w * scale).clip(-128, 127).astype('int8')
                weights[name] = {
                    'data': w_int8,
                    'scale': scale,
                    'shape': w.shape,
                }
        
        if path:
            import numpy as np
            np.savez(path, **{k: v['data'] for k, v in weights.items()})
        
        return weights
