"""
Spike Encoders and Decoders

Convert between continuous values and spike trains.

Usage:
    encoder = snn.encoder.Rate(T=100)
    spikes = encoder(images)  # (B, T, C, H, W)
    
    decoder = snn.decoder.Rate()
    values = decoder(spikes)  # (B, classes)

Author: Jiwoon Lee (@metr0jw)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import math

__all__ = [
    # Encoders
    'Rate', 'Poisson', 'Latency', 'Temporal', 'Delta', 'Phase',
    # Decoders
    'RateDecoder', 'LatencyDecoder', 'MaxDecoder',
    # Aliases (compatibility)
    'RateEncoder', 'PoissonEncoder', 'LatencyEncoder',
]


# =============================================================================
# Encoders
# =============================================================================

class Encoder(nn.Module):
    """Base class for spike encoders."""
    
    def __init__(self, T: int = 100):
        super().__init__()
        self.T = T
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Rate(Encoder):
    """
    Rate coding encoder.
    
    Converts intensity to spike probability per timestep.
    spike_prob = x * gain (each timestep independent)
    
    Args:
        T: Number of timesteps
        gain: Probability scaling factor (default: 1.0)
        
    Examples:
        >>> enc = Rate(T=100)
        >>> spikes = enc(images)  # images in [0, 1]
    """
    
    def __init__(self, T: int = 100, gain: float = 1.0):
        super().__init__(T)
        self.gain = gain
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (B, ...) with values in [0, 1]
            
        Returns:
            Spike tensor (B, T, ...) with binary values
        """
        # Expand for time dimension
        x_expanded = x.unsqueeze(1).expand(-1, self.T, *x.shape[1:])
        
        # Generate random tensor
        rand = torch.rand_like(x_expanded)
        
        # Spike where random < x * gain
        spikes = (rand < x_expanded * self.gain).float()
        
        return spikes


class Poisson(Encoder):
    """
    Poisson spike encoding.
    
    Similar to Rate but generates spikes from Poisson distribution.
    More biologically realistic but slightly slower.
    
    Args:
        T: Number of timesteps
        rate_scale: Firing rate scaling (default: 1.0)
    """
    
    def __init__(self, T: int = 100, rate_scale: float = 1.0):
        super().__init__(T)
        self.rate_scale = rate_scale
    
    def forward(self, x: Tensor) -> Tensor:
        # Rate as Poisson parameter
        rate = x * self.rate_scale
        
        # Generate spikes from Poisson process
        spikes = []
        for _ in range(self.T):
            spike = (torch.rand_like(x) < rate).float()
            spikes.append(spike)
        
        return torch.stack(spikes, dim=1)


class Latency(Encoder):
    """
    Latency (time-to-first-spike) encoding.
    
    Higher intensity → earlier spike.
    spike_time = T * (1 - x) for x in [0, 1]
    
    Args:
        T: Number of timesteps
        normalize: Normalize input to [0, 1] (default: True)
        
    Examples:
        >>> enc = Latency(T=100)
        >>> spikes = enc(images)  # High values spike early
    """
    
    def __init__(self, T: int = 100, normalize: bool = True):
        super().__init__(T)
        self.normalize = normalize
    
    def forward(self, x: Tensor) -> Tensor:
        if self.normalize:
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        # Calculate spike time (high value = early spike)
        spike_times = ((1.0 - x) * (self.T - 1)).long()
        spike_times = torch.clamp(spike_times, 0, self.T - 1)
        
        # Create spike tensor
        spikes = torch.zeros(x.shape[0], self.T, *x.shape[1:], device=x.device)
        
        # Set spikes at computed times
        for b in range(x.shape[0]):
            for idx in range(x.shape[1:].numel()):
                flat_idx = idx
                t = spike_times.view(x.shape[0], -1)[b, flat_idx].item()
                spikes.view(x.shape[0], self.T, -1)[b, t, flat_idx] = 1.0
        
        return spikes


class Temporal(Encoder):
    """
    Temporal coding with learned delays.
    
    Uses learnable parameters to determine spike times.
    Useful for temporal pattern recognition.
    """
    
    def __init__(self, T: int = 100, features: int = 784):
        super().__init__(T)
        self.delays = nn.Parameter(torch.rand(features))
    
    def forward(self, x: Tensor) -> Tensor:
        # Combine input with learned delays
        spike_times = ((1.0 - x * self.delays) * (self.T - 1)).long()
        spike_times = torch.clamp(spike_times, 0, self.T - 1)
        
        spikes = torch.zeros(x.shape[0], self.T, x.shape[1], device=x.device)
        
        for b in range(x.shape[0]):
            for f in range(x.shape[1]):
                t = spike_times[b, f].item()
                spikes[b, t, f] = 1.0
        
        return spikes


class Delta(Encoder):
    """
    Delta modulation encoding.
    
    Generates spikes when input changes significantly.
    Useful for event-based sensors (DVS cameras).
    
    Args:
        T: Number of timesteps
        threshold: Change threshold to trigger spike
    """
    
    def __init__(self, T: int = 100, threshold: float = 0.1):
        super().__init__(T)
        self.threshold = threshold
        self.prev_value: Optional[Tensor] = None
    
    def reset(self):
        self.prev_value = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Temporal input (B, T, ...) or static (B, ...)
        """
        if x.dim() == len(x.shape):
            # Already has time dimension
            spikes = []
            for t in range(x.shape[1]):
                current = x[:, t]
                if self.prev_value is None:
                    spike = torch.zeros_like(current)
                else:
                    diff = current - self.prev_value
                    spike = (torch.abs(diff) > self.threshold).float() * torch.sign(diff)
                spikes.append(spike)
                self.prev_value = current
            return torch.stack(spikes, dim=1)
        else:
            # Static input - treat as constant
            return Rate(self.T).forward(x)


class Phase(Encoder):
    """
    Phase encoding.
    
    Encodes values as phases in a periodic spike pattern.
    Useful for clock-synchronized hardware.
    
    Args:
        T: Number of timesteps
        freq: Base frequency
    """
    
    def __init__(self, T: int = 100, freq: float = 10.0):
        super().__init__(T)
        self.freq = freq
    
    def forward(self, x: Tensor) -> Tensor:
        # Phase determined by input value
        phases = x * 2 * math.pi
        
        spikes = []
        for t in range(self.T):
            theta = 2 * math.pi * self.freq * t / self.T
            # Spike when phase matches
            spike = (torch.cos(theta + phases) > 0.9).float()
            spikes.append(spike)
        
        return torch.stack(spikes, dim=1)


# =============================================================================
# Decoders
# =============================================================================

class Decoder(nn.Module):
    """Base class for spike decoders."""
    pass


class RateDecoder(Decoder):
    """
    Rate decoding - count spikes.
    
    Output = spike_count / T
    
    Args:
        dim: Time dimension to sum over (default: 1)
        normalize: Divide by timesteps (default: True)
    """
    
    def __init__(self, dim: int = 1, normalize: bool = True):
        super().__init__()
        self.dim = dim
        self.normalize = normalize
    
    def forward(self, spikes: Tensor) -> Tensor:
        """
        Args:
            spikes: (B, T, F) spike tensor
            
        Returns:
            (B, F) decoded values
        """
        counts = spikes.sum(dim=self.dim)
        if self.normalize:
            T = spikes.shape[self.dim]
            counts = counts / T
        return counts


class LatencyDecoder(Decoder):
    """
    Latency decoding - time to first spike.
    
    Output = 1 - (first_spike_time / T)
    Earlier spike = higher value.
    """
    
    def __init__(self, T: int = 100):
        super().__init__()
        self.T = T
    
    def forward(self, spikes: Tensor) -> Tensor:
        """
        Args:
            spikes: (B, T, F) spike tensor
            
        Returns:
            (B, F) decoded values
        """
        B, T, F = spikes.shape
        
        # Find first spike time for each feature
        # Use argmax on cumsum to find first 1
        cumsum = torch.cumsum(spikes, dim=1)
        first_spike = (cumsum >= 1).float().argmax(dim=1)  # (B, F)
        
        # Handle no-spike case
        no_spike = (spikes.sum(dim=1) == 0)
        first_spike[no_spike] = T
        
        # Convert to intensity (earlier = higher)
        return 1.0 - first_spike.float() / T


class MaxDecoder(Decoder):
    """
    Max spike count decoding.
    
    Returns argmax of spike counts (for classification).
    """
    
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
    
    def forward(self, spikes: Tensor) -> Tensor:
        """
        Args:
            spikes: (B, T, F) spike tensor
            
        Returns:
            (B,) predicted classes
        """
        counts = spikes.sum(dim=self.dim)
        return counts.argmax(dim=-1)


# =============================================================================
# Aliases for backwards compatibility
# =============================================================================

RateEncoder = Rate
PoissonEncoder = Poisson
LatencyEncoder = Latency
