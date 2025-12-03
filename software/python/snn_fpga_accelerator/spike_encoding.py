"""
Spike Encoding and Decoding Utilities

Provides various encoding schemes to convert analog data to spike trains
and decode spike trains back to meaningful outputs.

Author: Jiwoon Lee (@metr0jw)
Organization: Kwangwoon University, Seoul, South Korea
Contact: jwlee@linux.com
"""

import numpy as np
from numpy.random import Generator, default_rng
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import time

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


@dataclass
class SpikeEvent:
    """Represents a single spike event."""
    neuron_id: int
    timestamp: float
    weight: float = 1.0
    layer_id: Optional[int] = None


class SpikeEncoder:
    """Base class for spike encoding algorithms."""
    
    def __init__(self, num_neurons: int, duration: float):
        self.num_neurons = num_neurons
        self.duration = duration
    
    def encode(self, data: np.ndarray) -> List[SpikeEvent]:
        """Encode input data to spike events."""
        raise NotImplementedError


class PoissonEncoder(SpikeEncoder):
    """
    Poisson spike encoder.
    
    Converts analog values to Poisson-distributed spike trains where
    the firing rate is proportional to the input intensity.
    """
    
    def __init__(self, num_neurons: int, duration: float, 
                 max_rate: float = 100.0, min_rate: float = 0.1,
                 *, rng: Optional[Generator] = None, seed: Optional[int] = None):
        """
        Initialize Poisson encoder.
        
        Args:
            num_neurons: Number of input neurons
            duration: Encoding duration in seconds
            max_rate: Maximum firing rate in Hz
            min_rate: Minimum firing rate in Hz
            rng: Optional NumPy random generator for reproducible encoding
            seed: Optional seed used to initialise an internal generator when
                ``rng`` is not provided.
        """
        super().__init__(num_neurons, duration)
        self.max_rate = max_rate
        self.min_rate = min_rate
        if rng is not None and seed is not None:
            raise ValueError("Provide either 'rng' or 'seed', not both")
        self._rng: Generator = rng if rng is not None else default_rng(seed)
    
    def encode(self, data: np.ndarray) -> List[SpikeEvent]:
        """
        Encode data using Poisson process.
        
        Args:
            data: Input data array (values should be in [0, 1])
            
        Returns:
            List of spike events
        """
        if data.size != self.num_neurons:
            raise ValueError(f"Data size {data.size} doesn't match num_neurons {self.num_neurons}")
        
        spikes = []
        flat_data = data.flatten()
        
        for neuron_id, intensity in enumerate(flat_data):
            # Calculate firing rate based on intensity
            rate = self.min_rate + intensity * (self.max_rate - self.min_rate)
            
            # Generate Poisson spike train
            if rate > 0:
                # Expected number of spikes
                expected_spikes = rate * self.duration
                num_spikes = self._rng.poisson(expected_spikes)
                
                # Generate random spike times
                if num_spikes > 0:
                    spike_times = np.sort(self._rng.uniform(0, self.duration, num_spikes))
                    
                    for spike_time in spike_times:
                        spikes.append(SpikeEvent(
                            neuron_id=neuron_id,
                            timestamp=spike_time,
                            weight=1.0
                        ))
        
        return spikes


class TemporalEncoder(SpikeEncoder):
    """
    Temporal spike encoder.
    
    Encodes analog values as the timing of first spike, where higher
    values result in earlier spike times.
    """
    
    def __init__(self, num_neurons: int, duration: float):
        super().__init__(num_neurons, duration)
    
    def encode(self, data: np.ndarray) -> List[SpikeEvent]:
        """
        Encode data using temporal coding.
        
        Args:
            data: Input data array (values should be in [0, 1])
            
        Returns:
            List of spike events
        """
        spikes = []
        flat_data = data.flatten()
        
        for neuron_id, intensity in enumerate(flat_data):
            if intensity > 0:
                # Spike time inversely proportional to intensity
                spike_time = self.duration * (1.0 - intensity)
                
                spikes.append(SpikeEvent(
                    neuron_id=neuron_id,
                    timestamp=spike_time,
                    weight=intensity
                ))
        
        return spikes


class RateEncoder(SpikeEncoder):
    """
    Rate-based spike encoder.
    
    Converts analog values to regular spike trains where the frequency
    is proportional to the input value.
    """
    
    def __init__(self, num_neurons: int, duration: float, max_rate: float = 100.0):
        super().__init__(num_neurons, duration)
        self.max_rate = max_rate
    
    def encode(self, data: np.ndarray) -> List[SpikeEvent]:
        """
        Encode data using rate coding.
        
        Args:
            data: Input data array (values should be in [0, 1])
            
        Returns:
            List of spike events
        """
        spikes = []
        flat_data = data.flatten()
        
        for neuron_id, intensity in enumerate(flat_data):
            if intensity > 0:
                rate = intensity * self.max_rate
                inter_spike_interval = 1.0 / rate if rate > 0 else float('inf')
                
                current_time = 0
                while current_time < self.duration:
                    current_time += inter_spike_interval
                    if current_time < self.duration:
                        spikes.append(SpikeEvent(
                            neuron_id=neuron_id,
                            timestamp=current_time,
                            weight=1.0
                        ))
        
        return spikes


class LatencyEncoder(SpikeEncoder):
    """
    Latency-based encoder for temporal pattern recognition.
    
    Each input feature is encoded as a spike with a specific delay,
    allowing the network to learn temporal dependencies.
    """
    
    def __init__(self, num_neurons: int, duration: float, 
                 max_delay: Optional[float] = None):
        super().__init__(num_neurons, duration)
        self.max_delay = max_delay or duration * 0.8
    
    def encode(self, data: np.ndarray) -> List[SpikeEvent]:
        """Encode data with latency-based timing."""
        spikes = []
        flat_data = data.flatten()
        
        for neuron_id, value in enumerate(flat_data):
            if value > 0:
                # Delay is inversely proportional to input value
                delay = self.max_delay * (1.0 - value)
                
                if delay < self.duration:
                    spikes.append(SpikeEvent(
                        neuron_id=neuron_id,
                        timestamp=delay,
                        weight=1.0
                    ))
        
        return spikes


class SpikeDecoder:
    """Base class for spike decoding algorithms."""
    
    def __init__(self, num_outputs: int, duration: float):
        self.num_outputs = num_outputs
        self.duration = duration
    
    def decode(self, spikes: List[SpikeEvent]) -> np.ndarray:
        """Decode spike events to output values."""
        raise NotImplementedError


class PopulationDecoder(SpikeDecoder):
    """
    Population vector decoder.
    
    Decodes output based on spike counts within time windows.
    """
    
    def __init__(self, num_outputs: int, duration: float, 
                 time_window: float = 0.05):
        super().__init__(num_outputs, duration)
        self.time_window = time_window
    
    def decode(self, spikes: List[SpikeEvent]) -> np.ndarray:
        """Decode using population vector method."""
        spike_counts = np.zeros(self.num_outputs)
        
        for spike in spikes:
            if spike.neuron_id < self.num_outputs:
                spike_counts[spike.neuron_id] += 1
        
        # Normalize by duration
        return spike_counts / self.duration


class TemporalDecoder(SpikeDecoder):
    """
    Temporal decoder based on first spike times.
    
    Earlier spikes indicate higher confidence/activation.
    """
    
    def decode(self, spikes: List[SpikeEvent]) -> np.ndarray:
        """Decode using temporal information."""
        first_spike_times = np.full(self.num_outputs, self.duration)
        
        for spike in spikes:
            if spike.neuron_id < self.num_outputs:
                first_spike_times[spike.neuron_id] = min(
                    first_spike_times[spike.neuron_id], 
                    spike.timestamp
                )
        
        # Convert times to activation values (earlier = higher)
        activations = 1.0 - (first_spike_times / self.duration)
        activations[first_spike_times >= self.duration] = 0.0
        
        return activations


def encode_mnist_image(image: np.ndarray, encoder_type: str = "poisson",
                      duration: float = 0.1, **kwargs) -> List[SpikeEvent]:
    """
    Convenience function to encode MNIST images to spikes.
    
    Args:
        image: MNIST image (28x28)
        encoder_type: Type of encoder ("poisson", "temporal", "rate")
        duration: Encoding duration
        **kwargs: Additional encoder parameters
        
    Returns:
        List of spike events
    """
    # Flatten and normalize image
    flat_image = image.flatten() / 255.0
    num_pixels = flat_image.size
    
    # Select encoder
    if encoder_type == "poisson":
        encoder = PoissonEncoder(num_pixels, duration, **kwargs)
    elif encoder_type == "temporal":
        encoder = TemporalEncoder(num_pixels, duration)
    elif encoder_type == "rate":
        encoder = RateEncoder(num_pixels, duration, **kwargs)
    elif encoder_type == "latency":
        encoder = LatencyEncoder(num_pixels, duration, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    return encoder.encode(flat_image)


def spikes_to_raster(spikes: List[SpikeEvent], num_neurons: int, 
                    duration: float, time_bins: int = 100) -> np.ndarray:
    """
    Convert spike list to raster plot matrix.
    
    Args:
        spikes: List of spike events
        num_neurons: Number of neurons
        duration: Total duration
        time_bins: Number of time bins
        
    Returns:
        Raster matrix (num_neurons x time_bins)
    """
    raster = np.zeros((num_neurons, time_bins))
    time_step = duration / time_bins
    
    for spike in spikes:
        if spike.neuron_id < num_neurons:
            time_bin = int(spike.timestamp / time_step)
            if time_bin < time_bins:
                raster[spike.neuron_id, time_bin] += 1
    
    return raster


def calculate_spike_rate(spikes: List[SpikeEvent], neuron_id: int, 
                        duration: float, window_size: float = 0.01) -> np.ndarray:
    """
    Calculate instantaneous firing rate for a specific neuron.
    
    Args:
        spikes: List of spike events
        neuron_id: Target neuron ID
        duration: Total duration
        window_size: Time window for rate calculation
        
    Returns:
        Array of firing rates over time
    """
    # Filter spikes for target neuron
    neuron_spikes = [s.timestamp for s in spikes if s.neuron_id == neuron_id]
    
    # Create time axis
    time_points = np.arange(0, duration, window_size)
    rates = np.zeros(len(time_points))
    
    for i, t in enumerate(time_points):
        # Count spikes in window [t, t + window_size]
        spike_count = sum(1 for st in neuron_spikes 
                         if t <= st < t + window_size)
        rates[i] = spike_count / window_size
    
    return rates
