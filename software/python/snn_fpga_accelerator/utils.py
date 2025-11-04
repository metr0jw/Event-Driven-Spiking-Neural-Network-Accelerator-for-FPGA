"""
Utility functions and classes for SNN FPGA Accelerator.

Author: Jiwoon Lee (@metr0jw)
Organization: Kwangwoon University, Seoul, South Korea
Contact: jwlee@linux.com
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import logging
import pickle
import json
from pathlib import Path
import time

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    plt = cast(Any, None)

try:
    import h5py  # type: ignore
    if plt is None:
        raise RuntimeError("matplotlib is required for spike visualization")
except ImportError:  # pragma: no cover - optional dependency
    h5py = cast(Any, None)

from .spike_encoding import SpikeEvent


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('snn_fpga_accelerator')


def load_weights(filepath: Union[str, Path]) -> Dict:
    """
    Load weights from various file formats.
    
    Args:
        filepath: Path to weight file
    
    Returns:
        Dictionary containing weights and metadata
    """
    path = Path(filepath)

    if path.suffix in {'.h5', '.hdf5'}:
        return _load_weights_h5(path)
    if path.suffix == '.npz':
        return _load_weights_npz(path)
    if path.suffix == '.pkl':
        return _load_weights_pickle(path)
    else:
        raise ValueError(f"Unsupported weight file format: {path.suffix}")


def save_weights(weights: Dict, filepath: Union[str, Path]) -> None:
    """
    Save weights to file.
    
    Args:
        weights: Dictionary containing weights and metadata
        filepath: Output file path
    """
    path = Path(filepath)
    
    if path.suffix in {'.h5', '.hdf5'}:
        _save_weights_h5(weights, path)
    elif path.suffix == '.npz':
        _save_weights_npz(weights, path)
    elif path.suffix == '.pkl':
        _save_weights_pickle(weights, path)
    else:
        raise ValueError(f"Unsupported weight file format: {path.suffix}")
    
    logger.info(f"Weights saved to {filepath}")


def _load_weights_h5(filepath: Path) -> Dict:
    """Load weights from HDF5 file."""
    if h5py is None:
        raise RuntimeError("h5py is required to load HDF5 weight files")
    weights = {}
    
    with h5py.File(filepath, 'r') as f:
        # Load metadata
        weights['metadata'] = {}
        for key in f.attrs:
            weights['metadata'][key] = f.attrs[key]
        
        # Load weight matrices
        weights['layers'] = {}
        for layer_name in f.keys():
            if layer_name.startswith('layer_'):
                layer_data = {}
                layer_group = f[layer_name]
                
                # Load layer attributes
                for attr_name in layer_group.attrs:
                    layer_data[attr_name] = layer_group.attrs[attr_name]
                
                # Load datasets
                for dataset_name in layer_group.keys():
                    layer_data[dataset_name] = layer_group[dataset_name][:]
                
                weights['layers'][layer_name] = layer_data
    
    return weights


def _save_weights_h5(weights: Dict, filepath: Path) -> None:
    """Save weights to HDF5 file."""
    if h5py is None:
        raise RuntimeError("h5py is required to save HDF5 weight files")
    with h5py.File(filepath, 'w') as f:
        # Save metadata
        if 'metadata' in weights:
            for key, value in weights['metadata'].items():
                f.attrs[key] = value
        
        # Save layers
        if 'layers' in weights:
            for layer_name, layer_data in weights['layers'].items():
                layer_group = f.create_group(layer_name)
                
                for key, value in layer_data.items():
                    if isinstance(value, np.ndarray):
                        layer_group.create_dataset(key, data=value)
                    else:
                        layer_group.attrs[key] = value


def _load_weights_npz(filepath: Path) -> Dict:
    """Load weights from NumPy archive."""
    npz_data = np.load(filepath, allow_pickle=True)
    
    weights = {}
    for key in npz_data.files:
        weights[key] = npz_data[key].item() if npz_data[key].ndim == 0 else npz_data[key]
    
    return weights


def _save_weights_npz(weights: Dict, filepath: Path) -> None:
    """Save weights to NumPy archive."""
    np.savez_compressed(filepath, **weights)


def _load_weights_pickle(filepath: Path) -> Dict:
    """Load weights from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def _save_weights_pickle(weights: Dict, filepath: Path) -> None:
    """Save weights to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(weights, f)


def visualize_spikes(spikes: List[SpikeEvent], duration: float, 
                    num_neurons: Optional[int] = None, 
                    title: str = "Spike Raster Plot",
                    save_path: Optional[str] = None) -> None:
    """
    Create a raster plot of spike events.
    
    Args:
        spikes: List of spike events
        duration: Total duration of recording
        num_neurons: Number of neurons (if None, inferred from spikes)
        title: Plot title
        save_path: Path to save the plot (if None, display only)
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for spike visualization")

    if not spikes:
        logger.warning("No spikes to visualize")
        return
    
    if num_neurons is None:
        num_neurons = max(spike.neuron_id for spike in spikes) + 1
    
    # Extract spike times and neuron IDs
    spike_times = [spike.timestamp for spike in spikes]
    neuron_ids = [spike.neuron_id for spike in spikes]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create raster plot
    plt.scatter(spike_times, neuron_ids, s=2, alpha=0.7, c='black')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron ID')
    plt.title(title)
    plt.xlim(0, duration)
    plt.ylim(-0.5, num_neurons - 0.5)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    total_spikes = len(spikes)
    avg_rate = total_spikes / (duration * num_neurons)
    plt.text(0.02, 0.98, f'Total spikes: {total_spikes}\nAvg rate: {avg_rate:.2f} Hz', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Spike plot saved to {save_path}")
    else:
        plt.show()


def visualize_weight_matrix(weights: np.ndarray, title: str = "Weight Matrix",
                          save_path: Optional[str] = None) -> None:
    """
    Visualize a weight matrix as a heatmap.
    
    Args:
        weights: 2D weight matrix
        title: Plot title
        save_path: Path to save the plot
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for weight visualization")

    plt.figure(figsize=(10, 8))
    
    im = plt.imshow(weights, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, label='Weight Value')
    
    plt.xlabel('Input Neuron')
    plt.ylabel('Output Neuron')
    plt.title(title)
    
    # Add weight statistics
    mean_weight = np.mean(weights)
    std_weight = np.std(weights)
    min_weight = np.min(weights)
    max_weight = np.max(weights)
    
    stats_text = f'Mean: {mean_weight:.3f}\nStd: {std_weight:.3f}\n' \
                f'Range: [{min_weight:.3f}, {max_weight:.3f}]'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Weight matrix plot saved to {save_path}")
    else:
        plt.show()


def analyze_spike_statistics(spikes: List[SpikeEvent], duration: float) -> Dict:
    """
    Analyze statistical properties of spike trains.
    
    Args:
        spikes: List of spike events
        duration: Total duration
        
    Returns:
        Dictionary with statistics
    """
    if not spikes:
        return {'total_spikes': 0, 'mean_rate': 0.0}
    
    # Basic counts
    total_spikes = len(spikes)
    num_neurons = len(set(spike.neuron_id for spike in spikes))
    
    # Firing rates per neuron
    neuron_spike_counts = {}
    for spike in spikes:
        neuron_spike_counts[spike.neuron_id] = neuron_spike_counts.get(spike.neuron_id, 0) + 1
    
    firing_rates = {nid: count / duration for nid, count in neuron_spike_counts.items()}
    
    # Inter-spike intervals
    spike_times_by_neuron = {}
    for spike in spikes:
        if spike.neuron_id not in spike_times_by_neuron:
            spike_times_by_neuron[spike.neuron_id] = []
        spike_times_by_neuron[spike.neuron_id].append(spike.timestamp)
    
    all_isis = []
    for neuron_id, times in spike_times_by_neuron.items():
        if len(times) > 1:
            times = sorted(times)
            isis = np.diff(times)
            all_isis.extend(isis)
    
    # Calculate statistics
    stats = {
        'total_spikes': total_spikes,
        'num_active_neurons': num_neurons,
        'duration': duration,
        'mean_rate': total_spikes / (duration * num_neurons) if num_neurons > 0 else 0.0,
        'firing_rates': firing_rates,
    }
    
    if all_isis:
        stats.update({
            'mean_isi': np.mean(all_isis),
            'std_isi': np.std(all_isis),
            'cv_isi': np.std(all_isis) / np.mean(all_isis) if np.mean(all_isis) > 0 else 0.0,
        })
    
    return stats


def create_network_topology(architecture: str, layer_sizes: List[int], 
                          connectivity: float = 1.0) -> Dict:
    """
    Create network topology specification.
    
    Args:
        architecture: Network architecture type ('feedforward', 'recurrent', 'reservoir')
        layer_sizes: List of layer sizes
        connectivity: Connection probability (0.0 to 1.0)
        
    Returns:
        Network topology dictionary
    """
    topology = {
        'architecture': architecture,
        'layer_sizes': layer_sizes,
        'total_neurons': sum(layer_sizes),
        'connectivity': connectivity,
        'connections': []
    }
    
    if architecture == 'feedforward':
        # Create feedforward connections
        neuron_offset = 0
        for i in range(len(layer_sizes) - 1):
            input_layer_start = neuron_offset
            input_layer_end = neuron_offset + layer_sizes[i]
            neuron_offset += layer_sizes[i]
            
            output_layer_start = neuron_offset
            output_layer_end = neuron_offset + layer_sizes[i + 1]
            
            # Connect all neurons from layer i to layer i+1
            for pre_id in range(input_layer_start, input_layer_end):
                for post_id in range(output_layer_start, output_layer_end):
                    if np.random.random() < connectivity:
                        topology['connections'].append({
                            'pre_neuron': pre_id,
                            'post_neuron': post_id,
                            'weight': np.random.randn() * 0.1,
                            'delay': 1
                        })
    
    elif architecture == 'recurrent':
        # Create recurrent connections within each layer + feedforward
        neuron_offset = 0
        for i in range(len(layer_sizes)):
            layer_start = neuron_offset
            layer_end = neuron_offset + layer_sizes[i]
            
            # Recurrent connections within layer
            for pre_id in range(layer_start, layer_end):
                for post_id in range(layer_start, layer_end):
                    if pre_id != post_id and np.random.random() < connectivity:
                        topology['connections'].append({
                            'pre_neuron': pre_id,
                            'post_neuron': post_id,
                            'weight': np.random.randn() * 0.05,
                            'delay': 1
                        })
            
            # Feedforward connections to next layer
            if i < len(layer_sizes) - 1:
                next_layer_start = layer_end
                next_layer_end = layer_end + layer_sizes[i + 1]
                
                for pre_id in range(layer_start, layer_end):
                    for post_id in range(next_layer_start, next_layer_end):
                        if np.random.random() < connectivity:
                            topology['connections'].append({
                                'pre_neuron': pre_id,
                                'post_neuron': post_id,
                                'weight': np.random.randn() * 0.1,
                                'delay': 1
                            })
            
            neuron_offset = layer_end
    
    logger.info(f"Created {architecture} topology: {len(topology['connections'])} connections")
    return topology


def convert_spikes_to_events(spike_dict: Dict[int, List[float]]) -> List[SpikeEvent]:
    """
    Convert spike dictionary to list of SpikeEvent objects.
    
    Args:
        spike_dict: Dictionary mapping neuron_id -> list of spike times
        
    Returns:
        List of SpikeEvent objects
    """
    events = []
    for neuron_id, spike_times in spike_dict.items():
        for timestamp in spike_times:
            events.append(SpikeEvent(
                neuron_id=neuron_id,
                timestamp=timestamp,
                weight=1.0
            ))
    
    # Sort by timestamp
    events.sort(key=lambda x: x.timestamp)
    return events


def convert_events_to_spikes(events: List[SpikeEvent]) -> Dict[int, List[float]]:
    """
    Convert list of SpikeEvent objects to spike dictionary.
    
    Args:
        events: List of SpikeEvent objects
        
    Returns:
        Dictionary mapping neuron_id -> list of spike times
    """
    spike_dict = {}
    for event in events:
        if event.neuron_id not in spike_dict:
            spike_dict[event.neuron_id] = []
        spike_dict[event.neuron_id].append(event.timestamp)
    
    # Sort spike times for each neuron
    for neuron_id in spike_dict:
        spike_dict[neuron_id].sort()
    
    return spike_dict


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str) -> None:
        """Start timing a process."""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and record duration."""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            self.metrics[name] = duration
            del self.start_times[name]
            return duration
        return 0.0
    
    def record_metric(self, name: str, value: float) -> None:
        """Record a performance metric."""
        self.metrics[name] = value
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all recorded metrics."""
        return self.metrics.copy()
    
    def log_metrics(self) -> None:
        """Log all metrics."""
        for name, value in self.metrics.items():
            logger.info(f"Performance - {name}: {value:.4f}")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper())
    
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger.info(f"Logging setup complete - Level: {log_level}")


def validate_spike_events(events: List[SpikeEvent], max_neurons: int, 
                         duration: float) -> List[str]:
    """
    Validate spike events for consistency.
    
    Args:
        events: List of spike events to validate
        max_neurons: Maximum valid neuron ID
        duration: Maximum valid timestamp
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    for i, event in enumerate(events):
        if event.neuron_id < 0 or event.neuron_id >= max_neurons:
            errors.append(f"Event {i}: Invalid neuron_id {event.neuron_id}")
        
        if event.timestamp < 0 or event.timestamp > duration:
            errors.append(f"Event {i}: Invalid timestamp {event.timestamp}")
        
        if not isinstance(event.weight, (int, float)):
            errors.append(f"Event {i}: Invalid weight type {type(event.weight)}")
    
    return errors
