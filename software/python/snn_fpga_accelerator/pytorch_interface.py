"""
PyTorch Integration for SNN Accelerator

Provides utilities to convert PyTorch models to SNN format and
load weights to the FPGA accelerator.

Author: Jiwoon Lee (@metr0jw)
Organization: Kwangwoon University, Seoul, South Korea
Contact: jwlee@linux.com
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, cast, TYPE_CHECKING
import logging
from pathlib import Path

from .spike_encoding import SpikeEvent, PoissonEncoder
from .utils import logger

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
else:  # pragma: no cover - import side effects only during runtime
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except ImportError:
        torch = cast(Any, None)  # type: ignore
        nn = cast(Any, None)  # type: ignore

try:
    import h5py  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    h5py = cast(Any, None)  # type: ignore


class SNNLayer:
    """
    Represents a layer in the SNN that can be mapped to FPGA hardware.
    """
    
    def __init__(self, input_size: int, output_size: int, 
                 layer_type: str = "fully_connected", **layer_config):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_type = layer_type
        self.weights = None
        self.bias = None
        self.neuron_params = {}
        self.layer_config = layer_config  # Store Conv2d params: kernel_size, stride, padding, etc.
        
    def set_weights(self, weights: np.ndarray, bias: Optional[np.ndarray] = None):
        """Set layer weights and bias."""
        if self.layer_type == "fully_connected":
            expected_shape = (self.output_size, self.input_size)
            if weights.shape != expected_shape:
                raise ValueError(f"Weight shape {weights.shape} doesn't match expected {expected_shape}")
        elif self.layer_type == "convolutional":
            # For Conv2d: (out_channels, in_channels, kernel_h, kernel_w)
            expected_out_channels = self.layer_config.get('out_channels')
            expected_in_channels = self.layer_config.get('in_channels', 1)
            if weights.shape[0] != expected_out_channels or weights.shape[1] != expected_in_channels:
                raise ValueError(
                    f"Weight channels {weights.shape[:2]} doesn't match expected "
                    f"({expected_out_channels}, {expected_in_channels})"
                )
        
        self.weights = weights.copy()
        if bias is not None:
            self.bias = bias.copy()
    
    def set_neuron_parameters(self, threshold: float = 1.0, leak_rate: float = 0.1,
                            refractory_period: int = 5):
        """Set neuron parameters for this layer."""
        self.neuron_params = {
            'threshold': threshold,
            'leak_rate': leak_rate,
            'refractory_period': refractory_period
        }


class SNNModel:
    """
    Represents a complete SNN model that can be deployed to FPGA.
    """
    
    def __init__(self, name: str = "snn_model"):
        self.name = name
        self.layers = []
        self.total_neurons = 0
        self.neuron_id_map = {}  # Maps layer neurons to global neuron IDs
        
    def add_layer(self, layer: SNNLayer) -> None:
        """Add a layer to the model."""
        # Assign global neuron IDs
        start_id = self.total_neurons
        end_id = start_id + layer.output_size
        
        self.neuron_id_map[len(self.layers)] = (start_id, end_id)
        self.layers.append(layer)
        self.total_neurons = end_id
        
        logger.info(f"Added layer {len(self.layers)}: {layer.input_size} -> {layer.output_size}, "
                   f"neurons {start_id}-{end_id-1}")
    
    def get_layer_neuron_ids(self, layer_idx: int) -> Tuple[int, int]:
        """Get the neuron ID range for a specific layer."""
        return self.neuron_id_map[layer_idx]
    
    def save_weights(self, filepath: str) -> None:
        """Save model weights to HDF5 file."""
        if h5py is None:
            raise RuntimeError("h5py is required to save model weights")
        with h5py.File(filepath, 'w') as f:
            f.attrs['model_name'] = self.name
            f.attrs['num_layers'] = len(self.layers)
            f.attrs['total_neurons'] = self.total_neurons
            
            for i, layer in enumerate(self.layers):
                layer_grp = f.create_group(f'layer_{i}')
                layer_grp.attrs['input_size'] = layer.input_size
                layer_grp.attrs['output_size'] = layer.output_size
                layer_grp.attrs['layer_type'] = layer.layer_type
                
                # Save layer config (for convolutional layers)
                for key, value in layer.layer_config.items():
                    if isinstance(value, tuple):
                        layer_grp.attrs[f'config_{key}'] = str(value)
                    else:
                        layer_grp.attrs[f'config_{key}'] = value
                
                if layer.weights is not None:
                    layer_grp.create_dataset('weights', data=layer.weights)
                if layer.bias is not None:
                    layer_grp.create_dataset('bias', data=layer.bias)
                
                # Save neuron parameters
                for key, value in layer.neuron_params.items():
                    layer_grp.attrs[f'neuron_{key}'] = value
        
        logger.info(f"Model weights saved to {filepath}")
    
    def load_weights(self, filepath: str) -> None:
        """Load model weights from HDF5 file."""
        if h5py is None:
            raise RuntimeError("h5py is required to load model weights")
        with h5py.File(filepath, 'r') as f:
            model_name = f.attrs['model_name']
            num_layers = f.attrs['num_layers']
            
            self.layers = []
            self.total_neurons = 0
            self.neuron_id_map = {}
            
            for i in range(num_layers):
                layer_grp = f[f'layer_{i}']
                
                # Load layer config
                layer_config = {}
                for attr_name in layer_grp.attrs:
                    if attr_name.startswith('config_'):
                        param_name = attr_name[7:]  # Remove 'config_' prefix
                        value = layer_grp.attrs[attr_name]
                        # Convert string tuples back to tuples
                        if isinstance(value, (str, bytes)):
                            value_str = value.decode() if isinstance(value, bytes) else value
                            if value_str.startswith('(') and value_str.endswith(')'):
                                # Parse tuple
                                import ast
                                value = ast.literal_eval(value_str)
                        layer_config[param_name] = value
                
                # Create layer
                layer = SNNLayer(
                    input_size=layer_grp.attrs['input_size'],
                    output_size=layer_grp.attrs['output_size'],
                    layer_type=layer_grp.attrs['layer_type'].decode() if isinstance(layer_grp.attrs['layer_type'], bytes) else layer_grp.attrs['layer_type'],
                    **layer_config
                )
                
                # Load weights
                if 'weights' in layer_grp:
                    weights = layer_grp['weights'][:]
                    bias = layer_grp['bias'][:] if 'bias' in layer_grp else None
                    layer.set_weights(weights, bias)
                
                # Load neuron parameters
                neuron_params = {}
                for attr_name in layer_grp.attrs:
                    if attr_name.startswith('neuron_'):
                        param_name = attr_name[7:]  # Remove 'neuron_' prefix
                        neuron_params[param_name] = layer_grp.attrs[attr_name]
                
                if neuron_params:
                    layer.neuron_params = neuron_params
                
                self.add_layer(layer)
        
        logger.info(f"Model weights loaded from {filepath}")


def pytorch_to_snn(torch_model: nn.Module, input_shape: Tuple[int, ...],
                  conversion_params: Optional[Dict] = None) -> SNNModel:
    """
    Convert a PyTorch model to SNN format.
    
    Args:
        torch_model: PyTorch model to convert
        input_shape: Input tensor shape (excluding batch dimension)
        conversion_params: Parameters for conversion process
        
    Returns:
        SNNModel ready for FPGA deployment
    """
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for model conversion")
    if conversion_params is None:
        conversion_params = {
            'weight_scale': 128.0,
            'threshold_scale': 1.0,
            'leak_rate': 0.1,
            'refractory_period': 5
        }
    
    snn_model = SNNModel(name=f"converted_{torch_model.__class__.__name__}")
    
    # Set model to evaluation mode
    torch_model.eval()
    
    # Determine if input is flattened or not
    if len(input_shape) == 1:
        # Flattened input for fully connected layers
        current_size = input_shape[0]
        is_flattened = True
        input_channels = input_h = input_w = 0  # Will be set if needed
    else:
        # Spatial input for convolutional layers (C, H, W)
        current_size = int(np.prod(input_shape))
        is_flattened = False
        input_channels, input_h, input_w = input_shape[0], input_shape[-2], input_shape[-1]
    
    # Convert each layer
    layer_count = 0
    for name, module in torch_model.named_modules():
        if isinstance(module, nn.Linear):
            logger.info(f"Converting Linear layer: {name}")
            
            # Create SNN layer
            snn_layer = SNNLayer(
                input_size=current_size,
                output_size=module.out_features,
                layer_type="fully_connected"
            )
            
            # Convert weights
            weights = module.weight.detach().numpy()
            bias = module.bias.detach().numpy() if module.bias is not None else None
            
            # Scale weights for fixed-point representation
            weights_scaled = weights * conversion_params['weight_scale']
            weights_scaled = np.clip(weights_scaled, -128, 127)
            
            if bias is not None:
                bias_scaled = bias * conversion_params['weight_scale']
                bias_scaled = np.clip(bias_scaled, -128, 127)
            else:
                bias_scaled = None
            
            snn_layer.set_weights(weights_scaled, bias_scaled)
            
            # Set neuron parameters
            snn_layer.set_neuron_parameters(
                threshold=conversion_params['threshold_scale'],
                leak_rate=conversion_params['leak_rate'],
                refractory_period=conversion_params['refractory_period']
            )
            
            snn_model.add_layer(snn_layer)
            current_size = module.out_features
            layer_count += 1
            
        elif isinstance(module, nn.Conv2d):
            logger.info(f"Converting Conv2d layer: {name}")
            
            # Get convolution parameters
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
            stride = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
            padding = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
            
            # Calculate output feature map size
            # Assuming input shape is (C, H, W)
            if not is_flattened:
                # Use the spatial dimensions from input_shape
                current_h, current_w = input_h, input_w
            else:
                # Try to infer from flattened size
                current_h = current_w = int(np.sqrt(current_size // in_channels))
            
            # Calculate output spatial dimensions
            output_h = (current_h + 2 * int(padding[0]) - int(kernel_size[0])) // int(stride[0]) + 1
            output_w = (current_w + 2 * int(padding[1]) - int(kernel_size[1])) // int(stride[1]) + 1
            
            # Create SNN convolutional layer
            snn_layer = SNNLayer(
                input_size=in_channels * current_h * current_w,  # Flattened input
                output_size=out_channels * output_h * output_w,  # Flattened output
                layer_type="convolutional",
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                input_shape=(in_channels, current_h, current_w),
                output_shape=(out_channels, output_h, output_w)
            )
            
            # Convert weights: (out_channels, in_channels, kH, kW)
            weights = module.weight.detach().numpy()
            bias = module.bias.detach().numpy() if module.bias is not None else None
            
            # Scale weights for fixed-point representation
            weights_scaled = weights * conversion_params['weight_scale']
            weights_scaled = np.clip(weights_scaled, -128, 127)
            
            if bias is not None:
                bias_scaled = bias * conversion_params['weight_scale']
                bias_scaled = np.clip(bias_scaled, -128, 127)
            else:
                bias_scaled = None
            
            snn_layer.set_weights(weights_scaled, bias_scaled)
            
            # Set neuron parameters
            snn_layer.set_neuron_parameters(
                threshold=conversion_params['threshold_scale'],
                leak_rate=conversion_params['leak_rate'],
                refractory_period=conversion_params['refractory_period']
            )
            
            snn_model.add_layer(snn_layer)
            
            # Update current size and spatial dimensions for next layer
            current_size = out_channels * output_h * output_w
            input_h, input_w = output_h, output_w
            input_channels = out_channels
            layer_count += 1
            
        elif isinstance(module, nn.ReLU):
            # ReLU is handled by neuron threshold, no explicit conversion needed
            logger.info(f"ReLU activation {name} handled by neuron dynamics")
    
    logger.info(f"Converted {layer_count} layers to SNN format")
    return snn_model


def load_pytorch_weights(model_path: str, device: str = 'cpu') -> nn.Module:
    """
    Load PyTorch model from file.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
        
    Returns:
        Loaded PyTorch model
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to load PyTorch models")
    try:
        if model_path.endswith('.pth') or model_path.endswith('.pt'):
            # Load state dict
            state_dict = torch.load(model_path, map_location=device)
            
            # Try to infer model architecture (this is simplified)
            # In practice, you'd want to save the model architecture too
            logger.warning("Model architecture inference not implemented. "
                         "Please provide model definition separately.")
            return state_dict
        else:
            # Load full model
            model = torch.load(model_path, map_location=device)
            return model
            
    except Exception as e:
        logger.error(f"Failed to load PyTorch model: {e}")
        raise


def create_feedforward_snn(layer_sizes: List[int], 
                          neuron_params: Optional[Dict] = None) -> SNNModel:
    """
    Create a feedforward SNN model.
    
    Args:
        layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        neuron_params: Neuron parameters to use
        
    Returns:
        SNNModel with specified architecture
    """
    if neuron_params is None:
        neuron_params = {
            'threshold': 1.0,
            'leak_rate': 0.1,
            'refractory_period': 5
        }
    
    snn_model = SNNModel(name="feedforward_snn")
    
    for i in range(len(layer_sizes) - 1):
        layer = SNNLayer(
            input_size=layer_sizes[i],
            output_size=layer_sizes[i + 1],
            layer_type="fully_connected"
        )
        
        # Initialize random weights
        weights = np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * 0.1
        weights = np.clip(weights * 128, -128, 127)  # Scale and clip
        
        layer.set_weights(weights)
        layer.set_neuron_parameters(**neuron_params)
        
        snn_model.add_layer(layer)
    
    logger.info(f"Created feedforward SNN: {' -> '.join(map(str, layer_sizes))}")
    return snn_model


def simulate_snn_inference(snn_model: SNNModel, input_spikes: List[SpikeEvent],
                          duration: float = 0.1) -> List[SpikeEvent]:
    """
    Simulate SNN inference in software (for testing/debugging).
    
    Args:
        snn_model: SNN model to simulate
        input_spikes: Input spike events
        duration: Simulation duration
        
    Returns:
        Output spike events
    """
    # This is a simplified software simulation for testing
    # The actual computation should be done on FPGA
    
    logger.info("Running software SNN simulation for testing")
    
    # Initialize neuron states
    neuron_states = {}
    for i in range(snn_model.total_neurons):
        neuron_states[i] = {
            'membrane_potential': 0.0,
            'last_spike_time': -float('inf'),
            'refractory_until': 0.0
        }
    
    # Process input spikes layer by layer (simplified)
    current_spikes = input_spikes
    
    for layer_idx, layer in enumerate(snn_model.layers):
        layer_output_spikes = []
        start_id, end_id = snn_model.get_layer_neuron_ids(layer_idx)
        
        # Skip layers without weights (e.g., pooling)
        if layer.weights is None:
            current_spikes = current_spikes  # Pass through
            continue
        
        # Check if this is a convolutional layer
        is_conv = len(layer.weights.shape) == 4
        
        if is_conv:
            # For convolutional layers, simplified simulation: just pass spikes through
            # with some random weight contribution
            # Full convolution simulation would require spatial indexing
            for spike in current_spikes:
                # Generate output spikes for a subset of neurons
                num_outputs = min(layer.output_size, 10)  # Limit for simulation
                for neuron_idx in range(num_outputs):
                    if np.random.rand() > 0.5:  # Random activation
                        global_neuron_id = start_id + neuron_idx
                        layer_output_spikes.append(SpikeEvent(
                            neuron_id=global_neuron_id,
                            timestamp=spike.timestamp + 0.001,
                            weight=1.0,
                            layer_id=layer_idx
                        ))
        else:
            # Fully connected layer - proper simulation
            for spike in current_spikes:
                if spike.neuron_id < layer.input_size:
                    for neuron_idx in range(layer.output_size):
                        global_neuron_id = start_id + neuron_idx
                        weight = layer.weights[neuron_idx, spike.neuron_id]
                        
                        # Simple integrate-and-fire simulation
                        neuron_states[global_neuron_id]['membrane_potential'] += weight / 128.0
                        
                        # Check for spike
                        threshold = layer.neuron_params.get('threshold', 1.0)
                        if neuron_states[global_neuron_id]['membrane_potential'] >= threshold:
                            # Generate output spike
                            layer_output_spikes.append(SpikeEvent(
                                neuron_id=global_neuron_id,
                                timestamp=spike.timestamp + 0.001,  # Small delay
                                weight=1.0,
                                layer_id=layer_idx
                            ))
                            
                            # Reset neuron
                            neuron_states[global_neuron_id]['membrane_potential'] = 0.0
        
        current_spikes = layer_output_spikes
    
    logger.info(f"Software simulation complete: {len(current_spikes)} output spikes")
    return current_spikes


class CPUvsSNNComparator:
    """
    Compare inference results between CPU (traditional ANN) and SNN simulation.
    
    This utility helps validate that the SNN conversion maintains model accuracy
    and provides insights into the behavior differences between rate-coded SNNs
    and traditional ANNs.
    
    Example
    -------
    >>> comparator = CPUvsSNNComparator(torch_model, snn_model)
    >>> results = comparator.compare(input_data)
    >>> comparator.print_report(results)
    """
    
    def __init__(
        self,
        torch_model: Optional['nn.Module'] = None,
        snn_model: Optional[SNNModel] = None,
        device: str = 'cpu'
    ):
        """
        Initialize comparator.
        
        Parameters
        ----------
        torch_model : nn.Module, optional
            PyTorch model for CPU inference. If None, will create from snn_model.
        snn_model : SNNModel, optional
            SNN model for neuromorphic simulation. If None, will convert from torch_model.
        device : str
            Device for PyTorch inference ('cpu' or 'cuda').
        """
        self.torch_model = torch_model
        self.snn_model = snn_model
        self.device = device
        self._encoder = None
        
    def set_models(
        self,
        torch_model: Optional['nn.Module'] = None,
        snn_model: Optional[SNNModel] = None
    ) -> None:
        """Set or update models for comparison."""
        if torch_model is not None:
            self.torch_model = torch_model
        if snn_model is not None:
            self.snn_model = snn_model
            
    def _run_cpu_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on CPU using PyTorch model."""
        if torch is None:
            raise RuntimeError("PyTorch is required for CPU inference")
        if self.torch_model is None:
            raise RuntimeError("PyTorch model not set. Use set_models() first.")
            
        self.torch_model.eval()
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_data).float().to(self.device)
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            output = self.torch_model(input_tensor)
            return output.cpu().numpy()
    
    def _run_snn_inference(
        self,
        input_data: np.ndarray,
        duration: float = 0.1,
        max_rate: float = 100.0,
        num_repeats: int = 1
    ) -> Tuple[np.ndarray, List['SpikeEvent']]:
        """Run inference using SNN simulation."""
        if self.snn_model is None:
            raise RuntimeError("SNN model not set. Use set_models() first.")
        
        from .spike_encoding import PoissonEncoder
        
        # Flatten input if needed
        flat_input = input_data.flatten()
        
        # Encode to spikes
        encoder = PoissonEncoder(
            num_neurons=len(flat_input),
            duration=duration,
            max_rate=max_rate
        )
        
        all_output_spikes = []
        spike_counts = np.zeros(self.snn_model.layers[-1].output_size)
        
        for _ in range(num_repeats):
            spikes = encoder.encode(flat_input)
            output_spikes = simulate_snn_inference(self.snn_model, spikes, duration)
            all_output_spikes.extend(output_spikes)
            
            # Count output spikes per neuron
            output_start_id = self.snn_model.total_neurons - self.snn_model.layers[-1].output_size
            for spike in output_spikes:
                if spike.neuron_id >= output_start_id:
                    spike_counts[spike.neuron_id - output_start_id] += 1
        
        # Normalize by number of repeats
        spike_rates = spike_counts / num_repeats
        
        return spike_rates, all_output_spikes
    
    def compare(
        self,
        input_data: np.ndarray,
        duration: float = 0.1,
        max_rate: float = 100.0,
        num_repeats: int = 5
    ) -> Dict[str, Any]:
        """
        Compare CPU and SNN inference results.
        
        Parameters
        ----------
        input_data : np.ndarray
            Input data for inference.
        duration : float
            SNN simulation duration in seconds.
        max_rate : float
            Maximum spike rate for encoding.
        num_repeats : int
            Number of SNN inference repeats for averaging.
            
        Returns
        -------
        dict
            Comparison results including:
            - cpu_output: Raw CPU model output
            - cpu_prediction: Predicted class from CPU
            - snn_spike_rates: Spike rates from SNN
            - snn_prediction: Predicted class from SNN
            - snn_output_spikes: List of output spike events
            - agreement: Whether predictions match
            - cpu_confidence: Confidence of CPU prediction
            - snn_confidence: Confidence of SNN prediction
            - correlation: Correlation between outputs
        """
        import time
        
        results = {
            'input_shape': input_data.shape,
            'duration': duration,
            'max_rate': max_rate,
            'num_repeats': num_repeats,
        }
        
        # CPU inference
        if self.torch_model is not None:
            start_time = time.time()
            cpu_output = self._run_cpu_inference(input_data)
            results['cpu_time_ms'] = (time.time() - start_time) * 1000
            results['cpu_output'] = cpu_output.flatten()
            
            # Softmax for probabilities
            cpu_exp = np.exp(cpu_output - np.max(cpu_output))
            cpu_probs = cpu_exp / cpu_exp.sum()
            results['cpu_probabilities'] = cpu_probs.flatten()
            results['cpu_prediction'] = int(np.argmax(cpu_output))
            results['cpu_confidence'] = float(np.max(cpu_probs))
        else:
            results['cpu_output'] = None
            results['cpu_prediction'] = None
            results['cpu_confidence'] = None
            results['cpu_time_ms'] = None
        
        # SNN inference
        if self.snn_model is not None:
            start_time = time.time()
            snn_rates, snn_spikes = self._run_snn_inference(
                input_data, duration, max_rate, num_repeats
            )
            results['snn_time_ms'] = (time.time() - start_time) * 1000
            results['snn_spike_rates'] = snn_rates
            results['snn_output_spikes'] = snn_spikes
            results['snn_total_output_spikes'] = len(snn_spikes)
            
            # Normalize rates to probabilities
            if snn_rates.sum() > 0:
                snn_probs = snn_rates / snn_rates.sum()
                results['snn_probabilities'] = snn_probs
                results['snn_prediction'] = int(np.argmax(snn_rates))
                results['snn_confidence'] = float(np.max(snn_probs))
            else:
                results['snn_probabilities'] = np.zeros_like(snn_rates)
                results['snn_prediction'] = None
                results['snn_confidence'] = 0.0
        else:
            results['snn_spike_rates'] = None
            results['snn_prediction'] = None
            results['snn_confidence'] = None
            results['snn_time_ms'] = None
        
        # Compute comparison metrics
        if results['cpu_prediction'] is not None and results['snn_prediction'] is not None:
            results['agreement'] = results['cpu_prediction'] == results['snn_prediction']
            
            # Correlation between output distributions
            if results['cpu_output'] is not None and results['snn_spike_rates'] is not None:
                cpu_norm = results['cpu_output'] - results['cpu_output'].mean()
                snn_norm = results['snn_spike_rates'] - results['snn_spike_rates'].mean()
                
                if np.std(cpu_norm) > 0 and np.std(snn_norm) > 0:
                    correlation = np.corrcoef(cpu_norm, snn_norm)[0, 1]
                    results['correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
                else:
                    results['correlation'] = 0.0
        else:
            results['agreement'] = None
            results['correlation'] = None
            
        return results
    
    def compare_batch(
        self,
        inputs: np.ndarray,
        labels: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare CPU and SNN inference on a batch of inputs.
        
        Parameters
        ----------
        inputs : np.ndarray
            Batch of inputs with shape (N, ...).
        labels : np.ndarray, optional
            Ground truth labels for accuracy calculation.
        **kwargs
            Additional arguments passed to compare().
            
        Returns
        -------
        dict
            Batch comparison results including accuracy metrics.
        """
        batch_size = inputs.shape[0]
        
        cpu_predictions = []
        snn_predictions = []
        agreements = []
        correlations = []
        
        for i in range(batch_size):
            result = self.compare(inputs[i], **kwargs)
            
            if result['cpu_prediction'] is not None:
                cpu_predictions.append(result['cpu_prediction'])
            if result['snn_prediction'] is not None:
                snn_predictions.append(result['snn_prediction'])
            if result['agreement'] is not None:
                agreements.append(result['agreement'])
            if result['correlation'] is not None:
                correlations.append(result['correlation'])
        
        batch_results = {
            'batch_size': batch_size,
            'cpu_predictions': np.array(cpu_predictions) if cpu_predictions else None,
            'snn_predictions': np.array(snn_predictions) if snn_predictions else None,
            'agreement_rate': np.mean(agreements) if agreements else None,
            'mean_correlation': np.mean(correlations) if correlations else None,
            'std_correlation': np.std(correlations) if correlations else None,
        }
        
        if labels is not None:
            if cpu_predictions:
                batch_results['cpu_accuracy'] = np.mean(
                    np.array(cpu_predictions) == labels[:len(cpu_predictions)]
                )
            if snn_predictions:
                batch_results['snn_accuracy'] = np.mean(
                    np.array(snn_predictions) == labels[:len(snn_predictions)]
                )
                
        return batch_results
    
    @staticmethod
    def print_report(results: Dict[str, Any], verbose: bool = True) -> None:
        """
        Print a formatted comparison report.
        
        Parameters
        ----------
        results : dict
            Results from compare() or compare_batch().
        verbose : bool
            Whether to print detailed information.
        """
        print("\n" + "=" * 60)
        print("CPU vs SNN Comparison Report")
        print("=" * 60)
        
        # Single sample results
        if 'cpu_output' in results:
            print("\n[Input]")
            print(f"  Shape: {results.get('input_shape', 'N/A')}")
            print(f"  SNN Duration: {results.get('duration', 'N/A')}s")
            print(f"  Max Spike Rate: {results.get('max_rate', 'N/A')} Hz")
            
            print("\n[CPU Inference (PyTorch)]")
            if results.get('cpu_prediction') is not None:
                print(f"  Prediction: {results['cpu_prediction']}")
                print(f"  Confidence: {results['cpu_confidence']:.4f}")
                print(f"  Time: {results.get('cpu_time_ms', 'N/A'):.2f} ms")
                if verbose and results.get('cpu_probabilities') is not None:
                    print(f"  Probabilities: {results['cpu_probabilities']}")
            else:
                print("  (Not available)")
                
            print("\n[SNN Inference (Neuromorphic Simulation)]")
            if results.get('snn_prediction') is not None:
                print(f"  Prediction: {results['snn_prediction']}")
                print(f"  Confidence: {results['snn_confidence']:.4f}")
                print(f"  Time: {results.get('snn_time_ms', 'N/A'):.2f} ms")
                print(f"  Total Output Spikes: {results.get('snn_total_output_spikes', 0)}")
                if verbose and results.get('snn_spike_rates') is not None:
                    print(f"  Spike Rates: {results['snn_spike_rates']}")
            else:
                print("  (Not available - no output spikes)")
                
            print("\n[Comparison]")
            if results.get('agreement') is not None:
                status = "MATCH" if results['agreement'] else "MISMATCH"
                print(f"  Agreement: {status}")
                print(f"  Output Correlation: {results.get('correlation', 'N/A'):.4f}")
            else:
                print("  (Cannot compare - missing results)")
                
        # Batch results
        if 'batch_size' in results:
            print(f"\n[Batch Summary]")
            print(f"  Batch Size: {results['batch_size']}")
            
            if results.get('agreement_rate') is not None:
                print(f"  Agreement Rate: {results['agreement_rate']*100:.2f}%")
            if results.get('mean_correlation') is not None:
                print(f"  Mean Correlation: {results['mean_correlation']:.4f} +/- {results.get('std_correlation', 0):.4f}")
            if results.get('cpu_accuracy') is not None:
                print(f"  CPU Accuracy: {results['cpu_accuracy']*100:.2f}%")
            if results.get('snn_accuracy') is not None:
                print(f"  SNN Accuracy: {results['snn_accuracy']*100:.2f}%")
                
        print("\n" + "=" * 60)


if torch is not None and nn is not None:

    class TorchSNNLayer(nn.Module):
        """PyTorch-compatible SNN layer for training prior to FPGA deployment."""

        def __init__(
            self,
            in_features: int,
            out_features: int,
            tau_mem: float = 20.0,
            tau_syn: float = 5.0,
            threshold: float = 1.0,
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.tau_mem = tau_mem
            self.tau_syn = tau_syn
            self.threshold = threshold

            # Learnable parameters
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
            self.bias = nn.Parameter(torch.zeros(out_features))

            # State variables (reset for each sequence)
            self.register_buffer('mem', torch.zeros(1, out_features))
            self.register_buffer('syn', torch.zeros(1, out_features))
            self.register_buffer('last_spike', torch.zeros(1, out_features))

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':  # type: ignore[override]
            """Forward pass with LIF dynamics."""
            with torch.no_grad():
                batch_size = x.size(0)

                # Resize state variables if needed
                if self.mem.size(0) != batch_size:
                    self.mem = torch.zeros(batch_size, self.out_features, device=x.device)
                    self.syn = torch.zeros(batch_size, self.out_features, device=x.device)
                    self.last_spike = torch.zeros(batch_size, self.out_features, device=x.device)

                syn_input = torch.matmul(x, self.weight.t()) + self.bias
                self.syn = self.syn + (-self.syn + syn_input) / self.tau_syn

                self.mem = self.mem + (-self.mem + self.syn) / self.tau_mem

                spike: 'torch.Tensor' = spike_function(self.mem - self.threshold)

                self.last_spike = spike
                self.mem = self.mem * (1 - spike)

            return spike

        def reset_state(self) -> None:
            """Reset neuron states."""
            self.mem.zero_()
            self.syn.zero_()
            self.last_spike.zero_()

    def _spike_function_impl(x: 'torch.Tensor') -> 'torch.Tensor':
        """Hard threshold without surrogate gradients."""
        return (x > 0).float()

    spike_function = _spike_function_impl

else:

    class _TorchSNNLayerStub:
        """Stub layer used when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is required to use TorchSNNLayer")

        def reset_state(self) -> None:
            raise RuntimeError("PyTorch is required to use TorchSNNLayer")


    TorchSNNLayer = _TorchSNNLayerStub  # type: ignore[assignment]

    def _spike_function_stub(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("PyTorch is required to evaluate spike_function")

    spike_function = _spike_function_stub
