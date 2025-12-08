"""
FPGA Deployment Utilities

Convert trained models to FPGA-compatible format.
Handles weight quantization, configuration generation, and deployment.

Usage:
    from snn_fpga_accelerator import deploy
    
    # Export model for FPGA
    deploy.export(model, 'weights.npz')
    
    # Generate HW config
    config = deploy.gen_config(model)
    
    # Deploy to PYNQ
    fpga = deploy.PYNQ('snn.bit')
    fpga.load_weights(model)
    output = fpga.run(input)

Author: Jiwoon Lee (@metr0jw)
"""

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field, asdict
import json
import struct
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

__all__ = [
    # Quantization
    'quantize', 'dequantize', 'calibrate',
    # Export
    'export', 'export_onnx',
    # Config generation  
    'gen_config', 'LayerConfig', 'NetworkConfig',
    # FPGA interface
    'PYNQ', 'AXIInterface',
]


# =============================================================================
# Quantization
# =============================================================================

@dataclass
class QuantConfig:
    """Quantization configuration."""
    weight_bits: int = 8
    activation_bits: int = 16
    membrane_bits: int = 16
    symmetric: bool = True
    per_channel: bool = False


def quantize(
    tensor: Union[Tensor, np.ndarray],
    bits: int = 8,
    symmetric: bool = True,
    scale: float = None,
    zero_point: int = 0,
) -> Tuple[np.ndarray, float, int]:
    """
    Quantize tensor to fixed-point.
    
    Args:
        tensor: Float tensor to quantize
        bits: Bit width
        symmetric: Symmetric quantization (zero_point=0)
        scale: Scale factor (computed if None)
        zero_point: Zero point offset
        
    Returns:
        (quantized_tensor, scale, zero_point)
        
    Examples:
        >>> q_weights, scale, zp = quantize(weights, bits=8)
    """
    if isinstance(tensor, Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    if symmetric:
        # Symmetric quantization: [-max, max] -> [-2^(n-1), 2^(n-1)-1]
        max_val = max(abs(tensor.max()), abs(tensor.min()))
        qmax = 2 ** (bits - 1) - 1
        qmin = -2 ** (bits - 1)
        
        if scale is None:
            scale = max_val / qmax if max_val > 0 else 1.0
        
        zero_point = 0
    else:
        # Asymmetric quantization
        min_val, max_val = tensor.min(), tensor.max()
        qmax = 2 ** bits - 1
        qmin = 0
        
        if scale is None:
            scale = (max_val - min_val) / qmax if (max_val - min_val) > 0 else 1.0
            zero_point = int(-min_val / scale)
    
    # Quantize
    q_tensor = np.round(tensor / scale + zero_point).clip(qmin, qmax).astype(np.int8 if bits == 8 else np.int16)
    
    return q_tensor, scale, zero_point


def dequantize(
    q_tensor: np.ndarray,
    scale: float,
    zero_point: int = 0,
) -> np.ndarray:
    """Dequantize fixed-point to float."""
    return (q_tensor.astype(np.float32) - zero_point) * scale


def calibrate(
    model: nn.Module,
    dataloader,
    num_batches: int = 10,
) -> Dict[str, QuantConfig]:
    """
    Calibrate quantization ranges using calibration data.
    
    Args:
        model: Model to calibrate
        dataloader: Calibration data
        num_batches: Number of batches to use
        
    Returns:
        Per-layer quantization configs
    """
    configs = {}
    
    # Collect activation statistics
    activation_ranges = {}
    
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, Tensor):
                if name not in activation_ranges:
                    activation_ranges[name] = {'min': float('inf'), 'max': float('-inf')}
                activation_ranges[name]['min'] = min(
                    activation_ranges[name]['min'],
                    output.min().item()
                )
                activation_ranges[name]['max'] = max(
                    activation_ranges[name]['max'],
                    output.max().item()
                )
        return hook
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Run calibration
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            _ = model(data)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Create configs
    for name in activation_ranges:
        configs[name] = QuantConfig()
    
    return configs


# =============================================================================
# Export Functions
# =============================================================================

def export(
    model: nn.Module,
    path: str,
    format: str = 'npz',
    quantize: bool = True,
    bits: int = 8,
) -> Dict[str, any]:
    """
    Export model weights for FPGA deployment.
    
    Args:
        model: Trained model
        path: Output path
        format: 'npz', 'bin', or 'json'
        quantize: Quantize weights
        bits: Quantization bits
        
    Returns:
        Weight dictionary with metadata
        
    Examples:
        >>> export(model, 'weights.npz')
        >>> export(model, 'weights.bin', format='bin')
    """
    weights = {}
    metadata = {
        'layers': [],
        'quantize': quantize,
        'bits': bits,
    }
    
    for name, param in model.named_parameters():
        if 'weight' not in name and 'bias' not in name:
            continue
        
        w = param.detach().cpu().numpy()
        
        if quantize:
            w_q, scale, zp = globals()['quantize'](w, bits=bits)
            weights[name] = w_q
            metadata['layers'].append({
                'name': name,
                'shape': list(w.shape),
                'scale': float(scale),
                'zero_point': int(zp),
            })
        else:
            weights[name] = w
            metadata['layers'].append({
                'name': name,
                'shape': list(w.shape),
            })
    
    # Save
    path = Path(path)
    
    if format == 'npz':
        np.savez(path, **weights, metadata=json.dumps(metadata))
    
    elif format == 'bin':
        # Binary format for direct FPGA loading
        with open(path, 'wb') as f:
            # Header
            f.write(struct.pack('I', len(weights)))  # num layers
            
            for name, w in weights.items():
                # Layer header
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('I', len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack('I', len(w.shape)))
                for dim in w.shape:
                    f.write(struct.pack('I', dim))
                
                # Weight data
                f.write(w.tobytes())
    
    elif format == 'json':
        # JSON with base64 encoded weights
        import base64
        json_data = {
            'metadata': metadata,
            'weights': {
                name: base64.b64encode(w.tobytes()).decode('ascii')
                for name, w in weights.items()
            }
        }
        with open(path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    logger.info(f"Exported {len(weights)} layers to {path}")
    return {'weights': weights, 'metadata': metadata}


def export_onnx(model: nn.Module, path: str, input_shape: Tuple[int, ...]):
    """Export model to ONNX format."""
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(model, dummy_input, path)
    logger.info(f"Exported ONNX model to {path}")


# =============================================================================
# Hardware Configuration
# =============================================================================

@dataclass
class LayerConfig:
    """Configuration for a single layer."""
    type: str                    # 'linear', 'conv2d', 'lif', etc.
    in_features: int = 0
    out_features: int = 0
    kernel_size: int = 0
    stride: int = 1
    padding: int = 0
    threshold: int = 32768       # Q8.8 format (1.0 = 32768)
    leak: int = 230              # 0.9 in Q8.8
    weight_base_addr: int = 0
    weight_size: int = 0


@dataclass
class NetworkConfig:
    """Full network configuration for FPGA."""
    layers: List[LayerConfig] = field(default_factory=list)
    total_weights: int = 0
    max_neurons: int = 0
    timesteps: int = 100
    hw_version: str = '1.0'
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self, path: str = None) -> str:
        data = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, 'w') as f:
                f.write(data)
        return data
    
    def to_bin(self, path: str):
        """Export as binary config for FPGA."""
        with open(path, 'wb') as f:
            # Header
            f.write(struct.pack('4s', b'SNN\x00'))
            f.write(struct.pack('I', len(self.layers)))
            f.write(struct.pack('I', self.timesteps))
            
            # Layer configs
            for layer in self.layers:
                f.write(struct.pack(
                    'IIIIIIII',
                    0 if layer.type == 'linear' else 1,  # type
                    layer.in_features,
                    layer.out_features,
                    layer.threshold,
                    layer.leak,
                    layer.weight_base_addr,
                    layer.weight_size,
                    0  # reserved
                ))


def gen_config(model: nn.Module, timesteps: int = 100) -> NetworkConfig:
    """
    Generate FPGA configuration from model.
    
    Args:
        model: Trained model
        timesteps: Simulation timesteps
        
    Returns:
        NetworkConfig for FPGA deployment
    """
    from .neuron import SpikingNeuron
    from .layer import Linear, Conv2d
    
    config = NetworkConfig(timesteps=timesteps)
    weight_addr = 0
    max_neurons = 0
    
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            weight_size = module.in_features * module.out_features
            config.layers.append(LayerConfig(
                type='linear',
                in_features=module.in_features,
                out_features=module.out_features,
                weight_base_addr=weight_addr,
                weight_size=weight_size,
            ))
            weight_addr += weight_size
            max_neurons = max(max_neurons, module.out_features)
        
        elif isinstance(module, Conv2d):
            weight_size = (module.out_channels * module.in_channels * 
                          module.kernel_size[0] * module.kernel_size[1])
            config.layers.append(LayerConfig(
                type='conv2d',
                in_features=module.in_channels,
                out_features=module.out_channels,
                kernel_size=module.kernel_size[0],
                stride=module.stride[0],
                padding=module.padding[0],
                weight_base_addr=weight_addr,
                weight_size=weight_size,
            ))
            weight_addr += weight_size
        
        elif isinstance(module, SpikingNeuron):
            # Get threshold and tau
            thresh = module.thresh.item() if isinstance(module.thresh, Tensor) else module.thresh
            thresh_int = int(thresh * 32768)  # Q8.8
            
            if hasattr(module, 'tau'):
                tau = module.tau.item() if isinstance(module.tau, Tensor) else module.tau
            else:
                tau = 0.9
            leak_int = int(tau * 256)  # Q0.8
            
            config.layers.append(LayerConfig(
                type='lif',
                threshold=thresh_int,
                leak=leak_int,
            ))
    
    config.total_weights = weight_addr
    config.max_neurons = max_neurons
    
    return config


# =============================================================================
# FPGA Interface
# =============================================================================

class AXIInterface:
    """Low-level AXI interface for FPGA communication."""
    
    def __init__(self, base_addr: int = 0x40000000):
        self.base_addr = base_addr
        self.mmio = None
    
    def connect(self):
        """Connect to FPGA via MMIO."""
        try:
            from pynq import MMIO
            self.mmio = MMIO(self.base_addr, 0x10000)
            logger.info(f"Connected to AXI at {hex(self.base_addr)}")
            return True
        except ImportError:
            logger.warning("PYNQ not available - using simulation mode")
            return False
    
    def write(self, offset: int, value: int):
        """Write to AXI register."""
        if self.mmio:
            self.mmio.write(offset, value)
        else:
            logger.debug(f"[SIM] Write {hex(value)} to {hex(offset)}")
    
    def read(self, offset: int) -> int:
        """Read from AXI register."""
        if self.mmio:
            return self.mmio.read(offset)
        else:
            logger.debug(f"[SIM] Read from {hex(offset)}")
            return 0


class PYNQ:
    """
    PYNQ interface for FPGA deployment.
    
    Handles bitstream loading, weight programming, and inference.
    
    Args:
        bitstream: Path to .bit file
        
    Examples:
        >>> fpga = PYNQ('snn.bit')
        >>> fpga.load_weights(model)
        >>> output = fpga.run(input, T=100)
    """
    
    # Register offsets
    REG_CONTROL = 0x00
    REG_STATUS = 0x04
    REG_CONFIG = 0x08
    REG_TIMESTEPS = 0x0C
    REG_WEIGHT_ADDR = 0x10
    REG_WEIGHT_DATA = 0x14
    REG_SPIKE_IN = 0x18
    REG_SPIKE_OUT = 0x1C
    
    def __init__(self, bitstream: str = None, sim_mode: bool = False):
        self.bitstream = bitstream
        self.sim_mode = sim_mode
        self.overlay = None
        self.axi = AXIInterface()
        
        if not sim_mode and bitstream:
            self._load_bitstream()
    
    def _load_bitstream(self):
        """Load FPGA bitstream."""
        try:
            from pynq import Overlay
            self.overlay = Overlay(self.bitstream)
            self.axi.connect()
            logger.info(f"Loaded bitstream: {self.bitstream}")
        except ImportError:
            logger.warning("PYNQ not available - running in simulation mode")
            self.sim_mode = True
        except Exception as e:
            logger.error(f"Failed to load bitstream: {e}")
            self.sim_mode = True
    
    def load_weights(self, model: nn.Module, quantize: bool = True):
        """
        Program weights to FPGA.
        
        Args:
            model: Trained model
            quantize: Quantize weights to 8-bit
        """
        addr = 0
        
        for name, param in model.named_parameters():
            if 'weight' not in name:
                continue
            
            w = param.detach().cpu().numpy()
            
            if quantize:
                w_q, _, _ = globals()['quantize'](w, bits=8)
                w = w_q
            
            # Write weights
            w_flat = w.flatten()
            for i, val in enumerate(w_flat):
                self.axi.write(self.REG_WEIGHT_ADDR, addr + i)
                self.axi.write(self.REG_WEIGHT_DATA, int(val) & 0xFF)
            
            addr += len(w_flat)
            logger.debug(f"Loaded {name}: {w.shape} at addr {addr - len(w_flat)}")
        
        logger.info(f"Loaded {addr} weights to FPGA")
    
    def configure(self, config: NetworkConfig):
        """Configure FPGA with network parameters."""
        self.axi.write(self.REG_TIMESTEPS, config.timesteps)
        # Additional configuration...
        logger.info(f"Configured FPGA: {config.timesteps} timesteps")
    
    def run(
        self,
        input: Union[Tensor, np.ndarray],
        T: int = 100,
        return_spikes: bool = False,
        encoding: str = "rate",
        weight: int = 1,
    ) -> Union[Tensor, np.ndarray]:
        """
        Run inference on FPGA.
        
        Args:
            input: Input data (float tensor/ndarray), expected in [0,1] or normalized range
            T: Number of timesteps (1..1024)
            return_spikes: Return spike train or final counts
            encoding: 'rate' | 'temporal' | 'phase' (aligned with on-chip encoder semantics)
            weight: spike weight (8-bit signed)
            
        Returns:
            Output predictions or spikes
        """
        if isinstance(input, Tensor):
            input = input.detach().cpu().numpy()
        
        batch_size = input.shape[0]
        self.axi.write(self.REG_TIMESTEPS, T)
        
        outputs = []
        
        for b in range(batch_size):
            sample = input[b].flatten()
            # Prepare AER packet list for this sample
            aer_packets = self._encode_sample_to_aer(sample, T, encoding=encoding, weight=weight)
            
            # Start inference
            self.axi.write(self.REG_CONTROL, 0x01)
            
            # Feed AER stream to FPGA
            for pkt in aer_packets:
                self.axi.write(self.REG_SPIKE_IN, pkt)
            
            # Wait for completion (status bit0 asserted when done)
            while self.axi.read(self.REG_STATUS) & 0x01 == 0:
                pass
            
            # Read output (placeholder; real design may need burst read)
            output = self.axi.read(self.REG_SPIKE_OUT)
            outputs.append(output)
        
        return np.array(outputs)

    # ------------------------------------------------------------------
    # Host-side AER encoder (mirrors HLS encoder packing)
    # Packet: [31:18]=timestamp(14b), [17:10]=weight(8b, two's complement), [9:0]=neuron_id
    # ------------------------------------------------------------------
    def _encode_sample_to_aer(self, sample: np.ndarray, T: int, encoding: str = "rate", weight: int = 1):
        encoding = encoding.lower()
        num_channels = min(len(sample), 1024)
        weight_u8 = np.int8(weight).view(np.uint8)
        packets = []
        
        if encoding == "rate":
            for t in range(T):
                rnd = np.random.rand(num_channels)
                fires = rnd < sample[:num_channels]
                ids = np.nonzero(fires)[0]
                for nid in ids:
                    pkt = ((t & 0x3FFF) << 18) | (int(weight_u8) << 10) | (nid & 0x3FF)
                    packets.append(pkt)
        elif encoding == "temporal":
            # time-to-first-spike: earlier spike for larger value
            spike_times = ((1.0 - sample[:num_channels].clip(0, 1)) * (T - 1)).astype(np.int32)
            for nid, st in enumerate(spike_times):
                pkt = ((int(st) & 0x3FFF) << 18) | (int(weight_u8) << 10) | (nid & 0x3FF)
                packets.append(pkt)
        elif encoding == "phase":
            phase_acc = np.zeros(num_channels, dtype=np.int32)
            phase_threshold = 1 << 16
            phase_scale = (sample[:num_channels].clip(0, 1) * (1 << 12)).astype(np.int32)
            for t in range(T):
                phase_acc += phase_scale
                firing = phase_acc >= phase_threshold
                ids = np.nonzero(firing)[0]
                for nid in ids:
                    pkt = ((t & 0x3FFF) << 18) | (int(weight_u8) << 10) | (nid & 0x3FF)
                    packets.append(pkt)
                phase_acc[firing] -= phase_threshold
        else:
            raise ValueError(f"Unsupported encoding mode: {encoding}")
        return packets
    
    def close(self):
        """Release FPGA resources."""
        if self.overlay:
            # Clean up
            pass
        logger.info("FPGA resources released")
