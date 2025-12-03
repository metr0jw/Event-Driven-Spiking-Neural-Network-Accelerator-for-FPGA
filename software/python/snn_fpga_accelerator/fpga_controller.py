"""
FPGA Controller for SNN Layers
Hardware interface for PyTorch-compatible SNN layers on PYNQ-Z2

Author: Jiwoon Lee (@metr0jw)
Organization: Kwangwoon University, Seoul, South Korea
Contact: jwlee@linux.com
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import struct
import time
import logging

logger = logging.getLogger(__name__)

class SNNFPGAController:
    """
    FPGA controller for SNN accelerator hardware
    Provides interface between PyTorch layers and FPGA implementation
    """
    
    def __init__(self, bitstream_path: Optional[str] = None):
        self.bitstream_path = bitstream_path
        self.is_initialized = False
        self.layer_configs = {}
        self.memory_maps = {}
        
        # Hardware configuration constants
        self.AXI_BASE_ADDR = 0x43C00000
        self.MAX_LAYERS = 16
        self.MAX_WEIGHTS_PER_LAYER = 1024 * 1024  # 1M weights
        self.MAX_NEURONS_PER_LAYER = 256 * 256    # 64K neurons
        
        # Layer type IDs for hardware
        self.LAYER_TYPES = {
            'conv1d': 0x00,
            'conv2d': 0x01,
            'avgpool2d': 0x02,
            'maxpool2d': 0x03,
            'dense': 0x04
        }
        
        # Status flags
        self.STATUS_IDLE = 0x00
        self.STATUS_PROCESSING = 0x01
        self.STATUS_DONE = 0x02
        self.STATUS_ERROR = 0xFF
        
        try:
            # Try to import pynq for actual hardware
            from pynq import Overlay, allocate
            self.Overlay = Overlay
            self.allocate = allocate
            self.hardware_available = True
            logger.info("PYNQ hardware support available")
        except ImportError:
            # Fallback to simulation mode
            self.hardware_available = False
            logger.warning("PYNQ not available, running in simulation mode")
    
    def initialize_hardware(self) -> bool:
        """Initialize FPGA hardware with bitstream"""
        try:
            if self.hardware_available and self.bitstream_path:
                self.overlay = self.Overlay(self.bitstream_path)
                
                # Get IP blocks
                self.snn_accelerator = self.overlay.snn_accelerator_top
                
                # Reset the accelerator
                self._write_register('CONTROL', 0x01)  # Reset
                time.sleep(0.1)
                self._write_register('CONTROL', 0x00)  # Release reset
                
                # Check if hardware is responsive
                status = self._read_register('STATUS')
                if status == self.STATUS_IDLE:
                    self.is_initialized = True
                    logger.info("FPGA hardware initialized successfully")
                    return True
                else:
                    logger.error(f"Hardware initialization failed, status: {status}")
                    return False
            else:
                # Simulation mode
                self.is_initialized = True
                logger.info("Running in simulation mode")
                return True
                
        except Exception as e:
            logger.error(f"Hardware initialization error: {e}")
            return False
    
    def configure_layer(self, layer_id: int, config: Dict[str, Any]) -> bool:
        """Configure a single layer on the FPGA"""
        try:
            if layer_id >= self.MAX_LAYERS:
                raise ValueError(f"Layer ID {layer_id} exceeds maximum {self.MAX_LAYERS}")
            
            layer_type = config['layer_type']
            if layer_type not in self.LAYER_TYPES:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            
            # Store configuration
            self.layer_configs[layer_id] = config
            
            if self.hardware_available and self.is_initialized:
                # Configure layer on hardware
                base_addr = self._get_layer_base_address(layer_id)
                
                # Write layer type
                self._write_register_at_addr(base_addr + 0x00, self.LAYER_TYPES[layer_type])
                
                if layer_type == 'conv1d':
                    self._configure_conv1d_layer(base_addr, config)
                elif layer_type == 'conv2d':
                    self._configure_conv2d_layer(base_addr, config)
                elif layer_type == 'avgpool2d':
                    self._configure_avgpool2d_layer(base_addr, config)
                elif layer_type == 'maxpool2d':
                    self._configure_maxpool2d_layer(base_addr, config)
                
                logger.info(f"Layer {layer_id} ({layer_type}) configured successfully")
            else:
                logger.info(f"Layer {layer_id} ({layer_type}) configured in simulation mode")
            
            return True
            
        except Exception as e:
            logger.error(f"Layer configuration error: {e}")
            return False
    
    def configure_network(self, network_config: Dict[str, Any]) -> bool:
        """Configure entire network on FPGA"""
        try:
            num_layers = network_config['num_layers']
            layers = network_config['layers']
            
            if num_layers > self.MAX_LAYERS:
                raise ValueError(f"Network has {num_layers} layers, maximum is {self.MAX_LAYERS}")
            
            # Configure each layer
            for layer_config in layers:
                layer_id = layer_config['layer_id']
                if not self.configure_layer(layer_id, layer_config):
                    return False
            
            # Set network-level parameters
            if self.hardware_available and self.is_initialized:
                self._write_register('NUM_LAYERS', num_layers)
                
                # Configure layer routing
                for i in range(num_layers - 1):
                    src_layer = i
                    dst_layer = i + 1
                    self._write_register(f'LAYER_ROUTING_{src_layer}', dst_layer)
            
            logger.info(f"Network with {num_layers} layers configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Network configuration error: {e}")
            return False
    
    def process_spike_data(self, input_spikes: np.ndarray) -> np.ndarray:
        """Process spike data through the FPGA accelerator"""
        try:
            if not self.is_initialized:
                raise RuntimeError("Hardware not initialized")
            
            # Validate input dimensions
            if len(input_spikes.shape) != 5:
                raise ValueError("Input spikes must have shape (batch, channels, height, width, time)")
            
            batch_size, channels, height, width, time_steps = input_spikes.shape
            
            if self.hardware_available:
                return self._process_on_hardware(input_spikes)
            else:
                return self._process_simulation(input_spikes)
                
        except Exception as e:
            logger.error(f"Spike processing error: {e}")
            return np.zeros_like(input_spikes)
    
    def _process_on_hardware(self, input_spikes: np.ndarray) -> np.ndarray:
        """Process spikes on actual FPGA hardware"""
        batch_size, channels, height, width, time_steps = input_spikes.shape
        
        # Allocate FPGA memory buffers
        input_buffer = self.allocate(
            shape=(batch_size * channels * height * width * time_steps,),
            dtype=np.uint8
        )
        
        # Calculate output size based on network configuration
        output_size = self._calculate_output_size(input_spikes.shape)
        output_buffer = self.allocate(shape=(output_size,), dtype=np.uint8)
        
        try:
            # Convert spikes to packed format
            packed_input = self._pack_spikes(input_spikes)
            input_buffer[:len(packed_input)] = packed_input
            
            # Set input/output buffer addresses
            self._write_register('INPUT_BUFFER_ADDR', input_buffer.physical_address)
            self._write_register('OUTPUT_BUFFER_ADDR', output_buffer.physical_address)
            
            # Set data dimensions
            self._write_register('BATCH_SIZE', batch_size)
            self._write_register('INPUT_CHANNELS', channels)
            self._write_register('INPUT_HEIGHT', height)
            self._write_register('INPUT_WIDTH', width)
            self._write_register('TIME_STEPS', time_steps)
            
            # Start processing
            self._write_register('CONTROL', 0x02)  # Start processing
            
            # Wait for completion
            timeout_count = 0
            max_timeout = 10000  # 10 seconds
            
            while timeout_count < max_timeout:
                status = self._read_register('STATUS')
                if status == self.STATUS_DONE:
                    break
                elif status == self.STATUS_ERROR:
                    raise RuntimeError("FPGA processing error")
                
                time.sleep(0.001)
                timeout_count += 1
            
            if timeout_count >= max_timeout:
                raise RuntimeError("FPGA processing timeout")
            
            # Read output data
            output_data = np.array(output_buffer)
            output_spikes = self._unpack_spikes(output_data, input_spikes.shape)
            
            return output_spikes
            
        finally:
            # Clean up buffers
            del input_buffer
            del output_buffer
    
    def _process_simulation(self, input_spikes: np.ndarray) -> np.ndarray:
        """Simulate processing for development/testing"""
        logger.info("Processing spikes in simulation mode")
        
        # Simple passthrough simulation
        # In real implementation, this would simulate the layer operations
        output_spikes = input_spikes.copy()
        
        # Apply some basic transformations to simulate processing
        for layer_id, config in self.layer_configs.items():
            layer_type = config['layer_type']
            
            if layer_type == 'conv2d':
                # Simulate convolution by adding noise
                noise = np.random.normal(0, 0.1, output_spikes.shape)
                output_spikes = np.clip(output_spikes + noise, 0, 1)
                
            elif layer_type in ['avgpool2d', 'maxpool2d']:
                # Simulate pooling by downsampling
                if output_spikes.shape[2] > 1 and output_spikes.shape[3] > 1:
                    output_spikes = output_spikes[:, :, ::2, ::2, :]
        
        time.sleep(0.1)  # Simulate processing time
        return output_spikes
    
    def _configure_conv1d_layer(self, base_addr: int, config: Dict[str, Any]):
        """Configure Conv1D layer registers"""
        self._write_register_at_addr(base_addr + 0x04, config['in_channels'])
        self._write_register_at_addr(base_addr + 0x08, config['out_channels'])
        self._write_register_at_addr(base_addr + 0x0C, config['kernel_size'])
        self._write_register_at_addr(base_addr + 0x10, config['stride'])
        self._write_register_at_addr(base_addr + 0x14, config['padding'])
        self._write_register_at_addr(base_addr + 0x18, config['threshold'])
        self._write_register_at_addr(base_addr + 0x1C, config['decay_factor'])
        
        # Write weights to weight memory
        weights = config['weights']
        weight_addr = base_addr + 0x1000  # Weight memory offset
        for i, weight in enumerate(weights.flatten()):
            if i < self.MAX_WEIGHTS_PER_LAYER:
                self._write_register_at_addr(weight_addr + i * 4, int(weight))
    
    def _configure_conv2d_layer(self, base_addr: int, config: Dict[str, Any]):
        """Configure Conv2D layer registers"""
        self._write_register_at_addr(base_addr + 0x04, config['in_channels'])
        self._write_register_at_addr(base_addr + 0x08, config['out_channels'])
        self._write_register_at_addr(base_addr + 0x0C, config['kernel_size'])
        self._write_register_at_addr(base_addr + 0x10, config['stride'])
        self._write_register_at_addr(base_addr + 0x14, config['padding'])
        self._write_register_at_addr(base_addr + 0x18, config['threshold'])
        self._write_register_at_addr(base_addr + 0x1C, config['decay_factor'])
        
        # Write weights to weight memory
        weights = config['weights']
        weight_addr = base_addr + 0x1000  # Weight memory offset
        for i, weight in enumerate(weights.flatten()):
            if i < self.MAX_WEIGHTS_PER_LAYER:
                self._write_register_at_addr(weight_addr + i * 4, int(weight))
    
    def _configure_avgpool2d_layer(self, base_addr: int, config: Dict[str, Any]):
        """Configure AvgPool2D layer registers"""
        self._write_register_at_addr(base_addr + 0x04, config['kernel_size'])
        self._write_register_at_addr(base_addr + 0x08, config['stride'])
        self._write_register_at_addr(base_addr + 0x0C, config['padding'])
        self._write_register_at_addr(base_addr + 0x10, config['threshold'])
        self._write_register_at_addr(base_addr + 0x14, config['decay_factor'])
        self._write_register_at_addr(base_addr + 0x18, config['pooling_window_time'])
    
    def _configure_maxpool2d_layer(self, base_addr: int, config: Dict[str, Any]):
        """Configure MaxPool2D layer registers"""
        self._write_register_at_addr(base_addr + 0x04, config['kernel_size'])
        self._write_register_at_addr(base_addr + 0x08, config['stride'])
        self._write_register_at_addr(base_addr + 0x0C, config['padding'])
        self._write_register_at_addr(base_addr + 0x10, int(config['winner_take_all']))
        self._write_register_at_addr(base_addr + 0x14, config['pooling_window_time'])
    
    def _pack_spikes(self, spikes: np.ndarray) -> np.ndarray:
        """Pack spike data into efficient format for FPGA"""
        # Convert to binary and pack 8 spikes per byte
        binary_spikes = (spikes > 0).astype(np.uint8)
        
        # Flatten and pad to multiple of 8
        flat_spikes = binary_spikes.flatten()
        padding = (8 - len(flat_spikes) % 8) % 8
        if padding > 0:
            flat_spikes = np.pad(flat_spikes, (0, padding), mode='constant')
        
        # Pack 8 bits into each byte
        packed = np.zeros(len(flat_spikes) // 8, dtype=np.uint8)
        for i in range(len(packed)):
            byte_val = 0
            for j in range(8):
                if flat_spikes[i * 8 + j]:
                    byte_val |= (1 << j)
            packed[i] = byte_val
        
        return packed
    
    def _unpack_spikes(self, packed_data: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        """Unpack spike data from FPGA format"""
        # Unpack bits from bytes
        unpacked = []
        for byte_val in packed_data:
            for j in range(8):
                unpacked.append((byte_val >> j) & 1)
        
        # Reshape to original dimensions
        total_elements = np.prod(original_shape)
        unpacked = np.array(unpacked[:total_elements], dtype=np.float32)
        
        return unpacked.reshape(original_shape)
    
    def _calculate_output_size(self, input_shape: Tuple[int, ...]) -> int:
        """Calculate output buffer size based on network configuration"""
        # This is a simplified calculation
        # In practice, need to trace through all layers
        batch_size, channels, height, width, time_steps = input_shape
        
        # Assume worst case: no size reduction
        max_output_size = batch_size * channels * height * width * time_steps
        
        # Pack into bytes (8 spikes per byte)
        return (max_output_size + 7) // 8
    
    def _get_layer_base_address(self, layer_id: int) -> int:
        """Get base address for layer configuration registers"""
        return self.AXI_BASE_ADDR + (layer_id * 0x10000)
    
    def _write_register(self, reg_name: str, value: int):
        """Write to a named register"""
        if self.hardware_available and hasattr(self, 'snn_accelerator'):
            setattr(self.snn_accelerator, reg_name, value)
        else:
            logger.debug(f"Simulation: Write {reg_name} = {value}")
    
    def _read_register(self, reg_name: str) -> int:
        """Read from a named register"""
        if self.hardware_available and hasattr(self, 'snn_accelerator'):
            return getattr(self.snn_accelerator, reg_name)
        else:
            logger.debug(f"Simulation: Read {reg_name}")
            return self.STATUS_IDLE  # Default simulation value
    
    def _write_register_at_addr(self, addr: int, value: int):
        """Write to register at specific address"""
        if self.hardware_available and hasattr(self, 'overlay'):
            # Direct memory access
            self.overlay.write(addr, value)
        else:
            logger.debug(f"Simulation: Write addr {addr:08x} = {value}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from FPGA"""
        if self.hardware_available and self.is_initialized:
            return {
                'cycles_executed': self._read_register('CYCLE_COUNT'),
                'spikes_processed': self._read_register('SPIKE_COUNT'),
                'layer_latencies': [
                    self._read_register(f'LAYER_{i}_LATENCY') 
                    for i in range(len(self.layer_configs))
                ],
                'power_consumption': self._read_register('POWER_ESTIMATE'),
                'memory_bandwidth': self._read_register('MEMORY_BW_USED')
            }
        else:
            # Simulation stats
            return {
                'cycles_executed': 1000,
                'spikes_processed': 5000,
                'layer_latencies': [100] * len(self.layer_configs),
                'power_consumption': 250,  # mW
                'memory_bandwidth': 1200   # MB/s
            }
    
    def reset_accelerator(self):
        """Reset the FPGA accelerator"""
        if self.hardware_available and self.is_initialized:
            self._write_register('CONTROL', 0x01)  # Reset
            time.sleep(0.1)
            self._write_register('CONTROL', 0x00)  # Release reset
        
        logger.info("FPGA accelerator reset")
    
    def shutdown(self):
        """Shutdown and cleanup resources"""
        if self.hardware_available and hasattr(self, 'overlay'):
            del self.overlay
        
        self.is_initialized = False
        logger.info("FPGA controller shutdown")

# Integration with PyTorch layers
class PyTorchFPGABridge:
    """
    Bridge between PyTorch SNN layers and FPGA controller
    Provides seamless integration for PyTorch workflows
    """
    
    def __init__(self, fpga_controller: SNNFPGAController):
        self.fpga_controller = fpga_controller
        self.layer_mapping = {}
    
    def deploy_model(self, pytorch_model, input_example: np.ndarray) -> bool:
        """Deploy PyTorch SNN model to FPGA"""
        try:
            # Get model configuration
            if hasattr(pytorch_model, 'get_fpga_config'):
                config = pytorch_model.get_fpga_config()
            else:
                raise ValueError("Model must have get_fpga_config() method")
            
            # Configure FPGA
            success = self.fpga_controller.configure_network(config)
            
            if success:
                # Test with example input
                test_output = self.fpga_controller.process_spike_data(input_example)
                logger.info(f"Model deployed successfully, test output shape: {test_output.shape}")
            
            return success
            
        except Exception as e:
            logger.error(f"Model deployment error: {e}")
            return False
    
    def forward(self, input_spikes: np.ndarray) -> np.ndarray:
        """Forward pass through FPGA-deployed model"""
        return self.fpga_controller.process_spike_data(input_spikes)
    
    def benchmark_model(self, input_spikes: np.ndarray, num_runs: int = 10) -> Dict[str, Any]:
        """Benchmark model performance on FPGA"""
        latencies = []
        
        for _ in range(num_runs):
            start_time = time.time()
            output = self.forward(input_spikes)
            end_time = time.time()
            
            latencies.append(end_time - start_time)
        
        stats = self.fpga_controller.get_performance_stats()
        
        return {
            'avg_latency_ms': np.mean(latencies) * 1000,
            'min_latency_ms': np.min(latencies) * 1000,
            'max_latency_ms': np.max(latencies) * 1000,
            'throughput_fps': 1.0 / np.mean(latencies),
            'hardware_stats': stats
        }

# Example usage
if __name__ == "__main__":
    # Initialize FPGA controller
    controller = SNNFPGAController()
    
    if controller.initialize_hardware():
        # Create sample network configuration
        sample_config = {
            'num_layers': 3,
            'layers': [
                {
                    'layer_id': 0,
                    'layer_type': 'conv2d',
                    'in_channels': 1,
                    'out_channels': 32,
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1,
                    'threshold': 4096,  # Q8.8 format
                    'decay_factor': 230,  # 0.9 * 256
                    'weights': np.random.randint(-127, 127, (32, 1, 3, 3))
                },
                {
                    'layer_id': 1,
                    'layer_type': 'maxpool2d',
                    'kernel_size': 2,
                    'stride': 2,
                    'padding': 0,
                    'winner_take_all': True,
                    'pooling_window_time': 100
                },
                {
                    'layer_id': 2,
                    'layer_type': 'avgpool2d',
                    'kernel_size': 2,
                    'stride': 2,
                    'padding': 0,
                    'threshold': 2048,
                    'decay_factor': 230,
                    'pooling_window_time': 100
                }
            ]
        }
        
        # Configure network
        if controller.configure_network(sample_config):
            # Process sample data
            input_spikes = np.random.randint(0, 2, (1, 1, 28, 28, 100)).astype(np.float32)
            output_spikes = controller.process_spike_data(input_spikes)
            
            print(f"Input shape: {input_spikes.shape}")
            print(f"Output shape: {output_spikes.shape}")
            
            # Get performance stats
            stats = controller.get_performance_stats()
            print(f"Performance stats: {stats}")
        
        controller.shutdown()
    else:
        print("Failed to initialize FPGA hardware")
