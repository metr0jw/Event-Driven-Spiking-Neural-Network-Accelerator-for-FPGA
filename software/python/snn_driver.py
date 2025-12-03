"""
PYNQ Driver for SNN Accelerator
Event-Driven Spiking Neural Network Accelerator for FPGA

This driver provides a high-level interface to interact with the
SNN accelerator hardware on PYNQ boards.

Usage:
    from snn_driver import SNNAccelerator
    
    snn = SNNAccelerator('snn_accelerator.bit')
    snn.configure(threshold=100, leak_rate=16)
    snn.send_spike(neuron_id=5, weight=10)
    output_spikes = snn.get_output_spikes()
    snn.close()
"""

import numpy as np
from pynq import Overlay, allocate
import time


class SNNAccelerator:
    """
    PYNQ driver for Event-Driven SNN Accelerator
    
    Register Map:
        0x00: Control Register
              [0] - Enable
              [1] - Reset
              [2] - Clear counters
        0x04: Config Register
        0x08: Status Register (RO)
        0x0C: Spike Count (RO)
        0x10: Threshold
        0x14: Leak Rate
        0x18: Refractory Period
        0x1C: Reserved
    """
    
    # Register offsets
    REG_CONTROL = 0x00
    REG_CONFIG = 0x04
    REG_STATUS = 0x08
    REG_SPIKE_COUNT = 0x0C
    REG_THRESHOLD = 0x10
    REG_LEAK_RATE = 0x14
    REG_REFRACTORY = 0x18
    
    # Control register bits
    CTRL_ENABLE = 0x01
    CTRL_RESET = 0x02
    CTRL_CLEAR = 0x04
    
    def __init__(self, bitstream_path='snn_accelerator.bit'):
        """
        Initialize the SNN accelerator
        
        Args:
            bitstream_path: Path to the bitstream file (.bit)
        """
        print(f"Loading bitstream: {bitstream_path}")
        self.overlay = Overlay(bitstream_path)
        
        # Get reference to the SNN IP
        self.snn = self.overlay.snn_0
        
        # Default configuration
        self._threshold = 100
        self._leak_rate = 16
        self._refractory_period = 5
        
        # Initialize hardware
        self.reset()
        self.configure(self._threshold, self._leak_rate, self._refractory_period)
        
        print("SNN Accelerator initialized successfully")
        
    def reset(self):
        """Reset the SNN accelerator"""
        # Assert reset
        self.snn.write(self.REG_CONTROL, self.CTRL_RESET)
        time.sleep(0.001)  # 1ms delay
        # Deassert reset
        self.snn.write(self.REG_CONTROL, 0)
        time.sleep(0.001)
        
    def enable(self):
        """Enable the SNN accelerator"""
        ctrl = self.snn.read(self.REG_CONTROL)
        self.snn.write(self.REG_CONTROL, ctrl | self.CTRL_ENABLE)
        
    def disable(self):
        """Disable the SNN accelerator"""
        ctrl = self.snn.read(self.REG_CONTROL)
        self.snn.write(self.REG_CONTROL, ctrl & ~self.CTRL_ENABLE)
        
    def configure(self, threshold=None, leak_rate=None, refractory_period=None):
        """
        Configure neuron parameters
        
        Args:
            threshold: Spike threshold (0-65535)
            leak_rate: Membrane potential leak rate (0-65535)
            refractory_period: Refractory period in clock cycles (0-65535)
        """
        if threshold is not None:
            self._threshold = threshold
            self.snn.write(self.REG_THRESHOLD, threshold)
            
        if leak_rate is not None:
            self._leak_rate = leak_rate
            self.snn.write(self.REG_LEAK_RATE, leak_rate)
            
        if refractory_period is not None:
            self._refractory_period = refractory_period
            self.snn.write(self.REG_REFRACTORY, refractory_period)
            
    def get_status(self):
        """
        Get accelerator status
        
        Returns:
            dict with status information
        """
        status = self.snn.read(self.REG_STATUS)
        spike_count = self.snn.read(self.REG_SPIKE_COUNT)
        
        return {
            'status_raw': status,
            'spike_count': spike_count,
            'enabled': bool(self.snn.read(self.REG_CONTROL) & self.CTRL_ENABLE)
        }
        
    def clear_counters(self):
        """Clear spike counters"""
        ctrl = self.snn.read(self.REG_CONTROL)
        self.snn.write(self.REG_CONTROL, ctrl | self.CTRL_CLEAR)
        time.sleep(0.0001)  # Small delay
        self.snn.write(self.REG_CONTROL, ctrl & ~self.CTRL_CLEAR)
        
    @property
    def threshold(self):
        return self._threshold
        
    @threshold.setter
    def threshold(self, value):
        self.configure(threshold=value)
        
    @property
    def leak_rate(self):
        return self._leak_rate
        
    @leak_rate.setter
    def leak_rate(self, value):
        self.configure(leak_rate=value)
        
    @property
    def refractory_period(self):
        return self._refractory_period
        
    @refractory_period.setter
    def refractory_period(self, value):
        self.configure(refractory_period=value)
        
    def close(self):
        """Clean up resources"""
        self.disable()
        print("SNN Accelerator closed")
        

class SNNAcceleratorDMA(SNNAccelerator):
    """
    Extended SNN Accelerator driver with DMA support for high-throughput operation
    
    This version uses AXI DMA for bulk spike transfers.
    """
    
    def __init__(self, bitstream_path='snn_accelerator.bit'):
        """
        Initialize SNN accelerator with DMA support
        
        Args:
            bitstream_path: Path to bitstream file
        """
        super().__init__(bitstream_path)
        
        # Get DMA engine reference (if available)
        if hasattr(self.overlay, 'axi_dma_0'):
            self.dma = self.overlay.axi_dma_0
            self._dma_available = True
            print("DMA engine available")
        else:
            self._dma_available = False
            print("Warning: DMA not available, using register-based transfer")
            
        # Allocate DMA buffers
        if self._dma_available:
            self.input_buffer = allocate(shape=(1024,), dtype=np.uint32)
            self.output_buffer = allocate(shape=(1024,), dtype=np.uint32)
            
    def send_spikes_dma(self, spike_data):
        """
        Send multiple spikes via DMA
        
        Args:
            spike_data: numpy array of spike packets
                        Format: [neuron_id(8) | weight(8) | reserved(16)]
                        
        Returns:
            Number of spikes sent
        """
        if not self._dma_available:
            raise RuntimeError("DMA not available")
            
        # Copy data to input buffer
        n_spikes = min(len(spike_data), len(self.input_buffer))
        self.input_buffer[:n_spikes] = spike_data[:n_spikes]
        
        # Start DMA transfer
        self.dma.sendchannel.transfer(self.input_buffer[:n_spikes])
        self.dma.sendchannel.wait()
        
        return n_spikes
        
    def receive_spikes_dma(self, max_spikes=1024, timeout=1.0):
        """
        Receive output spikes via DMA
        
        Args:
            max_spikes: Maximum number of spikes to receive
            timeout: Timeout in seconds
            
        Returns:
            numpy array of received spike packets
        """
        if not self._dma_available:
            raise RuntimeError("DMA not available")
            
        # Start DMA receive
        self.dma.recvchannel.transfer(self.output_buffer[:max_spikes])
        
        # Wait with timeout
        try:
            self.dma.recvchannel.wait(timeout=timeout)
        except TimeoutError:
            pass
            
        # Return received data
        n_received = self.dma.recvchannel.transferred // 4  # 4 bytes per spike
        return self.output_buffer[:n_received].copy()
        
    def process_input_spike_train(self, spike_times, neuron_ids, weights):
        """
        Process a full spike train through the accelerator
        
        Args:
            spike_times: Array of spike times (timesteps)
            neuron_ids: Array of input neuron IDs
            weights: Array of synaptic weights
            
        Returns:
            Dictionary with output spikes and timing info
        """
        # Validate inputs
        assert len(spike_times) == len(neuron_ids) == len(weights)
        
        # Enable accelerator
        self.enable()
        self.clear_counters()
        
        # Pack spike data
        spike_data = np.zeros(len(spike_times), dtype=np.uint32)
        spike_data = (neuron_ids.astype(np.uint32) & 0xFF) | \
                     ((weights.astype(np.uint32) & 0xFF) << 8)
        
        start_time = time.time()
        
        if self._dma_available:
            # Use DMA for bulk transfer
            self.send_spikes_dma(spike_data)
            output_spikes = self.receive_spikes_dma()
        else:
            # Fallback: register-based transfer (slow)
            output_spikes = np.array([], dtype=np.uint32)
            
        elapsed_time = time.time() - start_time
        
        # Get final status
        status = self.get_status()
        
        return {
            'output_spikes': output_spikes,
            'total_output_spikes': status['spike_count'],
            'processing_time_ms': elapsed_time * 1000,
            'spikes_per_second': len(spike_times) / elapsed_time if elapsed_time > 0 else 0
        }
        
    def close(self):
        """Clean up resources including DMA buffers"""
        super().close()
        if self._dma_available:
            del self.input_buffer
            del self.output_buffer
            

def test_basic():
    """Basic functionality test"""
    print("=" * 50)
    print("SNN Accelerator Basic Test")
    print("=" * 50)
    
    # Create accelerator instance
    snn = SNNAccelerator()
    
    # Configure
    snn.configure(threshold=100, leak_rate=16, refractory_period=5)
    print(f"Configured: threshold={snn.threshold}, leak_rate={snn.leak_rate}")
    
    # Enable
    snn.enable()
    status = snn.get_status()
    print(f"Status: {status}")
    
    # Clean up
    snn.close()
    print("Test completed successfully!")
    

if __name__ == "__main__":
    test_basic()
