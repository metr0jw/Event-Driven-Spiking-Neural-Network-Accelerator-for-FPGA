"""Minimal XRT/pyxrt helper for SNN accelerator.

This backend writes AXI-Lite registers (mode/time_steps/encoder/learning) via
pyxrt's register API. Streaming helpers are stubbed for now to keep imports
lightweight; DMA/stream handling can be extended once kernel stream names are
finalised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:  # pyxrt is optional and only available on Xilinx platforms
    import xrt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    xrt = None  # type: ignore


@dataclass
class RegisterMap:
    """Offsets (in bytes) for AXI-Lite registers of snn_top_hls.

    Extracted from hardware/hls/hls_output/hls/impl/ip/hdl/verilog/snn_top_hls_ctrl_s_axi.v
    All offsets verified from HLS-generated control interface.
    """

    # HLS standard control registers
    ap_ctrl: int = 0x00        # AP_CTRL (bit0=start, bit1=done, bit2=idle, bit3=ready)
    gie: int = 0x04            # Global Interrupt Enable
    ier: int = 0x08            # Interrupt Enable Register
    isr: int = 0x0C            # Interrupt Status Register
    
    # User scalar registers
    ctrl_reg: int = 0x10       # ctrl_reg[31:0]
    config_reg: int = 0x18     # config_reg[31:0]
    mode_reg: int = 0x20       # mode_reg[31:0] (bits[1:0]=mode, bit[8]=encoder_en)
    time_steps_reg: int = 0x28 # time_steps_reg[31:0]
    
    # Learning params struct (5 x 32-bit words)
    learning_base: int = 0x30  # learning_params[0..4] at 0x30,0x34,0x38,0x3C,0x40
    
    # Encoder config struct (4 x 32-bit words)
    encoder_base: int = 0x48   # encoder_config[0..3] at 0x48,0x4C,0x50,0x54
    
    # Status/debug read-only registers
    status_reg: int = 0x5C     # status_reg[31:0]
    spike_count_reg: int = 0x6C    # spike_count_reg[31:0]
    weight_sum_reg: int = 0x7C     # weight_sum_reg[31:0]
    version_reg: int = 0x8C        # version_reg[31:0]
    
    # Reward signal (R-STDP)
    reward_signal: int = 0x9C  # reward_signal[7:0]


class XRTBackend:
    """Wrapper over pyxrt for register and DMA operations."""

    def __init__(self, xclbin_path: str, device_index: int = 0, reg_map: Optional[RegisterMap] = None) -> None:
        if xrt is None:
            raise RuntimeError("pyxrt is not installed; install Xilinx Runtime (XRT) to use this backend")

        self.device = xrt.device(device_index)
        xclbin = xrt.xclbin(xclbin_path)
        self.device.load_xclbin(xclbin)
        # Grab the first kernel; name is typically snn_top_hls:{snn_top_hls_1}
        kname = xclbin.get_kernels()[0].get_name()
        self.kernel = xrt.kernel(self.device, kname)
        self.regs = reg_map or RegisterMap()
        
        # Buffer cache for reuse (type: xrt.bo but annotated as Any to avoid import errors)
        self._spike_input_bo: Optional[Any] = None
        self._spike_output_bo: Optional[Any] = None
        self._raw_data_bo: Optional[Any] = None

    # ------------------------------------------------------------------
    # Register helpers
    # ------------------------------------------------------------------
    def write_reg(self, offset: int, value: int) -> None:
        self.kernel.write_register(offset, value)

    def set_ctrl(self, ctrl: int) -> None:
        self.write_reg(self.regs.ctrl_reg, ctrl)

    def set_config(self, config: int) -> None:
        self.write_reg(self.regs.config_reg, config)

    def set_mode(self, mode: int, encoder_enable: bool = False) -> None:
        payload = mode & 0x3
        if encoder_enable:
            payload |= (1 << 8)
        self.write_reg(self.regs.mode_reg, payload)

    def set_time_steps(self, steps: int) -> None:
        self.write_reg(self.regs.time_steps_reg, steps)

    def set_reward(self, reward: int) -> None:
        self.write_reg(self.regs.reward_signal, reward & 0xFF)
    
    def set_encoder_config(
        self,
        encoding_type: int = 0,  # 0=NONE, 1=RATE_POISSON, 2=LATENCY, 3=DELTA_SIGMA
        two_neuron_enable: bool = False,
        baseline: int = 128,
        num_steps: int = 100,
        rate_scale: int = 256,
        latency_window: int = 100,
        delta_threshold: int = 1000,
        delta_decay: int = 10,
        num_channels: int = 784,
        default_weight: int = 127
    ) -> None:
        """Configure on-chip spike encoder parameters.
        
        Args:
            encoding_type: 4-bit encoding type (0-3 defined, 4-15 reserved)
            two_neuron_enable: Enable ON/OFF neuron split (doubles channels)
            baseline: Baseline for two-neuron split (default 128 for uint8)
            num_steps: Total simulation timesteps (for rate/latency normalization)
            rate_scale: Rate coding threshold scale
            latency_window: Latency coding time window (timesteps)
            delta_threshold: Delta-sigma integration threshold
            delta_decay: Delta-sigma decay rate
            num_channels: Number of input channels (output = 2x if two_neuron_enable)
            default_weight: Default spike weight (0-255)
        """
        # Assuming encoder config registers start after basic control regs
        # This would need to match the actual HLS register map
        enc_base = 0x40  # Example offset for encoder config block
        
        # Pack encoding_type (4-bit) and two_neuron_enable (1-bit) into single register
        enc_ctrl = (encoding_type & 0xF) | ((1 if two_neuron_enable else 0) << 4)
        self.write_reg(enc_base + 0x00, enc_ctrl)
        
        self.write_reg(enc_base + 0x04, baseline & 0xFF)
        self.write_reg(enc_base + 0x08, num_steps & 0xFFFF)
        self.write_reg(enc_base + 0x0C, rate_scale & 0xFFFF)
        self.write_reg(enc_base + 0x10, latency_window & 0xFFFF)
        self.write_reg(enc_base + 0x14, delta_threshold & 0xFFFF)
        self.write_reg(enc_base + 0x18, delta_decay & 0xFFFF)
        self.write_reg(enc_base + 0x1C, num_channels & 0xFFFF)
        self.write_reg(enc_base + 0x20, default_weight & 0xFF)

    # ------------------------------------------------------------------
    # DMA Stream Helpers
    # ------------------------------------------------------------------
    def _allocate_bo(self, size: int) -> Any:
        """Allocate XRT buffer object."""
        if xrt is None:
            raise RuntimeError("XRT not available")
        # Create buffer in device memory (bank 0 by default)
        return xrt.bo(self.device, size, xrt.bo.flags.normal, 0)
    
    def send_spike_stream(self, spike_data: bytes, run_kernel: bool = True) -> None:
        """Send spike stream to s_axis_spikes via DMA.
        
        Args:
            spike_data: Packed spike packets (32-bit AER format per spike)
            run_kernel: Whether to trigger kernel execution after transfer
        """
        if xrt is None:
            raise RuntimeError("XRT not available")
        
        size = len(spike_data)
        if size == 0:
            return
        
        # Allocate or reuse buffer
        if self._spike_input_bo is None or self._spike_input_bo.size() < size:
            self._spike_input_bo = self._allocate_bo(size)
        
        # Write data to buffer
        self._spike_input_bo.write(spike_data, 0)
        
        # Sync to device
        self._spike_input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, size, 0)
        
        if run_kernel:
            # Start kernel (assumes AP_CTRL register at 0x00)
            self.write_reg(self.regs.ap_ctrl, 0x01)  # Set start bit
    
    def receive_spike_stream(self, max_size: int = 4096) -> bytes:
        """Receive spike output from m_axis_spikes via DMA.
        
        Args:
            max_size: Maximum buffer size to allocate
            
        Returns:
            Raw bytes from output spike stream
        """
        if xrt is None:
            raise RuntimeError("XRT not available")
        
        # Allocate or reuse buffer
        if self._spike_output_bo is None or self._spike_output_bo.size() < max_size:
            self._spike_output_bo = self._allocate_bo(max_size)
        
        # Wait for kernel completion (poll done bit)
        while True:
            status = self.kernel.read_register(self.regs.ap_ctrl)
            if status & 0x02:  # bit[1] = done
                break
        
        # Sync from device
        self._spike_output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, max_size, 0)
        
        # Read actual data size from spike_count_reg
        spike_count = self.kernel.read_register(self.regs.spike_count_reg)
        actual_size = min(spike_count * 4, max_size)  # 4 bytes per spike (32-bit AER)
        
        # Read data from buffer
        data = bytearray(actual_size)
        self._spike_output_bo.read(data, 0)
        return bytes(data)
    
    def send_raw_frame(self, raw_data: bytes, run_kernel: bool = False) -> None:
        """Send raw sensor data to s_axis_data for on-chip encoding.
        
        Args:
            raw_data: Raw pixel/sensor data (format depends on encoder config)
            run_kernel: Whether to trigger kernel execution (usually False, use send_spike_stream)
        """
        if xrt is None:
            raise RuntimeError("XRT not available")
        
        size = len(raw_data)
        if size == 0:
            return
        
        # Allocate or reuse buffer
        if self._raw_data_bo is None or self._raw_data_bo.size() < size:
            self._raw_data_bo = self._allocate_bo(size)
        
        # Write and sync
        self._raw_data_bo.write(raw_data, 0)
        self._raw_data_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, size, 0)
        
        if run_kernel:
            self.write_reg(self.regs.ap_ctrl, 0x01)
