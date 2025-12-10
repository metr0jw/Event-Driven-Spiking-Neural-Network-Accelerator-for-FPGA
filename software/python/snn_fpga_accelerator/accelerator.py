"""Main FPGA SNN Accelerator Interface.

The :class:`SNNAccelerator` class acts as the glue layer between the Python
software stack and the underlying FPGA implementation.  When the PYNQ runtime
and bitstream are available the class streams spike events directly to the
hardware.  When they are not available (for example on a pure software
development machine) the same API seamlessly falls back to a software
simulation that can be validated with Icarus Verilog or a lightweight Python
model.  This allows developers to iterate on both hardware and software using
only open-source tooling.

Author: Jiwoon Lee (@metr0jw)
Organization: Kwangwoon University, Seoul, South Korea
Contact: jwlee@linux.com
"""

from __future__ import annotations

import json
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

try:
    from pynq import Overlay, allocate  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Overlay = None   # type: ignore
    allocate = None  # type: ignore

# Optional XRT backend (pyxrt)
try:  # pragma: no cover - optional dependency
    from .xrt_backend import XRTBackend, RegisterMap
except Exception:  # pragma: no cover - keep import optional
    XRTBackend = None  # type: ignore
    RegisterMap = None  # type: ignore

from .exceptions import (
    FPGAError,
    TimeoutError as AcceleratorTimeoutError,
    WeightLoadError,
    ConfigurationError,
    CommunicationError,
    BitstreamError,
    KernelExecutionError,
    validate_parameter,
)
from .learning import RSTDPLearning, STDPLearning
from .pytorch_interface import SNNModel, simulate_snn_inference
from .spike_encoding import SpikeEvent
from .utils import logger, PerformanceMonitor


@dataclass
class SimulationResult:
    """Captures the outcome of a simulation run."""

    events: List[SpikeEvent]
    duration: float


class _SoftwareBackend:
    """Pure Python backend used when FPGA hardware is unavailable."""

    def __init__(self) -> None:
        self.model: Optional[SNNModel] = None
        self.learning_engine: Optional[Union[STDPLearning, RSTDPLearning]] = None
        self.learning_enabled: bool = False
        self.monitor = PerformanceMonitor()

    def configure_model(self, model: SNNModel) -> None:
        self.model = model
        logger.info(
            "Software backend configured: %d layers, %d neurons",
            len(model.layers),
            model.total_neurons,
        )

    def run(self, spikes: List[SpikeEvent], duration: float) -> SimulationResult:
        if self.model is None:
            raise RuntimeError("No SNN model configured for simulation")

        self.monitor.start_timer("software_simulation")
        events = simulate_snn_inference(self.model, spikes, duration)
        self.monitor.end_timer("software_simulation")

        if self.learning_enabled and self.learning_engine is not None:
            # Apply learning using the produced spikes
            synapse_map = {}
            updates = self.learning_engine.compute_weight_updates(
                synapse_map, duration
            ) if isinstance(self.learning_engine, RSTDPLearning) else []
            if updates:
                logger.debug("Applied %d learning updates in software backend", len(updates))

        return SimulationResult(events=events, duration=duration)

    def configure_learning(self, engine: Union[STDPLearning, RSTDPLearning]) -> None:
        self.learning_engine = engine
        logger.info("Software backend learning engine set to %s", engine.__class__.__name__)

    def enable_learning(self, enable: bool) -> None:
        self.learning_enabled = enable
        logger.info("Software backend learning %s", "enabled" if enable else "disabled")


class _HardwareBackend:
    """Thin wrapper around the PYNQ runtime for clarity."""

    def __init__(self, bitstream_path: str, fpga_ip: str) -> None:
        if Overlay is None or allocate is None:
            raise RuntimeError(
                "PYNQ runtime is not available. Install pynq or enable simulation_mode."
            )

        self.bitstream_path = bitstream_path
        self.fpga_ip = fpga_ip
        self.overlay = None
        self.dma = None
        self.ip = None

    def connect(self) -> None:
        logger.info("Loading FPGA bitstream: %s", self.bitstream_path)
        overlay_cls = cast(Any, Overlay)
        overlay = overlay_cls(self.bitstream_path)
        self.overlay = overlay
        overlay_any = cast(Any, overlay)
        self.ip = overlay_any.snn_accelerator_0
        self.dma = overlay_any.axi_dma_0
        self._initialize()
        logger.info("FPGA connection established")

    def _initialize(self) -> None:
        assert self.ip is not None
        self.ip.write(0x00, 0x1)
        time.sleep(0.01)
        self.ip.write(0x00, 0x0)
        self.ip.write(0x10, 1000)
        self.ip.write(0x14, 10)
        self.ip.write(0x18, 5)

    def disconnect(self) -> None:
        logger.info("Releasing FPGA overlay")
        if self.overlay is not None:
            del self.overlay
        self.overlay = None
        self.ip = None
        self.dma = None

    # Additional hardware specific helpers are kept forward compatible with the
    # legacy implementation and are defined in the main accelerator class.


class SNNAccelerator:
    """
    Event-driven Spiking Neural Network accelerator interface for PYNQ-Z2.
    
    This class provides the main interface between PyTorch/Python and the
    FPGA-based SNN accelerator hardware while also supporting a pure software
    simulation environment. Developers can therefore target Icarus Verilog or
    other open-source tools before moving to Vivado/Vitis.
    """
    
    def __init__(
        self,
        bitstream_path: Optional[str] = None,
        fpga_ip: str = "192.168.2.99",
        simulation_mode: bool = False,
        simulation_backend: str = "software",
        icarus_binary: str = "iverilog",
        vvp_binary: str = "vvp",
    ):
        """Initialise the SNN accelerator front-end.

        Parameters
        ----------
        bitstream_path:
            Path to the `.bit` file.  Ignored when ``simulation_mode`` is true.
        fpga_ip:
            IP address of the PYNQ board (for networked setups).
        simulation_mode:
            When ``True`` forces the accelerator to run without FPGA hardware
            using the requested ``simulation_backend``.
        simulation_backend:
            Currently ``"software"`` (default) for a Python based model.  The
            placeholder string can later be extended to invoke Icarus testbenches.
        icarus_binary / vvp_binary:
            Paths to the Icarus Verilog tooling (used by future backends).
        """

        self.bitstream_path = bitstream_path or "snn_accelerator.bit"
        self.fpga_ip = fpga_ip
        self.simulation_mode = simulation_mode or Overlay is None
        self.simulation_backend = simulation_backend.lower()
        self.icarus_binary = icarus_binary
        self.vvp_binary = vvp_binary
        self.use_xrt = False

        self._software_backend = _SoftwareBackend()
        self._hardware_backend: Optional[_HardwareBackend] = None
        if not self.simulation_mode:
            self._hardware_backend = _HardwareBackend(self.bitstream_path, self.fpga_ip)

        # Event processing primitives (hardware path only)
        self.spike_queue: "Queue[bytes]" = Queue()
        self.output_queue: "Queue[SpikeEvent]" = Queue()
        self.processing_thread: Optional[threading.Thread] = None
        self.is_running = False

        # Cached network / training state
        self.num_neurons: int = 0
        self.network_topology: Optional[Dict] = None
        self.neuron_parameters: Dict[str, Union[int, float]] = {}
        self.snn_model: Optional[SNNModel] = None
        self.learning_engine: Optional[Union[STDPLearning, RSTDPLearning]] = None
        self.learning_enabled: bool = False

        # Performance monitoring
        self.spike_count = 0
        self.last_spike_time = 0.0
        self.profiling_enabled = True
        
        # Timestep tracking
        self.current_timestep: int = 0
        self.timestep_dt: float = 0.001  # 1ms default timestep
        self.membrane_state: Optional[np.ndarray] = None
        self.spike_history: List[List[SpikeEvent]] = []  # History for multi-step

        # XRT backend handle (optional)
        self._xrt_backend: Optional[XRTBackend] = None
        self._xrt_regmap: Optional[RegisterMap] = None
        
    def load_bitstream(self, bitstream_path: Optional[str] = None) -> None:
        """Load the FPGA bitstream and initialize hardware (PYNQ path)."""
        if self.simulation_mode or self.use_xrt:
            logger.info("Simulation/XRT mode – skipping PYNQ bitstream load")
            return

        if bitstream_path:
            self.bitstream_path = bitstream_path

        if self._hardware_backend is None:
            self._hardware_backend = _HardwareBackend(self.bitstream_path, self.fpga_ip)
        self._hardware_backend.connect()

    def configure_xrt(self, xclbin_path: str, device_index: int = 0, reg_map: Optional[RegisterMap] = None) -> None:
        """Configure XRT backend for register writes (mode/time_steps/etc.)."""
        if XRTBackend is None:
            raise RuntimeError("XRT backend not available; install pyxrt/XRT runtime")
        self.use_xrt = True
        self._xrt_backend = XRTBackend(xclbin_path, device_index, reg_map)
        self._xrt_regmap = reg_map or RegisterMap()
        logger.info("XRT backend configured with xclbin %s", xclbin_path)
    
    def configure_encoder(
        self,
        encoding_type: str = "rate_poisson",
        num_steps: int = 100,
        two_neuron_enable: bool = False,
        baseline: int = 128,
        **encoder_params
    ) -> None:
        """Configure on-chip spike encoder parameters.
        
        Maps Python encoder configuration to HLS register writes via XRT backend.
        
        Args:
            encoding_type: Encoding method - "none", "rate_poisson", "latency", "delta_sigma"
            num_steps: Total simulation timesteps (for rate/latency normalization)
            two_neuron_enable: Enable ON/OFF neuron polarity split (doubles channels)
            baseline: Baseline value for two-neuron encoding (default 128 for uint8)
            **encoder_params: Additional encoder-specific parameters:
                - rate_scale: Rate coding threshold scale (default 256)
                - latency_window: Latency coding time window in timesteps (default num_steps)
                - delta_threshold: Delta-sigma integration threshold (default 1000)
                - delta_decay: Delta-sigma decay rate (default 10)
                - num_channels: Number of input channels (default 784 for MNIST)
                - default_weight: Default spike weight 0-255 (default 127)
        
        Raises:
            ConfigurationError: If encoding_type is invalid or parameters out of range
            CommunicationError: If XRT backend communication fails
        
        Example:
            >>> accel.configure_xrt("snn.xclbin")
            >>> accel.configure_encoder(
            ...     encoding_type="rate_poisson",
            ...     num_steps=100,
            ...     two_neuron_enable=True,
            ...     rate_scale=256,
            ...     num_channels=784
            ... )
        """
        # Encoding type mapping
        enc_type_map = {
            "none": 0,
            "rate_poisson": 1,
            "latency": 2,
            "delta_sigma": 3
        }
        
        # Validate encoding type
        try:
            validate_parameter(
                encoding_type,
                valid_options=list(enc_type_map.keys()),
                parameter_name="encoding_type"
            )
        except ValueError as e:
            raise ConfigurationError(
                str(e),
                parameter="encoding_type",
                value=encoding_type,
                error_code=4010
            ) from e
        
        enc_type_code = enc_type_map[encoding_type]
        
        # Extract encoder parameters with defaults
        rate_scale = encoder_params.get('rate_scale', 256)
        latency_window = encoder_params.get('latency_window', num_steps)
        delta_threshold = encoder_params.get('delta_threshold', 1000)
        delta_decay = encoder_params.get('delta_decay', 10)
        num_channels = encoder_params.get('num_channels', 784)
        default_weight = encoder_params.get('default_weight', 127)
        
        # Configure via XRT if available
        if self.use_xrt and self._xrt_backend is not None:
            self._xrt_backend.set_encoder_config(
                encoding_type=enc_type_code,
                two_neuron_enable=two_neuron_enable,
                baseline=baseline,
                num_steps=num_steps,
                rate_scale=rate_scale,
                latency_window=latency_window,
                delta_threshold=delta_threshold,
                delta_decay=delta_decay,
                num_channels=num_channels,
                default_weight=default_weight
            )
            logger.info(
                f"Configured on-chip encoder: type={encoding_type}, "
                f"num_steps={num_steps}, two_neuron={two_neuron_enable}, "
                f"channels={num_channels}"
            )
        elif self._hardware_backend is not None:
            # PYNQ path: Write to registers directly
            # This would require PYNQ IP driver implementation
            logger.warning("PYNQ encoder configuration not yet implemented")
        else:
            # Simulation mode: Store config for software encoder
            logger.info(f"Simulation mode: Encoder config stored (type={encoding_type})")
            self.encoder_config = {
                'encoding_type': encoding_type,
                'num_steps': num_steps,
                'two_neuron_enable': two_neuron_enable,
                'baseline': baseline,
                **encoder_params
            }
    
    def _run_simulation_xrt(self, duration: float, input_spikes: List[SpikeEvent],
                           encoding_type: int = 0, two_neuron_enable: bool = False,
                           baseline: int = 128, encoder_params: Optional[dict] = None) -> List[SpikeEvent]:
        """Run simulation using XRT backend with time_steps loop on FPGA.
        
        Args:
            duration: Simulation duration in seconds
            input_spikes: List of input spike events
            encoding_type: 0=NONE, 1=RATE_POISSON, 2=LATENCY, 3=DELTA_SIGMA
            two_neuron_enable: Enable ON/OFF neuron polarity split
            baseline: Baseline value for two-neuron encoding (default 128)
            encoder_params: Optional dict with rate_scale, latency_window, delta_threshold, etc.
        """
        if self._xrt_backend is None:
            raise RuntimeError("XRT backend not configured")
        
        # Calculate time steps from duration
        time_steps = int(duration / self.timestep_dt)
        logger.info("Running XRT simulation: duration=%.3fs, time_steps=%d, encoding=%d, two_neuron=%s",
                   duration, time_steps, encoding_type, two_neuron_enable)
        
        # Set control registers
        encoder_enable = (encoding_type != 0)  # Enable encoder for non-NONE types
        self._xrt_backend.set_mode(mode=0, encoder_enable=encoder_enable)  # MODE_INFERENCE
        self._xrt_backend.set_time_steps(time_steps)
        
        # Configure encoder if enabled
        if encoder_enable:
            params = encoder_params or {}
            self._xrt_backend.set_encoder_config(
                encoding_type=encoding_type,
                two_neuron_enable=two_neuron_enable,
                baseline=baseline,
                num_steps=time_steps,  # Pass total timesteps to encoder
                rate_scale=params.get('rate_scale', 256),
                latency_window=params.get('latency_window', 100),
                delta_threshold=params.get('delta_threshold', 1000),
                delta_decay=params.get('delta_decay', 10),
                num_channels=params.get('num_channels', 784),
                default_weight=params.get('default_weight', 127)
            )
        
        # Pack input spikes to bytes (32-bit AER format)
        spike_data = self._pack_spike_events(input_spikes)
        
        # Send spike stream and trigger kernel
        self._xrt_backend.send_spike_stream(spike_data, run_kernel=True)
        
        # Receive output spikes
        output_data = self._xrt_backend.receive_spike_stream(max_size=4096)
        
        # Unpack output spikes
        output_spikes = self._unpack_spike_events(output_data)
        
        logger.info("XRT simulation completed: %d input, %d output spikes", 
                   len(input_spikes), len(output_spikes))
        return output_spikes
    
    def _pack_spike_events(self, spikes: List[SpikeEvent]) -> bytes:
        """Pack spike events to 32-bit AER format bytes."""
        packed = bytearray()
        for spike in spikes:
            # AER format: [31:18] timestamp(14b), [17:10] weight(8b), [9:0] neuron_id
            timestamp_14b = int(spike.timestamp * 1000) & 0x3FFF  # 14-bit timestamp
            weight_8b = int(spike.weight * 127) & 0xFF  # 8-bit weight
            neuron_id_10b = int(spike.neuron_id) & 0x3FF  # 10-bit neuron ID
            
            word = (timestamp_14b << 18) | (weight_8b << 10) | neuron_id_10b
            packed.extend(struct.pack('<I', word))  # Little-endian 32-bit
        return bytes(packed)
    
    def _unpack_spike_events(self, data: bytes) -> List[SpikeEvent]:
        """Unpack 32-bit AER format bytes to spike events."""
        spikes = []
        for i in range(0, len(data), 4):
            if i + 4 > len(data):
                break
            word = struct.unpack('<I', data[i:i+4])[0]
            
            neuron_id = word & 0x3FF
            weight = ((word >> 10) & 0xFF) / 127.0
            timestamp = ((word >> 18) & 0x3FFF) / 1000.0
            
            spikes.append(SpikeEvent(neuron_id=neuron_id, timestamp=timestamp, weight=weight))
        return spikes
    
    def _initialize_hardware(self) -> None:
        """Initialize the SNN hardware IP."""
        if self._hardware_backend is not None:
            self._hardware_backend._initialize()
    
    def initialize(
        self,
        neurons_per_layer: List[int],
        weights: Optional[List[np.ndarray]] = None,
    ) -> None:
        """
        Initialize a simple feedforward SNN with specified layer sizes.
        
        This is a convenience method for quickly setting up a network topology.
        For more control, use configure_network() with an SNNModel directly.
        
        Parameters
        ----------
        neurons_per_layer:
            List of neuron counts for each layer (including input and output).
            E.g., [784, 128, 10] creates a 3-layer network.
        weights:
            Optional list of weight matrices. If None, random weights are used.
        """
        from .pytorch_interface import SNNModel, SNNLayer
        
        model = SNNModel(name=f"feedforward_{len(neurons_per_layer)}_layer")
        
        for i in range(len(neurons_per_layer) - 1):
            input_size = neurons_per_layer[i]
            output_size = neurons_per_layer[i + 1]
            
            # Use provided weights or initialize randomly
            if weights is not None and i < len(weights):
                layer_weights = weights[i]
            else:
                # Xavier/Glorot initialization
                limit = np.sqrt(6.0 / (input_size + output_size))
                layer_weights = np.random.uniform(-limit, limit, (input_size, output_size))
            
            layer = SNNLayer(
                input_size=input_size,
                output_size=output_size,
                layer_type="fully_connected",
                weights=layer_weights,
            )
            model.add_layer(layer)
        
        # Configure the network with the created model
        self.configure_network(model)
        logger.info(
            "Initialized %d-layer network: %s",
            len(neurons_per_layer) - 1,
            " -> ".join(map(str, neurons_per_layer))
        )
    
    def configure_network(
        self,
        config: Optional[Union[SNNModel, Dict, str, Path]] = None,
        num_neurons: Optional[int] = None,
        topology: Optional[Dict] = None,
    ) -> None:
        """
        Configure the network topology and parameters.

        Parameters
        ----------
        config:
            Either an :class:`SNNModel`, a configuration dictionary, or a path to
            a JSON/HDF5 description.  When *None*, the legacy ``num_neurons`` /
            ``topology`` arguments are used for backwards compatibility.
        num_neurons / topology:
            Legacy arguments preserved so that older scripts keep working.
        """
        if isinstance(config, (str, Path)):
            config = self._load_config_from_file(Path(config))

        if isinstance(config, SNNModel):
            self.snn_model = config
            self.num_neurons = config.total_neurons
            self.network_topology = {"layers": len(config.layers)}
            if self.simulation_mode:
                self._software_backend.configure_model(config)
            else:
                self._load_model_to_hardware(config)
            logger.info("Configured accelerator with supplied SNNModel")
            return

        if isinstance(config, dict):
            if "model" in config and isinstance(config["model"], SNNModel):
                self.configure_network(config["model"])
            elif "weights" in config:
                weights = np.array(config["weights"])
                self.num_neurons = config.get("num_neurons", weights.shape[0])
                if not self.simulation_mode:
                    self._load_weights(weights)
                logger.info("Configured accelerator from weight dictionary")
            else:
                self.network_topology = config
            return

        if num_neurons is not None:
            self.num_neurons = num_neurons
        if topology is not None:
            self.network_topology = topology
            if isinstance(topology.get("weights"), np.ndarray) and not self.simulation_mode:
                self._load_weights(topology["weights"])

        if self.simulation_mode and self.snn_model is None:
            logger.warning("Simulation mode active without an SNN model – results will be empty")

        logger.info("Network configured: %d neurons", self.num_neurons)
    
    def _load_weights(self, weights: np.ndarray) -> None:
        """Load synaptic weights to FPGA memory.
        
        Raises:
            WeightLoadError: If weight validation or transfer fails
            CommunicationError: If DMA communication fails
        """
        if self.simulation_mode:
            logger.debug("Simulation mode: skipping hardware weight load (%d weights)", weights.size)
            return

        # XRT path: Use s_axis_weights stream
        if self.use_xrt and self._xrt_backend is not None:
            try:
                # Convert weights to uint8 [0, 255] range
                weight_data = np.clip(weights * 127 + 128, 0, 255).astype(np.uint8)
                self._xrt_backend.load_weights(weight_data)
                logger.info("Uploaded %dx%d weights to FPGA via XRT", weight_data.shape[0], weight_data.shape[1])
                return
            except (WeightLoadError, CommunicationError):
                raise
            except Exception as e:
                raise WeightLoadError(
                    f"Failed to load weights: {e}",
                    weight_shape=weights.shape,
                    error_code=3010
                ) from e

        # PYNQ DMA path (legacy)
        if self._hardware_backend is None or self._hardware_backend.dma is None:
            raise RuntimeError("Hardware DMA engine not initialised")

        dma = self._hardware_backend.dma
        weight_buffer = allocate(shape=(weights.size,), dtype=np.int8)  # type: ignore[arg-type]
        weight_data = np.clip(weights * 128, -128, 127).astype(np.int8)
        weight_buffer[:] = weight_data.flatten()
        dma.sendchannel.transfer(weight_buffer)  # type: ignore[attr-defined]
        dma.sendchannel.wait()  # type: ignore[attr-defined]
        logger.info("Uploaded %d weights to FPGA DMA", weights.size)
    
    def send_spike_event(self, neuron_id: int, timestamp: float, weight: float = 1.0) -> None:
        """
        Send a spike event to the FPGA.
        
        Args:
            neuron_id: ID of the source neuron
            timestamp: Spike timestamp in seconds
            weight: Synaptic weight multiplier
        """
        # Convert to hardware format
        hw_timestamp = int(timestamp * 1e6)  # Convert to microseconds
        hw_weight = int(weight * 128)        # Fixed-point weight
        
        # Pack spike event (32-bit format)
        spike_data = struct.pack('<I', 
                               (neuron_id & 0xFF) | 
                               ((hw_weight & 0xFF) << 8) |
                               ((hw_timestamp & 0xFFFF) << 16))
        
        # Send via AXI stream interface
        self.spike_queue.put(spike_data)
        
    def start_processing(self) -> None:
        """Start the event processing thread."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._process_events)
            self.processing_thread.start()
            logger.info("Event processing started")
    
    def stop_processing(self) -> None:
        """Stop the event processing thread."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("Event processing stopped")
    
    def _process_events(self) -> None:
        """Main event processing loop (runs in separate thread)."""
        while self.is_running:
            try:
                # Process input spikes
                if not self.spike_queue.empty():
                    spike_data = self.spike_queue.get_nowait()
                    self._send_spike_to_hardware(spike_data)
                
                # Read output spikes
                output_spikes = self._read_output_spikes()
                if output_spikes:
                    for spike in output_spikes:
                        self.output_queue.put(spike)
                
                time.sleep(0.001)  # 1ms sleep to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Event processing error: {e}")
    
    def _send_spike_to_hardware(self, spike_data: bytes) -> None:
        """Send spike data to hardware via AXI interface."""
        if self.simulation_mode:
            return

        if self._hardware_backend is None or self._hardware_backend.ip is None:
            raise RuntimeError("Hardware backend not initialised")

        data_word = struct.unpack('<I', spike_data)[0]
        ip = self._hardware_backend.ip
        ip.write(0x100, data_word)
        ip.write(0x104, 0x1)
        ip.write(0x104, 0x0)
        self.spike_count += 1
    
    def _read_output_spikes(self) -> List[SpikeEvent]:
        """Read output spikes from hardware."""
        if self.simulation_mode:
            items: List[SpikeEvent] = []
            while not self.output_queue.empty():
                items.append(self.output_queue.get())
            return items

        if self._hardware_backend is None or self._hardware_backend.ip is None:
            return []

        output_spikes = []
        ip = self._hardware_backend.ip
        status = ip.read(0x108)
        if status & 0x1:
            spike_data = ip.read(0x10C)
            neuron_id = spike_data & 0xFF
            timestamp = ((spike_data >> 16) & 0xFFFF) / 1e6
            output_spikes.append(SpikeEvent(neuron_id=neuron_id, timestamp=timestamp, weight=1.0))
            ip.write(0x108, 0x1)
        return output_spikes
    
    def run_simulation(self, duration: float, input_spikes: List[SpikeEvent]) -> List[SpikeEvent]:
        """
        Run a simulation with given input spikes.
        
        Args:
            duration: Simulation duration in seconds
            input_spikes: List of input spike events
            
        Returns:
            List of output spike events
        """
        # Software simulation path
        if self.simulation_mode:
            normalized = self._normalize_spike_events(input_spikes)
            result = self._software_backend.run(normalized, duration)
            logger.info(
                "Software simulation completed: %d input spikes, %d output spikes",
                len(normalized),
                len(result.events),
            )
            return result.events
        
        # XRT hardware path
        if self.use_xrt and self._xrt_backend is not None:
            return self._run_simulation_xrt(duration, input_spikes)

        # PYNQ hardware path
        while not self.spike_queue.empty():
            self.spike_queue.get()
        while not self.output_queue.empty():
            self.output_queue.get()

        self.start_processing()
        start_time = time.time()
        for spike in sorted(input_spikes, key=lambda x: x.timestamp):
            self.send_spike_event(spike.neuron_id, spike.timestamp, spike.weight)

        elapsed = 0.0
        output_spikes: List[SpikeEvent] = []
        while elapsed < duration:
            while not self.output_queue.empty():
                output_spikes.append(self.output_queue.get())
            time.sleep(0.01)
            elapsed = time.time() - start_time

        self.stop_processing()
        while not self.output_queue.empty():
            output_spikes.append(self.output_queue.get())

        logger.info(
            "Hardware simulation completed: %d input spikes, %d output spikes",
            len(input_spikes),
            len(output_spikes),
        )
        return output_spikes

    # ------------------------------------------------------------------
    # High-level convenience APIs
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to hardware or initialise the software backend."""

        if self.simulation_mode:
            logger.info("Using software-only backend (simulation mode)")
            return True

        self.load_bitstream()
        return True

    def disconnect(self) -> None:
        """Release hardware resources (if in use)."""

        self.stop_processing()
        if self._hardware_backend is not None:
            self._hardware_backend.disconnect()

    def infer(
        self,
        input_spikes: Union[List[SpikeEvent], Sequence[SpikeEvent], np.ndarray],
        duration: Optional[float] = None,
        return_events: bool = False,
    ) -> Union[List[SpikeEvent], np.ndarray]:
        """Run inference and return spike rates or events.

        Parameters
        ----------
        input_spikes:
            Iterable of :class:`SpikeEvent` objects or an ``(N, 3)`` NumPy array
            containing ``(neuron_id, timestamp, weight)`` triplets.
        duration:
            Optional simulation time.  When ``None`` the duration is inferred
            from the latest spike timestamp plus 10 ms.
        return_events:
            When true the raw spike events are returned instead of firing rates.
            
        Note:
            Processes the entire simulation duration at once. For encoder-based
            workflows, use num_steps to specify the temporal resolution.
        """

        spikes_list = self._normalize_spike_events(input_spikes)
        if duration is None:
            duration = self._infer_duration(spikes_list)

        output_events = self.run_simulation(duration, spikes_list)
        if return_events:
            return output_events

        rates = self._spike_events_to_rates(output_events, duration)
        return rates

    def reset_state(self) -> None:
        """Set the step mode for inference.
        
        Parameters
        ----------
        mode : str
            Either "multi" for multi-step mode (process entire simulation at once)
            or "single" for single-step mode (process one timestep at a time).
        timestep_dt : float, optional
            Timestep duration in seconds for single-step mode. Default is 0.001 (1ms).
            
        Example
        -------
        >>> output = accelerator.infer(all_spikes, duration=0.1)
        
        """
        self.spike_history.clear()
        self.current_timestep = 0
        logger.info("Simulation state reset")

    def set_mode(self, mode: int, encoder_enable: bool = False) -> None:
        """Set hardware mode register (0=infer,1=STDP,2=checkpoint)."""
        if self.use_xrt and self._xrt_backend is not None:
            self._xrt_backend.set_mode(mode, encoder_enable)

    def set_simulation_steps(self, steps: int) -> None:
        """Explicitly program time_steps register when using XRT backend."""
        if self.use_xrt and self._xrt_backend is not None:
            self._xrt_backend.set_time_steps(steps)

    def set_reward_signal(self, reward: int) -> None:
        if self.use_xrt and self._xrt_backend is not None:
            self._xrt_backend.set_reward(reward)
    
    def get_spike_history(self) -> List[List[SpikeEvent]]:
        """Get the complete spike history.
        
        Returns
        -------
        history : List[List[SpikeEvent]]
            List where each element contains the spike events for that timestep.
        """
        return self.spike_history

    def infer_with_learning(
        self,
        input_spikes: Union[List[SpikeEvent], Sequence[SpikeEvent], np.ndarray],
        reward: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> np.ndarray:
        """Run inference while optionally applying learning updates."""

        if reward is not None and self.learning_engine is not None:
            self.apply_reward(reward)
        if self.learning_enabled and self.learning_engine is None:
            logger.warning("Learning enabled but no learning engine configured")
        return cast(
            np.ndarray,
            self.infer(input_spikes, duration=duration, return_events=False),
        )

    def configure_learning(self, learning_engine: Union[STDPLearning, RSTDPLearning]) -> None:
        """Attach a learning engine to the accelerator."""

        self.learning_engine = learning_engine
        if self.simulation_mode:
            self._software_backend.configure_learning(learning_engine)
        logger.info("Learning engine configured: %s", learning_engine.__class__.__name__)

    def enable_learning(self, enable: bool = True) -> None:
        """Enable or disable on-line learning."""

        self.learning_enabled = enable
        if self.simulation_mode:
            self._software_backend.enable_learning(enable)
        elif self._hardware_backend is not None and self._hardware_backend.ip is not None:
            ip = self._hardware_backend.ip
            ip.write(0x208, 1 if enable else 0)
        logger.info("Learning %s", "enabled" if enable else "disabled")

    def apply_reward(self, reward: float) -> None:
        """Apply an external reward signal (R-STDP)."""

        if self.learning_engine is None:
            logger.warning("No learning engine configured – reward ignored")
            return

        if isinstance(self.learning_engine, RSTDPLearning):
            timestamp = time.time()
            self.learning_engine.add_reward(reward, timestamp)
            logger.debug("Reward %.4f applied at timestamp %.3f", reward, timestamp)

    def verify_weights(self) -> bool:
        """Best-effort weight verification."""

        if self.simulation_mode:
            return True
        if self._hardware_backend is None or self._hardware_backend.ip is None:
            return False
        # Placeholder for real verification – the register map can expose a checksum
        logger.info("Hardware weight verification not yet implemented, assuming success")
        return True
    
    def set_learning_parameters(self, learning_rate: float, stdp_window: float) -> None:
        """Configure STDP learning parameters."""
        if self.simulation_mode:
            logger.debug(
                "Simulation mode: skipping hardware learning parameter configuration"
            )
            return

        if self._hardware_backend is None or self._hardware_backend.ip is None:
            raise RuntimeError("Hardware backend not initialised")

        lr_fixed = int(learning_rate * 1024)
        window_fixed = int(stdp_window * 1e6)
        ip = self._hardware_backend.ip
        ip.write(0x200, lr_fixed)
        ip.write(0x204, window_fixed)
        logger.info("Learning parameters set: lr=%s, window=%ss", learning_rate, stdp_window)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics from the accelerator."""
        return {
            'total_spikes_processed': self.spike_count,
            'last_spike_time': self.last_spike_time,
            'hardware_status': (
                self._hardware_backend.ip.read(0x08)
                if self._hardware_backend is not None and self._hardware_backend.ip is not None
                else 0
            ),
            'queue_depth': self.spike_queue.qsize(),
        }
    
    def reset(self) -> None:
        """Reset the SNN accelerator state."""
        if self._hardware_backend is not None and self._hardware_backend.ip is not None:
            ip = self._hardware_backend.ip
            ip.write(0x00, 0x1)
            time.sleep(0.01)
            ip.write(0x00, 0x0)

        self.spike_count = 0
        self.current_timestep = 0
        self.membrane_state = None
        self.spike_history = []
        
        while not self.spike_queue.empty():
            self.spike_queue.get()
        while not self.output_queue.empty():
            self.output_queue.get()
            
        logger.info("Accelerator reset (membrane state cleared)")
    
    def __enter__(self):
        """Context manager entry."""
        if not self.simulation_mode and self._hardware_backend is None:
            self.load_bitstream()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_processing()
        if self._hardware_backend is not None:
            self._hardware_backend.disconnect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_spike_events(
        self, spikes: Union[List[SpikeEvent], Sequence[SpikeEvent], np.ndarray]
    ) -> List[SpikeEvent]:
        """Convert various spike representations into a list of SpikeEvent."""

        if isinstance(spikes, np.ndarray):
            if spikes.ndim != 2 or spikes.shape[1] < 2:
                raise ValueError("Spike array must have shape (N, >=2)")
            events = [
                SpikeEvent(
                    neuron_id=int(row[0]),
                    timestamp=float(row[1]),
                    weight=float(row[2]) if row.shape[0] > 2 else 1.0,
                )
                for row in spikes
            ]
            return events

        if isinstance(spikes, SpikeEvent):
            return [spikes]

        if isinstance(spikes, (list, tuple)):
            normalized: List[SpikeEvent] = []
            for event in spikes:
                if isinstance(event, SpikeEvent):
                    normalized.append(event)
                elif isinstance(event, dict):
                    normalized.append(
                        SpikeEvent(
                            neuron_id=int(event.get("neuron_id", 0)),
                            timestamp=float(event.get("timestamp", 0.0)),
                            weight=float(event.get("weight", 1.0)),
                        )
                    )
                else:
                    neuron_id, timestamp, *rest = event  # type: ignore[misc]
                    weight = rest[0] if rest else 1.0
                    normalized.append(
                        SpikeEvent(
                            neuron_id=int(neuron_id),
                            timestamp=float(timestamp),
                            weight=float(weight),
                        )
                    )
            return normalized

        raise TypeError("Unsupported spike input type")

    def _spike_events_to_rates(self, events: List[SpikeEvent], duration: float) -> np.ndarray:
        """Convert spike events to firing rates.
        
        When an SNN model is configured, only return firing rates for output layer neurons.
        Otherwise return rates for all neurons.
        """

        # If we have a model, filter for output layer neurons only
        if self.snn_model and self.snn_model.layers:
            output_size = self.snn_model.layers[-1].output_size
            output_layer_start, _ = self.snn_model.get_layer_neuron_ids(len(self.snn_model.layers) - 1)
            
            # Count spikes only from output layer neurons
            counts = np.zeros(output_size, dtype=np.float32)
            for event in events:
                if output_layer_start <= event.neuron_id < output_layer_start + output_size:
                    local_id = event.neuron_id - output_layer_start
                    counts[local_id] += 1
            
            rates = counts / max(duration, 1e-6)
            return rates
        
        # No model configured - return all neuron rates
        if not events:
            return np.array([])

        max_neuron = max(event.neuron_id for event in events)
        counts = np.zeros(max_neuron + 1, dtype=np.float32)
        for event in events:
            counts[event.neuron_id] += 1

        rates = counts / max(duration, 1e-6)
        return rates

    @staticmethod
    def _infer_duration(events: List[SpikeEvent]) -> float:
        if not events:
            return 0.1
        return max(event.timestamp for event in events) + 0.01

    @staticmethod
    def _load_config_from_file(path: Path) -> Dict:
        logger.info("Loading network configuration from %s", path)
        if path.suffix in {".json", ".yaml", ".yml"}:
            with path.open() as fh:
                return json.load(fh)
        raise ValueError(f"Unsupported configuration format: {path.suffix}")

    def _load_model_to_hardware(self, model: SNNModel) -> None:
        if self.simulation_mode:
            return

        for layer in model.layers:
            if layer.weights is None:
                continue
            self._load_weights(layer.weights)
