"""
SNN FPGA Accelerator Package

Event-driven Spiking Neural Network accelerator for FPGA with PyTorch integration.

Author: Jiwoon Lee (@metr0jw)
Organization: Kwangwoon University, Seoul, South Korea
Contact: jwlee@linux.com
"""

__version__ = "0.1.0"
__author__ = "Jiwoon Lee"

from .accelerator import SNNAccelerator
from .pytorch_interface import pytorch_to_snn, SNNLayer, SNNModel, CPUvsSNNComparator
from .spike_encoding import PoissonEncoder, TemporalEncoder, RateEncoder
from .learning import STDPLearning, RSTDPLearning
from .utils import load_weights, save_weights, visualize_spikes
from .rtl_simulator import (
    IcarusSimulator,
    CocotbSimulator,
    RTLvsPythonComparator,
    check_simulation_tools,
    print_tool_status,
    ICARUS_AVAILABLE,
    VERILATOR_AVAILABLE,
    COCOTB_AVAILABLE,
)
from .hw_accurate_simulator import (
    FixedPoint,
    HWAccurateLIFNeuron,
    HWAccurateSTDPEngine,
    HWAccurateSNNSimulator,
    LIFNeuronParams,
    LIFNeuronState,
    STDPConfig,
    WeightUpdate,
    verify_lif_neuron,
    verify_stdp_engine,
)

__all__ = [
    # Core
    "SNNAccelerator",
    "pytorch_to_snn",
    "SNNLayer",
    "SNNModel",
    "CPUvsSNNComparator",
    # Encoders
    "PoissonEncoder",
    "TemporalEncoder",
    "RateEncoder",
    # Learning
    "STDPLearning",
    "RSTDPLearning",
    # Utilities
    "load_weights",
    "save_weights",
    "visualize_spikes",
    # RTL Simulation
    "IcarusSimulator",
    "CocotbSimulator",
    "RTLvsPythonComparator",
    "check_simulation_tools",
    "print_tool_status",
    "ICARUS_AVAILABLE",
    "VERILATOR_AVAILABLE",
    "COCOTB_AVAILABLE",
    # HW-Accurate Simulation
    "FixedPoint",
    "HWAccurateLIFNeuron",
    "HWAccurateSTDPEngine",
    "HWAccurateSNNSimulator",
    "LIFNeuronParams",
    "LIFNeuronState",
    "STDPConfig",
    "WeightUpdate",
    "verify_lif_neuron",
    "verify_stdp_engine",
]
