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
from .pytorch_interface import pytorch_to_snn, SNNLayer
from .spike_encoding import PoissonEncoder, TemporalEncoder, RateEncoder
from .learning import STDPLearning, RSTDPLearning
from .utils import load_weights, save_weights, visualize_spikes

__all__ = [
    "SNNAccelerator",
    "pytorch_to_snn",
    "SNNLayer", 
    "PoissonEncoder",
    "TemporalEncoder",
    "RateEncoder",
    "STDPLearning",
    "RSTDPLearning",
    "load_weights",
    "save_weights",
    "visualize_spikes",
]
