"""
SNN FPGA Accelerator - Event-Driven Spiking Neural Network Library

A PyTorch-like library for building, training, and deploying SNNs on FPGAs.
Supports surrogate gradient training and hardware-constrained learning.

Quick Start:
    import snn_fpga_accelerator as snn
    
    # Build model (like PyTorch)
    model = nn.Sequential(
        snn.Linear(784, 256),
        snn.LIF(),
        snn.Linear(256, 10),
        snn.LIF(),
    )
    
    # Train with surrogate gradient
    loss_fn = snn.loss.CrossEntropy()
    optimizer = torch.optim.Adam(model.parameters())
    trainer = snn.Trainer(model, optimizer, loss_fn)
    trainer.fit(train_loader, epochs=10)
    
    # Deploy to FPGA
    snn.deploy.export(model, 'weights.npz')
    fpga = snn.PYNQ('snn.bit')
    fpga.load_weights(model)
    output = fpga.run(test_data)

Author: Jiwoon Lee (@metr0jw)
Organization: Kwangwoon University
License: Apache 2.0
"""

__version__ = "0.2.0"
__author__ = "Jiwoon Lee"

# =============================================================================
# Core Imports - Neurons (like activation functions)
# =============================================================================
from .neuron import (
    # Spiking neurons
    LIF,              # Leaky Integrate-and-Fire (most common)
    IF,               # Integrate-and-Fire (simpler)
    ALIF,             # Adaptive LIF
    PLIF,             # Parametric LIF (learnable tau/thresh)
    Izhikevich,       # Biologically realistic
    
    # Surrogate gradients
    FastSigmoid,      # Default, fast
    ATan,             # Arctangent
    SuperSpike,       # SuperSpike
    SigmoidGrad,      # Sigmoid
    PiecewiseLinear,  # PWL
    get_surrogate,    # Get by name
    
    # Utilities
    reset_neurons,    # Reset all neuron states
    detach_states,    # Detach for TBPTT
    set_hw_mode,      # Enable HW constraints
    
    # Base class
    SpikingNeuron,
)

# =============================================================================
# Layers (like nn.Linear, nn.Conv2d)
# =============================================================================
from .layer import (
    # Linear
    Linear,           # FC layer with HW quantization
    SLinear,          # Linear + LIF combined
    
    # Convolutional
    Conv2d,           # Conv2d with HW quantization  
    SConv2d,          # Conv2d + LIF combined
    
    # Pooling
    AvgPool2d,
    MaxPool2d,
    
    # Normalization
    BatchNorm,
    LayerNorm,
    
    # Recurrent
    SRNN,             # Spiking RNN
    SLSTM,            # Spiking LSTM-like
    
    # Containers
    Sequential,       # nn.Sequential with reset()
    SNN,              # Base class for SNN models
    
    # Regularization
    Dropout,
)

# =============================================================================
# Encoding/Decoding
# =============================================================================
from .encoder import (
    # Encoders
    Rate,             # Rate coding (spike prob = intensity)
    Poisson,          # Poisson spike train
    Latency,          # Time-to-first-spike
    Temporal,         # Learnable temporal
    Delta,            # Event-based (DVS-like)
    Phase,            # Phase encoding
    
    # Decoders
    RateDecoder,      # Sum spike counts
    LatencyDecoder,   # First spike time
    MaxDecoder,       # Argmax for classification
    
    # Aliases
    RateEncoder,
    PoissonEncoder,
    LatencyEncoder,
)

# =============================================================================
# Training
# =============================================================================
from .training import (
    # Loss functions
    CrossEntropy,     # CE on spike counts
    MSE,              # MSE on spike rates
    SpikeCount,       # Spike count regularizer
    SpikeRate,        # Target spike rate loss
    MemPotential,     # Membrane potential loss
    
    # STDP learning
    STDP,             # Classic STDP
    RSTDP,            # Reward-modulated STDP
    STDPConfig,       # STDP parameters
    
    # Training utilities
    Trainer,          # High-level trainer
    accuracy,         # Accuracy metric
)

# =============================================================================
# Deployment (submodule)
# =============================================================================
from . import deploy
from .deploy import (
    # Quantization
    quantize,
    dequantize,
    calibrate,
    
    # Export
    export,
    export_onnx,
    
    # FPGA
    PYNQ,
    AXIInterface,
    
    # Config
    gen_config,
    NetworkConfig,
    LayerConfig,
)

# =============================================================================
# Hardware Simulation (submodule)
# =============================================================================
from . import hw_accurate_simulator as hw_sim
from .hw_accurate_simulator import (
    HWAccurateLIFNeuron,
    HWAccurateSTDPEngine,
    HWAccurateSNNSimulator,
    LIFNeuronParams,
    LIFNeuronState,
    verify_lif_neuron,
    verify_stdp_engine,
    FixedPoint,
)

# =============================================================================
# Legacy/Compatibility (will be deprecated)
# =============================================================================
from .accelerator import SNNAccelerator
from .pytorch_interface import pytorch_to_snn, SNNLayer, SNNModel
from .spike_encoding import SpikeEvent
from .learning import STDPLearning, RSTDPLearning, LearningConfig
from .utils import load_weights, save_weights, visualize_spikes

# =============================================================================
# Submodules
# =============================================================================
from . import neuron
from . import layer
from . import encoder
from . import training

# For loss functions as submodule
class loss:
    """Loss functions for SNN training."""
    CrossEntropy = CrossEntropy
    MSE = MSE
    SpikeCount = SpikeCount
    SpikeRate = SpikeRate
    MemPotential = MemPotential

# For surrogate gradients as submodule
class surrogate:
    """Surrogate gradient functions."""
    fast_sigmoid = FastSigmoid
    atan = ATan
    super_spike = SuperSpike
    sigmoid = SigmoidGrad
    pwl = PiecewiseLinear
    get = get_surrogate

# =============================================================================
# Functional API (like torch.nn.functional)
# =============================================================================
class functional:
    """Functional API for spiking operations."""
    
    @staticmethod
    def spike(mem, thresh: float = 1.0, surrogate: str = 'fast_sigmoid'):
        """Generate spikes from membrane potential."""
        spike_fn = get_surrogate(surrogate)
        return spike_fn(mem, thresh)
    
    @staticmethod
    def rate_encode(x, T: int = 100):
        """Rate encode input to spikes."""
        return Rate(T)(x)
    
    @staticmethod  
    def rate_decode(spikes, dim: int = 1):
        """Decode spike counts."""
        return spikes.sum(dim=dim)

F = functional  # Alias


# =============================================================================
# All exports
# =============================================================================
__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Neurons
    'LIF', 'IF', 'ALIF', 'PLIF', 'Izhikevich', 'SpikingNeuron',
    
    # Surrogate gradients
    'FastSigmoid', 'ATan', 'SuperSpike', 'SigmoidGrad', 'PiecewiseLinear',
    'get_surrogate', 'surrogate',
    
    # Neuron utilities
    'reset_neurons', 'detach_states', 'set_hw_mode',
    
    # Layers
    'Linear', 'SLinear', 'Conv2d', 'SConv2d',
    'AvgPool2d', 'MaxPool2d', 'BatchNorm', 'LayerNorm',
    'SRNN', 'SLSTM', 'Sequential', 'SNN', 'Dropout',
    
    # Encoding
    'Rate', 'Poisson', 'Latency', 'Temporal', 'Delta', 'Phase',
    'RateDecoder', 'LatencyDecoder', 'MaxDecoder',
    'RateEncoder', 'PoissonEncoder', 'LatencyEncoder',
    
    # Training
    'CrossEntropy', 'MSE', 'SpikeCount', 'SpikeRate', 'MemPotential',
    'STDP', 'RSTDP', 'STDPConfig',
    'Trainer', 'accuracy',
    'loss',
    
    # Deployment
    'deploy', 'quantize', 'dequantize', 'calibrate',
    'export', 'export_onnx',
    'PYNQ', 'AXIInterface',
    'gen_config', 'NetworkConfig', 'LayerConfig',
    
    # Hardware simulation
    'hw_sim',
    'HWAccurateLIFNeuron', 'HWAccurateSTDPEngine', 'HWAccurateSNNSimulator',
    'LIFNeuronParams', 'LIFNeuronState',
    'verify_lif_neuron', 'verify_stdp_engine', 'FixedPoint',
    
    # Submodules
    'neuron', 'layer', 'encoder', 'training',
    
    # Functional
    'functional', 'F',
    
    # Legacy (compatibility)
    'SNNAccelerator', 'pytorch_to_snn', 'SNNLayer', 'SNNModel',
    'SpikeEvent', 'STDPLearning', 'RSTDPLearning', 'LearningConfig',
    'load_weights', 'save_weights', 'visualize_spikes',
]
