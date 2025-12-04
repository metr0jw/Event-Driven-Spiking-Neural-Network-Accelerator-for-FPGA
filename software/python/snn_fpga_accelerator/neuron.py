"""
Spiking Neuron Modules

PyTorch-compatible spiking neuron modules that work like activation functions.
Supports surrogate gradient training and hardware-constrained quantization.

Usage:
    # Like activation functions
    x = nn.Linear(784, 128)(input)
    x = snn.LIF()(x)  # or snn.IF(), snn.ALIF()
    
    # With custom parameters
    lif = snn.LIF(thresh=1.0, tau=0.9, learn_thresh=True)
    
    # HW-constrained mode
    lif = snn.LIF(hw_mode=True)  # Uses 8-bit weights, 16-bit membrane

Author: Jiwoon Lee (@metr0jw)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, Callable, Union
import math

__all__ = [
    # Neurons
    'LIF', 'IF', 'ALIF', 'PLIF', 'Izhikevich',
    # Surrogate gradients  
    'FastSigmoid', 'ATan', 'SuperSpike', 'SigmoidGrad', 'PiecewiseLinear',
    # Utilities
    'reset_neurons', 'detach_states',
]


# =============================================================================
# Surrogate Gradient Functions
# =============================================================================

class SurrogateGradient(torch.autograd.Function):
    """Base class for surrogate gradient functions."""
    scale: float = 25.0
    
    @staticmethod
    def forward(ctx, input: Tensor, thresh: float = 1.0) -> Tensor:
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        return (input >= thresh).float()
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        raise NotImplementedError


class FastSigmoid(SurrogateGradient):
    """
    Fast sigmoid surrogate gradient.
    Gradient: scale / (1 + scale * |v - thresh|)^2
    
    Reference: Zenke & Ganguli (2018)
    """
    @staticmethod
    def forward(ctx, input: Tensor, thresh: float = 1.0) -> Tensor:
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        return (input >= thresh).float()
    
    @staticmethod  
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        input, = ctx.saved_tensors
        scale = FastSigmoid.scale
        grad = scale / (1.0 + scale * torch.abs(input - ctx.thresh)) ** 2
        return grad_output * grad, None


class ATan(SurrogateGradient):
    """
    Arctangent surrogate gradient.
    Gradient: scale / (1 + (scale * π * (v - thresh))^2)
    
    Reference: Fang et al. (2021)
    """
    @staticmethod
    def forward(ctx, input: Tensor, thresh: float = 1.0) -> Tensor:
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        return (input >= thresh).float()
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        input, = ctx.saved_tensors
        scale = ATan.scale
        alpha = math.pi * scale
        grad = scale / (1.0 + (alpha * (input - ctx.thresh)) ** 2)
        return grad_output * grad, None


class SuperSpike(SurrogateGradient):
    """
    SuperSpike surrogate gradient.
    Gradient: 1 / (scale * |v - thresh| + 1)^2
    
    Reference: Zenke & Ganguli (2018)
    """
    @staticmethod
    def forward(ctx, input: Tensor, thresh: float = 1.0) -> Tensor:
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        return (input >= thresh).float()
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        input, = ctx.saved_tensors
        scale = SuperSpike.scale
        grad = 1.0 / (scale * torch.abs(input - ctx.thresh) + 1.0) ** 2
        return grad_output * grad, None


class SigmoidGrad(SurrogateGradient):
    """
    Sigmoid surrogate gradient.
    Gradient: scale * sigmoid(scale * (v - thresh)) * (1 - sigmoid(...))
    """
    @staticmethod
    def forward(ctx, input: Tensor, thresh: float = 1.0) -> Tensor:
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        return (input >= thresh).float()
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        input, = ctx.saved_tensors
        scale = SigmoidGrad.scale
        sig = torch.sigmoid(scale * (input - ctx.thresh))
        grad = scale * sig * (1 - sig)
        return grad_output * grad, None


class PiecewiseLinear(SurrogateGradient):
    """
    Piecewise linear surrogate gradient.
    Gradient: max(0, 1 - |v - thresh|) * scale
    """
    @staticmethod
    def forward(ctx, input: Tensor, thresh: float = 1.0) -> Tensor:
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        return (input >= thresh).float()
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        input, = ctx.saved_tensors
        scale = PiecewiseLinear.scale
        grad = torch.clamp(1.0 - torch.abs(input - ctx.thresh), min=0) * scale
        return grad_output * grad, None


# Alias for convenience
spike_fn = FastSigmoid.apply


def get_surrogate(name: str = 'fast_sigmoid', scale: float = 25.0) -> Callable:
    """
    Get a surrogate gradient function by name.
    
    Args:
        name: One of 'fast_sigmoid', 'atan', 'super_spike', 'sigmoid', 'pwl'
        scale: Gradient scale factor
        
    Returns:
        Surrogate gradient function
    """
    surrogates = {
        'fast_sigmoid': FastSigmoid,
        'atan': ATan,
        'super_spike': SuperSpike,
        'sigmoid': SigmoidGrad,
        'pwl': PiecewiseLinear,
    }
    
    if name not in surrogates:
        raise ValueError(f"Unknown surrogate: {name}. Choose from {list(surrogates.keys())}")
    
    cls = surrogates[name]
    cls.scale = scale
    return cls.apply


# =============================================================================
# Hardware Constants (match lif_neuron.v)
# =============================================================================

# Shift-based tau lookup table
# leak_rate[2:0] = primary shift, leak_rate[7:3] = secondary shift
HW_TAU_TABLE = {
    # Single shift configurations
    1: 0.5,      # 1 - 1/2
    2: 0.75,     # 1 - 1/4
    3: 0.875,    # 1 - 1/8
    4: 0.9375,   # 1 - 1/16
    5: 0.96875,  # 1 - 1/32
    6: 0.984375, # 1 - 1/64
    7: 0.9921875,# 1 - 1/128
}


def tau_to_hw_leak_rate(tau: float) -> int:
    """
    Convert tau value to hardware leak_rate configuration.
    
    Returns the best shift configuration to approximate the target tau.
    Format: leak_rate[2:0] = shift1, leak_rate[7:3] = shift2 (0=disabled)
    """
    best_error = float('inf')
    best_config = 3  # Default: tau ≈ 0.875
    
    # Try single shift configurations
    for shift1 in range(1, 8):
        approx_tau = 1.0 - 1.0 / (1 << shift1)
        error = abs(approx_tau - tau)
        if error < best_error:
            best_error = error
            best_config = shift1
    
    # Try dual shift configurations for finer control
    for shift1 in range(1, 8):
        for shift2 in range(1, 8):
            if shift2 == shift1:
                continue
            approx_tau = 1.0 - 1.0 / (1 << shift1) - 1.0 / (1 << shift2)
            if approx_tau <= 0 or approx_tau >= 1:
                continue
            error = abs(approx_tau - tau)
            if error < best_error:
                best_error = error
                best_config = shift1 | (shift2 << 3)
    
    return best_config


def hw_leak_rate_to_tau(leak_rate: int) -> float:
    """
    Convert hardware leak_rate configuration to tau value.
    """
    shift1 = leak_rate & 0x07
    shift2_cfg = (leak_rate >> 3) & 0x1F
    shift2 = shift2_cfg & 0x07 if shift2_cfg != 0 else 0
    
    tau = 1.0
    if shift1 > 0:
        tau -= 1.0 / (1 << shift1)
    if shift2_cfg != 0 and shift2 > 0:
        tau -= 1.0 / (1 << shift2)
    return tau


# =============================================================================
# Spiking Neuron Modules
# =============================================================================

class SpikingNeuron(nn.Module):
    """
    Base class for all spiking neurons.
    
    Maintains membrane potential state and handles reset/detach.
    
    Hardware Mode (hw_mode=True):
        - Uses 16-bit unsigned membrane potential [0, 65535]
        - Uses 8-bit signed weights [-128, 127]
        - Uses shift-based exponential leak (no multiplier)
        - Matches lif_neuron.v exactly
    """
    
    def __init__(
        self,
        thresh: float = 1.0,
        reset: str = 'subtract',  # 'subtract' or 'zero'
        surrogate: str = 'fast_sigmoid',
        scale: float = 25.0,
        hw_mode: bool = False,
        learn_thresh: bool = False,
    ):
        super().__init__()
        
        # Threshold (optionally learnable)
        if learn_thresh:
            self.thresh = nn.Parameter(torch.tensor(thresh))
        else:
            self.register_buffer('thresh', torch.tensor(thresh))
        
        self.reset_mode = reset
        self.hw_mode = hw_mode
        self.surrogate_fn = get_surrogate(surrogate, scale)
        
        # State
        self.mem: Optional[Tensor] = None
        
        # HW constraints
        if hw_mode:
            self.mem_bits = 16
            self.weight_bits = 8
            self.mem_max = 65535  # 16-bit unsigned max
            self.mem_min = 0
    
    def init_state(self, shape: Tuple, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Initialize membrane potential."""
        return torch.zeros(shape, device=device, dtype=dtype)
    
    def reset_state(self) -> None:
        """Reset membrane potential to zero."""
        self.mem = None
    
    def detach_state(self) -> None:
        """Detach membrane potential from computation graph."""
        if self.mem is not None:
            self.mem = self.mem.detach()
    
    def fire(self, mem: Tensor) -> Tensor:
        """Generate spikes using surrogate gradient."""
        return self.surrogate_fn(mem, self.thresh.item() if isinstance(self.thresh, Tensor) else self.thresh)
    
    def reset_mem(self, mem: Tensor, spk: Tensor) -> Tensor:
        """Reset membrane potential after spike."""
        if self.reset_mode == 'subtract':
            return mem - spk * self.thresh
        else:  # zero reset
            return mem * (1 - spk)
    
    def quantize(self, x: Tensor) -> Tensor:
        """Quantize for HW compatibility (16-bit unsigned)."""
        if self.hw_mode:
            return torch.clamp(x, self.mem_min, self.mem_max)
        return x


class LIF(SpikingNeuron):
    """
    Leaky Integrate-and-Fire neuron.
    
    Software mode (hw_mode=False):
        mem[t] = tau * mem[t-1] + input[t]
        
    Hardware mode (hw_mode=True):
        mem[t] = mem[t-1] - (mem[t-1] >> shift1) - (mem[t-1] >> shift2) + input[t]
        This is equivalent to: mem[t] = tau * mem[t-1] + input[t]
        where tau = 1 - 2^(-shift1) - 2^(-shift2)
    
    Args:
        thresh: Firing threshold (default: 1.0)
        tau: Membrane time constant / leak factor (default: 0.9)
        reset: Reset mode - 'subtract' or 'zero' (default: 'subtract')
        surrogate: Surrogate gradient - 'fast_sigmoid', 'atan', 'super_spike', 'sigmoid', 'pwl'
        scale: Surrogate gradient scale (default: 25.0)
        hw_mode: Enable hardware constraints (default: False)
        learn_tau: Make tau learnable (default: False)
        learn_thresh: Make threshold learnable (default: False)
        
    Hardware Tau Approximations:
        tau=0.875  -> leak_rate=3  (shift=3)
        tau=0.9375 -> leak_rate=4  (shift=4)
        tau≈0.922  -> leak_rate=52 (shift1=4, shift2=6)
        tau≈0.906  -> leak_rate=35 (shift1=3, shift2=4)
        
    Examples:
        >>> lif = LIF()
        >>> spk = lif(input)
        
        >>> # With learnable parameters
        >>> lif = LIF(learn_tau=True, learn_thresh=True)
        
        >>> # Hardware-constrained (exact HW match)
        >>> lif = LIF(hw_mode=True, tau=0.9)
    """
    
    def __init__(
        self,
        thresh: float = 1.0,
        tau: float = 0.9,
        reset: str = 'subtract',
        surrogate: str = 'fast_sigmoid',
        scale: float = 25.0,
        hw_mode: bool = False,
        learn_tau: bool = False,
        learn_thresh: bool = False,
    ):
        super().__init__(thresh, reset, surrogate, scale, hw_mode, learn_thresh)
        
        # Tau (optionally learnable)
        if learn_tau:
            # Use sigmoid to keep tau in (0, 1)
            self.tau_param = nn.Parameter(torch.tensor(math.log(tau / (1 - tau))))
        else:
            self.register_buffer('tau_param', torch.tensor(tau))
        self.learn_tau = learn_tau
        
        # HW mode: compute shift configuration
        if hw_mode:
            self._hw_leak_rate = tau_to_hw_leak_rate(tau)
            self._hw_tau = hw_leak_rate_to_tau(self._hw_leak_rate)
            # Store shift values
            self._shift1 = self._hw_leak_rate & 0x07
            self._shift2_cfg = (self._hw_leak_rate >> 3) & 0x1F
            self._shift2 = self._shift2_cfg & 0x07 if self._shift2_cfg != 0 else 0
    
    @property
    def tau(self) -> Tensor:
        if self.learn_tau:
            return torch.sigmoid(self.tau_param)
        return self.tau_param
    
    def _hw_leak(self, mem: Tensor) -> Tensor:
        """
        Hardware-accurate shift-based leak.
        
        Matches lif_neuron.v exactly:
            leak_primary = v_mem >> shift1
            leak_secondary = v_mem >> shift2 (if enabled)
            v_mem_next = v_mem - leak_primary - leak_secondary
        """
        # Primary leak
        if self._shift1 > 0:
            leak_primary = torch.floor(mem / (2 ** self._shift1))
        else:
            leak_primary = torch.zeros_like(mem)
        
        # Secondary leak
        if self._shift2_cfg != 0 and self._shift2 > 0:
            leak_secondary = torch.floor(mem / (2 ** self._shift2))
        else:
            leak_secondary = torch.zeros_like(mem)
        
        # Apply leak
        mem_leaked = mem - leak_primary - leak_secondary
        
        # Saturate at 0
        return torch.clamp(mem_leaked, min=0)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, features) or (batch, time, features)
            
        Returns:
            Output spikes of same shape as input
        """
        # Initialize state if needed
        if self.mem is None:
            self.mem = self.init_state(x.shape, x.device, x.dtype)
        
        if self.hw_mode:
            # Hardware-accurate mode: shift-based leak
            self.mem = self._hw_leak(self.mem) + x
            # Quantize to 16-bit unsigned
            self.mem = self.quantize(self.mem)
        else:
            # Software mode: multiplicative leak
            self.mem = self.tau * self.mem + x
        
        # Fire
        spk = self.fire(self.mem)
        
        # Reset
        self.mem = self.reset_mem(self.mem, spk)
        
        return spk
    
    def get_hw_config(self) -> dict:
        """Get hardware configuration for this neuron."""
        if not self.hw_mode:
            return {}
        return {
            'leak_rate': self._hw_leak_rate,
            'shift1': self._shift1,
            'shift2': self._shift2,
            'effective_tau': self._hw_tau,
            'threshold': int(self.thresh.item()) if isinstance(self.thresh, Tensor) else int(self.thresh),
        }


class IF(SpikingNeuron):
    """
    Integrate-and-Fire neuron (no leak).
    
    mem[t] = mem[t-1] + input[t]
    spk[t] = fire(mem[t])
    mem[t] = reset(mem[t], spk[t])
    
    Simplest spiking neuron model. More hardware-efficient than LIF.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        if self.mem is None:
            self.mem = self.init_state(x.shape, x.device, x.dtype)
        
        # Integrate (no leak)
        self.mem = self.mem + x
        self.mem = self.quantize(self.mem)
        
        # Fire
        spk = self.fire(self.mem)
        
        # Reset
        self.mem = self.reset_mem(self.mem, spk)
        
        return spk


class ALIF(SpikingNeuron):
    """
    Adaptive Leaky Integrate-and-Fire neuron.
    
    Has an adaptive threshold that increases after each spike.
    Useful for temporal pattern recognition.
    
    mem[t] = tau_mem * mem[t-1] + input[t]
    thresh_adapt[t] = tau_thresh * thresh_adapt[t-1] + (1-tau_thresh) * spk[t-1] * beta
    spk[t] = fire(mem[t], thresh + thresh_adapt[t])
    """
    
    def __init__(
        self,
        thresh: float = 1.0,
        tau: float = 0.9,
        tau_thresh: float = 0.95,
        beta: float = 0.1,
        **kwargs
    ):
        super().__init__(thresh=thresh, **kwargs)
        self.register_buffer('tau_mem', torch.tensor(tau))
        self.register_buffer('tau_thresh', torch.tensor(tau_thresh))
        self.register_buffer('beta', torch.tensor(beta))
        self.thresh_adapt: Optional[Tensor] = None
    
    def reset_state(self) -> None:
        super().reset_state()
        self.thresh_adapt = None
    
    def forward(self, x: Tensor) -> Tensor:
        if self.mem is None:
            self.mem = self.init_state(x.shape, x.device, x.dtype)
            self.thresh_adapt = self.init_state(x.shape, x.device, x.dtype)
        
        # Leak + integrate
        self.mem = self.tau_mem * self.mem + x
        self.mem = self.quantize(self.mem)
        
        # Adaptive threshold
        effective_thresh = self.thresh + self.thresh_adapt
        
        # Fire with adaptive threshold
        spk = self.surrogate_fn(self.mem, effective_thresh.mean().item())
        
        # Update adaptive threshold
        self.thresh_adapt = self.tau_thresh * self.thresh_adapt + (1 - self.tau_thresh) * spk * self.beta
        
        # Reset
        self.mem = self.reset_mem(self.mem, spk)
        
        return spk


class PLIF(LIF):
    """
    Parametric Leaky Integrate-and-Fire neuron.
    
    LIF with learnable tau and threshold by default.
    Commonly used in deep SNN training.
    """
    
    def __init__(self, thresh: float = 1.0, tau: float = 0.9, **kwargs):
        super().__init__(thresh=thresh, tau=tau, learn_tau=True, learn_thresh=True, **kwargs)


class Izhikevich(SpikingNeuron):
    """
    Izhikevich neuron model.
    
    Can reproduce various spiking patterns (regular, bursting, chattering, etc.)
    More biologically realistic but more computationally expensive.
    
    dv/dt = 0.04*v^2 + 5*v + 140 - u + I
    du/dt = a*(b*v - u)
    if v >= 30: v = c, u = u + d
    """
    
    def __init__(
        self,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        dt: float = 1.0,
        **kwargs
    ):
        super().__init__(thresh=30.0, **kwargs)
        self.register_buffer('a', torch.tensor(a))
        self.register_buffer('b', torch.tensor(b))
        self.register_buffer('c', torch.tensor(c))
        self.register_buffer('d', torch.tensor(d))
        self.register_buffer('dt', torch.tensor(dt))
        self.u: Optional[Tensor] = None
    
    def reset_state(self) -> None:
        super().reset_state()
        self.u = None
    
    def forward(self, x: Tensor) -> Tensor:
        if self.mem is None:
            self.mem = torch.full(x.shape, -65.0, device=x.device, dtype=x.dtype)
            self.u = torch.full(x.shape, -14.0, device=x.device, dtype=x.dtype)
        
        # Scale input
        I = x * 10.0
        
        # Izhikevich dynamics
        v = self.mem
        dv = (0.04 * v ** 2 + 5 * v + 140 - self.u + I) * self.dt
        du = self.a * (self.b * v - self.u) * self.dt
        
        self.mem = v + dv
        self.u = self.u + du
        
        # Fire
        spk = self.fire(self.mem)
        
        # Reset
        self.mem = torch.where(spk.bool(), self.c, self.mem)
        self.u = torch.where(spk.bool(), self.u + self.d, self.u)
        
        return spk


# =============================================================================
# Utility Functions
# =============================================================================

def reset_neurons(model: nn.Module) -> None:
    """
    Reset all spiking neurons in a model.
    
    Call this between sequences/batches.
    
    Args:
        model: PyTorch model containing spiking neurons
    """
    for module in model.modules():
        if isinstance(module, SpikingNeuron):
            module.reset_state()


def detach_states(model: nn.Module) -> None:
    """
    Detach all neuron states from computation graph.
    
    Useful for TBPTT (Truncated Backpropagation Through Time).
    
    Args:
        model: PyTorch model containing spiking neurons
    """
    for module in model.modules():
        if isinstance(module, SpikingNeuron):
            module.detach_state()


def set_hw_mode(model: nn.Module, enabled: bool = True) -> None:
    """
    Enable/disable hardware mode for all neurons.
    
    Args:
        model: PyTorch model
        enabled: Enable HW constraints
    """
    for module in model.modules():
        if isinstance(module, SpikingNeuron):
            module.hw_mode = enabled
