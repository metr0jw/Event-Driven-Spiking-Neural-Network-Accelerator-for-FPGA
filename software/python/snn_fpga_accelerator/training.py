"""
Training Utilities for SNNs

Provides loss functions, training loops, and learning rules for SNNs.
Supports both surrogate gradient training and STDP-based learning.

Usage:
    # Surrogate gradient training
    loss_fn = snn.loss.CrossEntropy(T=100)
    loss = loss_fn(spk_out, target)
    
    # STDP training
    stdp = snn.STDP(model, lr=0.01)
    stdp.step(pre_spikes, post_spikes)

Author: Jiwoon Lee (@metr0jw)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
import math

__all__ = [
    # Loss functions
    'CrossEntropy', 'MSE', 'SpikeCount', 'SpikeRate', 'MemPotential',
    # STDP
    'STDP', 'RSTDP', 'STDPConfig',
    # Training utilities
    'Trainer', 'accuracy',
]


# =============================================================================
# Loss Functions
# =============================================================================

class CrossEntropy(nn.Module):
    """
    Cross entropy loss for spike trains.
    
    Uses spike counts (or rates) as logits.
    
    Args:
        T: Number of timesteps (for rate normalization)
        method: 'count' (sum spikes) or 'rate' (mean spikes)
        
    Examples:
        >>> loss_fn = CrossEntropy(T=100)
        >>> loss = loss_fn(output_spikes, targets)
    """
    
    def __init__(self, T: int = None, method: str = 'count'):
        super().__init__()
        self.T = T
        self.method = method
    
    def forward(self, spikes: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            spikes: (B, T, C) or (B, C) spike tensor
            target: (B,) class labels
            
        Returns:
            Cross entropy loss
        """
        # Sum over time if needed
        if spikes.dim() == 3:
            if self.method == 'rate':
                logits = spikes.mean(dim=1)
            else:
                logits = spikes.sum(dim=1)
        else:
            logits = spikes
        
        return F.cross_entropy(logits, target)


class MSE(nn.Module):
    """
    Mean squared error for spike trains.
    
    Compares spike rates to target rates.
    
    Args:
        method: 'count' or 'rate'
    """
    
    def __init__(self, method: str = 'rate'):
        super().__init__()
        self.method = method
    
    def forward(self, spikes: Tensor, target: Tensor) -> Tensor:
        if spikes.dim() == 3:
            if self.method == 'rate':
                pred = spikes.mean(dim=1)
            else:
                pred = spikes.sum(dim=1)
        else:
            pred = spikes
        
        return F.mse_loss(pred, target.float())


class SpikeCount(nn.Module):
    """
    Loss based on total spike count.
    
    Useful for regularization (encourage sparse spiking).
    
    Args:
        target_rate: Target spike rate per neuron
        penalty: Penalty for deviation from target
    """
    
    def __init__(self, target_rate: float = 0.1, penalty: float = 1.0):
        super().__init__()
        self.target_rate = target_rate
        self.penalty = penalty
    
    def forward(self, spikes: Tensor, target: Tensor = None) -> Tensor:
        # Calculate actual rate
        if spikes.dim() == 3:
            actual_rate = spikes.mean(dim=[0, 1])  # Per neuron average
        else:
            actual_rate = spikes.mean(dim=0)
        
        # Penalty for deviation from target
        loss = self.penalty * ((actual_rate - self.target_rate) ** 2).mean()
        
        return loss


class SpikeRate(nn.Module):
    """
    Spike rate loss for classification.
    
    Encourages correct class to have highest spike rate.
    
    Args:
        correct_rate: Target rate for correct class
        incorrect_rate: Target rate for incorrect classes
    """
    
    def __init__(self, correct_rate: float = 0.8, incorrect_rate: float = 0.2):
        super().__init__()
        self.correct_rate = correct_rate
        self.incorrect_rate = incorrect_rate
    
    def forward(self, spikes: Tensor, target: Tensor) -> Tensor:
        B, T, C = spikes.shape if spikes.dim() == 3 else (spikes.shape[0], 1, spikes.shape[1])
        
        rates = spikes.mean(dim=1) if spikes.dim() == 3 else spikes
        
        # Target rates
        target_rates = torch.full_like(rates, self.incorrect_rate)
        target_rates.scatter_(1, target.unsqueeze(1), self.correct_rate)
        
        return F.mse_loss(rates, target_rates)


class MemPotential(nn.Module):
    """
    Membrane potential loss.
    
    Uses final membrane potentials as logits (like standard ANN).
    Requires access to neuron states.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, mem: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(mem, target)


# =============================================================================
# STDP Learning
# =============================================================================

@dataclass
class STDPConfig:
    """STDP configuration parameters."""
    lr: float = 0.01               # Learning rate
    a_plus: float = 0.1            # LTP amplitude
    a_minus: float = 0.12          # LTD amplitude
    tau_plus: float = 20.0         # LTP time constant (ms)
    tau_minus: float = 20.0        # LTD time constant (ms)
    w_min: float = -1.0            # Min weight
    w_max: float = 1.0             # Max weight
    update_rule: str = 'additive'  # 'additive' or 'multiplicative'


class STDP:
    """
    Spike-Timing Dependent Plasticity.
    
    Updates weights based on relative timing of pre/post spikes.
    
    Args:
        model: SNN model to train
        connections: List of (pre_layer, post_layer) pairs to train
        config: STDP configuration
        
    Examples:
        >>> model = snn.Sequential(...)
        >>> stdp = STDP(model)
        >>> 
        >>> for batch in dataloader:
        ...     spikes = model.run(batch, T=100, return_all=True)
        ...     stdp.step()
    """
    
    def __init__(
        self,
        model: nn.Module,
        connections: List[Tuple[str, str]] = None,
        config: STDPConfig = None,
    ):
        self.model = model
        self.config = config or STDPConfig()
        self.connections = connections or self._auto_detect_connections()
        
        # Spike traces
        self.pre_traces: Dict[str, deque] = {}
        self.post_traces: Dict[str, deque] = {}
    
    def _auto_detect_connections(self) -> List[Tuple[str, str]]:
        """Auto-detect trainable connections."""
        connections = []
        layers = list(self.model.named_modules())
        
        for i, (name, module) in enumerate(layers):
            if hasattr(module, 'weight'):
                # Find corresponding neuron layer
                connections.append((name, name))
        
        return connections
    
    def record_spikes(self, name: str, spikes: Tensor, is_pre: bool = True):
        """Record spike times for STDP."""
        traces = self.pre_traces if is_pre else self.post_traces
        
        if name not in traces:
            traces[name] = deque(maxlen=100)
        
        # Record spike times
        spike_times = torch.where(spikes > 0)
        traces[name].append(spike_times)
    
    def step(self, pre_spikes: Tensor = None, post_spikes: Tensor = None):
        """
        Perform one STDP weight update step.
        
        If spikes not provided, uses recorded traces.
        """
        for pre_name, post_name in self.connections:
            # Get weight tensor
            module = dict(self.model.named_modules())[pre_name]
            if not hasattr(module, 'weight'):
                continue
            
            weight = module.weight
            
            # Calculate STDP update
            if pre_spikes is not None and post_spikes is not None:
                delta_w = self._compute_stdp(pre_spikes, post_spikes)
            else:
                # Use recorded traces
                delta_w = self._compute_stdp_from_traces(pre_name, post_name)
            
            # Apply update
            with torch.no_grad():
                if self.config.update_rule == 'additive':
                    weight.add_(self.config.lr * delta_w)
                else:  # multiplicative
                    weight.add_(self.config.lr * delta_w * (self.config.w_max - weight))
                
                # Clamp
                weight.clamp_(self.config.w_min, self.config.w_max)
    
    def _compute_stdp(self, pre: Tensor, post: Tensor) -> Tensor:
        """Compute STDP weight change."""
        # Pre before post: LTP
        # Post before pre: LTD
        
        B, T, N_pre = pre.shape if pre.dim() == 3 else (pre.shape[0], 1, pre.shape[1])
        _, _, N_post = post.shape if post.dim() == 3 else (post.shape[0], 1, post.shape[1])
        
        delta_w = torch.zeros(N_post, N_pre, device=pre.device)
        
        for t in range(T if pre.dim() == 3 else 1):
            pre_t = pre[:, t] if pre.dim() == 3 else pre
            post_t = post[:, t] if post.dim() == 3 else post
            
            # LTP: pre contributes to post spike
            ltp = torch.einsum('bi,bj->ij', post_t, pre_t) * self.config.a_plus
            
            # LTD: post doesn't follow pre
            ltd = torch.einsum('bi,bj->ij', pre_t, post_t) * self.config.a_minus
            
            delta_w += ltp - ltd
        
        return delta_w / (T * B)
    
    def _compute_stdp_from_traces(self, pre_name: str, post_name: str) -> Tensor:
        """Compute STDP from recorded spike traces."""
        # Simplified: just use most recent traces
        pre_traces = list(self.pre_traces.get(pre_name, []))
        post_traces = list(self.post_traces.get(post_name, []))
        
        if not pre_traces or not post_traces:
            return torch.zeros(1)
        
        # Use last recorded spikes
        pre = pre_traces[-1]
        post = post_traces[-1]
        
        return self._compute_stdp(pre, post)
    
    def reset(self):
        """Clear spike traces."""
        self.pre_traces.clear()
        self.post_traces.clear()


class RSTDP(STDP):
    """
    Reward-modulated STDP.
    
    STDP updates are modulated by reward signal.
    Uses eligibility traces for credit assignment.
    
    Args:
        model: SNN model
        config: STDP config
        tau_e: Eligibility trace time constant
        
    Examples:
        >>> rstdp = RSTDP(model)
        >>> rstdp.step()  # Accumulate eligibility
        >>> rstdp.apply_reward(reward=1.0)  # Apply reward
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: STDPConfig = None,
        tau_e: float = 0.95,
    ):
        super().__init__(model, config=config)
        self.tau_e = tau_e
        self.eligibility: Dict[str, Tensor] = {}
    
    def step(self, pre_spikes: Tensor = None, post_spikes: Tensor = None):
        """Update eligibility traces (don't modify weights yet)."""
        for pre_name, post_name in self.connections:
            module = dict(self.model.named_modules())[pre_name]
            if not hasattr(module, 'weight'):
                continue
            
            weight = module.weight
            
            # Calculate STDP update
            if pre_spikes is not None and post_spikes is not None:
                delta_w = self._compute_stdp(pre_spikes, post_spikes)
            else:
                delta_w = torch.zeros_like(weight)
            
            # Update eligibility trace
            if pre_name not in self.eligibility:
                self.eligibility[pre_name] = torch.zeros_like(weight)
            
            self.eligibility[pre_name] = (
                self.tau_e * self.eligibility[pre_name] + delta_w
            )
    
    def apply_reward(self, reward: float):
        """Apply reward to update weights."""
        for pre_name, _ in self.connections:
            module = dict(self.model.named_modules())[pre_name]
            if not hasattr(module, 'weight'):
                continue
            
            if pre_name not in self.eligibility:
                continue
            
            with torch.no_grad():
                module.weight.add_(
                    self.config.lr * reward * self.eligibility[pre_name]
                )
                module.weight.clamp_(self.config.w_min, self.config.w_max)
    
    def reset(self):
        super().reset()
        self.eligibility.clear()


# =============================================================================
# Training Utilities
# =============================================================================

def accuracy(output: Tensor, target: Tensor) -> float:
    """
    Compute classification accuracy.
    
    Args:
        output: (B, T, C) or (B, C) output spikes
        target: (B,) class labels
        
    Returns:
        Accuracy as float
    """
    if output.dim() == 3:
        # Sum over time
        counts = output.sum(dim=1)
    else:
        counts = output
    
    pred = counts.argmax(dim=-1)
    return (pred == target).float().mean().item()


class Trainer:
    """
    High-level trainer for SNNs.
    
    Handles training loop, evaluation, and logging.
    
    Args:
        model: SNN model
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        device: Device to train on
        
    Examples:
        >>> trainer = Trainer(model, optimizer, loss_fn)
        >>> trainer.fit(train_loader, val_loader, epochs=10)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        T: int = 100,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.T = T
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self, dataloader, encoder=None) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_acc = 0
        n_batches = 0
        
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            
            # Encode if needed
            if encoder:
                data = encoder(data)
            
            self.optimizer.zero_grad()
            
            # Reset neuron states
            if hasattr(self.model, 'reset'):
                self.model.reset()
            
            # Forward pass
            if data.dim() == 3:  # Temporal input
                outputs = []
                for t in range(data.shape[1]):
                    out = self.model(data[:, t])
                    outputs.append(out)
                output = torch.stack(outputs, dim=1)
            else:
                # Run for T timesteps
                outputs = []
                for _ in range(self.T):
                    out = self.model(data)
                    outputs.append(out)
                output = torch.stack(outputs, dim=1)
            
            # Compute loss
            loss = self.loss_fn(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_acc += accuracy(output, target)
            n_batches += 1
        
        return total_loss / n_batches, total_acc / n_batches
    
    @torch.no_grad()
    def evaluate(self, dataloader, encoder=None) -> Tuple[float, float]:
        """Evaluate on validation/test set."""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        n_batches = 0
        
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            
            if encoder:
                data = encoder(data)
            
            if hasattr(self.model, 'reset'):
                self.model.reset()
            
            if data.dim() == 3:
                outputs = []
                for t in range(data.shape[1]):
                    out = self.model(data[:, t])
                    outputs.append(out)
                output = torch.stack(outputs, dim=1)
            else:
                outputs = []
                for _ in range(self.T):
                    out = self.model(data)
                    outputs.append(out)
                output = torch.stack(outputs, dim=1)
            
            loss = self.loss_fn(output, target)
            
            total_loss += loss.item()
            total_acc += accuracy(output, target)
            n_batches += 1
        
        return total_loss / n_batches, total_acc / n_batches
    
    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 10,
        encoder=None,
        verbose: bool = True,
    ):
        """Full training loop."""
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, encoder)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader, encoder)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            
            if verbose:
                msg = f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
                if val_loader:
                    msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                print(msg)
        
        return self.history
