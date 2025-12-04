#!/usr/bin/env python3
"""
Mozafari et al. (2018) R-STDP Implementation using Our Library

This implements the reward-modulated STDP network for visual categorization
from "Bio-Inspired Digit Recognition Using Reward-Modulated Spike-Timing-
Dependent Plasticity in Deep Convolutional Networks" (Mozafari et al., 2018)

Using our snn_fpga_accelerator library components instead of SpykeTorch.

Architecture:
    Input -> Gabor filters (S1) -> Pooling (C1) -> STDP Conv (S2) -> Decision

Author: Based on Mozafari et al., adapted for our library
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'software', 'python'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import math
from typing import Optional, Tuple, List, Dict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import time

# Import our library
import snn_fpga_accelerator as snn
from snn_fpga_accelerator.neuron import IF  # Simple Integrate-and-Fire for STDP
from snn_fpga_accelerator.training import STDPConfig

# =============================================================================
# Utility Functions (Gabor, Intensity-to-Latency, etc.)
# =============================================================================

def create_gabor_bank(
    n_orientations: int = 4,
    kernel_size: int = 5,
    sigma: float = 1.0,
    freq: float = 0.25,
    n_scales: int = 1,
) -> Tensor:
    """
    Create a bank of Gabor filters for edge detection.
    
    Args:
        n_orientations: Number of orientation angles
        kernel_size: Size of each filter
        sigma: Gaussian envelope std
        freq: Spatial frequency
        n_scales: Number of scales (sigma scales)
        
    Returns:
        Tensor of shape (n_orientations * n_scales, 1, kernel_size, kernel_size)
    """
    filters = []
    
    for scale in range(n_scales):
        s = sigma * (1.5 ** scale)  # Scale progression
        
        for i in range(n_orientations):
            theta = i * math.pi / n_orientations
            
            # Create meshgrid
            x = torch.arange(kernel_size) - kernel_size // 2
            y = torch.arange(kernel_size) - kernel_size // 2
            y, x = torch.meshgrid(y, x, indexing='ij')
            
            # Rotate
            x_theta = x * math.cos(theta) + y * math.sin(theta)
            y_theta = -x * math.sin(theta) + y * math.cos(theta)
            
            # Gabor formula
            gabor = torch.exp(-0.5 * (x_theta**2 + y_theta**2) / s**2)
            gabor = gabor * torch.cos(2 * math.pi * freq * x_theta)
            
            # Normalize to zero mean, unit norm
            gabor = gabor - gabor.mean()
            gabor = gabor / (gabor.norm() + 1e-8)
            
            filters.append(gabor)
    
    # Stack: (n_filters, 1, H, W)
    return torch.stack(filters).unsqueeze(1).float()


def intensity_to_latency(
    image: Tensor,
    time_steps: int = 15,
    to_spike: bool = True,
) -> Tensor:
    """
    Convert pixel intensity to spike latency (rank-order coding).
    
    Higher intensity -> earlier spike.
    
    Args:
        image: (B, C, H, W) or (B, H, W) with values in [0, 1]
        time_steps: Number of time steps
        to_spike: If True, return spike tensor; else return latency values
        
    Returns:
        If to_spike: (B, T, C, H, W) spike tensor
        Else: (B, C, H, W) latency tensor
    """
    # Ensure 4D
    if image.dim() == 3:
        image = image.unsqueeze(1)
    
    B, C, H, W = image.shape
    
    # Latency: high intensity = early (low latency)
    # latency = (1 - intensity) * (time_steps - 1)
    latency = (1.0 - image) * (time_steps - 1)
    latency = latency.long().clamp(0, time_steps - 1)
    
    if not to_spike:
        return latency.float()
    
    # Convert to spike tensor
    spikes = torch.zeros(B, time_steps, C, H, W, device=image.device)
    
    # Set spike at corresponding time step
    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    t = latency[b, c, h, w].item()
                    if image[b, c, h, w] > 0.01:  # Only significant pixels spike
                        spikes[b, t, c, h, w] = 1.0
    
    return spikes


def intensity_to_latency_fast(
    image: Tensor,
    time_steps: int = 15,
) -> Tensor:
    """
    Fast vectorized version of intensity-to-latency encoding.
    
    Returns spike tensor (B, T, C, H, W)
    """
    if image.dim() == 3:
        image = image.unsqueeze(1)
    
    B, C, H, W = image.shape
    device = image.device
    
    # Latency from intensity
    latency = ((1.0 - image) * (time_steps - 1)).long().clamp(0, time_steps - 1)
    
    # Create time indices
    t_range = torch.arange(time_steps, device=device).view(1, time_steps, 1, 1, 1)
    
    # Create spike tensor where t == latency
    latency_expanded = latency.unsqueeze(1)  # (B, 1, C, H, W)
    spikes = (t_range == latency_expanded).float()
    
    # Mask out very low intensity pixels
    mask = (image > 0.01).unsqueeze(1).expand(-1, time_steps, -1, -1, -1)
    spikes = spikes * mask.float()
    
    return spikes


def local_normalization(image: Tensor, kernel_size: int = 7) -> Tensor:
    """
    Apply local contrast normalization.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Local mean
    avg_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=image.device) / (kernel_size ** 2)
    local_mean = F.conv2d(image, avg_kernel, padding=kernel_size//2)
    
    # Subtract local mean
    centered = image - local_mean
    
    # Local variance
    local_var = F.conv2d(centered ** 2, avg_kernel, padding=kernel_size//2)
    local_std = torch.sqrt(local_var + 1e-8)
    
    # Normalize
    normalized = centered / local_std
    
    # Scale to [0, 1]
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
    
    return normalized


def lateral_inhibition(
    potentials: Tensor,
    inhibition_radius: int = 2,
    inhibition_strength: float = 0.5,
) -> Tensor:
    """
    Apply lateral inhibition on potentials.
    Neurons with higher potential inhibit neighbors.
    
    Args:
        potentials: (B, C, H, W) membrane potentials
        inhibition_radius: Spatial radius of inhibition
        inhibition_strength: Strength of inhibition
        
    Returns:
        Inhibited potentials
    """
    B, C, H, W = potentials.shape
    
    # Max pool to find local winners
    k = 2 * inhibition_radius + 1
    local_max = F.max_pool2d(potentials, k, stride=1, padding=inhibition_radius)
    
    # Winners are where potential equals local max
    is_winner = (potentials >= local_max).float()
    
    # Inhibit non-winners
    inhibited = potentials * is_winner + potentials * (1 - is_winner) * (1 - inhibition_strength)
    
    return inhibited


def get_k_winners(
    potentials: Tensor,
    k: int = 1,
    spikes: Tensor = None,
    inhibition_radius: int = 0,
) -> Tuple[Tensor, Tensor]:
    """
    Select k neurons with highest potentials (winner-take-all).
    
    Similar to SpykeTorch's sf.get_k_winners.
    
    Args:
        potentials: (B, C, H, W) membrane potentials
        k: Number of winners
        spikes: Optional existing spikes to consider
        inhibition_radius: Spatial exclusion radius
        
    Returns:
        winners: Tensor of winner indices
        spikes: (B, C, H, W) spike tensor with only winners
    """
    B, C, H, W = potentials.shape
    device = potentials.device
    
    output_spikes = torch.zeros_like(potentials)
    winners_list = []
    
    for b in range(B):
        pot = potentials[b].clone()  # (C, H, W)
        batch_winners = []
        
        for _ in range(k):
            # Find global maximum
            max_val = pot.max()
            if max_val <= 0:
                break
                
            # Find position
            max_idx = (pot == max_val).nonzero()
            if len(max_idx) == 0:
                break
            
            c, h, w = max_idx[0]
            
            batch_winners.append((c.item(), h.item(), w.item()))
            output_spikes[b, c, h, w] = 1.0
            
            # Apply spatial inhibition
            if inhibition_radius > 0:
                h_start = max(0, h - inhibition_radius)
                h_end = min(H, h + inhibition_radius + 1)
                w_start = max(0, w - inhibition_radius)
                w_end = min(W, w + inhibition_radius + 1)
                pot[:, h_start:h_end, w_start:w_end] = -float('inf')
            else:
                pot[c, h, w] = -float('inf')
        
        winners_list.append(batch_winners)
    
    return winners_list, output_spikes


# =============================================================================
# STDP Convolution Layer (for unsupervised feature learning)
# =============================================================================

class STDPConv2d(nn.Module):
    """
    Convolutional layer with STDP learning.
    
    This implements the core S2/S4 layers in Mozafari's network.
    
    Features:
    - Convolutional synapses with STDP weight updates
    - Winner-take-all competition
    - Weight normalization
    - Anti-hebbian learning for losers
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        threshold: float = 1.0,
        n_winners: int = 1,
        inhibition_radius: int = 0,
        learning_rate: Tuple[float, float] = (0.004, -0.003),  # (LTP, LTD)
        weight_mean: float = 0.8,
        weight_std: float = 0.05,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.threshold = threshold
        self.n_winners = n_winners
        self.inhibition_radius = inhibition_radius
        self.lr_plus = learning_rate[0]  # LTP rate
        self.lr_minus = learning_rate[1]  # LTD rate (negative)
        
        # Initialize weights (similar to SpykeTorch)
        self.weight = nn.Parameter(
            torch.normal(weight_mean, weight_std, 
                        (out_channels, in_channels, kernel_size, kernel_size))
        )
        self.weight.data.clamp_(0, 1)  # Weights in [0, 1]
        
        # Tracking for STDP
        self.potentials: Optional[Tensor] = None
        self.input_spikes: Optional[Tensor] = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with IF neurons.
        
        Args:
            x: Input spikes (B, C, H, W)
            
        Returns:
            Output spikes after winner-take-all (B, out_C, H', W')
        """
        self.input_spikes = x
        
        # Convolve input with weights
        self.potentials = F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)
        
        # Winner-take-all
        winners, output_spikes = get_k_winners(
            self.potentials,
            k=self.n_winners,
            inhibition_radius=self.inhibition_radius,
        )
        
        return output_spikes
    
    def get_winning_features(self, output_spikes: Tensor) -> List[int]:
        """Get indices of output features that spiked."""
        B = output_spikes.shape[0]
        winning_features = []
        
        for b in range(B):
            # Sum over spatial dimensions
            feature_spikes = output_spikes[b].sum(dim=(1, 2))  # (out_C,)
            winners = (feature_spikes > 0).nonzero().squeeze(-1)
            winning_features.append(winners.tolist() if winners.dim() > 0 else [winners.item()] if winners.numel() > 0 else [])
        
        return winning_features
    
    def stdp(
        self,
        output_spikes: Tensor,
        use_stabilizer: bool = True,
        lr_scale: float = 1.0,
    ):
        """
        Apply STDP weight update.
        
        Args:
            output_spikes: Post-synaptic spikes (from forward)
            use_stabilizer: Use multiplicative STDP for stability
            lr_scale: Scale learning rate (for curriculum)
        """
        if self.input_spikes is None:
            return
        
        with torch.no_grad():
            # Unfold input to get patches
            input_unfolded = F.unfold(
                self.input_spikes, 
                self.kernel_size, 
                stride=self.stride, 
                padding=self.padding
            )  # (B, C*k*k, L)
            
            B, _, L = input_unfolded.shape
            H_out = W_out = int(math.sqrt(L))
            
            # Reshape input patches
            input_patches = input_unfolded.view(B, self.in_channels, self.kernel_size, self.kernel_size, H_out, W_out)
            
            # For each output neuron that spiked
            for b in range(B):
                for c in range(self.out_channels):
                    # Find spike locations
                    spike_locs = output_spikes[b, c].nonzero()
                    
                    for loc in spike_locs:
                        h, w = loc
                        
                        # Get input patch
                        patch = input_patches[b, :, :, :, h, w]  # (in_C, k, k)
                        
                        # STDP update
                        if use_stabilizer:
                            # Multiplicative STDP (more stable)
                            # LTP: Δw+ ∝ pre * (1 - w)
                            # LTD: Δw- ∝ (1 - pre) * w
                            ltp = self.lr_plus * lr_scale * patch * (1 - self.weight[c])
                            ltd = self.lr_minus * lr_scale * (1 - patch) * self.weight[c]
                        else:
                            # Additive STDP
                            ltp = self.lr_plus * lr_scale * patch
                            ltd = self.lr_minus * lr_scale * (1 - patch)
                        
                        self.weight[c] += ltp + ltd
            
            # Clamp weights
            self.weight.data.clamp_(0, 1)
    
    def reward_stdp(
        self,
        output_spikes: Tensor,
        reward: float,
        lr_scale: float = 1.0,
    ):
        """
        Apply reward-modulated STDP.
        
        If reward > 0: strengthen winning connections (LTP)
        If reward < 0: weaken winning connections (punishment/anti-STDP)
        
        Args:
            output_spikes: Post-synaptic spikes
            reward: +1 for correct, -1 for incorrect
            lr_scale: Learning rate scaling
        """
        if self.input_spikes is None:
            return
        
        with torch.no_grad():
            # Unfold input
            input_unfolded = F.unfold(
                self.input_spikes,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding
            )
            
            B, _, L = input_unfolded.shape
            H_out = W_out = int(math.sqrt(L))
            input_patches = input_unfolded.view(B, self.in_channels, self.kernel_size, self.kernel_size, H_out, W_out)
            
            # R-STDP update
            for b in range(B):
                for c in range(self.out_channels):
                    spike_locs = output_spikes[b, c].nonzero()
                    
                    for loc in spike_locs:
                        h, w = loc
                        patch = input_patches[b, :, :, :, h, w]
                        
                        if reward > 0:
                            # Reward: strengthen (LTP)
                            ltp = self.lr_plus * lr_scale * reward * patch * (1 - self.weight[c])
                            ltd = self.lr_minus * lr_scale * reward * (1 - patch) * self.weight[c]
                        else:
                            # Punishment: anti-STDP (weaken active, strengthen inactive)
                            ltp = self.lr_minus * lr_scale * (-reward) * patch * self.weight[c]
                            ltd = self.lr_plus * lr_scale * (-reward) * (1 - patch) * (1 - self.weight[c])
                        
                        self.weight[c] += ltp + ltd
            
            self.weight.data.clamp_(0, 1)


# =============================================================================
# Mozafari-style R-STDP Network
# =============================================================================

class MozafariRSTDPNetwork(nn.Module):
    """
    Mozafari et al. (2018) network architecture for digit recognition.
    
    Architecture:
        S1: Gabor filters (fixed)
        C1: Max pooling
        S2: STDP convolution (trainable)
        Decision: Global max pooling + voting
        
    For simplicity, we use a 2-class setup initially (0 vs 1).
    """
    
    def __init__(
        self,
        n_classes: int = 2,
        n_orientations: int = 4,
        s2_features: int = 200,  # Features per class (total = n_classes * s2_features)
        s2_kernel: int = 5,  # Reduced kernel for MNIST (28x28 -> ~4x4 after pooling)
        time_steps: int = 15,
        threshold: float = 15.0,
        n_winners: int = 5,
        learning_rate: Tuple[float, float] = (0.004, -0.003),
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.n_orientations = n_orientations
        self.time_steps = time_steps
        self.n_winners = n_winners
        
        # Features per class
        self.features_per_class = s2_features // n_classes
        self.total_features = self.features_per_class * n_classes
        
        # S1: Gabor filters (fixed, not learnable)
        self.register_buffer('gabor_filters', create_gabor_bank(n_orientations, kernel_size=5))
        
        # C1: Pooling parameters (smaller pooling for 28x28 MNIST)
        # After Gabor (padding=2): 28x28 -> 28x28
        # After pooling: 28x28 -> floor((28-3+2*1)/2)+1 = 14x14
        self.c1_pool_size = 3
        self.c1_stride = 2
        
        # S2: STDP convolution
        self.s2 = STDPConv2d(
            in_channels=n_orientations,
            out_channels=self.total_features,
            kernel_size=s2_kernel,
            threshold=threshold,
            n_winners=n_winners,
            inhibition_radius=3,
            learning_rate=learning_rate,
        )
        
        # Feature-to-class assignment
        # First features_per_class belong to class 0, next to class 1, etc.
        self.feature_class = torch.zeros(self.total_features, dtype=torch.long)
        for c in range(n_classes):
            start = c * self.features_per_class
            end = start + self.features_per_class
            self.feature_class[start:end] = c
    
    def s1_transform(self, x: Tensor) -> Tensor:
        """
        Apply Gabor filters (S1 layer).
        
        Args:
            x: Input image (B, 1, H, W)
            
        Returns:
            S1 features (B, n_orientations, H, W)
        """
        s1 = F.conv2d(x, self.gabor_filters, padding=2)
        # Take absolute value (detect both polarities)
        s1 = torch.abs(s1)
        # Normalize per sample
        s1 = s1 / (s1.max() + 1e-8)
        return s1
    
    def c1_transform(self, s1: Tensor) -> Tensor:
        """
        Apply pooling (C1 layer).
        
        Args:
            s1: S1 features (B, n_orientations, H, W)
            
        Returns:
            C1 features (B, n_orientations, H', W')
        """
        c1 = F.max_pool2d(s1, self.c1_pool_size, stride=self.c1_stride, padding=self.c1_pool_size // 2)
        return c1
    
    def encode_spikes(self, c1: Tensor) -> List[Tensor]:
        """
        Convert C1 features to temporal spike trains.
        
        Args:
            c1: C1 features (B, C, H, W)
            
        Returns:
            List of spike tensors for each time step
        """
        # Intensity-to-latency encoding
        spike_seq = intensity_to_latency_fast(c1, self.time_steps)
        
        # Split into list of time steps
        return [spike_seq[:, t] for t in range(self.time_steps)]
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, int]:
        """
        Forward pass for inference.
        
        Args:
            x: Input image (B, 1, 28, 28)
            
        Returns:
            s2_spikes: S2 layer output spikes
            predictions: Predicted class
            winning_class: Most active class
        """
        # S1: Gabor features
        s1 = self.s1_transform(x)
        
        # C1: Pooling
        c1 = self.c1_transform(s1)
        
        # Encode to spikes
        spike_list = self.encode_spikes(c1)
        
        # Process through S2 over time, accumulate spikes
        B = x.shape[0]
        total_s2_spikes = torch.zeros(B, self.total_features, device=x.device)
        last_s2_output = None
        
        for t, spikes in enumerate(spike_list):
            s2_out = self.s2(spikes)
            total_s2_spikes += s2_out.sum(dim=(2, 3))  # Sum over spatial
            last_s2_output = s2_out
        
        # Vote by class
        class_votes = torch.zeros(B, self.n_classes, device=x.device)
        for c in range(self.n_classes):
            mask = (self.feature_class == c).to(x.device)
            class_votes[:, c] = total_s2_spikes[:, mask].sum(dim=1)
        
        predictions = class_votes.argmax(dim=1)
        
        return last_s2_output, predictions, class_votes
    
    def train_step(
        self,
        x: Tensor,
        target: Tensor,
        use_rstdp: bool = True,
        lr_scale: float = 1.0,
    ) -> Tuple[int, bool]:
        """
        Single training step with R-STDP.
        
        Args:
            x: Input image (B, 1, 28, 28)
            target: Class labels (B,)
            use_rstdp: Use reward modulation
            lr_scale: Learning rate scaling
            
        Returns:
            prediction: Predicted class
            correct: Whether prediction was correct
        """
        B = x.shape[0]
        assert B == 1, "Training works with batch size 1"
        
        # S1 + C1
        s1 = self.s1_transform(x)
        c1 = self.c1_transform(s1)
        spike_list = self.encode_spikes(c1)
        
        # Accumulate winning features
        winning_features = []
        all_s2_outputs = []
        
        for spikes in spike_list:
            s2_out = self.s2(spikes)
            all_s2_outputs.append(s2_out)
            
            # Collect winners
            feat_spikes = s2_out.sum(dim=(2, 3))[0]  # (total_features,)
            winners = (feat_spikes > 0).nonzero().squeeze(-1)
            if winners.numel() > 0:
                winning_features.extend(winners.tolist() if winners.dim() > 0 else [winners.item()])
        
        # Determine winning class
        class_votes = torch.zeros(self.n_classes, device=x.device)
        for feat_idx in winning_features:
            class_votes[self.feature_class[feat_idx]] += 1
        
        prediction = class_votes.argmax().item()
        correct = (prediction == target[0].item())
        
        # R-STDP learning
        if use_rstdp and len(winning_features) > 0:
            # Get reward based on whether winning class matches target
            target_class = target[0].item()
            
            if correct:
                reward = 1.0  # Positive reward for correct
            else:
                reward = -1.0  # Punishment for incorrect
            
            # Apply R-STDP for each time step
            for s2_out in all_s2_outputs:
                if s2_out.sum() > 0:
                    self.s2.reward_stdp(s2_out, reward, lr_scale)
        else:
            # Unsupervised STDP (no reward modulation)
            for s2_out in all_s2_outputs:
                if s2_out.sum() > 0:
                    self.s2.stdp(s2_out, lr_scale=lr_scale)
        
        return prediction, correct


def train_mozafari_network(
    n_classes: int = 2,
    digits: List[int] = [0, 1],
    epochs: int = 2,
    samples_per_epoch: int = 1000,
    device: str = 'cpu',
):
    """
    Train Mozafari R-STDP network on MNIST subset.
    
    Args:
        n_classes: Number of classes
        digits: Which digits to use
        epochs: Number of training epochs
        samples_per_epoch: Samples per epoch
        device: Device to use
    """
    print("=" * 60)
    print("Mozafari et al. (2018) R-STDP Network Training")
    print("=" * 60)
    print(f"Classes: {n_classes} ({digits})")
    print(f"Epochs: {epochs}")
    print(f"Samples per epoch: {samples_per_epoch}")
    print()
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform,
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform,
    )
    
    # Filter to desired classes
    def filter_dataset(dataset, target_digits):
        indices = []
        digit_to_label = {d: i for i, d in enumerate(target_digits)}
        
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label in target_digits:
                indices.append(idx)
        
        return indices, digit_to_label
    
    train_indices, digit_to_label = filter_dataset(train_dataset, digits)
    test_indices, _ = filter_dataset(test_dataset, digits)
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")
    print()
    
    # Create network
    network = MozafariRSTDPNetwork(
        n_classes=n_classes,
        n_orientations=4,
        s2_features=200,
        s2_kernel=5,  # Smaller kernel for MNIST
        time_steps=15,
        threshold=15.0,
        n_winners=5,
        learning_rate=(0.004, -0.003),
    ).to(device)
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        # Shuffle training indices
        np.random.shuffle(train_indices)
        
        n_correct = 0
        n_total = 0
        
        start_time = time.time()
        
        for i, idx in enumerate(train_indices[:samples_per_epoch]):
            img, original_label = train_dataset[idx]
            label = digit_to_label[original_label]
            
            img = img.unsqueeze(0).to(device)  # (1, 1, 28, 28)
            target = torch.tensor([label], device=device)
            
            # Adaptive learning rate
            progress = (epoch * samples_per_epoch + i) / (epochs * samples_per_epoch)
            lr_scale = 1.0 - 0.5 * progress  # Decay from 1.0 to 0.5
            
            pred, correct = network.train_step(img, target, use_rstdp=True, lr_scale=lr_scale)
            
            if correct:
                n_correct += 1
            n_total += 1
            
            if (i + 1) % 200 == 0:
                acc = n_correct / n_total * 100
                elapsed = time.time() - start_time
                print(f"  [{i+1}/{samples_per_epoch}] Accuracy: {acc:.1f}% ({elapsed:.1f}s)")
        
        epoch_acc = n_correct / n_total * 100
        print(f"  Epoch accuracy: {epoch_acc:.1f}%")
        print()
    
    # Evaluation
    print("=" * 60)
    print("Evaluation")
    print("=" * 60)
    
    n_correct = 0
    n_total = 0
    
    network.eval()
    with torch.no_grad():
        for idx in test_indices[:500]:  # Test on 500 samples
            img, original_label = test_dataset[idx]
            label = digit_to_label[original_label]
            
            img = img.unsqueeze(0).to(device)
            
            _, pred, _ = network(img)
            
            if pred[0].item() == label:
                n_correct += 1
            n_total += 1
    
    test_acc = n_correct / n_total * 100
    print(f"Test Accuracy: {test_acc:.1f}% ({n_correct}/{n_total})")
    
    return network, test_acc


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    # Train on 2 classes (0 vs 1)
    network, acc = train_mozafari_network(
        n_classes=2,
        digits=[0, 1],
        epochs=2,
        samples_per_epoch=1000,
        device=device,
    )
    
    print()
    print("=" * 60)
    print(f"Final Test Accuracy: {acc:.1f}%")
    print("=" * 60)
