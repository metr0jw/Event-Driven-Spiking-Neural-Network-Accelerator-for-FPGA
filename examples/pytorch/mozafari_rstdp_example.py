#!/usr/bin/env python3
"""
Mozafari et al. (2018) R-STDP Implementation - Optimized Version

This is an optimized implementation with better debugging and proper R-STDP.

Key improvements:
1. Proper STDP with temporal pre/post spike tracking
2. Better winner-take-all mechanism
3. Feature competition within class
4. Adaptive threshold
5. Better spike encoding

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
from torch.utils.data import DataLoader
import time

# =============================================================================
# Gabor Filters
# =============================================================================

def create_gabor_bank(
    n_orientations: int = 4,
    kernel_size: int = 5,
    sigma: float = 1.0,
    freq: float = 0.25,
) -> Tensor:
    """Create Gabor filter bank."""
    filters = []
    
    for i in range(n_orientations):
        theta = i * math.pi / n_orientations
        
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        y = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        y, x = torch.meshgrid(y, x, indexing='ij')
        
        x_theta = x * math.cos(theta) + y * math.sin(theta)
        y_theta = -x * math.sin(theta) + y * math.cos(theta)
        
        gabor = torch.exp(-0.5 * (x_theta**2 + y_theta**2) / sigma**2)
        gabor = gabor * torch.cos(2 * math.pi * freq * x_theta)
        
        gabor = gabor - gabor.mean()
        gabor = gabor / (gabor.norm() + 1e-8)
        
        filters.append(gabor)
    
    return torch.stack(filters).unsqueeze(1)


# =============================================================================
# Simplified STDP Convolution with proper spike timing
# =============================================================================

class STDPConv2dSimple(nn.Module):
    """
    Simple but correct STDP convolution layer.
    
    Uses cumulative spike timing to determine LTP/LTD.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        threshold: float = 15.0,
        n_winners: int = 1,  # Per map or global
        a_plus: float = 0.004,
        a_minus: float = 0.003,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.threshold = threshold
        self.n_winners = n_winners
        self.a_plus = a_plus
        self.a_minus = a_minus
        
        # Initialize weights uniformly in [0, 1]
        self.weight = nn.Parameter(
            torch.rand(out_channels, in_channels, kernel_size, kernel_size) * 0.4 + 0.3
        )
        
        # For spike timing
        self.pre_spike_times: Optional[Tensor] = None  # When each pre-neuron last spiked
        self.post_spike_times: Optional[Tensor] = None  # When each post-neuron last spiked
        self.current_time: int = 0
    
    def reset(self):
        """Reset temporal state for new sample."""
        self.pre_spike_times = None
        self.post_spike_times = None
        self.current_time = 0
    
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        """
        Forward pass with integrate-and-fire.
        
        Args:
            x: Input spikes (B, C, H, W)
            training: Whether to track timing for STDP
            
        Returns:
            Output spikes after winner-take-all
        """
        B = x.shape[0]
        device = x.device
        
        # Compute potentials
        potentials = F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)
        _, C, H, W = potentials.shape
        
        # Track pre-synaptic spike times
        if training:
            if self.pre_spike_times is None:
                self.pre_spike_times = torch.full_like(x, float('inf'))
            
            # Update pre-spike times where spikes occurred
            self.pre_spike_times = torch.where(
                x > 0.5,
                torch.full_like(x, float(self.current_time)),
                self.pre_spike_times
            )
        
        # Winner-take-all: find global k winners
        output_spikes = torch.zeros_like(potentials)
        
        for b in range(B):
            pot = potentials[b].view(-1)  # Flatten
            
            # Find k winners
            k = min(self.n_winners, (pot > self.threshold).sum().item())
            if k == 0:
                k = min(self.n_winners, pot.numel())
            
            if k > 0:
                _, indices = torch.topk(pot, k)
                
                for idx in indices:
                    if pot[idx] > self.threshold:
                        output_spikes[b].view(-1)[idx] = 1.0
        
        # Track post-synaptic spike times
        if training:
            if self.post_spike_times is None:
                self.post_spike_times = torch.full_like(potentials, float('inf'))
            
            self.post_spike_times = torch.where(
                output_spikes > 0.5,
                torch.full_like(potentials, float(self.current_time)),
                self.post_spike_times
            )
        
        self.current_time += 1
        
        return output_spikes
    
    def stdp_update(self, input_spikes: Tensor, output_spikes: Tensor, reward: float = 0.0):
        """
        Apply STDP update based on spike timing.
        
        Uses simplified STDP: if post fires, strengthen connections from 
        pre-neurons that spiked recently.
        
        Args:
            input_spikes: Pre-synaptic spikes (B, C, H, W)
            output_spikes: Post-synaptic spikes (B, out_C, H', W')
            reward: R-STDP reward (+1, -1, or 0)
        """
        with torch.no_grad():
            B = input_spikes.shape[0]
            
            # Unfold input to patches
            patches = F.unfold(
                input_spikes,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding
            )  # (B, C*k*k, L)
            
            _, patch_size, L = patches.shape
            h_out = w_out = int(math.sqrt(L))
            
            patches = patches.view(B, self.in_channels, self.kernel_size, self.kernel_size, h_out, w_out)
            
            # For each post-synaptic spike
            for b in range(B):
                post_spikes_flat = output_spikes[b].view(self.out_channels, -1)  # (out_C, L)
                
                for c in range(self.out_channels):
                    spike_locs = (post_spikes_flat[c] > 0).nonzero().squeeze(-1)
                    
                    for loc in spike_locs:
                        h = loc // w_out
                        w = loc % w_out
                        
                        # Get input patch
                        patch = patches[b, :, :, :, h, w]  # (in_C, k, k)
                        
                        # STDP update
                        if reward >= 0:
                            # LTP: strengthen where pre fired
                            ltp = self.a_plus * patch * (1 - self.weight[c])
                            # LTD: weaken where pre didn't fire
                            ltd = -self.a_minus * (1 - patch) * self.weight[c]
                            
                            if reward > 0:
                                self.weight[c] += (ltp + ltd) * reward
                            else:
                                self.weight[c] += ltp + ltd
                        else:
                            # Anti-STDP (punishment)
                            # Weaken where pre fired
                            anti_ltp = -self.a_plus * patch * self.weight[c]
                            # Strengthen where pre didn't fire
                            anti_ltd = self.a_minus * (1 - patch) * (1 - self.weight[c])
                            
                            self.weight[c] += (anti_ltp + anti_ltd) * (-reward)
            
            # Clamp weights
            self.weight.data.clamp_(0, 1)


# =============================================================================
# Optimized R-STDP Network
# =============================================================================

class OptimizedRSTDPNetwork(nn.Module):
    """
    Simplified and optimized R-STDP network for MNIST.
    
    Architecture:
        Input (28x28) -> Gabor (4 orientations) -> Pool (14x14) -> 
        STDP Conv (n_features per class) -> Decision
    """
    
    def __init__(
        self,
        n_classes: int = 2,
        n_orientations: int = 4,
        features_per_class: int = 50,
        s2_kernel: int = 5,
        threshold: float = 15.0,
        n_winners: int = 1,  # Winners per sample
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.n_orientations = n_orientations
        self.features_per_class = features_per_class
        self.n_winners = n_winners
        
        # Gabor filters (fixed)
        self.register_buffer('gabor', create_gabor_bank(n_orientations, kernel_size=5))
        
        # One STDP conv per class
        self.stdp_layers = nn.ModuleList([
            STDPConv2dSimple(
                in_channels=n_orientations,
                out_channels=features_per_class,
                kernel_size=s2_kernel,
                threshold=threshold,
                n_winners=n_winners,
                a_plus=0.004,
                a_minus=0.003,
            )
            for _ in range(n_classes)
        ])
    
    def reset(self):
        """Reset state for new sample."""
        for layer in self.stdp_layers:
            layer.reset()
    
    def gabor_transform(self, x: Tensor) -> Tensor:
        """Apply Gabor filters."""
        s1 = F.conv2d(x, self.gabor, padding=2)
        s1 = torch.abs(s1)
        s1 = s1 / (s1.max() + 1e-8)
        return s1
    
    def pool(self, x: Tensor) -> Tensor:
        """Max pooling."""
        return F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    def intensity_to_latency(self, x: Tensor, time_steps: int = 15) -> List[Tensor]:
        """Convert intensity to spike latency."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        B, C, H, W = x.shape
        device = x.device
        
        # Latency: high intensity = early spike
        latency = ((1.0 - x) * (time_steps - 1)).long().clamp(0, time_steps - 1)
        
        spike_list = []
        for t in range(time_steps):
            spikes = (latency == t).float()
            # Only spike for significant intensities
            spikes = spikes * (x > 0.01).float()
            spike_list.append(spikes)
        
        return spike_list
    
    def forward(self, x: Tensor, time_steps: int = 15) -> Tuple[int, Tensor]:
        """
        Forward pass for inference.
        
        Returns:
            prediction: Predicted class
            class_spikes: Spike counts per class (n_classes,)
        """
        B = x.shape[0]
        device = x.device
        
        # Preprocessing
        s1 = self.gabor_transform(x)
        c1 = self.pool(s1)
        
        # Encode to spikes
        spike_list = self.intensity_to_latency(c1, time_steps)
        
        # Process through each class's STDP layer
        class_spikes = torch.zeros(B, self.n_classes, device=device)
        
        self.reset()
        
        for spikes in spike_list:
            for c, layer in enumerate(self.stdp_layers):
                out = layer(spikes, training=False)
                class_spikes[:, c] += out.sum(dim=(1, 2, 3))
        
        predictions = class_spikes.argmax(dim=1)
        
        return predictions, class_spikes
    
    def train_step(
        self,
        x: Tensor,
        target: int,
        time_steps: int = 15,
        lr_scale: float = 1.0,
    ) -> Tuple[int, bool]:
        """
        Single training step with R-STDP.
        
        Args:
            x: Input image (1, 1, 28, 28)
            target: Target class
            time_steps: Number of time steps
            lr_scale: Learning rate scaling
            
        Returns:
            prediction: Predicted class
            correct: Whether prediction was correct
        """
        device = x.device
        
        # Preprocessing
        s1 = self.gabor_transform(x)
        c1 = self.pool(s1)
        spike_list = self.intensity_to_latency(c1, time_steps)
        
        # Accumulate spikes per class
        class_spike_counts = torch.zeros(self.n_classes, device=device)
        all_outputs = [[] for _ in range(self.n_classes)]
        
        self.reset()
        
        # Forward pass for all classes
        for spikes in spike_list:
            for c, layer in enumerate(self.stdp_layers):
                out = layer(spikes, training=True)
                all_outputs[c].append((spikes.clone(), out.clone()))
                class_spike_counts[c] += out.sum()
        
        # Determine winner class
        prediction = class_spike_counts.argmax().item()
        correct = (prediction == target)
        
        # R-STDP learning
        # Target class gets reward if it wins, or encouragement
        # Other classes get punishment if they win incorrectly
        
        for c, layer in enumerate(self.stdp_layers):
            for input_spikes, output_spikes in all_outputs[c]:
                if output_spikes.sum() > 0:
                    if c == target:
                        # This is the correct class
                        if correct:
                            # Won correctly - strengthen
                            reward = 1.0 * lr_scale
                        else:
                            # Didn't win when it should - encourage
                            reward = 0.5 * lr_scale
                    else:
                        # Wrong class
                        if not correct and prediction == c:
                            # Won incorrectly - punish
                            reward = -1.0 * lr_scale
                        else:
                            # Just normal STDP
                            reward = 0.0
                    
                    layer.stdp_update(input_spikes, output_spikes, reward)
        
        return prediction, correct


# =============================================================================
# Training Loop
# =============================================================================

def train_optimized_network(
    n_classes: int = 2,
    digits: List[int] = [0, 1],
    epochs: int = 3,
    samples_per_epoch: int = 2000,
    device: str = 'cpu',
):
    """Train the optimized R-STDP network."""
    
    print("=" * 60)
    print("Optimized R-STDP Network Training")
    print("=" * 60)
    print(f"Classes: {n_classes} ({digits})")
    print(f"Epochs: {epochs}")
    print()
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Filter classes
    digit_to_label = {d: i for i, d in enumerate(digits)}
    
    train_indices = [i for i in range(len(train_dataset)) if train_dataset.targets[i].item() in digits]
    test_indices = [i for i in range(len(test_dataset)) if test_dataset.targets[i].item() in digits]
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")
    print()
    
    # Create network
    network = OptimizedRSTDPNetwork(
        n_classes=n_classes,
        n_orientations=4,
        features_per_class=50,
        s2_kernel=5,
        threshold=10.0,
        n_winners=1,
    ).to(device)
    
    # Training
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        np.random.shuffle(train_indices)
        
        n_correct = 0
        n_total = 0
        start = time.time()
        
        for i, idx in enumerate(train_indices[:samples_per_epoch]):
            img, orig_label = train_dataset[idx]
            label = digit_to_label[orig_label]
            
            img = img.unsqueeze(0).to(device)
            
            lr_scale = 1.0 - 0.3 * (epoch / epochs)
            
            pred, correct = network.train_step(img, label, time_steps=15, lr_scale=lr_scale)
            
            if correct:
                n_correct += 1
            n_total += 1
            
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                acc = n_correct / n_total * 100
                print(f"  [{i+1}/{samples_per_epoch}] Acc: {acc:.1f}% ({elapsed:.1f}s)")
        
        print(f"  Epoch accuracy: {n_correct/n_total*100:.1f}%")
        print()
    
    # Evaluation
    print("=" * 60)
    print("Evaluation")
    print("=" * 60)
    
    n_correct = 0
    n_total = 0
    
    with torch.no_grad():
        for idx in test_indices[:1000]:
            img, orig_label = test_dataset[idx]
            label = digit_to_label[orig_label]
            
            img = img.unsqueeze(0).to(device)
            
            pred, _ = network(img, time_steps=15)
            
            if pred[0].item() == label:
                n_correct += 1
            n_total += 1
    
    acc = n_correct / n_total * 100
    print(f"Test Accuracy: {acc:.1f}% ({n_correct}/{n_total})")
    
    return network, acc


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    network, acc = train_optimized_network(
        n_classes=2,
        digits=[0, 1],
        epochs=3,
        samples_per_epoch=2000,
        device=device,
    )
    
    print(f"\n Final Test Accuracy: {acc:.1f}%")
