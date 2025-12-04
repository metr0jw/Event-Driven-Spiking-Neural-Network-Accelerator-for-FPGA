#!/usr/bin/env python3
"""
Mozafari R-STDP V3 - SpykeTorch-accurate Implementation

Key insight from SpykeTorch:
1. Process one timestep at a time
2. First-spike-wins competition (temporal coding)
3. Feature-wise inhibition after winner is determined
4. Separate STDP layers per class

Author: Based on Mozafari et al.
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
from typing import Optional, Tuple, List
from torchvision import datasets, transforms
import time


def gabor_filters(n_orient: int = 4, size: int = 5) -> Tensor:
    """Create Gabor filter bank."""
    filters = []
    for i in range(n_orient):
        theta = i * math.pi / n_orient
        x = torch.arange(size, dtype=torch.float32) - size // 2
        y = torch.arange(size, dtype=torch.float32) - size // 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        x_t = xx * math.cos(theta) + yy * math.sin(theta)
        y_t = -xx * math.sin(theta) + yy * math.cos(theta)
        
        g = torch.exp(-0.5 * (x_t**2 + y_t**2))
        g = g * torch.cos(2 * math.pi * 0.25 * x_t)
        g = g - g.mean()
        g = g / (g.norm() + 1e-8)
        filters.append(g)
    
    return torch.stack(filters).unsqueeze(1)


def intensity_to_latency(img: Tensor, T: int = 15) -> Tensor:
    """
    Convert intensity [0,1] to latency encoding.
    High intensity = early spike (low latency).
    
    Returns: (B, T, C, H, W) spike tensor
    """
    if img.dim() == 3:
        img = img.unsqueeze(1)
    
    B, C, H, W = img.shape
    
    # Latency = (1 - intensity) * (T - 1)
    latency = ((1.0 - img.clamp(0, 1)) * (T - 1)).long()
    latency = latency.clamp(0, T - 1)
    
    # Create one-hot in time dimension
    spikes = torch.zeros(B, T, C, H, W, device=img.device)
    
    for t in range(T):
        spikes[:, t] = (latency == t).float() * (img > 0.01).float()
    
    return spikes


def k_winners(potentials: Tensor, k: int = 1, inhibit_radius: int = 0) -> Tensor:
    """
    Select k global winners from potentials.
    
    Args:
        potentials: (B, C, H, W)
        k: Number of winners
        inhibit_radius: Spatial inhibition radius
        
    Returns:
        (B, C, H, W) spike tensor
    """
    B, C, H, W = potentials.shape
    device = potentials.device
    
    output = torch.zeros_like(potentials)
    
    for b in range(B):
        pot = potentials[b].clone()
        
        for _ in range(k):
            max_val = pot.max()
            if max_val <= 0:
                break
            
            # Find max location
            idx = (pot == max_val).nonzero()
            if len(idx) == 0:
                break
            
            c, h, w = idx[0]
            output[b, c, h, w] = 1.0
            
            # Inhibit
            if inhibit_radius > 0:
                h_s = max(0, h - inhibit_radius)
                h_e = min(H, h + inhibit_radius + 1)
                w_s = max(0, w - inhibit_radius)
                w_e = min(W, w + inhibit_radius + 1)
                pot[:, h_s:h_e, w_s:w_e] = -1e9
            else:
                pot[c, h, w] = -1e9
    
    return output


class STDPLayer(nn.Module):
    """
    STDP convolutional layer that learns features.
    
    Uses multiplicative STDP for weight stability.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        threshold: float = float('inf'),  # Start with no threshold (use k-winners)
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.threshold = threshold
        
        # Weight initialization: uniform [0.2, 0.8]
        self.weight = nn.Parameter(
            torch.rand(out_channels, in_channels, kernel_size, kernel_size) * 0.6 + 0.2
        )
        
        # Membrane potential (accumulated over time)
        self.pot: Optional[Tensor] = None
        self.fired: Optional[Tensor] = None  # Track which neurons have fired
    
    def reset(self):
        self.pot = None
        self.fired = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Integrate input, return potential for winner selection.
        Does NOT apply threshold internally.
        """
        # Compute input contribution
        inp = F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)
        
        if self.pot is None:
            self.pot = inp.clone()
            self.fired = torch.zeros_like(inp, dtype=torch.bool)
        else:
            # Accumulate (only for neurons that haven't fired)
            self.pot = self.pot + inp * (~self.fired).float()
        
        return self.pot
    
    def mark_fired(self, spikes: Tensor):
        """Mark neurons as fired (for inhibition)."""
        if self.fired is not None:
            self.fired = self.fired | (spikes > 0.5)
    
    def get_inhibited_pot(self) -> Tensor:
        """Get potential with fired neurons set to -inf."""
        pot = self.pot.clone()
        pot[self.fired] = -1e9
        return pot


class RSTDPTrainer:
    """
    Trainer for STDP/R-STDP learning.
    """
    
    def __init__(
        self,
        layer: STDPLayer,
        a_plus: float = 0.004,
        a_minus: float = 0.003,
    ):
        self.layer = layer
        self.a_plus = a_plus
        self.a_minus = a_minus
        
        # Store spikes for learning
        self.pre_spikes: List[Tensor] = []
        self.post_spikes: List[Tensor] = []
    
    def reset(self):
        self.pre_spikes = []
        self.post_spikes = []
    
    def record(self, pre: Tensor, post: Tensor):
        """Record spike pair for later learning."""
        self.pre_spikes.append(pre.clone())
        self.post_spikes.append(post.clone())
    
    def apply_stdp(self, reward: float = 0.0, lr_scale: float = 1.0):
        """
        Apply STDP update based on recorded spikes.
        
        Uses multiplicative STDP:
        - LTP: Δw = a+ * pre * (1 - w)  [pre fired, strengthen unused weight]
        - LTD: Δw = -a- * (1-pre) * w   [pre didn't fire, weaken existing weight]
        
        With R-STDP:
        - Reward > 0: normal STDP
        - Reward < 0: anti-STDP
        """
        if len(self.pre_spikes) == 0:
            return
        
        with torch.no_grad():
            w = self.layer.weight
            
            for pre, post in zip(self.pre_spikes, self.post_spikes):
                if post.sum() == 0:
                    continue
                
                # Unfold pre to patches
                patches = F.unfold(
                    pre, self.layer.kernel_size,
                    stride=self.layer.stride,
                    padding=self.layer.padding
                )
                B, patch_size, L = patches.shape
                
                k = self.layer.kernel_size
                patches = patches.view(B, self.layer.in_channels, k, k, -1)
                
                # For each post spike
                B, C, H, W = post.shape
                
                for b in range(B):
                    for c in range(C):
                        spike_locs = (post[b, c] > 0.5).nonzero()
                        
                        for loc in spike_locs:
                            h, w_idx = loc[0].item(), loc[1].item()
                            flat_idx = h * W + w_idx
                            
                            if flat_idx >= patches.shape[-1]:
                                continue
                            
                            p = patches[b, :, :, :, flat_idx]  # (in_C, k, k)
                            
                            # Compute STDP update
                            if reward >= 0:
                                # Normal STDP
                                ltp = self.a_plus * lr_scale * p * (1 - w[c])
                                ltd = -self.a_minus * lr_scale * (1 - p) * w[c]
                                
                                if reward > 0:
                                    w[c] += reward * (ltp + ltd)
                                else:
                                    w[c] += ltp + ltd
                            else:
                                # Anti-STDP
                                anti_ltp = -self.a_plus * lr_scale * p * w[c]
                                anti_ltd = self.a_minus * lr_scale * (1 - p) * (1 - w[c])
                                
                                w[c] += (-reward) * (anti_ltp + anti_ltd)
            
            # Clamp weights
            w.data.clamp_(0, 1)


class MozafariV3Network(nn.Module):
    """
    Mozafari-style network with per-class STDP layers.
    """
    
    def __init__(
        self,
        n_classes: int = 2,
        features_per_class: int = 30,
        n_orient: int = 4,
        kernel_size: int = 5,
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.features_per_class = features_per_class
        self.n_orient = n_orient
        
        # Gabor (fixed)
        self.register_buffer('gabor', gabor_filters(n_orient, 5))
        
        # STDP layer per class
        self.layers = nn.ModuleList([
            STDPLayer(n_orient, features_per_class, kernel_size, stride=1, padding=0)
            for _ in range(n_classes)
        ])
        
        # Trainers
        self.trainers = [RSTDPTrainer(layer) for layer in self.layers]
    
    def preprocess(self, x: Tensor) -> Tensor:
        """Apply Gabor + pooling."""
        # Gabor
        s1 = F.conv2d(x, self.gabor, padding=2)
        s1 = torch.abs(s1)
        s1 = s1 / (s1.max() + 1e-8)
        
        # Pool: 28 -> 14
        c1 = F.max_pool2d(s1, 2, 2)
        
        return c1
    
    def reset(self):
        for layer in self.layers:
            layer.reset()
        for trainer in self.trainers:
            trainer.reset()
    
    def forward(self, x: Tensor, T: int = 15) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image (B, 1, 28, 28)
            T: Time steps
            
        Returns:
            prediction: (B,)
            class_spikes: (B, n_classes)
        """
        B = x.shape[0]
        device = x.device
        
        c1 = self.preprocess(x)
        spikes = intensity_to_latency(c1, T)  # (B, T, C, H, W)
        
        self.reset()
        
        class_spike_counts = torch.zeros(B, self.n_classes, device=device)
        
        for t in range(T):
            inp = spikes[:, t]  # (B, n_orient, H, W)
            
            for c, layer in enumerate(self.layers):
                pot = layer(inp)
                inhibited_pot = layer.get_inhibited_pot()
                
                # k-winners from this class's features
                out_spikes = k_winners(inhibited_pot, k=1, inhibit_radius=2)
                layer.mark_fired(out_spikes)
                
                class_spike_counts[:, c] += out_spikes.sum(dim=(1, 2, 3))
        
        predictions = class_spike_counts.argmax(dim=1)
        return predictions, class_spike_counts
    
    def train_step(self, x: Tensor, target: int, T: int = 15, lr: float = 1.0) -> Tuple[int, bool]:
        """
        Training step with R-STDP.
        """
        B = x.shape[0]
        assert B == 1
        device = x.device
        
        c1 = self.preprocess(x)
        spikes = intensity_to_latency(c1, T)
        
        self.reset()
        
        class_spike_counts = torch.zeros(self.n_classes, device=device)
        
        for t in range(T):
            inp = spikes[:, t]
            
            for c, layer in enumerate(self.layers):
                pot = layer(inp)
                inhibited_pot = layer.get_inhibited_pot()
                
                out_spikes = k_winners(inhibited_pot, k=1, inhibit_radius=2)
                layer.mark_fired(out_spikes)
                
                class_spike_counts[c] += out_spikes.sum()
                
                # Record for STDP
                self.trainers[c].record(inp, out_spikes)
        
        prediction = class_spike_counts.argmax().item()
        correct = (prediction == target)
        
        # Apply R-STDP
        for c, trainer in enumerate(self.trainers):
            if c == target:
                # Target class: reward if won, encourage if lost
                reward = 1.0 if correct else 0.5
            else:
                # Non-target: punish if won incorrectly
                reward = -1.0 if (prediction == c and not correct) else 0.0
            
            trainer.apply_stdp(reward=reward, lr_scale=lr)
        
        return prediction, correct


def train_v3(
    n_classes: int = 2,
    digits: List[int] = [0, 1],
    epochs: int = 5,
    samples: int = 3000,
    device: str = 'cpu',
):
    """Train V3 network."""
    
    print("=" * 60)
    print("Mozafari R-STDP V3")
    print("=" * 60)
    print(f"Classes: {n_classes}, Digits: {digits}")
    print(f"Epochs: {epochs}, Samples/epoch: {samples}")
    print()
    
    # Data
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    d2l = {d: i for i, d in enumerate(digits)}
    
    train_idx = [i for i in range(len(train_ds)) if train_ds.targets[i].item() in digits]
    test_idx = [i for i in range(len(test_ds)) if test_ds.targets[i].item() in digits]
    
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    print()
    
    # Network
    net = MozafariV3Network(
        n_classes=n_classes,
        features_per_class=30,
        n_orient=4,
        kernel_size=5,
    ).to(device)
    
    # Train
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        np.random.shuffle(train_idx)
        
        correct = 0
        total = 0
        t0 = time.time()
        
        for i, idx in enumerate(train_idx[:samples]):
            img, orig = train_ds[idx]
            label = d2l[orig]
            
            img = img.unsqueeze(0).to(device)
            lr = 1.0 - 0.5 * (epoch / epochs)
            
            pred, ok = net.train_step(img, label, T=15, lr=lr)
            
            if ok:
                correct += 1
            total += 1
            
            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{samples}] {correct/total*100:.1f}% ({time.time()-t0:.0f}s)")
        
        print(f"  --> {correct/total*100:.1f}%\n")
    
    # Test
    print("=" * 60)
    print("Test")
    print("=" * 60)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for idx in test_idx[:1000]:
            img, orig = test_ds[idx]
            label = d2l[orig]
            
            img = img.unsqueeze(0).to(device)
            pred, _ = net(img, T=15)
            
            if pred[0].item() == label:
                correct += 1
            total += 1
    
    acc = correct / total * 100
    print(f"Test Accuracy: {acc:.1f}% ({correct}/{total})")
    
    return net, acc


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    net, acc = train_v3(
        n_classes=2,
        digits=[0, 1],
        epochs=5,
        samples=3000,
        device=device,
    )
    
    print(f"\nFinal: {acc:.1f}%")
