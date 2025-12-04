#!/usr/bin/env python3
"""
MNIST Classification with Surrogate Gradient Training

This example demonstrates the new PyTorch-like API for building and training
Spiking Neural Networks with surrogate gradients - similar to snnTorch/SpikingJelly.

Features:
- Surrogate gradient backpropagation (FastSigmoid, ATan, etc.)
- LIF neurons as activation functions
- Rate/Poisson encoding
- HW-constrained training for FPGA deployment

Author: Jiwoon Lee (@metr0jw)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../software/python'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import snn_fpga_accelerator as snn
import time
from tqdm import tqdm


# =============================================================================
# Model Definitions
# =============================================================================

class SimpleSNN(nn.Module):
    """
    Simple SNN using LIF neurons as activation functions.
    Architecture: FC(784->256) -> LIF -> FC(256->10) -> LIF
    """
    
    def __init__(self, num_steps: int = 25):
        super().__init__()
        self.num_steps = num_steps
        
        # Layers
        self.fc1 = nn.Linear(784, 256)
        self.lif1 = snn.LIF(thresh=1.0, tau=0.9)
        
        self.fc2 = nn.Linear(256, 10)
        self.lif2 = snn.LIF(thresh=1.0, tau=0.9)
        
    def forward(self, x):
        """
        Args:
            x: Input spikes (batch, time, features) or (batch, features)
            
        Returns:
            Output spike counts (batch, classes)
        """
        # Flatten if image
        if x.dim() == 4:
            batch = x.size(0)
            x = x.view(batch, -1)  # (batch, 784)
        
        # Reset neuron states
        self.lif1.reset_state()
        self.lif2.reset_state()
        
        # Time loop
        spk_rec = []
        
        for t in range(self.num_steps):
            # Current injection (same input at each timestep, or use encoder)
            cur1 = self.fc1(x)
            spk1 = self.lif1(cur1)
            
            cur2 = self.fc2(spk1)
            spk2 = self.lif2(cur2)
            
            spk_rec.append(spk2)
        
        # Sum spikes over time
        return torch.stack(spk_rec, dim=0).sum(dim=0)  # (batch, 10)


class DeepSNN(nn.Module):
    """
    Deeper SNN with more layers.
    Architecture: FC(784->512) -> LIF -> FC(512->256) -> LIF -> FC(256->10) -> LIF
    """
    
    def __init__(self, num_steps: int = 25, hw_mode: bool = False):
        super().__init__()
        self.num_steps = num_steps
        
        # Layers
        self.fc1 = nn.Linear(784, 512)
        self.lif1 = snn.LIF(thresh=1.0, tau=0.95, hw_mode=hw_mode)
        
        self.fc2 = nn.Linear(512, 256)
        self.lif2 = snn.LIF(thresh=1.0, tau=0.95, hw_mode=hw_mode)
        
        self.fc3 = nn.Linear(256, 10)
        self.lif3 = snn.LIF(thresh=1.0, tau=0.95, hw_mode=hw_mode)
        
    def forward(self, x):
        if x.dim() == 4:
            batch = x.size(0)
            x = x.view(batch, -1)
        
        # Reset states
        self.lif1.reset_state()
        self.lif2.reset_state()
        self.lif3.reset_state()
        
        spk_rec = []
        mem_rec = []
        
        for t in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1 = self.lif1(cur1)
            
            cur2 = self.fc2(spk1)
            spk2 = self.lif2(cur2)
            
            cur3 = self.fc3(spk2)
            spk3 = self.lif3(cur3)
            
            spk_rec.append(spk3)
            mem_rec.append(self.lif3.mem.clone())
        
        return torch.stack(spk_rec, dim=0).sum(dim=0)


class ConvSNN(nn.Module):
    """
    Convolutional SNN for image classification.
    Architecture: Conv(1->32) -> LIF -> Pool -> Conv(32->64) -> LIF -> Pool -> FC
    """
    
    def __init__(self, num_steps: int = 25):
        super().__init__()
        self.num_steps = num_steps
        
        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.lif1 = snn.LIF(thresh=0.5, tau=0.9)
        self.pool1 = nn.AvgPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.lif2 = snn.LIF(thresh=0.5, tau=0.9)
        self.pool2 = nn.AvgPool2d(2)
        
        # FC layers
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.lif3 = snn.LIF(thresh=1.0, tau=0.9)
        
        self.fc2 = nn.Linear(256, 10)
        self.lif4 = snn.LIF(thresh=1.0, tau=0.9)
        
    def forward(self, x):
        # Reset all neurons
        self.lif1.reset_state()
        self.lif2.reset_state()
        self.lif3.reset_state()
        self.lif4.reset_state()
        
        spk_rec = []
        
        for t in range(self.num_steps):
            # Conv block 1
            cur = self.conv1(x)
            spk = self.lif1(cur)
            spk = self.pool1(spk)
            
            # Conv block 2
            cur = self.conv2(spk)
            spk = self.lif2(cur)
            spk = self.pool2(spk)
            
            # Flatten
            spk = spk.view(spk.size(0), -1)
            
            # FC block 1
            cur = self.fc1(spk)
            spk = self.lif3(cur)
            
            # FC block 2
            cur = self.fc2(spk)
            spk = self.lif4(cur)
            
            spk_rec.append(spk)
        
        return torch.stack(spk_rec, dim=0).sum(dim=0)


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # Loss on spike counts
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100. * correct / total


def test(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


# =============================================================================
# Main
# =============================================================================

def main():
    # Configuration
    batch_size = 128
    epochs = 5
    lr = 1e-3
    num_steps = 25  # Time steps for SNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("MNIST Classification with Surrogate Gradient SNN")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Time steps: {num_steps}")
    print(f"Learning rate: {lr}")
    print()
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model
    print("Creating SimpleSNN model...")
    model = SimpleSNN(num_steps=num_steps).to(device)
    print(model)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {params:,}")
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    best_acc = 0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        test_acc = test(model, test_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            # Save best model
            torch.save(model.state_dict(), 'best_snn_model.pth')
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Total Training Time: {elapsed:.1f}s")
    
    # Export for FPGA (optional)
    print("\n" + "=" * 60)
    print("Exporting for FPGA")
    print("=" * 60)
    
    # Quantize weights
    print("Quantizing weights to int8...")
    weights = {}
    for name, param in model.named_parameters():
        w = param.data.cpu().numpy()
        w_int8 = snn.quantize(w, bits=8)
        weights[name] = w_int8
        print(f"  {name}: {w.shape} -> int8")
    
    # Save quantized weights
    import numpy as np
    np.savez('snn_weights_int8.npz', **weights)
    print("Saved quantized weights to snn_weights_int8.npz")
    
    return best_acc


if __name__ == '__main__':
    main()
