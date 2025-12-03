#!/usr/bin/env python3
"""
MNIST Classification using SNN with HW-Accurate Neurons

This example demonstrates MNIST classification using:
1. Simple SNN with float weights (~84% accuracy, matches baseline)
2. HW-Accurate SNN with RTL-exact neurons (~59% accuracy)

The Simple SNN validates the spike encoding and decoding approach.
The HW-Accurate SNN demonstrates bit-exact behavior matching the Verilog RTL.

Results (typical):
- Softmax baseline: ~84%
- Simple SNN:       ~84% (100% of baseline) - validates approach
- HW-Accurate SNN:  ~59% (70% of baseline) - shows HW constraints

Usage:
    python mnist_snn_final.py

Author: Jiwoon Lee (@metr0jw)
"""

import numpy as np
import time
import logging
import sys
import os

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 'software/python'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import torch
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    logger.error("PyTorch and torchvision required: pip install torch torchvision")
    sys.exit(1)


def load_mnist():
    """Load MNIST dataset."""
    train_dataset = torchvision.datasets.MNIST(
        root='../data', train=True, download=True, 
        transform=transforms.ToTensor()
    )
    test_dataset = torchvision.datasets.MNIST(
        root='../data', train=False, download=True,
        transform=transforms.ToTensor()
    )
    
    train_images = train_dataset.data.numpy().astype(np.float32) / 255.0
    train_labels = train_dataset.targets.numpy()
    test_images = test_dataset.data.numpy().astype(np.float32) / 255.0
    test_labels = test_dataset.targets.numpy()
    
    return train_images, train_labels, test_images, test_labels


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def train_linear_classifier(X, y, n_epochs=20):
    """Train a linear (softmax) classifier."""
    n_samples, n_features = X.shape
    n_classes = 10
    
    W = np.random.randn(n_features, n_classes) * 0.01
    b = np.zeros(n_classes)
    lr = 0.5
    
    for epoch in range(n_epochs):
        logits = X @ W + b
        probs = softmax(logits)
        
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y] = 1
        
        d_logits = (probs - y_onehot) / n_samples
        d_W = X.T @ d_logits
        d_b = np.sum(d_logits, axis=0)
        
        W -= lr * d_W
        b -= lr * d_b
        
        if (epoch + 1) % 5 == 0:
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == y)
            logger.info(f"  Epoch {epoch+1}: {acc*100:.1f}%")
    
    return W, b


# =============================================================================
# Simplified SNN (High Accuracy)
# =============================================================================

class SimpleHWNeuron:
    """
    Simplified HW-compatible LIF neuron.
    Uses 16-bit membrane potential with saturation.
    """
    def __init__(self, threshold=30000, leak=100, refractory=3):
        self.threshold = threshold
        self.leak = leak
        self.refractory_max = refractory
        
        self.membrane = 0
        self.refractory = 0
        self.spiked = False
    
    def reset(self):
        self.membrane = 0
        self.refractory = 0
        self.spiked = False
    
    def update(self, weighted_input: int) -> bool:
        """Update with scaled weighted input."""
        self.spiked = False
        
        if self.refractory > 0:
            self.refractory -= 1
            return False
        
        # Leak
        self.membrane = max(0, self.membrane - self.leak)
        
        # Add input
        self.membrane = max(0, min(65535, self.membrane + weighted_input))
        
        # Check threshold
        if self.membrane >= self.threshold:
            self.spiked = True
            self.membrane = 0
            self.refractory = self.refractory_max
            return True
        
        return False


class SimpleSNN:
    """
    Simple SNN classifier with float weights.
    Achieves high accuracy for validation.
    """
    def __init__(self, weights, bias, time_steps=100, threshold=20000, leak=100):
        self.weights = weights.astype(np.float32)
        self.bias = bias
        self.time_steps = time_steps
        self.threshold = threshold
        self.leak = leak
        self.n_outputs = 10
        self.input_scale = 1000.0
        
    def predict(self, image, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Rate encode
        flat = image.flatten()
        
        # Create neurons
        neurons = [SimpleHWNeuron(self.threshold, self.leak) for _ in range(10)]
        spike_counts = np.zeros(10)
        
        for t in range(self.time_steps):
            # Generate input spikes
            input_spikes = (np.random.rand(784) < flat).astype(np.float32)
            
            # Weighted sum
            weighted_sums = input_spikes @ self.weights + self.bias
            
            for j in range(10):
                inp = int(weighted_sums[j] * self.input_scale)
                if neurons[j].update(inp):
                    spike_counts[j] += 1
        
        return int(np.argmax(spike_counts)), spike_counts
    
    def evaluate(self, images, labels, max_samples=500):
        n = min(len(images), max_samples)
        correct = 0
        
        for i in range(n):
            pred, _ = self.predict(images[i], seed=i)
            if pred == labels[i]:
                correct += 1
            
            if (i + 1) % 100 == 0:
                logger.info(f"  {i+1}/{n}: {correct/(i+1)*100:.1f}%")
        
        return correct / n * 100


# =============================================================================
# HW-Accurate SNN (Bit-Exact with RTL)
# =============================================================================

try:
    from snn_fpga_accelerator.hw_accurate_simulator import (
        HWAccurateLIFNeuron,
        LIFNeuronParams,
        WEIGHT_SCALE
    )
    HAS_HW_SIM = True
except ImportError:
    HAS_HW_SIM = False


class HWAccurateSNN:
    """
    SNN using the production HW-accurate LIF neurons.
    
    This version uses the HW neuron's tick interface properly,
    accumulating membrane potential each cycle.
    """
    def __init__(self, weights, bias, time_steps=100, threshold=8000, leak=5, scale=50.0):
        self.time_steps = time_steps
        self.n_outputs = 10
        
        # Keep weights as float for weighted sum calculation
        self.weights = weights.astype(np.float32)
        self.bias = bias
        
        # HW neuron params - key is proper threshold/leak balance
        self.neuron_params = LIFNeuronParams(
            threshold=threshold,
            leak_rate=leak,
            refractory_period=2
        )
        
        # Scale factor: smaller values work better with HW neuron
        self.scale = scale
        
        logger.info(f"HW-Accurate SNN initialized")
        logger.info(f"  Threshold: {threshold}, Leak: {leak}, Scale: {self.scale}")
    
    def predict(self, image, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        flat = image.flatten()
        
        # Create HW-accurate neurons
        neurons = [
            HWAccurateLIFNeuron(j, self.neuron_params)
            for j in range(10)
        ]
        
        spike_counts = np.zeros(10)
        
        for t in range(self.time_steps):
            # Generate input spikes for this timestep
            input_spikes = (np.random.rand(784) < flat).astype(np.float32)
            
            # Compute weighted sum (like a neuron integrating all inputs)
            weighted_sums = input_spikes @ self.weights + self.bias
            
            # Feed to each HW neuron
            for j in range(10):
                val = weighted_sums[j] * self.scale
                
                if val > 0:
                    # Excitatory - clip to 8-bit range
                    neurons[j].tick(
                        syn_valid=True,
                        syn_weight=min(int(val), 255),
                        syn_excitatory=True
                    )
                elif val < 0:
                    # Inhibitory
                    neurons[j].tick(
                        syn_valid=True,
                        syn_weight=min(int(-val), 255),
                        syn_excitatory=False
                    )
                else:
                    # No input this cycle - just leak
                    neurons[j].tick(syn_valid=False)
                
                if neurons[j].state.spike_out:
                    spike_counts[j] += 1
        
        return int(np.argmax(spike_counts)), spike_counts
    
    def evaluate(self, images, labels, max_samples=500):
        n = min(len(images), max_samples)
        correct = 0
        class_correct = np.zeros(10)
        class_total = np.zeros(10)
        
        for i in range(n):
            pred, _ = self.predict(images[i], seed=i)
            label = labels[i]
            class_total[label] += 1
            if pred == label:
                correct += 1
                class_correct[label] += 1
            
            if (i + 1) % 100 == 0:
                logger.info(f"  {i+1}/{n}: {correct/(i+1)*100:.1f}%")
        
        print("\nPer-class accuracy:")
        for c in range(10):
            if class_total[c] > 0:
                print(f"  Class {c}: {int(class_correct[c])}/{int(class_total[c])} = "
                      f"{class_correct[c]/class_total[c]*100:.1f}%")
        
        return correct / n * 100


def main():
    print("=" * 70)
    print("MNIST SNN Classification - Complete Validation")
    print("=" * 70)
    
    # Load data
    train_images, train_labels, test_images, test_labels = load_mnist()
    logger.info(f"Loaded {len(train_images)} train, {len(test_images)} test")
    
    # Train baseline
    n_train = 10000
    X_train = train_images[:n_train].reshape(n_train, -1)
    y_train = train_labels[:n_train]
    
    print("\n" + "=" * 70)
    print("1. Training Softmax Classifier (Baseline)")
    print("=" * 70)
    W, b = train_linear_classifier(X_train, y_train, n_epochs=20)
    
    # Baseline accuracy
    X_test = test_images[:500].reshape(500, -1)
    y_test = test_labels[:500]
    baseline_preds = np.argmax(X_test @ W + b, axis=1)
    baseline_acc = np.mean(baseline_preds == y_test) * 100
    print(f"\nSoftmax Baseline: {baseline_acc:.2f}%")
    
    # Simple SNN (high accuracy)
    print("\n" + "=" * 70)
    print("2. Simple SNN (Float Weights)")
    print("=" * 70)
    simple_snn = SimpleSNN(W, b, threshold=20000, leak=100)
    simple_acc = simple_snn.evaluate(test_images, test_labels, max_samples=500)
    print(f"\nSimple SNN: {simple_acc:.2f}%")
    
    # HW-Accurate SNN
    if HAS_HW_SIM:
        print("\n" + "=" * 70)
        print("3. HW-Accurate SNN (Bit-Exact with RTL)")
        print("=" * 70)
        
        # Use tuned parameters (threshold, leak, scale)
        hw_snn = HWAccurateSNN(W, b, threshold=5000, leak=2, scale=200)
        hw_acc = hw_snn.evaluate(test_images, test_labels, max_samples=500)
        print(f"\nHW-Accurate SNN: {hw_acc:.2f}%")
    else:
        print("\nHW-accurate simulator not available")
        hw_acc = 0
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Softmax Baseline:    {baseline_acc:.2f}%")
    print(f"Simple SNN:          {simple_acc:.2f}% ({simple_acc/baseline_acc*100:.1f}% of baseline)")
    if HAS_HW_SIM:
        print(f"HW-Accurate SNN:     {hw_acc:.2f}% ({hw_acc/baseline_acc*100:.1f}% of baseline)")
    
    print("\nNote: HW-accurate SNN has lower accuracy due to:")
    print("  - 8-bit weight quantization")
    print("  - Separate exc/inh processing matching RTL")
    print("  - Strict fixed-point arithmetic")
    print("\nThis demonstrates bit-exact RTL behavior, not optimal accuracy.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
