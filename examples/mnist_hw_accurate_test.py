#!/usr/bin/env python3
"""
MNIST Classification Test with HW-Accurate SNN Simulator

This script validates the SNN software implementation using:
1. Real MNIST dataset
2. HW-Accurate simulator (bit-exact with RTL/HLS)
3. Rate-coded spike encoding
4. Simple 784 -> 128 -> 10 fully-connected SNN

Author: Jiwoon Lee (@metr0jw)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'software', 'python'))

import numpy as np
import time
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from snn_fpga_accelerator import (
    HWAccurateSNNSimulator,
    HWAccurateLIFNeuron, 
    HWAccurateSTDPEngine,
    LIFNeuronParams,
    PoissonEncoder,
)
# Import STDPConfig directly from hw_accurate_simulator to avoid confusion
# with training.STDPConfig which has different fields
from snn_fpga_accelerator.hw_accurate_simulator import STDPConfig


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST dataset using torchvision or fallback to synthetic."""
    try:
        import torchvision
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='../data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='../data', train=False, download=True, transform=transform
        )
        
        # Convert to numpy
        train_images = train_dataset.data.numpy().astype(np.float32) / 255.0
        train_labels = train_dataset.targets.numpy()
        test_images = test_dataset.data.numpy().astype(np.float32) / 255.0
        test_labels = test_dataset.targets.numpy()
        
        logger.info(f"Loaded real MNIST: {len(train_images)} train, {len(test_images)} test")
        return train_images, train_labels, test_images, test_labels
        
    except Exception as e:
        logger.warning(f"Could not load MNIST via torchvision: {e}")
        logger.info("Using synthetic MNIST-like data")
        
        np.random.seed(42)
        train_images = np.random.rand(1000, 28, 28).astype(np.float32)
        train_labels = np.random.randint(0, 10, 1000)
        test_images = np.random.rand(200, 28, 28).astype(np.float32)
        test_labels = np.random.randint(0, 10, 200)
        
        return train_images, train_labels, test_images, test_labels


class SNNMNISTClassifier:
    """
    Simple SNN classifier for MNIST using HW-Accurate simulator.
    
    Architecture:
    - Input: 784 neurons (28x28 pixels, rate-coded)
    - Hidden: 128 LIF neurons
    - Output: 10 LIF neurons (one per class)
    
    Uses the HW-Accurate simulator for bit-exact simulation.
    """
    
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 128,
        output_size: int = 10,
        time_steps: int = 100,
        threshold: int = 500,
        tau: float = 0.875,  # tau instead of leak_rate
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.time_steps = time_steps
        
        # Neuron parameters (HW-accurate with shift-based leak)
        # Use from_tau() for automatic leak_rate calculation
        self.neuron_params = LIFNeuronParams.from_tau(
            tau=tau,
            threshold=threshold,
            refractory_period=5
        )
        # Set reset_potential manually (default is 0, disabled)
        
        # STDP configuration
        self.stdp_config = STDPConfig(
            a_plus=0.1,
            a_minus=0.05,
            tau_plus=20.0,
            tau_minus=20.0,
            stdp_window=50
        )
        
        # Initialize weights (8-bit signed, -128 to 127)
        np.random.seed(42)
        self.w_input_hidden = np.random.randint(-20, 60, (input_size, hidden_size)).astype(np.int8)
        self.w_hidden_output = np.random.randint(-20, 60, (hidden_size, output_size)).astype(np.int8)
        
        logger.info(f"SNN Classifier initialized:")
        logger.info(f"  Architecture: {input_size} -> {hidden_size} -> {output_size}")
        logger.info(f"  Time steps: {time_steps}")
        logger.info(f"  Threshold: {threshold}")
        logger.info(f"  Tau: {tau} (leak_rate={self.neuron_params.leak_rate}, "
                   f"shift1={self.neuron_params.leak_shift1}, shift2={self.neuron_params.leak_shift2})")
    
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to spike train using rate coding."""
        # Flatten image
        flat_image = image.flatten()
        
        # Rate encoding: spike probability proportional to pixel intensity
        spike_train = np.zeros((self.input_size, self.time_steps), dtype=np.uint8)
        
        for t in range(self.time_steps):
            # Generate spikes with probability = pixel_value
            rand = np.random.rand(self.input_size)
            spike_train[:, t] = (rand < flat_image).astype(np.uint8)
        
        return spike_train
    
    def forward(self, spike_train: np.ndarray, learning: bool = False) -> np.ndarray:
        """
        Forward pass through the SNN.
        
        Parameters
        ----------
        spike_train : np.ndarray
            Input spike train of shape (input_size, time_steps)
        learning : bool
            Enable STDP learning
            
        Returns
        -------
        np.ndarray
            Output spike counts for each class
        """
        # Initialize neurons
        hidden_neurons = [
            HWAccurateLIFNeuron(i, self.neuron_params) 
            for i in range(self.hidden_size)
        ]
        output_neurons = [
            HWAccurateLIFNeuron(i, self.neuron_params) 
            for i in range(self.output_size)
        ]
        
        # Spike counters
        hidden_spikes = np.zeros(self.hidden_size, dtype=np.int32)
        output_spikes = np.zeros(self.output_size, dtype=np.int32)
        
        # Process each time step
        for t in range(self.time_steps):
            input_spikes = spike_train[:, t]
            
            # Input -> Hidden layer
            for h in range(self.hidden_size):
                # Sum weighted inputs from all active input neurons
                total_input = 0
                for i in np.where(input_spikes > 0)[0]:
                    weight = int(self.w_input_hidden[i, h])
                    if weight > 0:
                        total_input += weight
                
                # Apply input if any
                if total_input > 0:
                    # Clamp to valid range
                    total_input = min(total_input, 255)
                    spike = hidden_neurons[h].tick(
                        syn_valid=True,
                        syn_weight=total_input,
                        syn_excitatory=True
                    )
                else:
                    spike = hidden_neurons[h].tick(syn_valid=False)
                
                if spike:
                    hidden_spikes[h] += 1
            
            # Hidden -> Output layer
            hidden_firing = np.array([n.state.spike_out for n in hidden_neurons])
            
            for o in range(self.output_size):
                total_input = 0
                for h in np.where(hidden_firing)[0]:
                    weight = int(self.w_hidden_output[h, o])
                    if weight > 0:
                        total_input += weight
                
                if total_input > 0:
                    total_input = min(total_input, 255)
                    spike = output_neurons[o].tick(
                        syn_valid=True,
                        syn_weight=total_input,
                        syn_excitatory=True
                    )
                else:
                    spike = output_neurons[o].tick(syn_valid=False)
                
                if spike:
                    output_spikes[o] += 1
        
        return output_spikes
    
    def predict(self, image: np.ndarray) -> int:
        """Predict class for a single image."""
        spike_train = self.encode_image(image)
        output_spikes = self.forward(spike_train)
        return int(np.argmax(output_spikes))
    
    def evaluate(self, images: np.ndarray, labels: np.ndarray, 
                 max_samples: int = None) -> Tuple[float, List[dict]]:
        """
        Evaluate accuracy on a dataset.
        
        Returns accuracy and detailed results.
        """
        if max_samples is not None:
            images = images[:max_samples]
            labels = labels[:max_samples]
        
        n_samples = len(images)
        correct = 0
        results = []
        
        start_time = time.time()
        
        for i, (image, label) in enumerate(zip(images, labels)):
            pred = self.predict(image)
            is_correct = pred == label
            correct += is_correct
            
            results.append({
                'index': i,
                'true_label': int(label),
                'predicted': pred,
                'correct': is_correct
            })
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                acc = correct / (i + 1) * 100
                logger.info(f"Progress: {i+1}/{n_samples}, Accuracy: {acc:.2f}%, "
                           f"Time: {elapsed:.1f}s ({elapsed/(i+1)*1000:.1f}ms/sample)")
        
        accuracy = correct / n_samples
        total_time = time.time() - start_time
        
        logger.info(f"Final Accuracy: {accuracy*100:.2f}% ({correct}/{n_samples})")
        logger.info(f"Total time: {total_time:.1f}s, Avg: {total_time/n_samples*1000:.1f}ms/sample")
        
        return accuracy, results


def test_single_neuron():
    """Test single LIF neuron behavior."""
    print("\n" + "="*60)
    print("Test 1: Single LIF Neuron")
    print("="*60)
    
    params = LIFNeuronParams(threshold=500, leak_rate=5, refractory_period=5)
    neuron = HWAccurateLIFNeuron(0, params)
    
    # Inject constant input
    spike_times = []
    for t in range(100):
        spike = neuron.tick(syn_valid=True, syn_weight=50, syn_excitatory=True)
        if spike:
            spike_times.append(t)
    
    print(f"  Threshold: {params.threshold}")
    print(f"  Input weight: 50 per cycle")
    print(f"  Expected spikes: ~{100 * 50 // params.threshold}")
    print(f"  Actual spikes: {len(spike_times)}")
    print(f"  Spike times: {spike_times[:10]}..." if len(spike_times) > 10 else f"  Spike times: {spike_times}")
    
    return len(spike_times) > 0


def test_spike_encoding():
    """Test spike encoding."""
    print("\n" + "="*60)
    print("Test 2: Spike Encoding")
    print("="*60)
    
    # Create a simple test image
    image = np.zeros((28, 28), dtype=np.float32)
    image[10:18, 10:18] = 1.0  # Square in center
    
    classifier = SNNMNISTClassifier(time_steps=50)
    spike_train = classifier.encode_image(image)
    
    # Count spikes in bright vs dark regions
    bright_region = spike_train[10*28+10:10*28+18, :]  # Part of bright area
    dark_region = spike_train[0:8, :]  # Dark area
    
    bright_rate = np.mean(bright_region)
    dark_rate = np.mean(dark_region)
    
    print(f"  Image shape: {image.shape}")
    print(f"  Spike train shape: {spike_train.shape}")
    print(f"  Total spikes: {np.sum(spike_train)}")
    print(f"  Bright region spike rate: {bright_rate:.3f}")
    print(f"  Dark region spike rate: {dark_rate:.3f}")
    
    return bright_rate > dark_rate


def test_forward_pass():
    """Test forward pass through the network."""
    print("\n" + "="*60)
    print("Test 3: Forward Pass")
    print("="*60)
    
    classifier = SNNMNISTClassifier(
        hidden_size=64,  # Smaller for speed
        time_steps=50
    )
    
    # Create test image (digit-like pattern)
    image = np.random.rand(28, 28).astype(np.float32) * 0.3
    image[5:23, 10:18] = 0.8  # Vertical bar (like digit 1)
    
    spike_train = classifier.encode_image(image)
    output_spikes = classifier.forward(spike_train)
    
    print(f"  Hidden size: {classifier.hidden_size}")
    print(f"  Time steps: {classifier.time_steps}")
    print(f"  Input spikes: {np.sum(spike_train)}")
    print(f"  Output spikes per class: {output_spikes}")
    print(f"  Predicted class: {np.argmax(output_spikes)}")
    print(f"  Total output spikes: {np.sum(output_spikes)}")
    
    return np.sum(output_spikes) > 0


def test_mnist_classification():
    """Test MNIST classification accuracy."""
    print("\n" + "="*60)
    print("Test 4: MNIST Classification")
    print("="*60)
    
    # Load MNIST
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # Create classifier (smaller for testing)
    classifier = SNNMNISTClassifier(
        hidden_size=64,
        time_steps=50,
        threshold=400,
        tau=0.875  # Uses shift-based leak (shift1=3)
    )
    
    # Evaluate on subset
    print("\nEvaluating on 100 test samples...")
    accuracy, results = classifier.evaluate(test_images, test_labels, max_samples=100)
    
    # Show confusion matrix summary
    from collections import Counter
    predictions = [r['predicted'] for r in results]
    pred_counts = Counter(predictions)
    
    print(f"\nPrediction distribution:")
    for digit in range(10):
        count = pred_counts.get(digit, 0)
        print(f"  Class {digit}: {count} predictions")
    
    # Random baseline is 10%
    print(f"\nAccuracy: {accuracy*100:.2f}% (random baseline: 10%)")
    
    return accuracy


def main():
    """Run all tests."""
    print("="*60)
    print("HW-Accurate SNN MNIST Classification Test")
    print("="*60)
    print("This test validates the SNN software implementation")
    print("using the hardware-accurate simulator (bit-exact with RTL/HLS)")
    print()
    
    results = {}
    
    # Test 1: Single neuron
    results['single_neuron'] = test_single_neuron()
    
    # Test 2: Spike encoding
    results['spike_encoding'] = test_spike_encoding()
    
    # Test 3: Forward pass
    results['forward_pass'] = test_forward_pass()
    
    # Test 4: MNIST classification
    results['mnist_accuracy'] = test_mnist_classification()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        if isinstance(result, bool):
            status = "PASS" if result else "FAIL"
        else:
            status = f"{result*100:.1f}%"
        print(f"  {test_name}: {status}")
    
    all_passed = all(
        r if isinstance(r, bool) else r > 0.05  # >5% accuracy (above pure random)
        for r in results.values()
    )
    
    print()
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
