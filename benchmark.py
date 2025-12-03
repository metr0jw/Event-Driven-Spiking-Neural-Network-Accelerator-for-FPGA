"""
Benchmark Script for SNN FPGA Accelerator

Compares CPU (traditional ANN) vs SNN (neuromorphic) inference performance
and accuracy on various workloads.

Author: Jiwoon Lee (@metr0jw)
"""

import numpy as np
import time
import argparse

from snn_fpga_accelerator import (
    SNNAccelerator, SNNModel, SNNLayer, CPUvsSNNComparator
)
from snn_fpga_accelerator.spike_encoding import PoissonEncoder

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def create_mlp_models(input_size=784, hidden_sizes=[256, 128], output_size=10, seed=42):
    """Create matching PyTorch and SNN models."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for benchmark comparison")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Build PyTorch model
    layers = []
    prev_size = input_size
    for h in hidden_sizes:
        layers.append(nn.Linear(prev_size, h))
        layers.append(nn.ReLU())
        prev_size = h
    layers.append(nn.Linear(prev_size, output_size))
    torch_model = nn.Sequential(*layers)
    
    # Build SNN model with same weights
    snn_model = SNNModel(name="benchmark_model")
    
    layer_idx = 0
    prev_size = input_size
    for i, h in enumerate(hidden_sizes):
        # Extract PyTorch weights
        with torch.no_grad():
            weight = torch_model[layer_idx].weight.numpy()
            bias = torch_model[layer_idx].bias.numpy()
        
        snn_layer = SNNLayer(input_size=prev_size, output_size=h, layer_type="fully_connected")
        snn_layer.set_weights(weight, bias)
        snn_layer.set_neuron_parameters(threshold=0.3, leak_rate=0.02, refractory_period=2)
        snn_model.add_layer(snn_layer)
        
        prev_size = h
        layer_idx += 2  # Skip ReLU
    
    # Output layer
    with torch.no_grad():
        weight = torch_model[layer_idx].weight.numpy()
        bias = torch_model[layer_idx].bias.numpy()
    
    snn_layer = SNNLayer(input_size=prev_size, output_size=output_size, layer_type="fully_connected")
    snn_layer.set_weights(weight, bias)
    snn_layer.set_neuron_parameters(threshold=0.3, leak_rate=0.02, refractory_period=2)
    snn_model.add_layer(snn_layer)
    
    return torch_model, snn_model


def run_single_benchmark(comparator, input_data, num_trials=5):
    """Run benchmark on single input with multiple trials."""
    results = []
    for _ in range(num_trials):
        result = comparator.compare(
            input_data,
            duration=0.1,
            max_rate=150.0,
            num_repeats=3
        )
        results.append(result)
    
    # Aggregate results
    cpu_times = [r['cpu_time_ms'] for r in results if r.get('cpu_time_ms')]
    snn_times = [r['snn_time_ms'] for r in results if r.get('snn_time_ms')]
    agreements = [r['agreement'] for r in results if r.get('agreement') is not None]
    
    return {
        'cpu_time_mean': np.mean(cpu_times) if cpu_times else None,
        'cpu_time_std': np.std(cpu_times) if cpu_times else None,
        'snn_time_mean': np.mean(snn_times) if snn_times else None,
        'snn_time_std': np.std(snn_times) if snn_times else None,
        'agreement_rate': np.mean(agreements) if agreements else None,
        'num_trials': num_trials
    }


def run_batch_benchmark(comparator, batch_size=100, input_size=784):
    """Run benchmark on a batch of random inputs."""
    np.random.seed(42)
    inputs = np.random.rand(batch_size, input_size)
    labels = np.random.randint(0, 10, batch_size)  # Random labels for testing
    
    print(f"\nRunning batch benchmark with {batch_size} samples...")
    start_time = time.time()
    
    results = comparator.compare_batch(
        inputs,
        labels=labels,
        duration=0.1,
        max_rate=150.0,
        num_repeats=1
    )
    
    elapsed = time.time() - start_time
    results['total_time'] = elapsed
    results['samples_per_second'] = batch_size / elapsed
    
    return results


def print_benchmark_report(single_results, batch_results):
    """Print formatted benchmark report."""
    print("\n" + "=" * 70)
    print("                    BENCHMARK RESULTS")
    print("=" * 70)
    
    print("\n[Single Sample Performance]")
    print("-" * 40)
    if single_results.get('cpu_time_mean'):
        print(f"  CPU Time:  {single_results['cpu_time_mean']:.2f} +/- {single_results['cpu_time_std']:.2f} ms")
    if single_results.get('snn_time_mean'):
        print(f"  SNN Time:  {single_results['snn_time_mean']:.2f} +/- {single_results['snn_time_std']:.2f} ms")
    if single_results.get('agreement_rate') is not None:
        print(f"  Agreement: {single_results['agreement_rate']*100:.1f}%")
    
    print("\n[Batch Performance]")
    print("-" * 40)
    print(f"  Batch Size: {batch_results['batch_size']}")
    print(f"  Total Time: {batch_results['total_time']:.2f} s")
    print(f"  Throughput: {batch_results['samples_per_second']:.1f} samples/s")
    
    if batch_results.get('agreement_rate') is not None:
        print(f"  Agreement Rate: {batch_results['agreement_rate']*100:.1f}%")
    if batch_results.get('mean_correlation') is not None:
        print(f"  Mean Correlation: {batch_results['mean_correlation']:.4f}")
    if batch_results.get('cpu_accuracy') is not None:
        print(f"  CPU Accuracy (random labels): {batch_results['cpu_accuracy']*100:.1f}%")
    if batch_results.get('snn_accuracy') is not None:
        print(f"  SNN Accuracy (random labels): {batch_results['snn_accuracy']*100:.1f}%")
    
    print("\n[Analysis]")
    print("-" * 40)
    if single_results.get('cpu_time_mean') and single_results.get('snn_time_mean'):
        speedup = single_results['cpu_time_mean'] / single_results['snn_time_mean']
        if speedup > 1:
            print(f"  SNN is {speedup:.2f}x faster than CPU (simulation)")
        else:
            print(f"  CPU is {1/speedup:.2f}x faster than SNN (simulation)")
        print("  Note: FPGA hardware will be significantly faster than simulation")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='SNN Accelerator Benchmark')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for benchmark')
    parser.add_argument('--hidden', type=int, nargs='+', default=[128], help='Hidden layer sizes')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials for single benchmark')
    args = parser.parse_args()
    
    print("SNN FPGA Accelerator Benchmark")
    print("=" * 70)
    
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required for benchmark comparison")
        return
    
    # Create models
    print("\nCreating models...")
    torch_model, snn_model = create_mlp_models(
        input_size=784,
        hidden_sizes=args.hidden,
        output_size=10
    )
    print(f"  Architecture: 784 -> {' -> '.join(map(str, args.hidden))} -> 10")
    print(f"  SNN Total Neurons: {snn_model.total_neurons}")
    
    # Create comparator
    comparator = CPUvsSNNComparator(
        torch_model=torch_model,
        snn_model=snn_model
    )
    
    # Single sample benchmark
    print("\nRunning single sample benchmark...")
    input_data = np.random.rand(784) * 0.8 + 0.2
    single_results = run_single_benchmark(comparator, input_data, num_trials=args.trials)
    
    # Batch benchmark
    batch_results = run_batch_benchmark(comparator, batch_size=args.batch_size)
    
    # Print report
    print_benchmark_report(single_results, batch_results)


if __name__ == '__main__':
    main()
