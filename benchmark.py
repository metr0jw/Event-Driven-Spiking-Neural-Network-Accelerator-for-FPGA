"""
Benchmark Script for PYNQ-Z2 SNN Accelerator
"""

import time
import numpy as np
from snn_fpga_accelerator import SNNAccelerator
from snn_fpga_accelerator.spike_encoding import rate_encode

def benchmark_inference(accelerator, num_samples=100):
    """Benchmark inference performance."""
    print(f"Benchmarking {num_samples} inference samples...")
    
    times = []
    for i in range(num_samples):
        # Generate random input
        input_data = np.random.rand(784)
        spikes = rate_encode(input_data, num_steps=100)
        
        # Time inference
        start_time = time.time()
        output = accelerator.infer(spikes)
        inference_time = time.time() - start_time
        
        times.append(inference_time)
        
        if i % 10 == 0:
            print(f"Sample {i}: {inference_time*1000:.2f} ms")
    
    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    throughput = 1000 / avg_time
    
    print(f"\nBenchmark Results:")
    print(f"Average time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"Throughput: {throughput:.1f} samples/second")

def main():
    print("📊 PYNQ-Z2 SNN Accelerator Benchmark")
    
    accelerator = SNNAccelerator(simulation_mode=True)
    benchmark_inference(accelerator)

if __name__ == '__main__':
    main()
