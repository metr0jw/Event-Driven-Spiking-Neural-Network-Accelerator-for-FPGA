"""
Performance Benchmark Suite for SNN Accelerator.

This module provides lightweight performance benchmarks focused on:
- Configuration overhead
- Memory footprint estimation  
- Resource utilization estimates
- Computational complexity analysis

For full latency/throughput/accuracy benchmarks, hardware is required.

Usage:
    pytest tests/test_benchmark.py -v -s
"""

import numpy as np
import pytest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from snn_fpga_accelerator.accelerator import SNNAccelerator


class TestConfigurationBenchmark:
    """Benchmark configuration overhead."""
    
    def test_config_encoder_latency(self):
        """Benchmark encoder configuration time."""
        acc = SNNAccelerator(simulation_mode=True)
        
        times = []
        for _ in range(100):
            start = time.perf_counter()
            acc.configure_encoder(
                encoding_type='rate_poisson',
                num_steps=100,
                rate_scale=256,
                two_neuron_enable=False
            )
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"\nEncoder Configuration Time: {mean_time:.3f} ± {std_time:.3f} ms")
        
        # Configuration should be fast (< 10ms)
        assert mean_time < 10, f"Config too slow: {mean_time:.3f} ms"
    
    def test_config_network_latency(self):
        """Benchmark network configuration time."""
        acc = SNNAccelerator(simulation_mode=True)
        
        # Create simple network config
        config = {
            'num_neurons': 138,  # 128 hidden + 10 output
            'topology': {'layers': 2}
        }
        
        times = []
        for _ in range(50):
            start = time.perf_counter()
            acc.configure_network(config=config)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        mean_time = np.mean(times)
        print(f"\nNetwork Configuration Time: {mean_time:.3f} ms")


class TestMemoryFootprint:
    """Memory footprint estimation tests."""
    
    def test_weight_memory_requirements(self):
        """Calculate weight memory requirements."""
        # Network configuration
        input_size = 784   # MNIST 28x28
        hidden_size = 128
        output_size = 10
        
        # Weights (16-bit fixed-point)
        w1_count = input_size * hidden_size
        w2_count = hidden_size * output_size
        
        w1_bytes = w1_count * 2  # 16-bit = 2 bytes
        w2_bytes = w2_count * 2
        
        total_weight_bytes = w1_bytes + w2_bytes
        total_weight_kb = total_weight_bytes / 1024
        
        print("\n" + "="*60)
        print("Weight Memory Footprint")
        print("="*60)
        print(f"  Input -> Hidden:  {w1_count:,} weights ({w1_bytes/1024:.2f} KB)")
        print(f"  Hidden -> Output: {w2_count:,} weights ({w2_bytes/1024:.2f} KB)")
        print(f"  Total weights:    {w1_count + w2_count:,} ({total_weight_kb:.2f} KB)")
        print("="*60)
        
        # Should fit in PYNQ-Z2 BRAM (512 KB total)
        assert total_weight_kb < 200, f"Weights too large: {total_weight_kb:.2f} KB"
    
    def test_neuron_state_memory(self):
        """Calculate neuron state memory requirements."""
        hidden_size = 128
        output_size = 10
        total_neurons = hidden_size + output_size
        
        # Per neuron state (membrane potential, threshold, etc.)
        # Assuming 32-bit fixed-point per state variable
        state_vars_per_neuron = 3  # v_mem, v_thresh, last_spike_time
        bytes_per_var = 4  # 32-bit
        
        total_bytes = total_neurons * state_vars_per_neuron * bytes_per_var
        total_kb = total_bytes / 1024
        
        print("\n" + "="*60)
        print("Neuron State Memory")
        print("="*60)
        print(f"  Total neurons:     {total_neurons}")
        print(f"  State vars/neuron: {state_vars_per_neuron}")
        print(f"  Bytes/var:         {bytes_per_var}")
        print(f"  Total memory:      {total_kb:.2f} KB")
        print("="*60)
        
        assert total_kb < 10, f"Neuron state too large: {total_kb:.2f} KB"
    
    def test_spike_buffer_memory(self):
        """Calculate spike buffer memory requirements."""
        max_spikes_per_timestep = 1000
        timesteps_buffered = 10
        
        # Spike format: (neuron_id, timestamp, weight)
        # neuron_id: 16-bit, timestamp: 32-bit, weight: 16-bit = 8 bytes
        bytes_per_spike = 8
        
        buffer_bytes = max_spikes_per_timestep * timesteps_buffered * bytes_per_spike
        buffer_kb = buffer_bytes / 1024
        
        print("\n" + "="*60)
        print("Spike Buffer Memory")
        print("="*60)
        print(f"  Max spikes/step: {max_spikes_per_timestep}")
        print(f"  Steps buffered:  {timesteps_buffered}")
        print(f"  Bytes/spike:     {bytes_per_spike}")
        print(f"  Total buffer:    {buffer_kb:.2f} KB")
        print("="*60)
        
        assert buffer_kb < 100, f"Spike buffer too large: {buffer_kb:.2f} KB"
    
    def test_total_memory_footprint(self):
        """Calculate total on-chip memory footprint."""
        # Weight memory
        weights_kb = ((784 * 128) + (128 * 10)) * 2 / 1024
        
        # Neuron state
        neurons_kb = (128 + 10) * 3 * 4 / 1024
        
        # Spike buffers
        spikes_kb = 1000 * 10 * 8 / 1024
        
        # Encoder buffers (for on-chip encoding)
        encoder_kb = 784 * 100 / 1024  # 784 channels, 100 timesteps
        
        total_kb = weights_kb + neurons_kb + spikes_kb + encoder_kb
        total_mb = total_kb / 1024
        
        print("\n" + "="*60)
        print("Total Memory Footprint")
        print("="*60)
        print(f"  Weights:       {weights_kb:>8.2f} KB")
        print(f"  Neuron state:  {neurons_kb:>8.2f} KB")
        print(f"  Spike buffers: {spikes_kb:>8.2f} KB")
        print(f"  Encoder data:  {encoder_kb:>8.2f} KB")
        print(f"  {'─'*40}")
        print(f"  Total:         {total_kb:>8.2f} KB ({total_mb:.3f} MB)")
        print(f"  PYNQ-Z2 BRAM:  {512:>8.0f} KB")
        print(f"  Utilization:   {(total_kb/512)*100:>7.1f}%")
        print("="*60)
        
        # Should use < 80% of BRAM
        assert total_kb < 512 * 0.8, f"Memory usage too high: {total_kb:.2f} KB"


class TestResourceUtilization:
    """Resource utilization estimates for FPGA."""
    
    def test_lut_utilization_estimate(self):
        """Estimate LUT utilization."""
        # PYNQ-Z2 (xc7z020): 53,200 LUTs
        available_luts = 53200
        
        # Estimated LUT usage
        encoder_luts = 5000      # On-chip encoder
        neuron_luts = 3000       # LIF neuron logic
        router_luts = 2000       # Spike router
        stdp_luts = 4000         # STDP learning
        control_luts = 1000      # Control logic
        
        total_luts = encoder_luts + neuron_luts + router_luts + stdp_luts + control_luts
        utilization = (total_luts / available_luts) * 100
        
        print("\n" + "="*60)
        print("LUT Utilization Estimate")
        print("="*60)
        print(f"  Encoder:       {encoder_luts:>6,} LUTs")
        print(f"  Neurons:       {neuron_luts:>6,} LUTs")
        print(f"  Router:        {router_luts:>6,} LUTs")
        print(f"  STDP:          {stdp_luts:>6,} LUTs")
        print(f"  Control:       {control_luts:>6,} LUTs")
        print(f"  {'─'*40}")
        print(f"  Total:         {total_luts:>6,} LUTs")
        print(f"  Available:     {available_luts:>6,} LUTs")
        print(f"  Utilization:   {utilization:>6.1f}%")
        print("="*60)
        
        # Should use < 60% of LUTs
        assert utilization < 60, f"LUT usage too high: {utilization:.1f}%"
    
    def test_dsp_utilization_estimate(self):
        """Estimate DSP48E1 utilization."""
        # PYNQ-Z2: 220 DSP slices
        available_dsps = 220
        
        # DSP usage
        mac_operations = 50      # Weight * input multiplications
        stdp_mults = 20          # STDP weight updates
        encoder_mults = 10       # Encoder scaling
        
        total_dsps = mac_operations + stdp_mults + encoder_mults
        utilization = (total_dsps / available_dsps) * 100
        
        print("\n" + "="*60)
        print("DSP Utilization Estimate")
        print("="*60)
        print(f"  MAC operations: {mac_operations:>4} DSPs")
        print(f"  STDP learning:  {stdp_mults:>4} DSPs")
        print(f"  Encoder:        {encoder_mults:>4} DSPs")
        print(f"  {'─'*40}")
        print(f"  Total:          {total_dsps:>4} DSPs")
        print(f"  Available:      {available_dsps:>4} DSPs")
        print(f"  Utilization:    {utilization:>5.1f}%")
        print("="*60)
        
        assert utilization < 50, f"DSP usage too high: {utilization:.1f}%"


class TestComputationalComplexity:
    """Computational complexity analysis."""
    
    def test_mac_operations_per_inference(self):
        """Calculate MAC operations per inference."""
        input_size = 784
        hidden_size = 128
        output_size = 10
        num_timesteps = 100
        
        # Sparse computation: only active neurons compute
        # Assume 10% sparsity (10% of neurons fire)
        sparsity = 0.1
        
        # Layer 1: input -> hidden
        mac_layer1_per_step = int(input_size * hidden_size * sparsity)
        
        # Layer 2: hidden -> output
        mac_layer2_per_step = int(hidden_size * output_size * sparsity)
        
        total_mac_per_step = mac_layer1_per_step + mac_layer2_per_step
        total_mac_per_inference = total_mac_per_step * num_timesteps
        
        # GOPs (Giga Operations)
        gops = total_mac_per_inference / 1e9
        
        print("\n" + "="*60)
        print("Computational Complexity (10% sparsity)")
        print("="*60)
        print(f"  Layer 1 MACs/step:  {mac_layer1_per_step:>10,}")
        print(f"  Layer 2 MACs/step:  {mac_layer2_per_step:>10,}")
        print(f"  Total MACs/step:    {total_mac_per_step:>10,}")
        print(f"  Timesteps:          {num_timesteps:>10,}")
        print(f"  {'─'*40}")
        print(f"  Total MACs:         {total_mac_per_inference:>10,}")
        print(f"  GOPs:               {gops:>10.3f}")
        print("="*60)
        
        # At 100 MHz, DSP can do 100M MACs/sec
        # With 80 DSPs -> 8 GMAC/sec
        # This inference needs < 1 GMAC, so should complete in < 125ms
        print(f"\nAt 100 MHz with 80 DSPs:")
        print(f"  Throughput: 8.0 GMAC/sec")
        print(f"  Time/inference: {(total_mac_per_inference / 8e9) * 1000:.2f} ms")
    
    def test_encoder_complexity(self):
        """Calculate encoder computational complexity."""
        num_pixels = 784  # MNIST
        num_timesteps = 100
        
        # Rate encoding: random comparison per pixel per timestep
        comparisons = num_pixels * num_timesteps
        
        # With two-neuron encoding: 2x output channels
        two_neuron_overhead = 2.0
        
        print("\n" + "="*60)
        print("Encoder Complexity")
        print("="*60)
        print(f"  Pixels:             {num_pixels}")
        print(f"  Timesteps:          {num_timesteps}")
        print(f"  Comparisons:        {comparisons:,}")
        print(f"  Two-neuron factor:  {two_neuron_overhead}x")
        print(f"  Total operations:   {int(comparisons * two_neuron_overhead):,}")
        print("="*60)


def test_print_comprehensive_report():
    """Print comprehensive resource report."""
    print("\n" + "="*70)
    print(" "*15 + "SNN ACCELERATOR RESOURCE SUMMARY")
    print("="*70)
    
    # Memory
    print("\n[MEMORY FOOTPRINT]")
    weights_kb = 201.25
    state_kb = 1.62
    spikes_kb = 78.13
    encoder_kb = 76.56
    total_kb = weights_kb + state_kb + spikes_kb + encoder_kb
    
    print(f"  Weights:       {weights_kb:>8.2f} KB")
    print(f"  Neuron state:  {state_kb:>8.2f} KB")
    print(f"  Spike buffers: {spikes_kb:>8.2f} KB")
    print(f"  Encoder:       {encoder_kb:>8.2f} KB")
    print(f"  Total:         {total_kb:>8.2f} KB ({(total_kb/512)*100:.1f}% of 512 KB)")
    
    # Logic
    print("\n[LOGIC UTILIZATION]")
    total_luts = 15000
    total_dsps = 80
    print(f"  LUTs:  {total_luts:>6,} / 53,200 ({(total_luts/53200)*100:.1f}%)")
    print(f"  DSPs:  {total_dsps:>6} / 220    ({(total_dsps/220)*100:.1f}%)")
    
    # Performance
    print("\n[PERFORMANCE ESTIMATES]")
    print(f"  Clock frequency:     100 MHz")
    print(f"  Throughput (10% sp): ~8 GMAC/sec")
    print(f"  Inference latency:   ~10-50 ms")
    print(f"  Power (estimated):   ~2-3 W")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
