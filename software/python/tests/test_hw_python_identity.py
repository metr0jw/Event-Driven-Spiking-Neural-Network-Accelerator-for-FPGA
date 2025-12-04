#!/usr/bin/env python3
"""
Hardware-Python Identity Verification Tests

This test verifies that the Python implementations exactly match the
Verilog RTL hardware implementations, bit-for-bit.

Tests:
1. LIF Neuron shift-based leak
2. Synaptic integration
3. Spike generation
4. Refractory period
5. STDP learning

Author: Jiwoon Lee (@metr0jw)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from snn_fpga_accelerator.hw_accurate_simulator import (
    HWAccurateLIFNeuron,
    LIFNeuronParams,
    LIFNeuronState,
)
from snn_fpga_accelerator.neuron import (
    LIF,
    tau_to_hw_leak_rate,
    hw_leak_rate_to_tau,
)


def test_shift_based_leak():
    """
    Test that shift-based leak exactly matches Verilog RTL.
    
    Verilog:
        leak_primary = v_mem >> shift1
        leak_secondary = v_mem >> shift2 (if enabled)
        v_mem_next = v_mem - leak_primary - leak_secondary
    """
    print("=" * 60)
    print("Test 1: Shift-Based Leak")
    print("=" * 60)
    
    test_cases = [
        # (v_mem, leak_rate, expected_leak, expected_v_mem_next)
        # leak_rate = shift1 | (shift2 << 3)
        
        # Single shift configurations
        (1000, 3, 125, 875),      # shift=3: 1000 >> 3 = 125, 1000 - 125 = 875
        (1000, 4, 62, 938),       # shift=4: 1000 >> 4 = 62, 1000 - 62 = 938
        (256, 3, 32, 224),        # shift=3: 256 >> 3 = 32, 256 - 32 = 224
        (256, 4, 16, 240),        # shift=4: 256 >> 4 = 16, 256 - 16 = 240
        (65535, 3, 8191, 57344),  # Max value test
        (7, 3, 0, 7),             # Small value (shift result = 0)
        
        # Dual shift configurations
        # leak_rate = shift1 | (shift2 << 3) = 3 | (6 << 3) = 3 | 48 = 51
        (1000, 51, 125 + 15, 860),  # shift1=3, shift2=6: 125 + 15 = 140
        (1000, 52, 62 + 15, 923),   # shift1=4, shift2=6: 62 + 15 = 77
    ]
    
    all_passed = True
    
    for v_mem, leak_rate, expected_leak, expected_next in test_cases:
        # Create neuron with specific leak_rate
        params = LIFNeuronParams(
            threshold=65535,  # High threshold to prevent spiking
            leak_rate=leak_rate,
            refractory_period=0,
        )
        neuron = HWAccurateLIFNeuron(neuron_id=0, params=params)
        neuron.state.v_mem = v_mem
        
        # Tick without input (apply leak only)
        neuron.tick(syn_valid=False, enable=True)
        
        actual_next = neuron.state.v_mem
        
        shift1 = leak_rate & 0x07
        shift2_cfg = (leak_rate >> 3) & 0x1F
        shift2 = shift2_cfg & 0x07 if shift2_cfg != 0 else 0
        tau = params.tau
        
        passed = actual_next == expected_next
        all_passed = all_passed and passed
        
        status = "PASS" if passed else "FAIL"
        print(f"  v_mem={v_mem:5d}, leak_rate={leak_rate:2d} (shift1={shift1}, shift2={shift2}, tau={tau:.4f})")
        print(f"    Expected: {expected_next:5d}, Actual: {actual_next:5d} [{status}]")
    
    print(f"\nTest 1 Result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_synaptic_integration():
    """
    Test that synaptic integration matches Verilog RTL.
    
    Verilog:
        syn_contribution = syn_excitatory ? syn_weight : -syn_weight
        v_mem_next = saturate(v_mem + syn_contribution)
        
    Note: We test with very high threshold (65535) but need to be careful:
    when v_mem_next >= threshold, spike occurs and membrane resets.
    To test saturation without spike, we use threshold=65536 (effectively disabled)
    by setting threshold to max value (65535) but testing values that don't reach it.
    """
    print("\n" + "=" * 60)
    print("Test 2: Synaptic Integration")
    print("=" * 60)
    
    # Test cases with high threshold to avoid spike
    test_cases = [
        # (v_mem, syn_weight, syn_excitatory, threshold, expected_next)
        (100, 50, True, 65535, 150),      # Excitatory
        (100, 50, False, 65535, 50),      # Inhibitory
        (100, 150, False, 65535, 0),      # Inhibitory saturate at 0
        (0, 255, True, 65535, 255),       # From zero
        (1000, 0, True, 65535, 1000),     # Zero weight
    ]
    
    all_passed = True
    
    for v_mem, syn_weight, syn_excitatory, threshold, expected_next in test_cases:
        params = LIFNeuronParams(
            threshold=threshold,
            leak_rate=0,      # No leak (shift=0)
            refractory_period=0,
        )
        neuron = HWAccurateLIFNeuron(neuron_id=0, params=params)
        neuron.state.v_mem = v_mem
        
        # Tick with synaptic input
        neuron.tick(syn_valid=True, syn_weight=syn_weight, 
                   syn_excitatory=syn_excitatory, enable=True)
        
        actual_next = neuron.state.v_mem
        passed = actual_next == expected_next
        all_passed = all_passed and passed
        
        status = "PASS" if passed else "FAIL"
        exc_str = "exc" if syn_excitatory else "inh"
        print(f"  v_mem={v_mem:5d}, weight={syn_weight:3d} ({exc_str})")
        print(f"    Expected: {expected_next:5d}, Actual: {actual_next:5d} [{status}]")
    
    # Special test for saturation at max (need to test _saturate_16bit directly)
    print("\n  Testing saturation at max (direct _saturate_16bit):")
    params = LIFNeuronParams(threshold=65535, leak_rate=0, refractory_period=0)
    neuron = HWAccurateLIFNeuron(neuron_id=0, params=params)
    
    # Test overflow saturation
    overflow_result = neuron._saturate_16bit(65500 + 100)  # 65600 should saturate to 65535
    underflow_result = neuron._saturate_16bit(-100)  # -100 should saturate to 0
    
    overflow_pass = overflow_result == 65535
    underflow_pass = underflow_result == 0
    
    print(f"    65500 + 100 = 65600 -> saturate to 65535: Actual={overflow_result} [{'PASS' if overflow_pass else 'FAIL'}]")
    print(f"    -100 -> saturate to 0: Actual={underflow_result} [{'PASS' if underflow_pass else 'FAIL'}]")
    
    all_passed = all_passed and overflow_pass and underflow_pass
    
    print(f"\nTest 2 Result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_spike_generation():
    """
    Test that spike generation matches Verilog RTL.
    
    Verilog:
        spike_condition = (v_mem_next >= threshold) && (refrac_counter == 0)
    """
    print("\n" + "=" * 60)
    print("Test 3: Spike Generation")
    print("=" * 60)
    
    test_cases = [
        # (v_mem, threshold, syn_weight, expected_spike)
        (900, 1000, 150, True),    # Should spike (900 + 150 >= 1000)
        (900, 1000, 50, False),    # Should not spike (900 + 50 < 1000)
        (999, 1000, 1, True),      # Exactly at threshold
        (0, 1000, 1000, True),     # From zero to threshold
        (500, 1000, 400, False),   # Below threshold
    ]
    
    all_passed = True
    
    for v_mem, threshold, syn_weight, expected_spike in test_cases:
        params = LIFNeuronParams(
            threshold=threshold,
            leak_rate=0,
            refractory_period=5,
        )
        neuron = HWAccurateLIFNeuron(neuron_id=0, params=params)
        neuron.state.v_mem = v_mem
        
        # Tick with synaptic input
        actual_spike = neuron.tick(syn_valid=True, syn_weight=syn_weight,
                                   syn_excitatory=True, enable=True)
        
        passed = actual_spike == expected_spike
        all_passed = all_passed and passed
        
        # Check membrane reset after spike
        if expected_spike:
            mem_reset_ok = neuron.state.v_mem == 0
            refrac_ok = neuron.state.refrac_counter == params.refractory_period
            passed = passed and mem_reset_ok and refrac_ok
        
        status = "PASS" if passed else "FAIL"
        print(f"  v_mem={v_mem:4d}, thresh={threshold:4d}, weight={syn_weight:3d}")
        print(f"    Expected spike: {expected_spike}, Actual: {actual_spike} [{status}]")
    
    print(f"\nTest 3 Result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_refractory_period():
    """
    Test that refractory period matches Verilog RTL.
    
    During refractory:
        - refrac_counter decrements each cycle
        - v_mem held at reset_potential
        - No spikes generated
    """
    print("\n" + "=" * 60)
    print("Test 4: Refractory Period")
    print("=" * 60)
    
    params = LIFNeuronParams(
        threshold=100,
        leak_rate=3,
        refractory_period=5,
    )
    neuron = HWAccurateLIFNeuron(neuron_id=0, params=params)
    
    # Trigger a spike
    neuron.state.v_mem = 50
    spike = neuron.tick(syn_valid=True, syn_weight=100, syn_excitatory=True, enable=True)
    
    print(f"  Initial spike triggered: {spike}")
    print(f"  Refractory counter: {neuron.state.refrac_counter}")
    
    all_passed = spike == True
    all_passed = all_passed and (neuron.state.refrac_counter == 5)
    
    # Tick through refractory period
    for i in range(5):
        old_refrac = neuron.state.refrac_counter
        spike = neuron.tick(syn_valid=True, syn_weight=200, syn_excitatory=True, enable=True)
        
        # Should not spike during refractory
        passed = (spike == False) and (neuron.state.v_mem == 0)
        passed = passed and (neuron.state.refrac_counter == old_refrac - 1)
        all_passed = all_passed and passed
        
        print(f"  Tick {i+1}: spike={spike}, refrac={neuron.state.refrac_counter}, v_mem={neuron.state.v_mem}")
    
    # Now should be able to spike again
    spike = neuron.tick(syn_valid=True, syn_weight=200, syn_excitatory=True, enable=True)
    print(f"  After refractory: spike={spike} (expected True)")
    all_passed = all_passed and (spike == True)
    
    print(f"\nTest 4 Result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_tau_conversion():
    """
    Test tau to leak_rate conversion and back.
    """
    print("\n" + "=" * 60)
    print("Test 5: Tau Conversion")
    print("=" * 60)
    
    test_taus = [0.5, 0.75, 0.8, 0.85, 0.875, 0.9, 0.92, 0.9375, 0.95, 0.96]
    
    all_passed = True
    
    for target_tau in test_taus:
        leak_rate = tau_to_hw_leak_rate(target_tau)
        actual_tau = hw_leak_rate_to_tau(leak_rate)
        
        shift1 = leak_rate & 0x07
        shift2_cfg = (leak_rate >> 3) & 0x1F
        shift2 = shift2_cfg & 0x07 if shift2_cfg != 0 else 0
        
        error = abs(actual_tau - target_tau)
        passed = error < 0.1  # Allow 10% error for approximation
        all_passed = all_passed and passed
        
        status = "PASS" if passed else "FAIL"
        print(f"  Target tau={target_tau:.4f} -> leak_rate={leak_rate:3d} (shift1={shift1}, shift2={shift2})")
        print(f"    Actual tau={actual_tau:.4f}, error={error:.4f} [{status}]")
    
    print(f"\nTest 5 Result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_pytorch_hw_mode():
    """
    Test that PyTorch LIF in hw_mode matches HWAccurateLIFNeuron.
    
    NOTE: There is a fundamental difference in update equations:
    - Verilog RTL: v_mem = syn_valid ? (v_mem + weight) : (v_mem - leak)
    - PyTorch SNN: v_mem = tau * v_mem + input (leak and input applied together)
    
    This test verifies the shift-based leak calculation is identical.
    For exact HW matching in inference, use HWAccurateLIFNeuron.
    """
    print("\n" + "=" * 60)
    print("Test 6: PyTorch hw_mode Leak Calculation")
    print("=" * 60)
    
    # Test configuration
    tau = 0.875  # shift=3
    threshold = 1000
    
    # Create PyTorch neuron in hw_mode
    lif_torch = LIF(thresh=float(threshold), tau=tau, hw_mode=True, reset='zero')
    
    # Create HW-accurate neuron
    params = LIFNeuronParams.from_tau(tau, threshold=threshold, refractory_period=0)
    lif_hw = HWAccurateLIFNeuron(neuron_id=0, params=params)
    
    print(f"  PyTorch hw_config: {lif_torch.get_hw_config()}")
    print(f"  HW tau: {lif_hw.params.tau:.4f}")
    
    # Test leak-only sequence (no synaptic input, just decay)
    print("\n  Testing leak-only (decay) sequence:")
    
    # Set same initial membrane potential
    initial_mem = 1000
    lif_torch.mem = torch.tensor([[float(initial_mem)]])
    lif_hw.state.v_mem = initial_mem
    
    all_passed = True
    
    for t in range(10):
        # PyTorch: apply leak with zero input
        x_torch = torch.tensor([[0.0]])
        _ = lif_torch(x_torch)
        mem_torch = int(lif_torch.mem.item())
        
        # HW: apply leak (no synaptic input)
        _ = lif_hw.tick(syn_valid=False, enable=True)
        mem_hw = lif_hw.state.v_mem
        
        # Compare
        mem_diff = abs(mem_torch - mem_hw)
        passed = mem_diff == 0
        all_passed = all_passed and passed
        
        status = "PASS" if passed else "FAIL"
        print(f"    t={t}: PyTorch mem={mem_torch:4d}, HW mem={mem_hw:4d}, diff={mem_diff} [{status}]")
    
    # Test that leak formula is identical
    print("\n  Testing leak formula for various membrane values:")
    
    test_mems = [100, 256, 500, 1000, 2000, 10000, 50000, 65535]
    
    for initial_mem in test_mems:
        # Reset
        lif_torch.reset_state()
        lif_torch.mem = torch.tensor([[float(initial_mem)]])
        lif_hw.reset()
        lif_hw.state.v_mem = initial_mem
        
        # Apply one leak step
        _ = lif_torch(torch.tensor([[0.0]]))
        mem_torch = int(lif_torch.mem.item())
        
        _ = lif_hw.tick(syn_valid=False, enable=True)
        mem_hw = lif_hw.state.v_mem
        
        passed = abs(mem_torch - mem_hw) == 0
        all_passed = all_passed and passed
        
        expected_leak = initial_mem >> 3  # shift=3
        expected_next = initial_mem - expected_leak
        
        status = "PASS" if passed else "FAIL"
        print(f"    initial={initial_mem:5d} -> PyTorch={mem_torch:5d}, HW={mem_hw:5d}, expected={expected_next:5d} [{status}]")
    
    print(f"\nTest 6 Result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def main():
    """Run all tests."""
    print("=" * 60)
    print("Hardware-Python Identity Verification")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Shift-Based Leak", test_shift_based_leak()))
    results.append(("Synaptic Integration", test_synaptic_integration()))
    results.append(("Spike Generation", test_spike_generation()))
    results.append(("Refractory Period", test_refractory_period()))
    results.append(("Tau Conversion", test_tau_conversion()))
    results.append(("PyTorch hw_mode", test_pytorch_hw_mode()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    print()
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
