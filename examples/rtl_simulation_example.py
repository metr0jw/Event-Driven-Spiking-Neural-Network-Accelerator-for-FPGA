#!/usr/bin/env python3
"""
RTL Simulation Example

Demonstrates how to simulate the SNN accelerator Verilog RTL from Python
using Icarus Verilog and Cocotb.

Requirements:
    - Icarus Verilog: apt install iverilog
    - Cocotb (optional): pip install cocotb
    - Verilator (optional): apt install verilator

Author: Jiwoon Lee (@metr0jw)
"""

import sys
import os
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "software" / "python"))

from snn_fpga_accelerator import (
    SNNAccelerator,
    SNNModel,
    SNNLayer,
    RateEncoder,
    check_simulation_tools,
    print_tool_status,
    ICARUS_AVAILABLE,
)

# Import conditionally
if ICARUS_AVAILABLE:
    from snn_fpga_accelerator import (
        IcarusSimulator,
        RTLvsPythonComparator,
    )


def check_tools():
    """Check available simulation tools."""
    print("\n" + "=" * 60)
    print("RTL Simulation Tools Check")
    print("=" * 60)
    
    print_tool_status()
    
    tools = check_simulation_tools()
    if not any(tools.values()):
        print("\nNote: To run RTL simulations, install at least one simulator:")
        print("  Icarus Verilog (recommended):")
        print("    Ubuntu/Debian: sudo apt install iverilog")
        print("    MacOS: brew install icarus-verilog")
        print("")
        print("  Cocotb (for Python testbenches):")
        print("    pip install cocotb")
        print("")
        print("  Verilator (for high-performance simulation):")
        print("    Ubuntu/Debian: sudo apt install verilator")
        return False
    return True


def demo_python_only_simulation():
    """
    Demonstrate Python-only behavioral simulation.
    This works without any RTL tools installed.
    """
    print("\n" + "=" * 60)
    print("Python Behavioral Simulation (Software-Only)")
    print("=" * 60)
    
    # Create accelerator in software mode
    accel = SNNAccelerator(simulation_mode=True)
    print(f"Backend: Software Simulation")
    print(f"Simulation mode: {accel.simulation_mode}")
    
    # Create an SNN model
    model = SNNModel()
    model.add_layer(SNNLayer("input", 784, "input"))
    model.add_layer(SNNLayer("hidden", 100, "lif", threshold=1.0, decay=0.9))
    model.add_layer(SNNLayer("output", 10, "lif", threshold=0.8, decay=0.95))
    
    # Configure network with the model
    accel.configure_network(model)
    
    # Encode input
    import numpy as np
    np.random.seed(42)
    
    encoder = RateEncoder(num_neurons=784, duration=0.01, max_rate=100)
    test_input = np.random.rand(784)
    spikes = encoder.encode(test_input)
    
    print(f"\nInput encoding:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Encoded spikes: {len(spikes)}")
    
    # Run simulation using infer method
    output = accel.infer(spikes, duration=0.01, return_events=True)
    print(f"\nSimulation output:")
    print(f"  Output spikes: {len(output)}")
    
    # Get timing info
    print(f"\nTiming (software simulation):")
    print(f"  Note: This is behavioral simulation, not cycle-accurate")
    print(f"  For cycle-accurate timing, use RTL simulation")


def demo_precompiled_rtl_tests():
    """
    Run pre-compiled RTL test binaries.
    These are module-level tests for individual components.
    """
    if not ICARUS_AVAILABLE:
        print("\nSkipping RTL tests (iverilog not installed)")
        return
    
    print("\n" + "=" * 60)
    print("Pre-compiled RTL Module Tests")
    print("=" * 60)
    
    # Initialize simulator
    project_root = Path(__file__).parent.parent
    rtl_dir = project_root / "hardware" / "hdl" / "rtl"
    
    simulator = IcarusSimulator(rtl_dir=str(rtl_dir))
    
    # List available tests
    tests = simulator.list_precompiled_tests()
    print(f"\nAvailable pre-compiled tests: {len(tests)}")
    for t in tests:
        print(f"  - {t}")
    
    # Run all tests
    print("\nRunning all RTL tests...")
    print("-" * 60)
    
    results = simulator.run_all_tests(verbose=True)
    
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Tests passed: {results['passed_tests']}/{results['total_tests']}")
    print(f"  Assertions: {results['total_pass_assertions']} pass, {results['total_fail_assertions']} fail")
    print(f"  All passed: {'YES' if results['all_passed'] else 'NO'}")


def demo_individual_rtl_test():
    """
    Run a single RTL test and examine output.
    """
    if not ICARUS_AVAILABLE:
        print("\nSkipping individual RTL test (iverilog not installed)")
        return
    
    print("\n" + "=" * 60)
    print("Individual RTL Test: LIF Neuron")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    rtl_dir = project_root / "hardware" / "hdl" / "rtl"
    
    simulator = IcarusSimulator(rtl_dir=str(rtl_dir))
    
    # Run LIF neuron test
    result = simulator.run_precompiled_test('lif_neuron')
    
    print(f"\nTest: lif_neuron")
    print(f"Success: {result['success']}")
    print(f"Passes: {result['passes']}")
    print(f"Fails: {result['fails']}")
    
    print("\nTest Output (excerpt):")
    print("-" * 40)
    # Print first 30 lines of output
    lines = result['stdout'].split('\n')[:30]
    for line in lines:
        print(line)
    if len(result['stdout'].split('\n')) > 30:
        print("... (truncated)")


def demo_cocotb_setup():
    """
    Show how to set up Cocotb for interactive RTL testing.
    """
    print("\n" + "=" * 60)
    print("Cocotb Test Setup (for Advanced RTL Testing)")
    print("=" * 60)
    
    try:
        from snn_fpga_accelerator.rtl_simulator import CocotbSimulator, COCOTB_AVAILABLE
        
        if not COCOTB_AVAILABLE:
            print("\nCocotb not installed. Install with: pip install cocotb")
            print("\nCocotb allows writing Python testbenches that directly")
            print("interact with Verilog signals during simulation.")
            return
        
        # Create Cocotb setup
        project_root = Path(__file__).parent.parent
        rtl_dir = project_root / "hardware" / "hdl" / "rtl"
        
        if not rtl_dir.exists():
            print(f"RTL directory not found: {rtl_dir}")
            return
        
        cocotb_sim = CocotbSimulator(
            rtl_dir=str(rtl_dir),
            top_module="snn_accelerator_top"
        )
        
        # Generate files
        output_dir = project_root / "outputs" / "cocotb_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        makefile = cocotb_sim.generate_makefile(str(output_dir))
        test_file = cocotb_sim.generate_test_template(str(output_dir))
        
        print(f"\nGenerated Cocotb test files:")
        print(f"  Makefile: {makefile}")
        print(f"  Test file: {test_file}")
        
        print(f"\nTo run cocotb tests:")
        print(f"  cd {output_dir}")
        print(f"  make")
        
        print(f"\nCocotb test template includes:")
        print(f"  - test_basic_spike: Single spike propagation test")
        print(f"  - test_spike_train: Multiple spike processing test")
        print(f"  - test_membrane_dynamics: LIF neuron membrane test")
        
    except Exception as e:
        print(f"Cocotb setup error: {e}")


def main():
    """Main example runner."""
    print("\n" + "#" * 60)
    print("# SNN Accelerator RTL Simulation Examples")
    print("#" * 60)
    
    # Check available tools
    if not check_tools():
        print("\n(Running Python-only simulation)")
    
    # Always available: Python behavioral simulation
    demo_python_only_simulation()
    
    # Pre-compiled RTL tests (requires Icarus Verilog)
    demo_precompiled_rtl_tests()
    
    # Individual RTL test with detailed output
    demo_individual_rtl_test()
    
    # Cocotb setup guide
    demo_cocotb_setup()
    
    print("\n" + "#" * 60)
    print("# Examples Complete")
    print("#" * 60)
    print("\nSummary:")
    print("  - Python behavioral simulation: Always available")
    print("  - Pre-compiled RTL tests: Requires Icarus Verilog (iverilog)")
    print("  - Cocotb tests: Requires cocotb + iverilog/verilator")
    print("  - Hardware execution: Requires PYNQ board + bitstream")


if __name__ == "__main__":
    main()
