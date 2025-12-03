"""
RTL Simulation Backend for SNN Accelerator

Provides Python interfaces to simulate actual Verilog RTL code using:
1. Cocotb - Direct Python testbench for RTL simulation
2. Icarus Verilog - Open-source Verilog simulator
3. Verilator - High-performance cycle-accurate simulation

This allows comparing Python behavioral model with actual RTL implementation.

Author: Jiwoon Lee (@metr0jw)
"""

import os
import subprocess
import tempfile
import struct
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from .spike_encoding import SpikeEvent
from .utils import logger

# Check for simulation tool availability
ICARUS_AVAILABLE = False
VERILATOR_AVAILABLE = False
COCOTB_AVAILABLE = False

try:
    result = subprocess.run(['iverilog', '-V'], capture_output=True, timeout=5)
    ICARUS_AVAILABLE = result.returncode == 0
except (FileNotFoundError, subprocess.TimeoutExpired):
    pass

try:
    result = subprocess.run(['verilator', '--version'], capture_output=True, timeout=5)
    VERILATOR_AVAILABLE = result.returncode == 0
except (FileNotFoundError, subprocess.TimeoutExpired):
    pass

try:
    import cocotb
    COCOTB_AVAILABLE = True
except ImportError:
    pass


@dataclass
class RTLSimulationResult:
    """Results from RTL simulation."""
    output_spikes: List[SpikeEvent]
    simulation_time_ns: int
    cycle_count: int
    register_values: Dict[str, int]
    waveform_file: Optional[str] = None


class IcarusSimulator:
    """
    Icarus Verilog based RTL simulator.
    
    Compiles and runs Verilog RTL using iverilog/vvp for cycle-accurate
    simulation of the SNN accelerator hardware.
    """
    
    def __init__(
        self,
        rtl_dir: Optional[str] = None,
        top_module: str = "snn_accelerator_top",
        include_dirs: Optional[List[str]] = None
    ):
        """
        Initialize Icarus simulator.
        
        Parameters
        ----------
        rtl_dir : str, optional
            Path to RTL source directory. Defaults to project hardware/hdl/rtl.
        top_module : str
            Name of the top-level module to simulate.
        include_dirs : list, optional
            Additional include directories for Verilog compilation.
        """
        if not ICARUS_AVAILABLE:
            raise RuntimeError(
                "Icarus Verilog not found. Install with: apt install iverilog"
            )
        
        # Find RTL directory
        if rtl_dir is None:
            # Try to find it relative to this file
            pkg_dir = Path(__file__).parent
            project_root = pkg_dir.parent.parent.parent.parent
            rtl_dir = project_root / "hardware" / "hdl" / "rtl"
            if not rtl_dir.exists():
                raise FileNotFoundError(f"RTL directory not found: {rtl_dir}")
        
        self.rtl_dir = Path(rtl_dir)
        self.top_module = top_module
        self.include_dirs = include_dirs or []
        self.compiled_binary: Optional[Path] = None
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        
        # Check for pre-compiled binaries
        self._precompiled_dir = self.rtl_dir.parent / "sim" / "test_results"
        
    def list_precompiled_tests(self) -> List[str]:
        """List available pre-compiled test binaries."""
        if not self._precompiled_dir.exists():
            return []
        return [f.stem for f in self._precompiled_dir.glob("*.vvp")]
    
    def run_all_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Run all pre-compiled RTL tests.
        
        Parameters
        ----------
        verbose : bool
            Print detailed test output.
            
        Returns
        -------
        dict
            Summary of all test results.
        """
        tests = self.list_precompiled_tests()
        results = []
        total_pass = 0
        total_fail = 0
        
        for test in tests:
            try:
                result = self.run_precompiled_test(test)
                results.append({
                    'name': test,
                    'success': result['success'],
                    'passes': result['passes'],
                    'fails': result['fails']
                })
                total_pass += result['passes']
                total_fail += result['fails']
                
                if verbose:
                    status = "PASS" if result['success'] else "FAIL"
                    print(f"{test:20s} : {status} (pass={result['passes']}, fail={result['fails']})")
                    
            except Exception as e:
                results.append({
                    'name': test,
                    'success': False,
                    'error': str(e)
                })
                if verbose:
                    print(f"{test:20s} : ERROR - {e}")
        
        return {
            'tests': results,
            'total_tests': len(tests),
            'passed_tests': sum(1 for r in results if r.get('success', False)),
            'total_pass_assertions': total_pass,
            'total_fail_assertions': total_fail,
            'all_passed': all(r.get('success', False) for r in results)
        }
    
    def run_precompiled_test(self, test_name: str) -> Dict[str, Any]:
        """
        Run a pre-compiled test binary.
        
        Parameters
        ----------
        test_name : str
            Name of the test (e.g., 'lif_neuron', 'spike_router').
            
        Returns
        -------
        dict
            Test results including stdout output.
        """
        binary_path = self._precompiled_dir / f"{test_name}.vvp"
        if not binary_path.exists():
            available = self.list_precompiled_tests()
            raise FileNotFoundError(
                f"Test '{test_name}' not found. Available: {available}"
            )
        
        # Use absolute path and run from sim directory (parent of test_results)
        sim_dir = self._precompiled_dir.parent
        result = subprocess.run(
            ['vvp', str(binary_path.resolve())],
            capture_output=True,
            text=True,
            cwd=str(sim_dir),
            timeout=30
        )
        
        # Parse output for pass/fail counts
        output = result.stdout
        passes = output.upper().count('PASS')
        fails = output.upper().count('FAIL')
        
        return {
            'test_name': test_name,
            'success': result.returncode == 0 and fails == 0,
            'passes': passes,
            'fails': fails,
            'stdout': output,
            'stderr': result.stderr
        }
        
    def _collect_rtl_files(self) -> List[Path]:
        """Collect all Verilog source files."""
        rtl_files = []
        for subdir in ['neurons', 'synapses', 'layers', 'router', 'common', 'interfaces', 'top']:
            subpath = self.rtl_dir / subdir
            if subpath.exists():
                rtl_files.extend(subpath.glob('*.v'))
        return rtl_files
    
    def compile(self, output_path: Optional[str] = None) -> str:
        """
        Compile Verilog RTL to simulation binary.
        
        Parameters
        ----------
        output_path : str, optional
            Path for compiled binary. Uses temp file if not specified.
            
        Returns
        -------
        str
            Path to compiled binary.
        """
        rtl_files = self._collect_rtl_files()
        if not rtl_files:
            raise FileNotFoundError("No Verilog files found in RTL directory")
        
        logger.info(f"Compiling {len(rtl_files)} Verilog files...")
        
        # Create temp directory if needed
        if output_path is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            output_path = Path(self._temp_dir.name) / "sim.vvp"
        else:
            output_path = Path(output_path)
        
        # Build iverilog command
        cmd = ['iverilog', '-g2012', '-o', str(output_path)]
        
        # Add include directories
        for inc_dir in self.include_dirs:
            cmd.extend(['-I', str(inc_dir)])
        cmd.extend(['-I', str(self.rtl_dir)])
        
        # Add source files
        cmd.extend([str(f) for f in rtl_files])
        
        # Run compilation
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Compilation failed:\n{result.stderr}")
            raise RuntimeError(f"iverilog compilation failed: {result.stderr}")
        
        self.compiled_binary = output_path
        logger.info(f"Compilation successful: {output_path}")
        return str(output_path)
    
    def run(
        self,
        input_spikes: List[SpikeEvent],
        duration_ns: int = 100000,
        dump_vcd: bool = False
    ) -> RTLSimulationResult:
        """
        Run RTL simulation with given input spikes.
        
        Parameters
        ----------
        input_spikes : list
            Input spike events to inject.
        duration_ns : int
            Simulation duration in nanoseconds.
        dump_vcd : bool
            Whether to dump VCD waveform file.
            
        Returns
        -------
        RTLSimulationResult
            Simulation results including output spikes.
        """
        if self.compiled_binary is None:
            self.compile()
        
        # Create stimulus file
        stim_file = self._create_stimulus_file(input_spikes)
        
        # Run vvp
        cmd = ['vvp', str(self.compiled_binary)]
        env = os.environ.copy()
        env['SIM_DURATION'] = str(duration_ns)
        env['STIMULUS_FILE'] = str(stim_file)
        
        if dump_vcd:
            vcd_path = Path(self._temp_dir.name) / "waves.vcd" if self._temp_dir else Path("waves.vcd")
            env['VCD_FILE'] = str(vcd_path)
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
        
        # Parse output
        output_spikes = self._parse_output(result.stdout)
        
        return RTLSimulationResult(
            output_spikes=output_spikes,
            simulation_time_ns=duration_ns,
            cycle_count=duration_ns // 10,  # Assuming 100MHz clock
            register_values={},
            waveform_file=str(vcd_path) if dump_vcd else None
        )
    
    def _create_stimulus_file(self, spikes: List[SpikeEvent]) -> Path:
        """Create stimulus file for testbench."""
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory()
        
        stim_path = Path(self._temp_dir.name) / "stimulus.txt"
        
        with open(stim_path, 'w') as f:
            for spike in spikes:
                # Format: timestamp_ns neuron_id weight
                ts_ns = int(spike.timestamp * 1e9)
                f.write(f"{ts_ns} {spike.neuron_id} {int(spike.weight * 128)}\n")
        
        return stim_path
    
    def _parse_output(self, stdout: str) -> List[SpikeEvent]:
        """Parse simulation output for spike events."""
        spikes = []
        for line in stdout.split('\n'):
            if line.startswith('SPIKE:'):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        ts_ns = int(parts[1])
                        neuron_id = int(parts[2])
                        spikes.append(SpikeEvent(
                            neuron_id=neuron_id,
                            timestamp=ts_ns * 1e-9,
                            weight=1.0
                        ))
                    except ValueError:
                        pass
        return spikes
    
    def cleanup(self):
        """Clean up temporary files."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None


class CocotbSimulator:
    """
    Cocotb-based RTL simulator for interactive Python testing.
    
    Provides direct Python control over RTL simulation using cocotb,
    allowing fine-grained stimulus generation and response checking.
    """
    
    def __init__(
        self,
        rtl_dir: Optional[str] = None,
        top_module: str = "snn_accelerator_top",
        simulator: str = "icarus"
    ):
        """
        Initialize Cocotb simulator.
        
        Parameters
        ----------
        rtl_dir : str, optional
            Path to RTL source directory.
        top_module : str
            Top-level module name.
        simulator : str
            Underlying simulator ('icarus', 'verilator', 'questa').
        """
        if not COCOTB_AVAILABLE:
            raise RuntimeError(
                "cocotb not found. Install with: pip install cocotb"
            )
        
        if rtl_dir is None:
            pkg_dir = Path(__file__).parent
            project_root = pkg_dir.parent.parent.parent.parent
            rtl_dir = project_root / "hardware" / "hdl" / "rtl"
        
        self.rtl_dir = Path(rtl_dir)
        self.top_module = top_module
        self.simulator = simulator
        
    def generate_makefile(self, output_dir: str) -> str:
        """
        Generate Makefile for cocotb simulation.
        
        Parameters
        ----------
        output_dir : str
            Directory to write Makefile.
            
        Returns
        -------
        str
            Path to generated Makefile.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Collect RTL files
        rtl_files = []
        for subdir in ['neurons', 'synapses', 'layers', 'router', 'common', 'interfaces', 'top']:
            subpath = self.rtl_dir / subdir
            if subpath.exists():
                rtl_files.extend([str(f) for f in subpath.glob('*.v')])
        
        makefile_content = f"""
# Cocotb Makefile for SNN Accelerator RTL Simulation
# Auto-generated by rtl_simulator.py

SIM ?= {self.simulator}
TOPLEVEL_LANG ?= verilog
TOPLEVEL = {self.top_module}
MODULE = test_snn_rtl

VERILOG_SOURCES = \\
    {' '.join(rtl_files)}

EXTRA_ARGS += -I{self.rtl_dir}

include $(shell cocotb-config --makefiles)/Makefile.sim
"""
        
        makefile_path = output_path / "Makefile"
        with open(makefile_path, 'w') as f:
            f.write(makefile_content)
        
        return str(makefile_path)
    
    def generate_test_template(self, output_dir: str) -> str:
        """
        Generate Python test template for cocotb.
        
        Parameters
        ----------
        output_dir : str
            Directory to write test file.
            
        Returns
        -------
        str
            Path to generated test file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        test_content = '''"""
Cocotb Test for SNN Accelerator RTL
Auto-generated by rtl_simulator.py
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ClockCycles
from cocotb.result import TestFailure
import numpy as np


async def reset_dut(dut, cycles=10):
    """Reset the DUT."""
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, cycles)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)


async def send_spike(dut, neuron_id, weight=1):
    """Send a spike to the accelerator."""
    dut.spike_in_valid.value = 1
    dut.spike_in_neuron_id.value = neuron_id
    dut.spike_in_weight.value = int(weight * 128)
    await RisingEdge(dut.clk)
    dut.spike_in_valid.value = 0


async def wait_for_output_spike(dut, timeout_cycles=1000):
    """Wait for output spike."""
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if dut.spike_out_valid.value == 1:
            return {
                'neuron_id': int(dut.spike_out_neuron_id.value),
                'timestamp': cocotb.utils.get_sim_time(units='ns')
            }
    return None


@cocotb.test()
async def test_basic_spike(dut):
    """Test basic spike propagation through the network."""
    # Start clock
    clock = Clock(dut.clk, 10, units='ns')  # 100MHz
    cocotb.start_soon(clock.start())
    
    # Reset
    await reset_dut(dut)
    
    # Send input spike
    await send_spike(dut, neuron_id=0, weight=1.0)
    
    # Wait for processing
    await ClockCycles(dut.clk, 100)
    
    dut._log.info("Basic spike test completed")


@cocotb.test()
async def test_spike_train(dut):
    """Test spike train processing."""
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Send multiple spikes
    for i in range(10):
        await send_spike(dut, neuron_id=i % 64, weight=0.5)
        await ClockCycles(dut.clk, 10)
    
    # Wait for output
    await ClockCycles(dut.clk, 500)
    
    dut._log.info("Spike train test completed")


@cocotb.test()
async def test_membrane_dynamics(dut):
    """Test LIF neuron membrane potential dynamics."""
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Send sub-threshold spikes and verify accumulation
    for _ in range(5):
        await send_spike(dut, neuron_id=0, weight=0.2)
        await ClockCycles(dut.clk, 5)
    
    await ClockCycles(dut.clk, 100)
    
    dut._log.info("Membrane dynamics test completed")
'''
        
        test_path = output_path / "test_snn_rtl.py"
        with open(test_path, 'w') as f:
            f.write(test_content)
        
        return str(test_path)


class RTLvsPythonComparator:
    """
    Compare RTL simulation results with Python behavioral model.
    
    Useful for validating that the hardware implementation matches
    the software model behavior.
    """
    
    def __init__(
        self,
        rtl_simulator: Optional[IcarusSimulator] = None,
        python_model: Optional[Any] = None
    ):
        """
        Initialize comparator.
        
        Parameters
        ----------
        rtl_simulator : IcarusSimulator, optional
            RTL simulator instance.
        python_model : SNNModel, optional
            Python behavioral model for comparison.
        """
        self.rtl_sim = rtl_simulator
        self.python_model = python_model
        
    def compare(
        self,
        input_spikes: List[SpikeEvent],
        duration: float = 0.001
    ) -> Dict[str, Any]:
        """
        Compare RTL and Python model outputs.
        
        Parameters
        ----------
        input_spikes : list
            Input spikes to both simulators.
        duration : float
            Simulation duration in seconds.
            
        Returns
        -------
        dict
            Comparison results.
        """
        results = {
            'input_spike_count': len(input_spikes),
            'duration': duration,
        }
        
        # Run RTL simulation
        if self.rtl_sim is not None:
            try:
                rtl_result = self.rtl_sim.run(
                    input_spikes,
                    duration_ns=int(duration * 1e9)
                )
                results['rtl_output_spikes'] = len(rtl_result.output_spikes)
                results['rtl_cycle_count'] = rtl_result.cycle_count
            except Exception as e:
                results['rtl_error'] = str(e)
                results['rtl_output_spikes'] = None
        
        # Run Python simulation
        if self.python_model is not None:
            try:
                from .pytorch_interface import simulate_snn_inference
                py_output = simulate_snn_inference(
                    self.python_model,
                    input_spikes,
                    duration
                )
                results['python_output_spikes'] = len(py_output)
            except Exception as e:
                results['python_error'] = str(e)
                results['python_output_spikes'] = None
        
        # Compare results
        if results.get('rtl_output_spikes') is not None and \
           results.get('python_output_spikes') is not None:
            results['spike_count_match'] = (
                results['rtl_output_spikes'] == results['python_output_spikes']
            )
            results['spike_count_diff'] = abs(
                results['rtl_output_spikes'] - results['python_output_spikes']
            )
        
        return results
    
    @staticmethod
    def print_report(results: Dict[str, Any]) -> None:
        """Print comparison report."""
        print("\n" + "=" * 60)
        print("RTL vs Python Model Comparison Report")
        print("=" * 60)
        
        print(f"\n[Input]")
        print(f"  Spike Count: {results.get('input_spike_count', 'N/A')}")
        print(f"  Duration: {results.get('duration', 'N/A')}s")
        
        print(f"\n[RTL Simulation]")
        if 'rtl_error' in results:
            print(f"  Error: {results['rtl_error']}")
        else:
            print(f"  Output Spikes: {results.get('rtl_output_spikes', 'N/A')}")
            print(f"  Cycle Count: {results.get('rtl_cycle_count', 'N/A')}")
        
        print(f"\n[Python Model]")
        if 'python_error' in results:
            print(f"  Error: {results['python_error']}")
        else:
            print(f"  Output Spikes: {results.get('python_output_spikes', 'N/A')}")
        
        print(f"\n[Comparison]")
        if results.get('spike_count_match') is not None:
            status = "MATCH" if results['spike_count_match'] else "MISMATCH"
            print(f"  Spike Count: {status}")
            print(f"  Difference: {results.get('spike_count_diff', 'N/A')}")
        else:
            print("  (Cannot compare - missing results)")
        
        print("\n" + "=" * 60)


def check_simulation_tools() -> Dict[str, bool]:
    """
    Check availability of simulation tools.
    
    Returns
    -------
    dict
        Dictionary with tool availability status.
    """
    return {
        'icarus_verilog': ICARUS_AVAILABLE,
        'verilator': VERILATOR_AVAILABLE,
        'cocotb': COCOTB_AVAILABLE,
    }


def print_tool_status():
    """Print simulation tool availability."""
    tools = check_simulation_tools()
    
    print("\nRTL Simulation Tool Status:")
    print("-" * 40)
    for tool, available in tools.items():
        status = "Available" if available else "Not Found"
        print(f"  {tool}: {status}")
    
    if not any(tools.values()):
        print("\nTo enable RTL simulation, install:")
        print("  - Icarus Verilog: apt install iverilog")
        print("  - Cocotb: pip install cocotb")
        print("  - Verilator: apt install verilator")
