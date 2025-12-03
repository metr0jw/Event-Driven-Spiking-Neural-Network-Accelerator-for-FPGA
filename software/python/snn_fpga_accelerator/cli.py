"""
Command Line Interface for SNN FPGA Accelerator

Provides convenient commands for common operations like flashing bitstreams,
running tests, and managing models.

Author: Jiwoon Lee (@metr0jw)
Organization: Kwangwoon University, Seoul, South Korea
Contact: jwlee@linux.com
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
import json

from .accelerator import SNNAccelerator
from .pytorch_interface import pytorch_to_snn, load_pytorch_weights
from .utils import logger, setup_logging


def flash_bitstream():
    """Flash bitstream to FPGA."""
    parser = argparse.ArgumentParser(description='Flash bitstream to FPGA')
    parser.add_argument('--bitstream', '-b', type=str, required=True,
                        help='Path to bitstream file (.bit)')
    parser.add_argument('--ip', type=str, default='192.168.2.99',
                        help='PYNQ board IP address')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        setup_logging('DEBUG')
    
    bitstream_path = Path(args.bitstream)
    if not bitstream_path.exists():
        logger.error(f"Bitstream file not found: {bitstream_path}")
        return 1
    
    try:
        accelerator = SNNAccelerator(bitstream_path=str(bitstream_path), 
                                   fpga_ip=args.ip)
        accelerator.load_bitstream()
        logger.info("Bitstream flashed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Failed to flash bitstream: {e}")
        return 1


def run_tests():
    """Run test suite."""
    parser = argparse.ArgumentParser(description='Run SNN accelerator tests')
    parser.add_argument('--hardware', action='store_true',
                        help='Run hardware tests (requires FPGA)')
    parser.add_argument('--software', action='store_true', default=True,
                        help='Run software tests')
    parser.add_argument('--pattern', '-p', type=str, default='test_*.py',
                        help='Test file pattern')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        setup_logging('DEBUG')
    
    test_commands = []
    
    if args.software:
        # Run software tests
        test_commands.append(['python', '-m', 'pytest', 
                            'tests/software/', '-v'])
    
    if args.hardware:
        # Run hardware tests
        test_commands.append(['python', '-m', 'pytest', 
                            'tests/hardware/', '-v'])
    
    all_passed = True
    
    for cmd in test_commands:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Tests passed!")
            print(result.stdout)
        else:
            logger.error("Tests failed!")
            print(result.stderr)
            all_passed = False
    
    return 0 if all_passed else 1


def convert_model():
    """Convert PyTorch model to SNN format."""
    parser = argparse.ArgumentParser(description='Convert PyTorch model to SNN')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input PyTorch model file (.pth)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output SNN weights file (.h5)')
    parser.add_argument('--input-shape', type=str, default='784',
                        help='Input shape (comma-separated)')
    parser.add_argument('--config', '-c', type=str,
                        help='Conversion config file (.json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        setup_logging('DEBUG')
    
    # Parse input shape
    try:
        input_shape = tuple(map(int, args.input_shape.split(',')))
    except ValueError:
        logger.error("Invalid input shape format. Use comma-separated integers.")
        return 1
    
    # Load conversion config
    conversion_params = {
        'weight_scale': 128.0,
        'threshold_scale': 1.0,
        'leak_rate': 0.1,
        'refractory_period': 5
    }
    
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                conversion_params.update(config_data.get('conversion_params', {}))
    
    # Set output path
    output_path = args.output
    if not output_path:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_snn_weights.h5"
    
    try:
        # Load PyTorch model
        logger.info(f"Loading PyTorch model from {args.input}")
        torch_model = load_pytorch_weights(args.input)
        
        # Convert to SNN
        logger.info("Converting to SNN format...")
        snn_model = pytorch_to_snn(torch_model, input_shape, conversion_params)
        
        # Save SNN weights
        logger.info(f"Saving SNN weights to {output_path}")
        snn_model.save_weights(output_path)
        
        logger.info("Conversion completed successfully!")
        logger.info(f"SNN model: {snn_model.total_neurons} neurons, "
                   f"{len(snn_model.layers)} layers")
        
        return 0
    
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1


def benchmark():
    """Run performance benchmarks."""
    parser = argparse.ArgumentParser(description='Run performance benchmarks')
    parser.add_argument('--hardware', action='store_true',
                        help='Benchmark hardware performance')
    parser.add_argument('--software', action='store_true', default=True,
                        help='Benchmark software performance')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Benchmark duration in seconds')
    parser.add_argument('--spike-rate', type=float, default=1000.0,
                        help='Input spike rate (spikes/second)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output benchmark results file')
    
    args = parser.parse_args()
    
    results = {}
    
    if args.software:
        logger.info("Running software benchmark...")
        # Implement software benchmark
        results['software'] = run_software_benchmark(args.duration, args.spike_rate)
    
    if args.hardware:
        logger.info("Running hardware benchmark...")
        # Implement hardware benchmark
        try:
            results['hardware'] = run_hardware_benchmark(args.duration, args.spike_rate)
        except Exception as e:
            logger.error(f"Hardware benchmark failed: {e}")
            results['hardware'] = {'error': str(e)}
    
    # Print results
    print("\n=== Benchmark Results ===")
    for platform, data in results.items():
        print(f"\n{platform.upper()}:")
        if 'error' in data:
            print(f"  Error: {data['error']}")
        else:
            for metric, value in data.items():
                print(f"  {metric}: {value}")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Benchmark results saved to {args.output}")
    
    return 0


def run_software_benchmark(duration, spike_rate):
    """Run software performance benchmark."""
    import time
    import numpy as np
    from .spike_encoding import PoissonEncoder
    from .pytorch_interface import simulate_snn_inference, create_feedforward_snn
    
    # Create test model
    snn_model = create_feedforward_snn([100, 50, 10])
    encoder = PoissonEncoder(100, 0.1, spike_rate)
    
    # Generate test data
    test_data = np.random.rand(100)
    
    start_time = time.time()
    iterations = 0
    
    while time.time() - start_time < duration:
        # Encode input
        input_spikes = encoder.encode(test_data)
        
        # Run simulation
        output_spikes = simulate_snn_inference(snn_model, input_spikes)
        
        iterations += 1
    
    elapsed = time.time() - start_time
    
    return {
        'iterations': iterations,
        'iterations_per_second': iterations / elapsed,
        'elapsed_time': elapsed,
        'platform': 'software'
    }


def run_hardware_benchmark(duration, spike_rate):
    """Run hardware performance benchmark."""
    import time
    import numpy as np
    from .spike_encoding import PoissonEncoder
    
    # Initialize accelerator
    accelerator = SNNAccelerator()
    accelerator.load_bitstream()
    
    # Configure test network
    accelerator.configure_network(100, {})
    
    encoder = PoissonEncoder(100, 0.1, spike_rate)
    
    start_time = time.time()
    iterations = 0
    total_input_spikes = 0
    total_output_spikes = 0
    
    while time.time() - start_time < duration:
        # Generate test data
        test_data = np.random.rand(100)
        input_spikes = encoder.encode(test_data)
        
        # Run on FPGA
        output_spikes = accelerator.run_simulation(0.1, input_spikes)
        
        total_input_spikes += len(input_spikes)
        total_output_spikes += len(output_spikes)
        iterations += 1
    
    elapsed = time.time() - start_time
    
    return {
        'iterations': iterations,
        'iterations_per_second': iterations / elapsed,
        'input_spikes': total_input_spikes,
        'output_spikes': total_output_spikes,
        'input_spike_rate': total_input_spikes / elapsed,
        'output_spike_rate': total_output_spikes / elapsed,
        'elapsed_time': elapsed,
        'platform': 'hardware'
    }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='SNN FPGA Accelerator CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  flash      Flash bitstream to FPGA
  test       Run test suite
  convert    Convert PyTorch model to SNN
  benchmark  Run performance benchmarks

Use 'snn-cli <command> --help' for command-specific help.
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Flash command
    flash_parser = subparsers.add_parser('flash', help='Flash bitstream to FPGA')
    flash_parser.set_defaults(func=flash_bitstream)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run test suite')
    test_parser.set_defaults(func=run_tests)
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert PyTorch model')
    convert_parser.set_defaults(func=convert_model)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.set_defaults(func=benchmark)
    
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    
    # Override sys.argv for subcommand parsing
    sys.argv = ['snn-cli', args.command] + sys.argv[2:]
    
    return args.func()


if __name__ == '__main__':
    sys.exit(main())
