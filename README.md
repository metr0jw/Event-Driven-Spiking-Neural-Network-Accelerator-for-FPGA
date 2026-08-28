# Event-Driven SNN Accelerator for FPGA

Energy-efficient, spike-triggered SNN accelerator on PYNQ-Z2 with a native Python/PyTorch-style software stack.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Current Status

- FPGA target: `Zynq-7020 (PYNQ-Z2)`
- Canonical PL clock: `80 MHz`
- Maintained workflow: native library-first
- Supported scenarios:
  1. GPU training -> FPGA inference
  2. FPGA on-chip STDP train + inference

## Quick Start

```bash
git clone https://github.com/metr0jw/Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA.git
cd Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA
./setup.sh
```

```python
import numpy as np
from snn_fpga_accelerator import SNNAccelerator

accel = SNNAccelerator(simulation_mode=True)
output = accel.forward(np.array([[0, 0.0, 1.0]], dtype=np.float32))
```

## Project Direction

This repository uses a native workflow built around `snn_fpga_accelerator`.

- Maintained path: native PyTorch-style training/export/runtime
- Removed from supported path: `SpikingJelly auto-conversion`

## Features

- Fixed-point LIF neuron and event-router hardware
- On-chip STDP and R-STDP support
- Native Python/PyTorch-style API
- Host/runtime tooling for export, parity, and board execution
- RTL + HLS co-design for inference and learning

## Public Documentation

- [Developer Guide](docs/developer_guide.md)
- [API Reference](docs/api_reference.md)
- [Architecture](docs/architecture.md)
- [User Guide](docs/user_guide.md)

## Examples

```bash
python examples/pytorch/mnist_training_example.py
python examples/pytorch/r_stdp_learning_example.py
python examples/pytorch/mozafari_rstdp_faithful.py
```

## Project Structure

```
hardware/
software/python/
examples/
docs/
scripts/
tests/
```

## Citation

```bibtex
@article{lee2026hardware,
  title={Hardware-Software Co-Design for Event-Driven SNN Deployment on Low-Cost Neuromorphic FPGAs},
  author={Lee, Jiwoon and Chakraborty, Souvik and Alam, Syed Bahauddin and Park, Cheolsoo},
  journal={arXiv preprint arXiv:2604.22179},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE)

## Successor

Active development continues in [SpikeMold](https://github.com/jiwoonl/SpikeMold).
This repository remains the publication V1 source.
