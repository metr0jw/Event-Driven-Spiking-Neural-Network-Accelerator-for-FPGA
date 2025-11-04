import numpy as np
import pytest

from snn_fpga_accelerator import utils
from snn_fpga_accelerator.spike_encoding import SpikeEvent


def _sample_spikes():
    return [
        SpikeEvent(neuron_id=0, timestamp=0.0, weight=1.0),
        SpikeEvent(neuron_id=1, timestamp=0.01, weight=0.5),
    ]


def test_visualize_spikes_requires_matplotlib(monkeypatch):
    monkeypatch.setattr(utils, "plt", None)

    with pytest.raises(RuntimeError, match="matplotlib is required"):
        utils.visualize_spikes(_sample_spikes(), duration=0.1)


def test_visualize_weight_matrix_requires_matplotlib(monkeypatch):
    monkeypatch.setattr(utils, "plt", None)
    weights = np.ones((2, 2), dtype=float)

    with pytest.raises(RuntimeError, match="matplotlib is required"):
        utils.visualize_weight_matrix(weights)


def test_analyze_spike_statistics_basic():
    stats = utils.analyze_spike_statistics(_sample_spikes(), duration=0.1)

    assert stats["total_spikes"] == 2
    assert stats["num_active_neurons"] == 2
    assert stats["duration"] == pytest.approx(0.1)
    assert stats["mean_rate"] > 0
