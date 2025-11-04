import numpy as np
import pytest

from snn_fpga_accelerator.accelerator import SNNAccelerator
from snn_fpga_accelerator.learning import LearningConfig, RSTDPLearning
from snn_fpga_accelerator.pytorch_interface import SNNLayer, SNNModel
from snn_fpga_accelerator.spike_encoding import SpikeEvent


def _build_single_synapse_model() -> SNNModel:
    model = SNNModel(name="test_model")
    layer = SNNLayer(input_size=1, output_size=1)
    layer.set_weights(np.array([[128.0]], dtype=np.float32))
    model.add_layer(layer)
    return model


def test_simulation_infer_produces_rates() -> None:
    accelerator = SNNAccelerator(simulation_mode=True)
    accelerator.configure_network(_build_single_synapse_model())

    spikes = [SpikeEvent(neuron_id=0, timestamp=0.0, weight=1.0)]
    rates = accelerator.infer(spikes, duration=0.1)

    assert isinstance(rates, np.ndarray)
    assert rates.size >= 1
    assert rates[0] > 0


def test_normalize_spike_events_accepts_numpy() -> None:
    accelerator = SNNAccelerator(simulation_mode=True)
    spikes_array = np.array([[0, 0.01, 1.0], [1, 0.02, 0.5]], dtype=float)

    normalized = accelerator._normalize_spike_events(spikes_array)

    assert len(normalized) == 2
    assert normalized[0].neuron_id == 0
    assert normalized[1].weight == 0.5


def test_infer_with_learning_without_engine_returns_rates() -> None:
    accelerator = SNNAccelerator(simulation_mode=True)
    accelerator.configure_network(_build_single_synapse_model())
    accelerator.enable_learning(True)

    spikes = [SpikeEvent(neuron_id=0, timestamp=0.0, weight=1.0)]
    rates = accelerator.infer_with_learning(spikes, duration=0.1)

    assert isinstance(rates, np.ndarray)
    assert rates.size >= 1


def test_infer_with_rstdp_learning_applies_reward() -> None:
    accelerator = SNNAccelerator(simulation_mode=True)
    accelerator.configure_network(_build_single_synapse_model())
    learning_engine = RSTDPLearning(LearningConfig(learning_rate=0.01))
    accelerator.configure_learning(learning_engine)
    accelerator.enable_learning(True)

    spikes = [SpikeEvent(neuron_id=0, timestamp=0.0, weight=1.0)]
    accelerator.apply_reward(0.5)
    rates = accelerator.infer_with_learning(spikes, duration=0.1)

    assert isinstance(rates, np.ndarray)
    assert rates.size >= 1


def test_connect_returns_true_in_simulation() -> None:
    accelerator = SNNAccelerator(simulation_mode=True)
    assert accelerator.connect() is True
    accelerator.disconnect()


@pytest.mark.skip(reason="Requires PYNQ overlay environment")
def test_hardware_connect_disconnect_no_overlay() -> None:
    accelerator = SNNAccelerator(simulation_mode=False)
    with pytest.raises(RuntimeError):
        accelerator.connect()
