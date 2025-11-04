import numpy as np
import pytest

from snn_fpga_accelerator.learning import LearningConfig, STDPLearning, RSTDPLearning
from snn_fpga_accelerator.spike_encoding import SpikeEvent


@pytest.fixture
def stdp() -> STDPLearning:
    return STDPLearning(LearningConfig(learning_rate=0.05))


def test_stdp_updates_increase_with_pre_before_post(stdp: STDPLearning) -> None:
    pre_spike = SpikeEvent(neuron_id=0, timestamp=0.0)
    post_spike = SpikeEvent(neuron_id=1, timestamp=0.01)

    stdp.add_spike(pre_spike, is_post_synaptic=False)
    stdp.add_spike(post_spike, is_post_synaptic=True)

    synapse_map = {(0, 1): 0.0}
    updates = stdp.compute_weight_updates(synapse_map)

    assert updates, "Expected at least one weight update"
    update = updates[0]
    assert update["pre_neuron"] == 0
    assert update["post_neuron"] == 1
    assert update["weight_change"] > 0


def test_stdp_ltd_when_post_before_pre(stdp: STDPLearning) -> None:
    post_spike = SpikeEvent(neuron_id=1, timestamp=0.0)
    pre_spike = SpikeEvent(neuron_id=0, timestamp=0.01)

    stdp.add_spike(post_spike, is_post_synaptic=True)
    stdp.add_spike(pre_spike, is_post_synaptic=False)

    synapse_map = {(0, 1): 0.0}
    updates = stdp.compute_weight_updates(synapse_map)

    assert updates, "Expected LTD update"
    assert updates[0]["weight_change"] < 0


def test_rstdp_modulates_with_reward() -> None:
    config = LearningConfig(learning_rate=0.01, reward_window=0.5)
    rstdp = RSTDPLearning(config)

    pre_spike = SpikeEvent(neuron_id=0, timestamp=0.0)
    post_spike = SpikeEvent(neuron_id=1, timestamp=0.01)

    rstdp.add_spike(pre_spike, is_post_synaptic=False)
    rstdp.add_spike(post_spike, is_post_synaptic=True)

    synapse_map = {(0, 1): 0.0}
    rstdp.update_eligibility_traces(synapse_map)

    rstdp.add_reward(1.0, timestamp=0.02)
    updates_without_reward = rstdp.compute_weight_updates(synapse_map, current_time=1.0)

    rstdp.add_reward(1.0, timestamp=0.99)
    updates_with_reward = rstdp.compute_weight_updates(synapse_map, current_time=1.0)

    change_no_reward = updates_without_reward[0]["weight_change"] if updates_without_reward else 0.0
    change_with_reward = updates_with_reward[0]["weight_change"] if updates_with_reward else 0.0

    assert change_with_reward >= change_no_reward
