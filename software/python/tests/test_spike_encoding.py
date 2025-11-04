import numpy as np
import pytest

from snn_fpga_accelerator.spike_encoding import (
    PoissonEncoder,
    TemporalEncoder,
    RateEncoder,
    LatencyEncoder,
    PopulationDecoder,
    TemporalDecoder,
    SpikeEvent,
    spikes_to_raster,
    calculate_spike_rate,
)


def test_poisson_encoder_same_seed_reproducible():
    data = np.array([1.0, 0.5, 0.25], dtype=float)
    encoder_a = PoissonEncoder(
        num_neurons=3,
        duration=0.05,
        max_rate=120.0,
        min_rate=0.0,
        seed=123,
    )
    encoder_b = PoissonEncoder(
        num_neurons=3,
        duration=0.05,
        max_rate=120.0,
        min_rate=0.0,
        seed=123,
    )

    events_a = encoder_a.encode(data)
    events_b = encoder_b.encode(data)

    assert events_a == events_b


def test_poisson_encoder_rejects_rng_and_seed():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        PoissonEncoder(
            num_neurons=2,
            duration=0.01,
            max_rate=50.0,
            min_rate=0.0,
            rng=rng,
            seed=42,
        )


def test_poisson_encoder_size_mismatch_raises():
    encoder = PoissonEncoder(num_neurons=2, duration=0.01, max_rate=50.0, min_rate=0.0, seed=7)
    with pytest.raises(ValueError):
        encoder.encode(np.array([1.0]))


def test_temporal_encoder_produces_expected_times():
    encoder = TemporalEncoder(num_neurons=3, duration=1.0)
    spikes = encoder.encode(np.array([0.0, 0.5, 1.0], dtype=float))

    assert len(spikes) == 2
    spike_map = {event.neuron_id: event.timestamp for event in spikes}
    assert pytest.approx(spike_map[1]) == 0.5
    assert pytest.approx(spike_map[2]) == 0.0


def test_rate_encoder_generates_regular_spikes():
    encoder = RateEncoder(num_neurons=1, duration=0.01, max_rate=1000.0)
    spikes = encoder.encode(np.array([1.0], dtype=float))

    expected_times = np.arange(0.001, 0.01, 0.001)
    assert np.allclose([event.timestamp for event in spikes], expected_times)


def test_latency_encoder_applies_max_delay():
    encoder = LatencyEncoder(num_neurons=2, duration=1.0, max_delay=0.4)
    spikes = encoder.encode(np.array([1.0, 0.5], dtype=float))

    spike_map = {event.neuron_id: event.timestamp for event in spikes}
    assert pytest.approx(spike_map[0]) == 0.0
    assert pytest.approx(spike_map[1]) == 0.2


def test_population_decoder_counts_spikes():
    decoder = PopulationDecoder(num_outputs=3, duration=0.1, time_window=0.05)
    spikes = [
        SpikeEvent(neuron_id=0, timestamp=0.01, weight=1.0),
        SpikeEvent(neuron_id=2, timestamp=0.02, weight=1.0),
        SpikeEvent(neuron_id=2, timestamp=0.03, weight=1.0),
    ]

    rates = decoder.decode(spikes)
    expected = np.array([10.0, 0.0, 20.0], dtype=float)
    assert np.allclose(rates, expected)


def test_temporal_decoder_uses_first_spike_time():
    decoder = TemporalDecoder(num_outputs=2, duration=1.0)
    spikes = [
        SpikeEvent(neuron_id=0, timestamp=0.2, weight=1.0),
        SpikeEvent(neuron_id=1, timestamp=0.5, weight=1.0),
        SpikeEvent(neuron_id=0, timestamp=0.1, weight=1.0),
    ]

    activations = decoder.decode(spikes)
    expected = np.array([0.9, 0.5], dtype=float)
    assert np.allclose(activations, expected)


def test_spikes_to_raster_populates_bins():
    spikes = [
        SpikeEvent(neuron_id=0, timestamp=0.05, weight=1.0),
        SpikeEvent(neuron_id=1, timestamp=0.15, weight=1.0),
    ]
    raster = spikes_to_raster(spikes, num_neurons=2, duration=0.2, time_bins=4)

    expected = np.zeros((2, 4))
    expected[0, 1] = 1.0
    expected[1, 2] = 1.0
    assert np.array_equal(raster, expected)


def test_calculate_spike_rate_counts_within_windows():
    spikes = [
        SpikeEvent(neuron_id=0, timestamp=0.01, weight=1.0),
        SpikeEvent(neuron_id=0, timestamp=0.03, weight=1.0),
        SpikeEvent(neuron_id=0, timestamp=0.08, weight=1.0),
    ]

    rates = calculate_spike_rate(spikes, neuron_id=0, duration=0.1, window_size=0.05)
    expected = np.array([40.0, 20.0], dtype=float)
    assert np.allclose(rates, expected)
