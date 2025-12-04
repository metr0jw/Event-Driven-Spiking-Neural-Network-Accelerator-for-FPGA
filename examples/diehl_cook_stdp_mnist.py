#!/usr/bin/env python3
"""
Diehl & Cook (2015) STDP-based MNIST Classification

Reimplementation using our SNN FPGA Accelerator library.
Original paper: "Unsupervised learning of digit recognition using STDP"

Architecture:
- Input: 784 Poisson neurons (rate-coded MNIST pixels)
- Excitatory: 400 LIF neurons with adaptive threshold
- Inhibitory: 400 LIF neurons (lateral inhibition / WTA)
- STDP learning on input->excitatory connections

Author: Jiwoon Lee (@metr0jw)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'software', 'python'))

import numpy as np
import time
from typing import Tuple, Dict, List, Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our library
from snn_fpga_accelerator.neuron import LIF
from snn_fpga_accelerator.encoder import PoissonEncoder
from snn_fpga_accelerator.hw_accurate_simulator import (
    HWAccurateLIFNeuron, LIFNeuronParams
)

import torch
import torch.nn as nn


# =============================================================================
# Network Parameters (matching Diehl & Cook 2015)
# =============================================================================

class DiehlCookParams:
    """Parameters from the original paper."""
    # Network size
    n_input = 784           # 28x28 MNIST
    n_e = 400               # Excitatory neurons
    n_i = 400               # Inhibitory neurons (= n_e)
    
    # Timing
    single_example_time = 350  # ms
    resting_time = 150         # ms
    dt = 0.5                   # ms (timestep)
    
    # Neuron parameters (converted to our format)
    v_rest_e = -65.0        # mV
    v_rest_i = -60.0        # mV
    v_reset_e = -65.0       # mV
    v_reset_i = -45.0       # mV
    v_thresh_e = -52.0      # mV (base, before theta)
    v_thresh_i = -40.0      # mV
    refrac_e = 5            # ms
    refrac_i = 2            # ms
    
    # Time constants
    tau_e = 100.0           # ms (membrane)
    tau_i = 10.0            # ms (membrane)
    tau_ge = 1.0            # ms (exc synaptic)
    tau_gi = 2.0            # ms (inh synaptic)
    
    # STDP parameters
    tc_pre_ee = 20.0        # ms
    tc_post_1_ee = 20.0     # ms
    tc_post_2_ee = 40.0     # ms
    nu_ee_pre = 0.0001      # LTD learning rate
    nu_ee_post = 0.01       # LTP learning rate
    wmax_ee = 1.0           # max weight
    
    # Adaptive threshold
    theta_plus = 0.05       # mV (threshold increase after spike)
    tc_theta = 1e7          # ms (theta decay time constant)
    offset = 20.0           # mV (threshold offset)
    
    # Weight normalization
    weight_ee_input = 78.0  # Target column sum for normalization
    
    # Input encoding
    input_intensity = 2.0   # Initial intensity multiplier


# =============================================================================
# Exponential STDP Implementation
# =============================================================================

class ExponentialSTDP:
    """
    Triplet STDP rule from Diehl & Cook.
    
    Pre-before-post (LTP): w += nu_post * pre * post2_before
    Post-before-pre (LTD): w -= nu_pre * post1
    
    With exponential traces:
    - pre trace: decays with tc_pre
    - post1 trace: decays with tc_post_1 (for LTD)
    - post2 trace: decays with tc_post_2 (for LTP, slower)
    """
    
    def __init__(
        self,
        n_pre: int,
        n_post: int,
        tc_pre: float = 20.0,
        tc_post_1: float = 20.0,
        tc_post_2: float = 40.0,
        nu_pre: float = 0.0001,
        nu_post: float = 0.01,
        wmax: float = 1.0,
        wmin: float = 0.0,
        dt: float = 0.5
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.tc_pre = tc_pre
        self.tc_post_1 = tc_post_1
        self.tc_post_2 = tc_post_2
        self.nu_pre = nu_pre
        self.nu_post = nu_post
        self.wmax = wmax
        self.wmin = wmin
        self.dt = dt
        
        # Decay factors
        self.decay_pre = np.exp(-dt / tc_pre)
        self.decay_post_1 = np.exp(-dt / tc_post_1)
        self.decay_post_2 = np.exp(-dt / tc_post_2)
        
        # Traces
        self.pre_trace = np.zeros(n_pre)
        self.post1_trace = np.zeros(n_post)
        self.post2_trace = np.zeros(n_post)
        
        # Weights
        self.weights = np.random.uniform(0, wmax, (n_pre, n_post)).astype(np.float32)
    
    def reset_traces(self):
        """Reset all traces to zero."""
        self.pre_trace[:] = 0
        self.post1_trace[:] = 0
        self.post2_trace[:] = 0
    
    def step(self, pre_spikes: np.ndarray, post_spikes: np.ndarray):
        """
        Update traces and weights based on spikes.
        
        Args:
            pre_spikes: Binary array of pre-synaptic spikes (n_pre,)
            post_spikes: Binary array of post-synaptic spikes (n_post,)
        """
        # Store post2 before update for triplet rule
        post2_before = self.post2_trace.copy()
        
        # Decay traces
        self.pre_trace *= self.decay_pre
        self.post1_trace *= self.decay_post_1
        self.post2_trace *= self.decay_post_2
        
        # Update traces on spikes
        pre_spike_idx = np.where(pre_spikes > 0)[0]
        post_spike_idx = np.where(post_spikes > 0)[0]
        
        # Pre-synaptic spike: update pre trace, apply LTD
        if len(pre_spike_idx) > 0:
            self.pre_trace[pre_spike_idx] = 1.0
            # LTD: w -= nu_pre * post1
            for i in pre_spike_idx:
                self.weights[i, :] -= self.nu_pre * self.post1_trace
        
        # Post-synaptic spike: update post traces, apply LTP
        if len(post_spike_idx) > 0:
            # LTP: w += nu_post * pre * post2_before
            for j in post_spike_idx:
                self.weights[:, j] += self.nu_post * self.pre_trace * post2_before[j]
            
            # Update post traces
            self.post1_trace[post_spike_idx] = 1.0
            self.post2_trace[post_spike_idx] = 1.0
        
        # Clip weights
        np.clip(self.weights, self.wmin, self.wmax, out=self.weights)
    
    def normalize_weights(self, target_sum: float):
        """Normalize weights so each column sums to target_sum."""
        col_sums = np.sum(self.weights, axis=0)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        factors = target_sum / col_sums
        self.weights *= factors


# =============================================================================
# LIF Neuron with Adaptive Threshold
# =============================================================================

class AdaptiveLIFNeuron:
    """
    LIF neuron with adaptive threshold (theta).
    
    Threshold increases by theta_plus after each spike,
    then decays exponentially with time constant tc_theta.
    """
    
    def __init__(
        self,
        n_neurons: int,
        v_rest: float = -65.0,
        v_reset: float = -65.0,
        v_thresh_base: float = -52.0,
        tau_m: float = 100.0,
        tau_ge: float = 1.0,
        tau_gi: float = 2.0,
        refrac: int = 5,
        theta_plus: float = 0.05,
        tc_theta: float = 1e7,
        offset: float = 20.0,
        dt: float = 0.5,
        adaptive_threshold: bool = True
    ):
        self.n = n_neurons
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh_base = v_thresh_base
        self.tau_m = tau_m
        self.tau_ge = tau_ge
        self.tau_gi = tau_gi
        self.refrac = int(refrac / dt)  # Convert ms to timesteps
        self.theta_plus = theta_plus
        self.tc_theta = tc_theta
        self.offset = offset
        self.dt = dt
        self.adaptive_threshold = adaptive_threshold
        
        # Decay factors
        self.decay_m = np.exp(-dt / tau_m)
        self.decay_ge = np.exp(-dt / tau_ge)
        self.decay_gi = np.exp(-dt / tau_gi)
        self.decay_theta = np.exp(-dt / tc_theta) if adaptive_threshold else 1.0
        
        # State
        self.v = np.ones(n_neurons) * (v_rest - 40.0)  # Initial: rest - 40mV
        self.ge = np.zeros(n_neurons)  # Excitatory conductance
        self.gi = np.zeros(n_neurons)  # Inhibitory conductance
        self.theta = np.ones(n_neurons) * offset  # Adaptive threshold
        self.refrac_count = np.zeros(n_neurons, dtype=np.int32)
        
    def reset_state(self):
        """Reset neuron state (keep theta)."""
        self.v[:] = self.v_rest - 40.0
        self.ge[:] = 0
        self.gi[:] = 0
        self.refrac_count[:] = 0
    
    def step(self, I_ext_e: np.ndarray = None, I_ext_i: np.ndarray = None) -> np.ndarray:
        """
        Advance one timestep.
        
        Args:
            I_ext_e: External excitatory input (added to ge)
            I_ext_i: External inhibitory input (added to gi)
            
        Returns:
            Binary spike array
        """
        # Add external input to conductances
        if I_ext_e is not None:
            self.ge += I_ext_e
        if I_ext_i is not None:
            self.gi += I_ext_i
        
        # Calculate synaptic currents
        # I_synE = ge * (-v)  (reversal at 0)
        # I_synI = gi * (-100 - v)  (reversal at -100mV)
        I_synE = self.ge * (-self.v)
        I_synI = self.gi * (-100.0 - self.v)
        
        # Membrane dynamics: dv/dt = (v_rest - v + I_syn) / tau_m
        dv = ((self.v_rest - self.v) + (I_synE + I_synI)) / self.tau_m * self.dt
        
        # Update membrane (only if not refractory)
        active = self.refrac_count == 0
        self.v[active] += dv[active]
        
        # Decay conductances
        self.ge *= self.decay_ge
        self.gi *= self.decay_gi
        
        # Decay theta (adaptive threshold)
        if self.adaptive_threshold:
            self.theta *= self.decay_theta
        
        # Check threshold: v > (theta - offset + v_thresh_base)
        effective_thresh = self.theta - self.offset + self.v_thresh_base
        spikes = (self.v > effective_thresh) & active
        
        # Handle spikes
        spike_idx = np.where(spikes)[0]
        if len(spike_idx) > 0:
            self.v[spike_idx] = self.v_reset
            self.refrac_count[spike_idx] = self.refrac
            if self.adaptive_threshold:
                self.theta[spike_idx] += self.theta_plus
        
        # Decrement refractory counter
        self.refrac_count[self.refrac_count > 0] -= 1
        
        return spikes.astype(np.float32)


class InhibitoryLIFNeuron:
    """Simple LIF neuron for inhibitory population."""
    
    def __init__(
        self,
        n_neurons: int,
        v_rest: float = -60.0,
        v_reset: float = -45.0,
        v_thresh: float = -40.0,
        tau_m: float = 10.0,
        tau_ge: float = 1.0,
        tau_gi: float = 2.0,
        refrac: int = 2,
        dt: float = 0.5
    ):
        self.n = n_neurons
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.tau_m = tau_m
        self.refrac = int(refrac / dt)
        self.dt = dt
        
        self.decay_m = np.exp(-dt / tau_m)
        self.decay_ge = np.exp(-dt / tau_ge)
        self.decay_gi = np.exp(-dt / tau_gi)
        
        self.v = np.ones(n_neurons) * (v_rest - 40.0)
        self.ge = np.zeros(n_neurons)
        self.gi = np.zeros(n_neurons)
        self.refrac_count = np.zeros(n_neurons, dtype=np.int32)
    
    def reset_state(self):
        self.v[:] = self.v_rest - 40.0
        self.ge[:] = 0
        self.gi[:] = 0
        self.refrac_count[:] = 0
    
    def step(self, I_ext_e: np.ndarray = None) -> np.ndarray:
        if I_ext_e is not None:
            self.ge += I_ext_e
        
        I_synE = self.ge * (-self.v)
        I_synI = self.gi * (-85.0 - self.v)
        
        dv = ((self.v_rest - self.v) + (I_synE + I_synI)) / self.tau_m * self.dt
        
        active = self.refrac_count == 0
        self.v[active] += dv[active]
        
        self.ge *= self.decay_ge
        self.gi *= self.decay_gi
        
        spikes = (self.v > self.v_thresh) & active
        
        spike_idx = np.where(spikes)[0]
        if len(spike_idx) > 0:
            self.v[spike_idx] = self.v_reset
            self.refrac_count[spike_idx] = self.refrac
        
        self.refrac_count[self.refrac_count > 0] -= 1
        
        return spikes.astype(np.float32)


# =============================================================================
# Diehl & Cook Network
# =============================================================================

class DiehlCookNetwork:
    """
    Complete Diehl & Cook (2015) network.
    
    Architecture:
    - Input (X): 784 Poisson neurons
    - Excitatory (Ae): 400 adaptive LIF neurons
    - Inhibitory (Ai): 400 LIF neurons
    
    Connections:
    - XeAe: Input to excitatory (STDP learning)
    - AeAi: Excitatory to inhibitory (one-to-one)
    - AiAe: Inhibitory to excitatory (all-to-all except self)
    """
    
    def __init__(self, params: DiehlCookParams = None, test_mode: bool = False):
        self.params = params or DiehlCookParams()
        self.test_mode = test_mode
        self.dt = self.params.dt
        
        p = self.params
        
        # Create neuron populations
        logger.info(f"Creating network: {p.n_input} -> {p.n_e} (exc) -> {p.n_i} (inh)")
        
        self.exc_neurons = AdaptiveLIFNeuron(
            n_neurons=p.n_e,
            v_rest=p.v_rest_e,
            v_reset=p.v_reset_e,
            v_thresh_base=p.v_thresh_e,
            tau_m=p.tau_e,
            tau_ge=p.tau_ge,
            tau_gi=p.tau_gi,
            refrac=p.refrac_e,
            theta_plus=p.theta_plus,
            tc_theta=p.tc_theta,
            offset=p.offset,
            dt=p.dt,
            adaptive_threshold=not test_mode
        )
        
        self.inh_neurons = InhibitoryLIFNeuron(
            n_neurons=p.n_i,
            v_rest=p.v_rest_i,
            v_reset=p.v_reset_i,
            v_thresh=p.v_thresh_i,
            tau_m=p.tau_i,
            tau_ge=p.tau_ge,
            tau_gi=p.tau_gi,
            refrac=p.refrac_i,
            dt=p.dt
        )
        
        # Create STDP for input->excitatory
        self.stdp = ExponentialSTDP(
            n_pre=p.n_input,
            n_post=p.n_e,
            tc_pre=p.tc_pre_ee,
            tc_post_1=p.tc_post_1_ee,
            tc_post_2=p.tc_post_2_ee,
            nu_pre=p.nu_ee_pre,
            nu_post=p.nu_ee_post,
            wmax=p.wmax_ee,
            wmin=0.0,
            dt=p.dt
        )
        
        # Fixed connections
        # AeAi: one-to-one (exc neuron i -> inh neuron i)
        self.w_ei = np.eye(p.n_e, p.n_i) * 10.4  # Strong connection
        
        # AiAe: all-to-all except diagonal (lateral inhibition)
        self.w_ie = np.ones((p.n_i, p.n_e)) * 17.0
        np.fill_diagonal(self.w_ie, 0)  # No self-inhibition
        
        # Neuron assignments (for classification)
        self.assignments = np.zeros(p.n_e, dtype=np.int32)
        
        # Result monitoring
        self.spike_counts = np.zeros(p.n_e)
        
        # Input intensity (adaptive)
        self.input_intensity = p.input_intensity
        
        logger.info("Network created successfully")
    
    def reset_state(self):
        """Reset neuron states between examples."""
        self.exc_neurons.reset_state()
        self.inh_neurons.reset_state()
        self.stdp.reset_traces()
        self.spike_counts[:] = 0
    
    def load_weights(self, weight_path: str):
        """Load pre-trained weights."""
        try:
            # Load input->exc weights
            w_data = np.load(os.path.join(weight_path, 'XeAe.npy'))
            if w_data.ndim == 2 and w_data.shape[1] == 3:
                # Sparse format: (pre_id, post_id, weight)
                for row in w_data:
                    i, j, w = int(row[0]), int(row[1]), row[2]
                    self.stdp.weights[i, j] = w
            else:
                self.stdp.weights = w_data
            
            # Load theta
            theta_data = np.load(os.path.join(weight_path, 'theta_A.npy'))
            self.exc_neurons.theta = theta_data
            
            logger.info(f"Loaded weights from {weight_path}")
        except Exception as e:
            logger.warning(f"Could not load weights: {e}")
    
    def save_weights(self, weight_path: str, suffix: str = ''):
        """Save current weights."""
        os.makedirs(weight_path, exist_ok=True)
        
        # Save in sparse format
        sparse_weights = []
        for i in range(self.stdp.weights.shape[0]):
            for j in range(self.stdp.weights.shape[1]):
                if self.stdp.weights[i, j] > 0:
                    sparse_weights.append([i, j, self.stdp.weights[i, j]])
        np.save(os.path.join(weight_path, f'XeAe{suffix}.npy'), np.array(sparse_weights))
        
        # Save theta
        np.save(os.path.join(weight_path, f'theta_A{suffix}.npy'), self.exc_neurons.theta)
        
        logger.info(f"Saved weights to {weight_path}")
    
    def run_example(
        self,
        input_rates: np.ndarray,
        duration_ms: float,
        learning: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Run network on one example.
        
        Args:
            input_rates: Poisson rates for each input neuron (Hz or normalized)
            duration_ms: Duration in milliseconds
            learning: Enable STDP learning
            
        Returns:
            exc_spike_counts: Spike count per excitatory neuron
            total_spikes: Total excitatory spikes
        """
        n_steps = int(duration_ms / self.dt)
        
        # Scale rates for Poisson generation
        # rate -> probability per timestep
        spike_prob = input_rates * self.dt / 1000.0 * self.input_intensity
        spike_prob = np.clip(spike_prob, 0, 1)
        
        self.spike_counts[:] = 0
        
        for t in range(n_steps):
            # Generate input spikes (Poisson)
            input_spikes = (np.random.rand(self.params.n_input) < spike_prob).astype(np.float32)
            
            # Input -> Excitatory
            I_exc = np.dot(input_spikes, self.stdp.weights)
            
            # Run excitatory neurons
            exc_spikes = self.exc_neurons.step(I_ext_e=I_exc)
            
            # Excitatory -> Inhibitory
            I_inh_from_exc = np.dot(exc_spikes, self.w_ei)
            inh_spikes = self.inh_neurons.step(I_ext_e=I_inh_from_exc)
            
            # Inhibitory -> Excitatory (lateral inhibition)
            I_exc_from_inh = np.dot(inh_spikes, self.w_ie)
            self.exc_neurons.gi += I_exc_from_inh
            
            # STDP learning
            if learning and not self.test_mode:
                self.stdp.step(input_spikes, exc_spikes)
            
            # Count spikes
            self.spike_counts += exc_spikes
        
        return self.spike_counts.copy(), int(np.sum(self.spike_counts))
    
    def normalize_weights(self):
        """Normalize input weights."""
        self.stdp.normalize_weights(self.params.weight_ee_input)
    
    def update_assignments(self, result_monitor: np.ndarray, labels: np.ndarray):
        """
        Update neuron-to-class assignments based on spike responses.
        
        Each neuron is assigned to the class that causes it to fire most.
        """
        assignments = np.zeros(self.params.n_e, dtype=np.int32)
        max_rates = np.zeros(self.params.n_e)
        
        for digit in range(10):
            digit_mask = labels == digit
            if np.sum(digit_mask) > 0:
                avg_rates = np.mean(result_monitor[digit_mask], axis=0)
                for i in range(self.params.n_e):
                    if avg_rates[i] > max_rates[i]:
                        max_rates[i] = avg_rates[i]
                        assignments[i] = digit
        
        self.assignments = assignments
        return assignments
    
    def predict(self, spike_counts: np.ndarray) -> int:
        """
        Predict digit based on spike counts and neuron assignments.
        
        Uses summed spike rates per assigned class.
        """
        summed_rates = np.zeros(10)
        num_assignments = np.zeros(10)
        
        for digit in range(10):
            mask = self.assignments == digit
            num_assignments[digit] = np.sum(mask)
            if num_assignments[digit] > 0:
                summed_rates[digit] = np.sum(spike_counts[mask]) / num_assignments[digit]
        
        return int(np.argmax(summed_rates))


# =============================================================================
# Data Loading
# =============================================================================

def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST dataset."""
    try:
        import torchvision
        import torchvision.transforms as transforms
        
        transform = transforms.ToTensor()
        
        train_dataset = torchvision.datasets.MNIST(
            root='../data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='../data', train=False, download=True, transform=transform
        )
        
        train_images = train_dataset.data.numpy().astype(np.float32) / 255.0
        train_labels = train_dataset.targets.numpy()
        test_images = test_dataset.data.numpy().astype(np.float32) / 255.0
        test_labels = test_dataset.targets.numpy()
        
        logger.info(f"Loaded MNIST: {len(train_images)} train, {len(test_images)} test")
        return train_images, train_labels, test_images, test_labels
        
    except Exception as e:
        logger.error(f"Could not load MNIST: {e}")
        raise


# =============================================================================
# Training
# =============================================================================

def train(
    network: DiehlCookNetwork,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    num_examples: int = 60000,
    update_interval: int = 10000,
    weight_update_interval: int = 100,
    save_interval: int = 10000,
    save_path: str = './weights/'
):
    """
    Train the network using STDP.
    """
    p = network.params
    result_monitor = np.zeros((update_interval, p.n_e))
    input_numbers = []
    
    logger.info(f"Starting training for {num_examples} examples...")
    start_time = time.time()
    
    j = 0
    example_idx = 0
    
    while example_idx < num_examples:
        # Get input image
        img_idx = example_idx % len(train_images)
        image = train_images[img_idx].flatten()
        label = train_labels[img_idx]
        
        # Convert to rates (0-255 -> scaled by intensity)
        rates = image / 8.0 * 255.0  # Match original scaling
        
        # Reset network state
        network.reset_state()
        
        # Run example
        spike_counts, total_spikes = network.run_example(
            rates, 
            duration_ms=p.single_example_time,
            learning=True
        )
        
        # Check if enough spikes (adaptive intensity)
        if total_spikes < 5:
            network.input_intensity += 1
            # Run resting period without recording
            network.reset_state()
            continue
        
        # Record results
        result_monitor[j % update_interval] = spike_counts
        input_numbers.append(label)
        
        # Normalize weights periodically
        if (example_idx + 1) % weight_update_interval == 0:
            network.normalize_weights()
        
        # Update assignments
        if (example_idx + 1) % update_interval == 0 and example_idx > 0:
            network.update_assignments(
                result_monitor,
                np.array(input_numbers[-update_interval:])
            )
            
            # Calculate performance
            correct = 0
            for k in range(update_interval):
                pred = network.predict(result_monitor[k])
                if pred == input_numbers[-(update_interval - k)]:
                    correct += 1
            acc = correct / update_interval * 100
            
            elapsed = time.time() - start_time
            logger.info(f"Example {example_idx + 1}/{num_examples}, "
                       f"Accuracy: {acc:.1f}%, "
                       f"Time: {elapsed:.1f}s")
        
        # Save weights periodically
        if (example_idx + 1) % save_interval == 0:
            network.save_weights(save_path, suffix=f'_{example_idx + 1}')
        
        # Reset input intensity
        network.input_intensity = p.input_intensity
        
        # Run resting period
        network.reset_state()
        
        j += 1
        example_idx += 1
    
    # Final save
    network.save_weights(save_path)
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.1f}s")
    
    return network


def test(
    network: DiehlCookNetwork,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    num_examples: int = 10000
) -> float:
    """
    Test the network.
    """
    p = network.params
    
    logger.info(f"Testing on {num_examples} examples...")
    start_time = time.time()
    
    correct = 0
    
    for i in range(min(num_examples, len(test_images))):
        image = test_images[i].flatten()
        label = test_labels[i]
        
        rates = image / 8.0 * 255.0
        
        network.reset_state()
        spike_counts, total_spikes = network.run_example(
            rates,
            duration_ms=p.single_example_time,
            learning=False
        )
        
        # Handle low spike count
        if total_spikes < 5:
            network.input_intensity += 1
            network.reset_state()
            spike_counts, total_spikes = network.run_example(
                rates,
                duration_ms=p.single_example_time,
                learning=False
            )
            network.input_intensity = p.input_intensity
        
        pred = network.predict(spike_counts)
        if pred == label:
            correct += 1
        
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Tested {i + 1}/{num_examples}, "
                       f"Accuracy: {correct / (i + 1) * 100:.1f}%, "
                       f"Time: {elapsed:.1f}s")
    
    accuracy = correct / min(num_examples, len(test_images)) * 100
    logger.info(f"Final test accuracy: {accuracy:.2f}%")
    
    return accuracy


# =============================================================================
# Visualization
# =============================================================================

def visualize_weights(network: DiehlCookNetwork, save_path: str = None):
    """Visualize learned input weights."""
    try:
        import matplotlib.pyplot as plt
        
        n_e_sqrt = int(np.sqrt(network.params.n_e))
        n_in_sqrt = int(np.sqrt(network.params.n_input))
        
        # Rearrange weights into grid
        weights = network.stdp.weights
        num_values = n_e_sqrt * n_in_sqrt
        rearranged = np.zeros((num_values, num_values))
        
        for i in range(n_e_sqrt):
            for j in range(n_e_sqrt):
                neuron_idx = i + j * n_e_sqrt
                weight_patch = weights[:, neuron_idx].reshape((n_in_sqrt, n_in_sqrt))
                rearranged[i*n_in_sqrt:(i+1)*n_in_sqrt, 
                          j*n_in_sqrt:(j+1)*n_in_sqrt] = weight_patch
        
        plt.figure(figsize=(12, 12))
        plt.imshow(rearranged, cmap='hot_r', vmin=0, vmax=network.params.wmax_ee)
        plt.colorbar(label='Weight')
        plt.title('Learned Input Weights (XeAe)')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved weight visualization to {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available for visualization")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Diehl & Cook (2015) STDP MNIST Classification")
    print("Using SNN FPGA Accelerator Library")
    print("=" * 60)
    
    # Load data
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # Create network
    params = DiehlCookParams()
    
    # Training mode
    train_mode = True
    
    if train_mode:
        network = DiehlCookNetwork(params, test_mode=False)
        
        # Train (use fewer examples for quick test)
        num_train = 10000  # Use 60000 * 3 for full training
        train(
            network,
            train_images,
            train_labels,
            num_examples=num_train,
            update_interval=min(10000, num_train),
            save_path='./weights_diehl_cook/'
        )
        
        # Visualize learned weights
        visualize_weights(network, save_path='./weights_diehl_cook/weights_visualization.png')
        
        # Test
        test(network, test_images, test_labels, num_examples=1000)
        
    else:
        # Test mode with pre-trained weights
        network = DiehlCookNetwork(params, test_mode=True)
        network.load_weights('./weights_diehl_cook/')
        
        # Need to build assignments from training data
        logger.info("Building assignments from training data...")
        result_monitor = []
        labels = []
        
        for i in range(10000):
            image = train_images[i].flatten()
            label = train_labels[i]
            rates = image / 8.0 * 255.0
            
            network.reset_state()
            spike_counts, _ = network.run_example(rates, params.single_example_time, learning=False)
            result_monitor.append(spike_counts)
            labels.append(label)
        
        network.update_assignments(np.array(result_monitor), np.array(labels))
        
        # Test
        test(network, test_images, test_labels, num_examples=10000)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
