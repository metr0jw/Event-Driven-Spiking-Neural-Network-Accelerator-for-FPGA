"""
Learning Algorithms for SNN

Implements STDP (Spike-Timing Dependent Plasticity) and R-STDP (Reward-modulated STDP)
learning algorithms for the FPGA SNN accelerator.

Author: Jiwoon Lee (@metr0jw)
Organization: Kwangwoon University, Seoul, South Korea
Contact: jwlee@linux.com
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
import time
from dataclasses import dataclass
from collections import defaultdict, deque

from .spike_encoding import SpikeEvent
from .utils import logger


@dataclass
class LearningConfig:
    """Configuration parameters for learning algorithms."""
    learning_rate: float = 0.01
    a_plus: float = 0.1          # LTP amplitude
    a_minus: float = 0.12        # LTD amplitude  
    tau_plus: float = 0.020      # LTP time constant (20ms)
    tau_minus: float = 0.020     # LTD time constant (20ms)
    w_min: float = -1.0          # Minimum weight
    w_max: float = 1.0           # Maximum weight
    eligibility_decay: float = 0.95  # For R-STDP
    reward_window: float = 0.1   # Reward time window


class STDPLearning:
    """
    Spike-Timing Dependent Plasticity learning algorithm.
    
    Implements the classical STDP rule where synaptic weights are modified
    based on the relative timing of pre- and post-synaptic spikes.
    """
    
    def __init__(self, config: LearningConfig):
        self.config = config
        self.pre_spike_traces = defaultdict(lambda: deque(maxlen=1000))
        self.post_spike_traces = defaultdict(lambda: deque(maxlen=1000))
        self.weight_updates = []
        
        logger.info(f"STDP Learning initialized: lr={config.learning_rate}, "
                   f"tau_plus={config.tau_plus}, tau_minus={config.tau_minus}")
    
    def add_spike(self, spike: SpikeEvent, is_post_synaptic: bool = False) -> None:
        """
        Add a spike event for STDP processing.
        
        Args:
            spike: Spike event
            is_post_synaptic: True if this is a post-synaptic spike
        """
        if is_post_synaptic:
            self.post_spike_traces[spike.neuron_id].append(spike.timestamp)
        else:
            self.pre_spike_traces[spike.neuron_id].append(spike.timestamp)
    
    def compute_weight_updates(self, synapse_map: Dict[Tuple[int, int], float]) -> List[Dict]:
        """
        Compute STDP weight updates for all synapses.
        
        Args:
            synapse_map: Dictionary mapping (pre_id, post_id) -> current_weight
            
        Returns:
            List of weight update dictionaries
        """
        updates = []
        
        for (pre_id, post_id), current_weight in synapse_map.items():
            weight_change = self._compute_stdp_update(pre_id, post_id)
            
            if abs(weight_change) > 1e-6:  # Only apply significant updates
                new_weight = np.clip(
                    current_weight + weight_change,
                    self.config.w_min,
                    self.config.w_max
                )
                
                updates.append({
                    'pre_neuron': pre_id,
                    'post_neuron': post_id,
                    'old_weight': current_weight,
                    'new_weight': new_weight,
                    'weight_change': weight_change,
                    'update_type': 'stdp'
                })
        
        logger.debug(f"Computed {len(updates)} STDP weight updates")
        return updates
    
    def _compute_stdp_update(self, pre_id: int, post_id: int) -> float:
        """Compute STDP weight change for a specific synapse."""
        total_change = 0.0
        
        pre_spikes = list(self.pre_spike_traces[pre_id])
        post_spikes = list(self.post_spike_traces[post_id])
        
        # LTP: pre before post (Δt = t_post - t_pre > 0)
        for post_time in post_spikes:
            for pre_time in pre_spikes:
                dt = post_time - pre_time
                if 0 < dt <= 5 * self.config.tau_plus:  # Within window
                    ltp = self.config.a_plus * np.exp(-dt / self.config.tau_plus)
                    total_change += ltp
        
        # LTD: post before pre (Δt = t_post - t_pre < 0)
        for pre_time in pre_spikes:
            for post_time in post_spikes:
                dt = post_time - pre_time
                if -5 * self.config.tau_minus <= dt < 0:  # Within window
                    ltd = -self.config.a_minus * np.exp(dt / self.config.tau_minus)
                    total_change += ltd
        
        return self.config.learning_rate * total_change
    
    def reset_traces(self) -> None:
        """Clear all spike traces."""
        self.pre_spike_traces.clear()
        self.post_spike_traces.clear()
        logger.debug("STDP traces reset")


class RSTDPLearning:
    """
    Reward-modulated Spike-Timing Dependent Plasticity (R-STDP).
    
    Extends STDP with reward signals that modulate weight changes.
    Uses eligibility traces to associate weight changes with delayed rewards.
    """
    
    def __init__(self, config: LearningConfig):
        self.config = config
        self.stdp = STDPLearning(config)
        
        # Eligibility traces for each synapse
        self.eligibility_traces = defaultdict(float)
        self.reward_history = deque(maxlen=1000)
        
        # Statistics
        self.total_reward = 0.0
        self.reward_count = 0
        
        logger.info(f"R-STDP Learning initialized: eligibility_decay={config.eligibility_decay}, "
                   f"reward_window={config.reward_window}")
    
    def add_spike(self, spike: SpikeEvent, is_post_synaptic: bool = False) -> None:
        """Add spike event (delegates to STDP)."""
        self.stdp.add_spike(spike, is_post_synaptic)
    
    def add_reward(self, reward: float, timestamp: float) -> None:
        """
        Add a reward signal.
        
        Args:
            reward: Reward value (positive or negative)
            timestamp: Time when reward was received
        """
        self.reward_history.append((timestamp, reward))
        self.total_reward += reward
        self.reward_count += 1
        
        logger.debug(f"Reward added: {reward} at time {timestamp}")
    
    def update_eligibility_traces(self, synapse_map: Dict[Tuple[int, int], float]) -> None:
        """
        Update eligibility traces based on recent STDP changes.
        
        Args:
            synapse_map: Current synapse weights
        """
        # Get STDP updates
        stdp_updates = self.stdp.compute_weight_updates(synapse_map)
        
        # Update eligibility traces
        for update in stdp_updates:
            synapse_key = (update['pre_neuron'], update['post_neuron'])
            
            # Add STDP change to eligibility trace
            self.eligibility_traces[synapse_key] += update['weight_change']
            
            # Apply decay
            self.eligibility_traces[synapse_key] *= self.config.eligibility_decay
    
    def compute_weight_updates(self, synapse_map: Dict[Tuple[int, int], float], 
                             current_time: float) -> List[Dict]:
        """
        Compute R-STDP weight updates.
        
        Args:
            synapse_map: Current synapse weights
            current_time: Current simulation time
            
        Returns:
            List of weight update dictionaries
        """
        # Update eligibility traces
        self.update_eligibility_traces(synapse_map)
        
        # Calculate reward signal for current time window
        reward_signal = self._get_recent_reward(current_time)
        
        updates = []
        
        # Apply reward-modulated updates
        for (pre_id, post_id), current_weight in synapse_map.items():
            synapse_key = (pre_id, post_id)
            eligibility = self.eligibility_traces.get(synapse_key, 0.0)
            
            if abs(eligibility) > 1e-6 and abs(reward_signal) > 1e-6:
                # R-STDP weight change
                weight_change = reward_signal * eligibility
                
                new_weight = np.clip(
                    current_weight + weight_change,
                    self.config.w_min,
                    self.config.w_max
                )
                
                updates.append({
                    'pre_neuron': pre_id,
                    'post_neuron': post_id,
                    'old_weight': current_weight,
                    'new_weight': new_weight,
                    'weight_change': weight_change,
                    'eligibility': eligibility,
                    'reward': reward_signal,
                    'update_type': 'r-stdp'
                })
                
                # Decay eligibility trace after use
                self.eligibility_traces[synapse_key] *= 0.9
        
        logger.debug(f"Computed {len(updates)} R-STDP weight updates "
                    f"(reward={reward_signal:.4f})")
        return updates
    
    def _get_recent_reward(self, current_time: float) -> float:
        """Get aggregated reward signal for recent time window."""
        total_reward = 0.0
        count = 0
        
        for timestamp, reward in self.reward_history:
            if current_time - timestamp <= self.config.reward_window:
                total_reward += reward
                count += 1
        
        return total_reward / max(count, 1)
    
    def get_learning_stats(self) -> Dict:
        """Get learning statistics."""
        avg_reward = self.total_reward / max(self.reward_count, 1)
        active_traces = sum(1 for trace in self.eligibility_traces.values() 
                           if abs(trace) > 1e-6)
        
        return {
            'total_reward': self.total_reward,
            'average_reward': avg_reward,
            'reward_count': self.reward_count,
            'active_eligibility_traces': active_traces,
            'total_eligibility_traces': len(self.eligibility_traces)
        }
    
    def reset_traces(self) -> None:
        """Reset all learning traces."""
        self.stdp.reset_traces()
        self.eligibility_traces.clear()
        self.reward_history.clear()
        logger.debug("R-STDP traces reset")


class TripleSTDP:
    """
    Triplet STDP model that considers triplets of spikes for more
    stable learning in recurrent networks.
    """
    
    def __init__(self, config: LearningConfig):
        self.config = config
        self.pre_traces_1 = defaultdict(float)   # First-order pre traces
        self.pre_traces_2 = defaultdict(float)   # Second-order pre traces  
        self.post_traces_1 = defaultdict(float)  # First-order post traces
        self.post_traces_2 = defaultdict(float)  # Second-order post traces
        
        # Additional triplet parameters
        self.a2_plus = 7.5e-10   # Triplet LTP coefficient
        self.a2_minus = 7e-3     # Triplet LTD coefficient
        self.tau_x = 0.02        # Pre trace time constant
        self.tau_y = 0.02        # Post trace time constant
        
        logger.info("Triplet STDP initialized")
    
    def process_spike(self, neuron_id: int, timestamp: float, 
                     is_post: bool, synapse_map: Dict) -> List[Dict]:
        """Process a spike and compute triplet STDP updates."""
        updates = []
        
        if is_post:
            # Post-synaptic spike
            self._update_post_traces(neuron_id, timestamp)
            
            # Find all pre-synaptic connections
            for (pre_id, post_id), weight in synapse_map.items():
                if post_id == neuron_id:
                    # Compute LTP update
                    r1 = self.pre_traces_1[pre_id]
                    r2 = self.pre_traces_2[pre_id]
                    
                    dw = (self.config.a_plus * r1 + 
                          self.a2_plus * r2 * self.post_traces_2[neuron_id])
                    
                    if abs(dw) > 1e-8:
                        updates.append({
                            'pre_neuron': pre_id,
                            'post_neuron': post_id,
                            'weight_change': dw,
                            'update_type': 'triplet_ltp'
                        })
        else:
            # Pre-synaptic spike
            self._update_pre_traces(neuron_id, timestamp)
            
            # Find all post-synaptic connections
            for (pre_id, post_id), weight in synapse_map.items():
                if pre_id == neuron_id:
                    # Compute LTD update
                    o1 = self.post_traces_1[post_id]
                    o2 = self.post_traces_2[post_id]
                    
                    dw = -(self.config.a_minus * o1 + 
                           self.a2_minus * o2 * self.pre_traces_2[neuron_id])
                    
                    if abs(dw) > 1e-8:
                        updates.append({
                            'pre_neuron': pre_id,
                            'post_neuron': post_id,
                            'weight_change': dw,
                            'update_type': 'triplet_ltd'
                        })
        
        return updates
    
    def _update_pre_traces(self, neuron_id: int, timestamp: float) -> None:
        """Update pre-synaptic traces."""
        # Decay existing traces
        self.pre_traces_1[neuron_id] *= np.exp(-0.001 / self.tau_x)
        self.pre_traces_2[neuron_id] *= np.exp(-0.001 / self.tau_x)
        
        # Add spike contribution
        self.pre_traces_1[neuron_id] += 1.0
        self.pre_traces_2[neuron_id] += 1.0
    
    def _update_post_traces(self, neuron_id: int, timestamp: float) -> None:
        """Update post-synaptic traces."""
        # Decay existing traces
        self.post_traces_1[neuron_id] *= np.exp(-0.001 / self.tau_y)
        self.post_traces_2[neuron_id] *= np.exp(-0.001 / self.tau_y)
        
        # Add spike contribution
        self.post_traces_1[neuron_id] += 1.0
        self.post_traces_2[neuron_id] += 1.0


class AdaptiveLearning:
    """
    Adaptive learning that adjusts learning rates based on performance.
    """
    
    def __init__(self, base_config: LearningConfig):
        self.base_config = base_config
        self.current_lr = base_config.learning_rate
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.update_count = 0
        
        # Adaptation parameters
        self.lr_decay = 0.99
        self.lr_boost = 1.01
        self.performance_threshold = 0.1
        
    def adapt_learning_rate(self, performance_metric: float) -> None:
        """Adapt learning rate based on performance."""
        self.performance_history.append(performance_metric)
        
        if len(self.performance_history) >= 10:
            recent_trend = np.mean(list(self.performance_history)[-5:]) - \
                          np.mean(list(self.performance_history)[-10:-5])
            
            if recent_trend > self.performance_threshold:
                # Performance improving, boost learning rate
                self.current_lr *= self.lr_boost
            else:
                # Performance stagnating, decay learning rate
                self.current_lr *= self.lr_decay
            
            # Clamp learning rate
            self.current_lr = np.clip(self.current_lr, 1e-6, 1.0)
            
            logger.debug(f"Adapted learning rate to {self.current_lr:.6f} "
                        f"(trend: {recent_trend:.4f})")
    
    def get_current_config(self) -> LearningConfig:
        """Get current learning configuration."""
        config = LearningConfig()
        config.__dict__.update(self.base_config.__dict__)
        config.learning_rate = self.current_lr
        return config


def create_reward_function(task_type: str) -> Callable[[List[SpikeEvent], any], float]:
    """
    Create reward function for different learning tasks.
    
    Args:
        task_type: Type of task ('classification', 'regression', 'reinforcement')
        
    Returns:
        Reward function
    """
    if task_type == 'classification':
        def classification_reward(output_spikes: List[SpikeEvent], target_class: int) -> float:
            """Reward based on classification accuracy."""
            if not output_spikes:
                return -1.0
            
            # Find most active output neuron
            spike_counts = defaultdict(int)
            for spike in output_spikes:
                spike_counts[spike.neuron_id] += 1
            
            predicted_class = max(spike_counts, key=spike_counts.get)
            return 1.0 if predicted_class == target_class else -0.5
        
        return classification_reward
    
    elif task_type == 'regression':
        def regression_reward(output_spikes: List[SpikeEvent], target_value: float) -> float:
            """Reward based on regression error."""
            if not output_spikes:
                return -1.0
            
            # Convert spikes to value (simplified)
            spike_value = len(output_spikes) / 100.0  # Normalize
            error = abs(spike_value - target_value)
            return -error  # Negative error as reward
        
        return regression_reward
    
    else:
        def default_reward(output_spikes: List[SpikeEvent], target: any) -> float:
            """Default reward function."""
            return 0.0
        
        return default_reward
