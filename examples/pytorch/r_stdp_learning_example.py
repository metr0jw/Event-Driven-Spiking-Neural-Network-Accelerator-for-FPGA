"""
Reward-modulated STDP (R-STDP) Learning Example

This example demonstrates how to implement and use R-STDP learning
for reinforcement learning tasks on the SNN FPGA accelerator.

Author: Jiwoon Lee (@metr0jw)
Organization: Kwangwoon University, Seoul, South Korea
Contact: jwlee@linux.com
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from typing import List, Dict, Tuple
import random

# Add the software package to path
sys.path.append('../../software/python')

from snn_fpga_accelerator import (
    SNNAccelerator, PoissonEncoder, RSTDPLearning, 
    LearningConfig, create_feedforward_snn
)
from snn_fpga_accelerator.spike_encoding import SpikeEvent
from snn_fpga_accelerator.learning import create_reward_function
from snn_fpga_accelerator.utils import (
    visualize_spikes, analyze_spike_statistics, PerformanceMonitor
)


class SimpleNavigationTask:
    """
    Simple 2D navigation task for testing R-STDP learning.
    
    Agent needs to learn to navigate to a target location to receive reward.
    """
    
    def __init__(self, grid_size=8, target_pos=None):
        self.grid_size = grid_size
        self.agent_pos = [0, 0]  # Start position
        self.target_pos = target_pos or [grid_size-1, grid_size-1]  # Target position
        self.max_steps = grid_size * 2  # Maximum steps per episode
        self.current_step = 0
        
        # Actions: 0=up, 1=right, 2=down, 3=left
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.num_actions = len(self.actions)
        
        # State encoding (position to input neurons)
        self.state_encoding_size = grid_size * grid_size
        
    def reset(self):
        """Reset environment for new episode."""
        self.agent_pos = [0, 0]
        self.current_step = 0
        return self.get_state()
    
    def get_state(self):
        """Get current state as one-hot encoded position."""
        state = np.zeros(self.state_encoding_size)
        state_idx = self.agent_pos[0] * self.grid_size + self.agent_pos[1]
        state[state_idx] = 1.0
        return state
    
    def step(self, action):
        """Take action and return new state, reward, done."""
        if action < 0 or action >= self.num_actions:
            return self.get_state(), -0.1, False  # Invalid action penalty
        
        # Move agent
        dx, dy = self.actions[action]
        new_x = max(0, min(self.grid_size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.grid_size - 1, self.agent_pos[1] + dy))
        
        # Check if position actually changed (wall collision)
        if new_x == self.agent_pos[0] and new_y == self.agent_pos[1]:
            reward = -0.05  # Small penalty for hitting wall
        else:
            self.agent_pos = [new_x, new_y]
            reward = -0.01  # Small step penalty
        
        self.current_step += 1
        
        # Check if reached target
        done = False
        if self.agent_pos == self.target_pos:
            reward = 1.0  # Big reward for reaching target
            done = True
        elif self.current_step >= self.max_steps:
            reward = -0.5  # Penalty for timeout
            done = True
        
        return self.get_state(), reward, done
    
    def get_distance_to_target(self):
        """Get Manhattan distance to target."""
        return abs(self.agent_pos[0] - self.target_pos[0]) + \
               abs(self.agent_pos[1] - self.target_pos[1])


class SNNRLAgent:
    """
    SNN-based Reinforcement Learning Agent using R-STDP.
    """
    
    def __init__(self, state_size, num_actions, learning_config=None):
        self.state_size = state_size
        self.num_actions = num_actions
        
        # Create network architecture
        # Input layer -> Hidden layer -> Action layer
        self.hidden_size = 32
        self.total_neurons = state_size + self.hidden_size + num_actions
        
        # Create SNN model
        self.snn_model = create_feedforward_snn(
            layer_sizes=[state_size, self.hidden_size, num_actions],
            neuron_params={
                'threshold': 0.8,
                'leak_rate': 0.05,
                'refractory_period': 3
            }
        )
        
        # Learning configuration
        if learning_config is None:
            learning_config = LearningConfig(
                learning_rate=0.01,
                a_plus=0.1,
                a_minus=0.12,
                tau_plus=0.020,
                tau_minus=0.020,
                eligibility_decay=0.95,
                reward_window=0.1
            )
        
        # Initialize R-STDP learning
        self.r_stdp = RSTDPLearning(learning_config)
        
        # Spike encoding
        self.encoder = PoissonEncoder(
            num_neurons=state_size,
            duration=0.05,  # 50ms simulation time
            max_rate=100.0
        )
        
        # Create synapse map for learning
        self.synapse_map = self._create_synapse_map()
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_steps = []
        self.learning_stats = []
        
    def _create_synapse_map(self):
        """Create initial synapse connectivity map."""
        synapse_map = {}
        
        # Input to hidden connections
        for i in range(self.state_size):
            for j in range(self.state_size, self.state_size + self.hidden_size):
                weight = np.random.randn() * 0.1
                synapse_map[(i, j)] = weight
        
        # Hidden to action connections
        hidden_start = self.state_size
        action_start = self.state_size + self.hidden_size
        
        for i in range(hidden_start, action_start):
            for j in range(action_start, action_start + self.num_actions):
                weight = np.random.randn() * 0.1
                synapse_map[(i, j)] = weight
        
        return synapse_map
    
    def select_action(self, state, exploration_rate=0.1):
        """
        Select action based on SNN output spikes.
        
        Args:
            state: Current state vector
            exploration_rate: Probability of random action
            
        Returns:
            Selected action and spike information
        """
        # Exploration vs exploitation
        if np.random.random() < exploration_rate:
            return np.random.randint(self.num_actions), None, None
        
        # Encode state to spikes
        input_spikes = self.encoder.encode(state)
        
        # Simulate SNN forward pass (simplified software simulation)
        output_spikes = self._simulate_network(input_spikes)
        
        # Extract action layer spikes
        action_start = self.state_size + self.hidden_size
        action_spikes = [s for s in output_spikes 
                        if s.neuron_id >= action_start]
        
        if not action_spikes:
            # No output spikes, random action
            return np.random.randint(self.num_actions), input_spikes, output_spikes
        
        # Count spikes per action
        action_counts = np.zeros(self.num_actions)
        for spike in action_spikes:
            action_idx = spike.neuron_id - action_start
            action_counts[action_idx] += 1
        
        # Select action with most spikes
        selected_action = np.argmax(action_counts)
        
        return selected_action, input_spikes, output_spikes
    
    def _simulate_network(self, input_spikes):
        """
        Simplified SNN simulation for action selection.
        
        This would be replaced by FPGA acceleration in deployment.
        """
        # Initialize neuron states
        neuron_potentials = np.zeros(self.total_neurons)
        spike_times = {i: [] for i in range(self.total_neurons)}
        output_spikes = []
        
        # Simulation parameters
        dt = 0.001  # 1ms time step
        duration = 0.05  # 50ms simulation
        threshold = 0.8
        leak_rate = 0.05
        
        # Process input spikes
        for spike in input_spikes:
            spike_times[spike.neuron_id].append(spike.timestamp)
        
        # Simulate over time
        for t in np.arange(0, duration, dt):
            # Apply input spikes
            for neuron_id, times in spike_times.items():
                if neuron_id < self.state_size:  # Input neurons
                    for spike_time in times:
                        if abs(t - spike_time) < dt/2:
                            neuron_potentials[neuron_id] = threshold + 0.1
            
            # Update membrane potentials
            neuron_potentials *= (1 - leak_rate * dt)
            
            # Propagate spikes through network
            for (pre_id, post_id), weight in self.synapse_map.items():
                if neuron_potentials[pre_id] > threshold:
                    neuron_potentials[post_id] += weight * 0.1
            
            # Check for spikes and reset
            for neuron_id in range(self.total_neurons):
                if neuron_potentials[neuron_id] > threshold:
                    output_spikes.append(SpikeEvent(
                        neuron_id=neuron_id,
                        timestamp=t,
                        weight=1.0
                    ))
                    neuron_potentials[neuron_id] = 0.0  # Reset
        
        return output_spikes
    
    def learn_from_experience(self, state_spikes, action_spikes, reward, timestamp):
        """
        Update weights using R-STDP based on experience.
        
        Args:
            state_spikes: Input spikes from state encoding
            action_spikes: Output spikes from action selection
            reward: Reward received
            timestamp: Current time
        """
        # Add spikes to R-STDP learning
        for spike in state_spikes:
            self.r_stdp.add_spike(spike, is_post_synaptic=False)
        
        for spike in action_spikes:
            self.r_stdp.add_spike(spike, is_post_synaptic=True)
        
        # Add reward signal
        self.r_stdp.add_reward(reward, timestamp)
        
        # Compute weight updates
        weight_updates = self.r_stdp.compute_weight_updates(
            self.synapse_map, timestamp
        )
        
        # Apply weight updates
        for update in weight_updates:
            synapse_key = (update['pre_neuron'], update['post_neuron'])
            if synapse_key in self.synapse_map:
                self.synapse_map[synapse_key] = update['new_weight']
        
        return len(weight_updates)
    
    def train_episode(self, env, exploration_rate=0.1, verbose=False):
        """
        Train agent for one episode.
        
        Args:
            env: Environment instance
            exploration_rate: Exploration rate for action selection
            verbose: Print debug information
            
        Returns:
            Episode statistics
        """
        state = env.reset()
        total_reward = 0
        steps = 0
        updates_made = 0
        timestamp = 0.0
        
        episode_spikes = []
        
        while True:
            # Select action
            action, state_spikes, action_spikes = self.select_action(
                state, exploration_rate
            )
            
            if state_spikes:
                episode_spikes.extend(state_spikes)
            if action_spikes:
                episode_spikes.extend(action_spikes)
            
            # Take action in environment
            next_state, reward, done = env.step(action)
            
            # Learn from experience
            if state_spikes and action_spikes:
                num_updates = self.learn_from_experience(
                    state_spikes, action_spikes, reward, timestamp
                )
                updates_made += num_updates
            
            total_reward += reward
            steps += 1
            timestamp += 0.1  # 100ms between actions
            
            if verbose:
                print(f"Step {steps}: Action={action}, Reward={reward:.3f}, "
                      f"Pos={env.agent_pos}, Distance={env.get_distance_to_target()}")
            
            if done:
                break
            
            state = next_state
        
        # Store episode statistics
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)
        
        # Get learning statistics
        learning_stats = self.r_stdp.get_learning_stats()
        self.learning_stats.append(learning_stats)
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'updates_made': updates_made,
            'final_distance': env.get_distance_to_target(),
            'episode_spikes': episode_spikes,
            'learning_stats': learning_stats
        }


def run_r_stdp_experiment(num_episodes=200, grid_size=6):
    """
    Run R-STDP learning experiment on navigation task.
    
    Args:
        num_episodes: Number of training episodes
        grid_size: Size of the navigation grid
        
    Returns:
        Training results and statistics
    """
    
    print(f"Starting R-STDP Learning Experiment")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Episodes: {num_episodes}")
    print("-" * 50)
    
    # Create environment
    env = SimpleNavigationTask(grid_size=grid_size)
    state_size = env.state_encoding_size
    num_actions = env.num_actions
    
    # Create learning configuration
    learning_config = LearningConfig(
        learning_rate=0.02,
        a_plus=0.15,
        a_minus=0.18,
        tau_plus=0.025,
        tau_minus=0.025,
        eligibility_decay=0.9,
        reward_window=0.2
    )
    
    # Create agent
    agent = SNNRLAgent(state_size, num_actions, learning_config)
    
    # Performance monitoring
    monitor = PerformanceMonitor()
    
    # Training loop
    success_episodes = 0
    exploration_rate = 0.3  # Start with high exploration
    exploration_decay = 0.995
    min_exploration = 0.05
    
    print("Training started...")
    
    for episode in range(num_episodes):
        monitor.start_timer('episode_time')
        
        # Train one episode
        episode_stats = agent.train_episode(
            env, 
            exploration_rate=exploration_rate,
            verbose=(episode % 50 == 0)  # Verbose every 50 episodes
        )
        
        episode_time = monitor.end_timer('episode_time')
        
        # Track success
        if episode_stats['final_distance'] == 0:
            success_episodes += 1
        
        # Decay exploration rate
        exploration_rate = max(min_exploration, 
                             exploration_rate * exploration_decay)
        
        # Print progress
        if episode % 20 == 0:
            recent_rewards = agent.episode_rewards[-20:]
            avg_reward = np.mean(recent_rewards)
            success_rate = success_episodes / (episode + 1) * 100
            
            print(f"Episode {episode:3d}: "
                  f"Reward={episode_stats['total_reward']:6.2f}, "
                  f"Steps={episode_stats['steps']:2d}, "
                  f"Avg Reward={avg_reward:6.2f}, "
                  f"Success Rate={success_rate:5.1f}%, "
                  f"Exploration={exploration_rate:.3f}")
    
    print("\nTraining completed!")
    print(f"Final success rate: {success_episodes/num_episodes*100:.1f}%")
    
    # Analyze results
    analyze_results(agent, env, num_episodes)
    
    return agent, env


def analyze_results(agent, env, num_episodes):
    """Analyze and visualize training results."""
    
    print("\n=== Training Analysis ===")
    
    # Performance statistics
    final_rewards = agent.episode_rewards[-50:]  # Last 50 episodes
    print(f"Average reward (last 50 episodes): {np.mean(final_rewards):.3f}")
    print(f"Best episode reward: {np.max(agent.episode_rewards):.3f}")
    print(f"Average episode length: {np.mean(agent.episode_steps):.1f}")
    
    # Learning statistics
    if agent.learning_stats:
        final_stats = agent.learning_stats[-1]
        print(f"Total reward accumulated: {final_stats['total_reward']:.2f}")
        print(f"Average reward per episode: {final_stats['average_reward']:.3f}")
        print(f"Active eligibility traces: {final_stats['active_eligibility_traces']}")
    
    # Plot training curves
    plot_training_results(agent, num_episodes)
    
    # Test final policy
    test_final_policy(agent, env, num_test_episodes=10)


def plot_training_results(agent, num_episodes):
    """Plot training performance curves."""
    
    episodes = range(len(agent.episode_rewards))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward curve
    axes[0, 0].plot(episodes, agent.episode_rewards, alpha=0.6, label='Episode Reward')
    
    # Moving average
    window = min(20, len(agent.episode_rewards) // 10)
    if window > 1:
        moving_avg = np.convolve(agent.episode_rewards, 
                               np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(agent.episode_rewards)), 
                       moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Learning Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode length
    axes[0, 1].plot(episodes, agent.episode_steps, alpha=0.6)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode Length')
    axes[0, 1].set_title('Episode Length Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning statistics
    if agent.learning_stats:
        total_rewards = [stats['total_reward'] for stats in agent.learning_stats]
        avg_rewards = [stats['average_reward'] for stats in agent.learning_stats]
        
        axes[1, 0].plot(episodes, total_rewards, label='Cumulative Reward')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Cumulative Reward')
        axes[1, 0].set_title('Learning Progress')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(episodes, avg_rewards, label='Average Reward')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].set_title('Average Reward Trend')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('r_stdp_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_final_policy(agent, env, num_test_episodes=10):
    """Test the final learned policy."""
    
    print(f"\n=== Testing Final Policy ({num_test_episodes} episodes) ===")
    
    test_rewards = []
    test_steps = []
    success_count = 0
    
    for episode in range(num_test_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < env.max_steps:
            # Use greedy policy (no exploration)
            action, _, _ = agent.select_action(state, exploration_rate=0.0)
            
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                if env.get_distance_to_target() == 0:
                    success_count += 1
                break
        
        test_rewards.append(total_reward)
        test_steps.append(steps)
        
        print(f"Test Episode {episode+1}: "
              f"Reward={total_reward:.2f}, "
              f"Steps={steps}, "
              f"Success={'Yes' if env.get_distance_to_target() == 0 else 'No'}")
    
    print(f"\nTest Results:")
    print(f"Average reward: {np.mean(test_rewards):.3f} ± {np.std(test_rewards):.3f}")
    print(f"Average steps: {np.mean(test_steps):.1f} ± {np.std(test_steps):.1f}")
    print(f"Success rate: {success_count/num_test_episodes*100:.1f}%")


def deploy_to_fpga_demo(agent, env):
    """
    Demonstrate deployment to FPGA accelerator.
    
    This function shows how the trained R-STDP agent could be deployed
    to the FPGA for real-time inference.
    """
    
    print("\n=== FPGA Deployment Demo ===")
    
    try:
        # Initialize FPGA accelerator
        with SNNAccelerator() as accelerator:
            # Configure network with learned weights
            accelerator.configure_network(
                num_neurons=agent.total_neurons,
                topology={'weights': agent.synapse_map}
            )
            
            # Enable R-STDP learning on FPGA
            accelerator.set_learning_parameters(
                learning_rate=0.02,
                stdp_window=0.025
            )
            accelerator.enable_learning(True)
            
            print("FPGA configured successfully")
            
            # Run a few test episodes on FPGA
            for episode in range(3):
                state = env.reset()
                print(f"\nFPGA Test Episode {episode+1}:")
                
                for step in range(env.max_steps):
                    # Encode state to spikes
                    input_spikes = agent.encoder.encode(state)
                    
                    # Run inference on FPGA
                    output_spikes = accelerator.run_simulation(
                        duration=0.05,
                        input_spikes=input_spikes
                    )
                    
                    # Decode action from output spikes
                    action = decode_action_from_spikes(output_spikes, agent)
                    
                    # Take action
                    state, reward, done = env.step(action)
                    
                    print(f"  Step {step+1}: Action={action}, Reward={reward:.3f}")
                    
                    if done:
                        print(f"  Episode completed in {step+1} steps")
                        if env.get_distance_to_target() == 0:
                            print("  SUCCESS: Reached target!")
                        break
                
    except Exception as e:
        print(f"FPGA not available: {e}")
        print("This demo requires FPGA hardware and bitstream")


def decode_action_from_spikes(output_spikes, agent):
    """Decode action from output spikes."""
    action_start = agent.state_size + agent.hidden_size
    action_counts = np.zeros(agent.num_actions)
    
    for spike in output_spikes:
        if spike.neuron_id >= action_start:
            action_idx = spike.neuron_id - action_start
            if action_idx < agent.num_actions:
                action_counts[action_idx] += 1
    
    if np.sum(action_counts) > 0:
        return np.argmax(action_counts)
    else:
        return np.random.randint(agent.num_actions)


def main():
    """Main function to run R-STDP learning experiment."""
    
    print("R-STDP Learning Example for SNN FPGA Accelerator")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run the main experiment
    agent, env = run_r_stdp_experiment(
        num_episodes=300,
        grid_size=6
    )
    
    # Optional: Demonstrate FPGA deployment
    # deploy_to_fpga_demo(agent, env)
    
    print("\nExperiment completed!")
    print("Check 'r_stdp_training_results.png' for training curves.")


if __name__ == '__main__':
    main()
