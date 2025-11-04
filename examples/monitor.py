# snn_accelerator/monitor.py
"""
Real-time monitoring and visualization of SNN activity

Author: Jiwoon Lee (@metr0jw)
Organization: Kwangwoon University, Seoul, South Korea
Contact: jwlee@linux.com
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np

class SNNMonitor:
    """Real-time monitoring of SNN activity"""
    
    def __init__(self, network: SNNNetwork, window_size: int = 1000):
        self.network = network
        self.window_size = window_size
        
        # Data buffers
        self.spike_times = deque(maxlen=window_size)
        self.spike_neurons = deque(maxlen=window_size)
        self.spike_rates = {}
        
        # Plotting
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
    def update_plot(self, frame):
        """Update plot with latest spike data"""
        # Get new spikes
        new_spikes = self.network.fpga.get_output_spikes(timeout=0.01)
        
        for spike in new_spikes:
            self.spike_times.append(spike.timestamp / 1000.0)  # Convert to ms
            self.spike_neurons.append(spike.neuron_id)
        
        # Clear and redraw
        self.ax1.clear()
        self.ax2.clear()
        
        # Raster plot
        if self.spike_times:
            self.ax1.scatter(self.spike_times, self.spike_neurons, 
                           c='black', s=1, alpha=0.5)
            self.ax1.set_xlabel('Time (ms)')
            self.ax1.set_ylabel('Neuron ID')
            self.ax1.set_title('Spike Raster Plot')
            
        # Spike rate histogram
        if len(self.spike_neurons) > 0:
            unique, counts = np.unique(list(self.spike_neurons), return_counts=True)
            self.ax2.bar(unique, counts)
            self.ax2.set_xlabel('Neuron ID')
            self.ax2.set_ylabel('Spike Count')
            self.ax2.set_title('Spike Count Distribution')
        
        plt.tight_layout()
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        ani = animation.FuncAnimation(self.fig, self.update_plot, 
                                    interval=100, blit=False)
        plt.show()

# Usage
network = SNNNetwork(fpga_ip="192.168.1.100")
network.connect()
monitor = SNNMonitor(network)
monitor.start_monitoring()
