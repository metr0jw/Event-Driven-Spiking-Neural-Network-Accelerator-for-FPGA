"""
PyTorch MNIST Training Example for SNN FPGA Accelerator

This example shows how to train a SNN model in PyTorch and then
deploy it to the FPGA accelerator for inference.

Author: Jiwoon Lee (@metr0jw)
Organization: Kwangwoon University, Seoul, South Korea
Contact: jwlee@linux.com
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import sys

# Add the software package to path
sys.path.append('../../software/python')

from snn_fpga_accelerator import (
    SNNAccelerator, pytorch_to_snn, PoissonEncoder, 
    SNNModel, create_feedforward_snn
)
from snn_fpga_accelerator.pytorch_interface import TorchSNNLayer, spike_function


class SpikingMLP(nn.Module):
    """
    Spiking Multi-Layer Perceptron for MNIST classification.
    
    This model uses PyTorch-compatible spiking layers that can be
    trained with backpropagation and then converted to run on FPGA.
    """
    
    def __init__(self, input_size=784, hidden_sizes=[256, 128], num_classes=10,
                 tau_mem=20.0, tau_syn=5.0, threshold=1.0):
        super(SpikingMLP, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_timesteps = 100  # Number of time steps for simulation
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(TorchSNNLayer(
                in_features=prev_size,
                out_features=hidden_size,
                tau_mem=tau_mem,
                tau_syn=tau_syn,
                threshold=threshold
            ))
            prev_size = hidden_size
        
        # Output layer
        layers.append(TorchSNNLayer(
            in_features=prev_size,
            out_features=num_classes,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            threshold=threshold
        ))
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        """
        Forward pass through the spiking network.
        
        Args:
            x: Input tensor (batch_size, channels, height, width) or (batch_size, features)
            
        Returns:
            Output spikes aggregated over time
        """
        batch_size = x.size(0)
        
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(batch_size, -1)
        
        # Reset all layer states
        for layer in self.layers:
            layer.reset_state()
        
        # Simulate over time steps
        output_spikes = torch.zeros(batch_size, self.num_classes, device=x.device)
        
        for t in range(self.num_timesteps):
            # Convert input to spikes (Poisson encoding)
            spike_input = torch.rand_like(x) < (x * 0.1)  # Poisson approximation
            current_activity = spike_input.float()
            
            # Forward through layers
            for layer in self.layers:
                current_activity = layer(current_activity)
            
            # Accumulate output spikes
            output_spikes += current_activity
        
        return output_spikes
    
    def reset_states(self):
        """Reset all neuron states."""
        for layer in self.layers:
            layer.reset_state()


def load_mnist_data(batch_size=128, download=True):
    """Load and preprocess MNIST dataset."""
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=download, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=download, transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001,
                device='cpu', save_path='snn_mnist_model.pth'):
    """
    Train the spiking neural network model.
    
    Args:
        model: SNN model to train
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_path: Path to save the trained model
    """
    
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training history
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"Training SNN model on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Training loop
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct_train/total_train:.2f}%'
            })
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct_train / total_train
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Test evaluation
        test_acc = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        # Update learning rate
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')
        print(f'Test Acc: {test_acc:.2f}%')
        print('-' * 50)
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'model_config': {
            'input_size': model.input_size,
            'num_classes': model.num_classes,
            'hidden_sizes': [layer.out_features for layer in model.layers[:-1]],
        }
    }, save_path)
    
    print(f'Model saved to {save_path}')
    
    # Plot training curves
    plot_training_curves(train_losses, train_accuracies, test_accuracies)
    
    return model, train_losses, test_accuracies


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def plot_training_curves(train_losses, train_accuracies, test_accuracies):
    """Plot training curves."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def convert_and_deploy_to_fpga(model_path, test_loader, device='cpu'):
    """
    Convert trained PyTorch model to SNN format and deploy to FPGA.
    
    Args:
        model_path: Path to saved PyTorch model
        test_loader: Test data loader for validation
        device: Device for PyTorch operations
    """
    
    print("Converting PyTorch model to SNN format...")
    
    # Load the trained model
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    # Recreate model
    model = SpikingMLP(
        input_size=model_config['input_size'],
        hidden_sizes=model_config['hidden_sizes'],
        num_classes=model_config['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Convert to SNN format
    snn_model = pytorch_to_snn(
        model, 
        input_shape=(784,),
        conversion_params={
            'weight_scale': 128.0,
            'threshold_scale': 1.0,
            'leak_rate': 0.1,
            'refractory_period': 5
        }
    )
    
    # Save SNN weights
    snn_model.save_weights('snn_mnist_weights.h5')
    print("SNN weights saved to snn_mnist_weights.h5")
    
    # Initialize FPGA accelerator (if available)
    try:
        with SNNAccelerator() as accelerator:
            # Load bitstream and configure network
            accelerator.configure_network(
                num_neurons=snn_model.total_neurons,
                topology={'weights': snn_model.layers[0].weights}
            )
            
            # Test FPGA inference on a few samples
            print("Testing FPGA inference...")
            test_fpga_inference(accelerator, snn_model, test_loader, num_samples=10)
            
    except Exception as e:
        print(f"FPGA not available or error occurred: {e}")
        print("Running software simulation instead...")
        
        # Test software simulation
        test_software_inference(snn_model, test_loader, num_samples=10)


def test_fpga_inference(accelerator, snn_model, test_loader, num_samples=10):
    """Test inference on FPGA accelerator."""
    
    correct = 0
    total = 0
    
    # Get first batch
    data_iter = iter(test_loader)
    data, targets = next(data_iter)
    
    # Test on limited samples
    for i in range(min(num_samples, len(data))):
        image = data[i].numpy().squeeze()
        target = targets[i].item()
        
        # Encode image to spikes
        encoder = PoissonEncoder(num_neurons=784, duration=0.1, max_rate=100)
        input_spikes = encoder.encode(image.flatten())
        
        # Run inference on FPGA
        output_spikes = accelerator.run_simulation(
            duration=0.1,
            input_spikes=input_spikes
        )
        
        # Decode output
        output_counts = np.zeros(10)
        for spike in output_spikes:
            if spike.neuron_id < 10:  # Assuming last 10 neurons are output
                output_counts[spike.neuron_id] += 1
        
        predicted = np.argmax(output_counts)
        
        print(f"Sample {i}: Target={target}, Predicted={predicted}, "
              f"Correct={predicted==target}")
        
        if predicted == target:
            correct += 1
        total += 1
    
    accuracy = 100.0 * correct / total
    print(f"FPGA Inference Accuracy: {accuracy:.2f}% ({correct}/{total})")


def test_software_inference(snn_model, test_loader, num_samples=10):
    """Test software simulation of SNN model."""
    
    correct = 0
    total = 0
    
    # Get first batch
    data_iter = iter(test_loader)
    data, targets = next(data_iter)
    
    for i in range(min(num_samples, len(data))):
        image = data[i].numpy().squeeze()
        target = targets[i].item()
        
        # Encode image to spikes
        encoder = PoissonEncoder(num_neurons=784, duration=0.1, max_rate=100)
        input_spikes = encoder.encode(image.flatten())
        
        # Run software simulation
        from snn_fpga_accelerator.pytorch_interface import simulate_snn_inference
        output_spikes = simulate_snn_inference(snn_model, input_spikes, duration=0.1)
        
        # Decode output (simplified)
        output_counts = np.zeros(10)
        for spike in output_spikes:
            output_layer_start = snn_model.total_neurons - 10
            if spike.neuron_id >= output_layer_start:
                output_idx = spike.neuron_id - output_layer_start
                output_counts[output_idx] += 1
        
        predicted = np.argmax(output_counts) if np.sum(output_counts) > 0 else 0
        
        print(f"Sample {i}: Target={target}, Predicted={predicted}, "
              f"Correct={predicted==target}")
        
        if predicted == target:
            correct += 1
        total += 1
    
    accuracy = 100.0 * correct / total
    print(f"Software Simulation Accuracy: {accuracy:.2f}% ({correct}/{total})")


def main():
    parser = argparse.ArgumentParser(description='PyTorch SNN MNIST Training')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model-path', type=str, default='snn_mnist_model.pth',
                        help='path to save/load model')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train the model')
    parser.add_argument('--deploy', action='store_true', default=False,
                        help='convert and deploy to FPGA')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_mnist_data(batch_size=args.batch_size)
    
    if args.train:
        # Create and train model
        model = SpikingMLP(
            input_size=784,
            hidden_sizes=[256, 128],
            num_classes=10,
            tau_mem=20.0,
            tau_syn=5.0,
            threshold=1.0
        )
        
        print("Model architecture:")
        print(model)
        
        # Train the model
        trained_model, train_losses, test_accuracies = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            device=device,
            save_path=args.model_path
        )
        
        print(f"Best test accuracy: {max(test_accuracies):.2f}%")
    
    if args.deploy:
        # Convert and deploy to FPGA
        if os.path.exists(args.model_path):
            convert_and_deploy_to_fpga(args.model_path, test_loader, device)
        else:
            print(f"Model file {args.model_path} not found. Train the model first.")


if __name__ == '__main__':
    main()
