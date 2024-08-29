import torch
import argparse
from MASModelCNN import ParallelizationModel
from sklearn.model_selection import train_test_split  # Import train_test_split for dataset splitting

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="Simulated parallel processing on MNIST using MAS")
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--processors', type=int, default=4, help='Number of simulated processors (nodes) to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='Learning rate (default: 0.01)')

    args = parser.parse_args()

    # Check if GPU is available and requested
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("***Using device: GPU***")
    else:
        device = torch.device("cpu")
        print("***Using device: CPU***")

    # Load MNIST dataset
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Convert the dataset to numpy arrays, keeping the 2D shape for convolution operations
    X = mnist_dataset.data.numpy().reshape(-1, 1, 28, 28) / 255.0
    y = mnist_dataset.targets.numpy()

    # Split the dataset into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Adjust behavior based on the number of processors
    if args.processors == 1:
        # Use the entire dataset for training (48000 examples) when only one processor is specified
        X_train = X_train
        y_train = y_train

    # Initialize the Parallelization Model
    model = ParallelizationModel(X_train, y_train, X_test, y_test, device, args)

    # Run the simulation for a single step (can be extended to multiple steps)
    model.step()

if __name__ == '__main__':
    main()
