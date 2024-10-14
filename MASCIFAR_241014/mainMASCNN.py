import torch
import argparse
from MASModelCNN import ParallelizationModel
from sklearn.model_selection import train_test_split  # Import train_test_split for dataset splitting
from torchvision import datasets, transforms  # Import necessary libraries

def check_latency_range(value):
    # Split the input string into two parts and validate them as positive integers
    try:
        x, y = map(int, value.split(','))
        if x < 0 or y < 0:
            raise argparse.ArgumentTypeError("Both x and y must be positive integers.")
        if x >= y:
            raise argparse.ArgumentTypeError("x must be less than y.")
        return x, y
    except ValueError:
        raise argparse.ArgumentTypeError("Latency range must be in the format 'x,y' where x and y are integers.")

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="Simulated parallel processing on CIFAR-10 using MAS")
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--processors', type=int, default=4, help='Number of simulated processors (nodes) to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='Learning rate (default: 0.01)')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping (default: 5)')
    parser.add_argument('--latency', type=check_latency_range, default="1,10", help="Latency range in ms as 'x,y' where x < y")

    args = parser.parse_args()

    # Check if GPU is available and requested
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("***Using device: GPU***")
    else:
        device = torch.device("cpu")
        print("***Using device: CPU***")

    # CIFAR-10 dataset loading with normalization and data augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally for data augmentation
        transforms.RandomCrop(32, padding=4),  # Randomly crop images to introduce variation
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # Normalize with CIFAR-10's mean and std
    ])

    cifar10_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

    # Convert the dataset to numpy arrays, keeping the 2D shape for convolution operations
    X = cifar10_dataset.data  # Use the CIFAR-10 data array directly
    y = cifar10_dataset.targets

    # Split the dataset into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if args.processors == 1:
        # Use the entire dataset for training when only one processor is specified
        X_train = X_train
        y_train = y_train

    # Initialize the Parallelization Model
    model = ParallelizationModel(X_train, y_train, X_test, y_test, device, args)

    # Run the simulation for a single step (can be extended to multiple steps)
    model.step()

if __name__ == '__main__':
    main()

