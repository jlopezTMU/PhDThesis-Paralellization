import torch
import argparse
from MASModelCNN import ParallelizationModel
from sklearn.model_selection import train_test_split  # Import train_test_split for dataset splitting
from torchvision import datasets, transforms  # Import necessary libraries

def check_latency_range(value):
    # Split the input string into two parts and validate them as positive integers
    try:
        x, y = map(float, value.split(','))
        if x < 0 or y < 0:
            raise argparse.ArgumentTypeError("Both x and y must be positive integers.")
        if x >= y:
            raise argparse.ArgumentTypeError("x must be less than y.")
        return x, y
    except ValueError:
        raise argparse.ArgumentTypeError("Latency range must be in the format 'x,y' where x and y are integers.")

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="Simulated parallel processing on MNIST using MAS")
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--processors', type=int, default=4, help='Number of simulated processors (nodes) to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='Learning rate (default: 0.01)')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping (default: 5)')
    parser.add_argument('--latency', type=check_latency_range, default="1,10", help="Latency range in ms as 'x,y' where x < y")
    parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'ADAMW', 'SGDM', 'RMSP'],
                        help='Optimizer to use (default: ADAM)')
    parser.add_argument('--loss', type=str, default='CE', choices=['CE', 'LSCE', 'FC', 'WCE'],
                        help='Loss function to use (default: CE)')

    # Added activation argument
    parser.add_argument('--activation', type=str, default='RELU', choices=['RELU', 'LEAKY_RELU', 'ELU', 'SELU', 'GELU', 'MISH'],
                        help='Activation function to use (default: RELU)')  # Added line

    args = parser.parse_args()

    # Check if GPU is available and requested
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("***Using device: GPU***")
    else:
        device = torch.device("cpu")
        print("***Using device: CPU***")

    # MNIST dataset loading with normalization
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST's mean and std
        transforms.Lambda(lambda x: x.expand(3, -1, -1))  # Expand grayscale images to 3 channels for VGG compatibility
    ])

    mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Convert the dataset to numpy arrays, keeping the 2D shape for convolution operations
    X = mnist_dataset.data.numpy()  # Use the MNIST data array directly
    y = mnist_dataset.targets.numpy()

    # Split the dataset into training (80%) and testing (20%)
    Training_ds, Testing_ds, Training_lbls, Testing_lbls = train_test_split(X, y, test_size=0.2, random_state=42)

    if args.processors == 1:
        # Use the entire dataset for training when only one processor is specified
        Training_ds = Training_ds
        Training_lbls = Training_lbls

    # Initialize the Parallelization Model
    model = ParallelizationModel(Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args)

    # Run the simulation for a single step (can be extended to multiple steps)
    model.step()

if __name__ == '__main__':
    main()
