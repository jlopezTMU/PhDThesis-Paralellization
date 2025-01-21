import torch
import argparse
import numpy as np
from MASModelCNN import ParallelizationModel
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

def check_latency_range(value):
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
    parser = argparse.ArgumentParser(description="Simulated parallel processing on MNIST/CIFAR10 using MAS")
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--processors', type=int, default=4, help='Number of simulated processors (nodes) to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='Learning rate (default: 0.01)')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping (default: 5)')
    parser.add_argument('--latency', type=check_latency_range, default="1,10",
                        help="Latency range in ms as 'x,y' where x < y")
    parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'ADAMW', 'SGDM', 'RMSP'],
                        help='Optimizer to use (default: ADAM)')
    parser.add_argument('--loss', type=str, default='CE', choices=['CE', 'LSCE', 'FC', 'WCE'],
                        help='Loss function to use (default: CE)')
    parser.add_argument('--activation', type=str, default='RELU',
                        choices=['RELU', 'LEAKY_RELU', 'ELU', 'SELU', 'GELU', 'MISH'],
                        help='Activation function to use (default: RELU)')

    # New argument for dataset selection (MNIST or CIFAR10)
    parser.add_argument('--ds', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'],
                        help='Dataset to use: MNIST or CIFAR10 (default: MNIST)')

    args = parser.parse_args()

    # Decide on device
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("***Using device: GPU***")
    else:
        device = torch.device("cpu")
        print("***Using device: CPU***")

    # -------------------------------------------------------
    # Load dataset based on --ds argument
    # -------------------------------------------------------
    if args.ds == 'MNIST':
        transform_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.expand(3, -1, -1))  # Expand grayscale to 3 channels
        ])
        mnist_dataset = datasets.MNIST(root='./data', train=True,
                                       transform=transform_mnist, download=True)
        # Convert the dataset to numpy arrays
        X = mnist_dataset.data.numpy()  # shape: (N, 28, 28)
        y = mnist_dataset.targets.numpy()

    else:  # args.ds == 'CIFAR10'
        transform_cifar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        cifar_dataset = datasets.CIFAR10(root='./data', train=True,
                                         transform=transform_cifar, download=True)
        X = cifar_dataset.data  # shape: (50000, 32, 32, 3)
        y = np.array(cifar_dataset.targets)

        # FIX: Transpose to channels-first (N, C, H, W)
        X = np.transpose(X, (0, 3, 1, 2))  # shape becomes (50000, 3, 32, 32)

    # -------------------------------------------------------
    # Split the dataset into training (80%) and testing (20%)
    # -------------------------------------------------------
    Training_ds, Testing_ds, Training_lbls, Testing_lbls = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # If only one processor, just keep the entire training set
    if args.processors == 1:
        Training_ds = Training_ds
        Training_lbls = Training_lbls

    # -------------------------------------------------------
    # Initialize the Parallelization Model and run the step
    # -------------------------------------------------------
    model = ParallelizationModel(Training_ds, Training_lbls,
                                 Testing_ds, Testing_lbls,
                                 device, args)

    model.step()

if __name__ == '__main__':
    main()
