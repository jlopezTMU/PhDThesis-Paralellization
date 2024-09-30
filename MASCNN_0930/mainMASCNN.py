import torch
import argparse
from MASModelCNN import ParallelizationModel
from sklearn.model_selection import train_test_split

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
    parser = argparse.ArgumentParser(description="Simulated parallel processing on MNIST using MAS")
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--processors', type=int, default=4, help='Number of simulated processors (nodes) to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.009, metavar='LR', help='Learning rate (default: 0.009)')
    parser.add_argument('--patience_epochs', type=int, default=5, help='Patience for early stopping in epochs (default: 5)')
    parser.add_argument('--latency', type=check_latency_range, default="1,10", help="Latency range in ms as 'x,y' where x < y")

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

    # Convert the dataset to numpy arrays
    X = mnist_dataset.data.numpy().reshape(-1, 1, 28, 28) / 255.0
    y = mnist_dataset.targets.numpy()

    # Split the dataset into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Calculate number of batches per epoch
    batches_per_epoch = len(X_train) // args.batch_size
    if len(X_train) % args.batch_size != 0:
        batches_per_epoch += 1  # Account for the last partial batch

    # Calculate adjusted patience in mini-batches
    adjusted_patience = args.patience_epochs * batches_per_epoch
    args.patience = adjusted_patience

    # Print the calculated patience for confirmation
    print(f"Calculated patience: {args.patience} mini-batches (equivalent to {args.patience_epochs} epochs)")

    # Initialize the Parallelization Model
    model = ParallelizationModel(X_train, y_train, X_test, y_test, device, args)

    # Run the simulation
    model.step()

if __name__ == '__main__':
    main()
