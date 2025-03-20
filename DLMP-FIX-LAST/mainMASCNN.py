import argparse
import torch
import numpy as np
from torchvision import datasets, transforms
from MASModelCNN import ParallelizationModel

def main():
    parser = argparse.ArgumentParser(description="MAS CNN Simulation")
    parser.add_argument('--gpu', action='store_true', help="Use GPU if available")
    parser.add_argument('--ds', type=str, required=True, help="Dataset to use: MNIST or CIFAR-10")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=0.0070, help="Learning rate")
    parser.add_argument('--patience', type=int, default=99, help="Patience for early stopping")
    parser.add_argument('--processors', type=int, default=1, help="Number of processing nodes")
    parser.add_argument('--latency', type=str, default="0,1",
                        help="Latency range in ms as 'min,max'")
    args = parser.parse_args()

    # Determine device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"***Using device: {device}***")

    # Convert latency argument from string "m,n" to a tuple of integers (m, n)
    try:
        latency_parts = args.latency.split(',')
        args.latency = (int(latency_parts[0]), int(latency_parts[1]))
    except Exception as e:
        raise ValueError("Latency must be specified as two integers separated by a comma, e.g. '0,1'") from e

    # Set architecture based on dataset
    if args.ds.upper() == "MNIST":
        args.arch = "LeNet5"
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        # MNIST data is in PIL Image format for train_dataset.data is a Tensor already?
        # The torchvision MNIST returns train_dataset.data as a tensor.
        X_train = train_dataset.data.numpy()
        y_train = train_dataset.targets.numpy()
        X_test = test_dataset.data.numpy()
        y_test = test_dataset.targets.numpy()
    elif args.ds.upper() == "CIFAR-10":
        args.arch = "VGG11"
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        # For CIFAR-10, the images are PIL Images but the transform converts them to tensors.
        # We convert them to numpy arrays of shape (N, 3, H, W)
        X_train = np.stack([img.numpy() for img, _ in train_dataset])
        y_train = np.array([label for _, label in train_dataset])
        X_test = np.stack([img.numpy() for img, _ in test_dataset])
        y_test = np.array([label for _, label in test_dataset])
    else:
        raise ValueError("Dataset not supported. Use MNIST or CIFAR-10")

    # Create simulation model (ParallelizationModel) using the dataset arrays
    model = ParallelizationModel(Training_ds=X_train,
                                 Training_lbls=y_train,
                                 Testing_ds=X_test,
                                 Testing_lbls=y_test,
                                 device=device,
                                 args=args)

    # Run simulation steps (the scheduler calls each agent's step)
    # Here we run one global step; adjust as needed for your simulation.
    model.step()

if __name__ == '__main__':
    main()

