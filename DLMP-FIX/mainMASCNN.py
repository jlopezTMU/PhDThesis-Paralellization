import argparse
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from MASModelCNN import ParallelizationModel
from trainMASCNN import get_vgg11_model

def check_latency_range(value):
    try:
        x, y = map(int, value.split(','))
        if x > y:
            raise argparse.ArgumentTypeError("Latency range should be in format 'x,y' where x <= y")
        return x, y
    except ValueError:
        raise argparse.ArgumentTypeError("Latency range must be two integers separated by a comma.")

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent CNN Training Simulation")
    parser.add_argument('--gpu', action='store_true', help="Use GPU if available")
    parser.add_argument('--processors', type=int, required=True, help="Number of processing nodes")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--ds', type=str, choices=['MNIST', 'CIFAR-10'], required=True, help="Dataset")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping")
    parser.add_argument('--latency', type=check_latency_range, default=(0, 0), help="Latency range in ms as 'x,y'")

    args = parser.parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"***Using device: {'GPU' if device.type == 'cuda' else 'CPU'}***")

    if args.ds == "MNIST":
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    elif args.ds == "CIFAR-10":
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    X_train = np.array([np.array(image[0]) for image in dataset])
    y_train = np.array([label for _, label in dataset])

    X_test = np.array([np.array(image[0]) for image in test_dataset])
    y_test = np.array([label for _, label in test_dataset])

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = ParallelizationModel(X_train=X_train, y_train=y_train,
                                 X_test=X_test, y_test=y_test,
                                 device=device, args=args)

    model.step()

if __name__ == "__main__":
    main()

