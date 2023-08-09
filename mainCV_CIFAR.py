import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from torch.multiprocessing import Pool, set_start_method
import argparse
import time
import os

from trainCV_CIFAR import train

from sklearn.model_selection import train_test_split

import numpy as np


# Necessary for CUDA multiprocessing
set_start_method('spawn', force=True)

##
def parallel_kfold():
    kf = KFold(n_splits=args.folds)
    start_time = time.time()  # Start the timer before initializing the pool

    with Pool(processes=args.processors) as pool:
        results = pool.starmap(train, [(fold, train_idx, val_idx, X, y, device, args) for fold, (train_idx, val_idx) in enumerate(kf.split(X))])

    end_time = time.time()  # Stop the timer after the processing finishes

    end__time = end_time - start_time
    elapsed_time = end_time - start_time  # Calculate the total elapsed time for all processing across all processors

    return results, elapsed_time

def test_model(best_model, X_test, y_test, device):
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.int64).to(device)
        outputs = best_model(X_test)
        test_accuracy = (outputs.argmax(1) == y_test).float().mean().item()
    return test_accuracy

def main():

    # Split the data into training and testing (80-20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"*** The dataset has been split in 80% training with {len(y_train)} records and 20% testing with {len(y_test)} records ***")
    results, total_elapsed_time = parallel_kfold()  # Receive the total elapsed time

    losses = [result[0] for result in results]
    accuracies = [result[1] for result in results]
    best_model_idx = accuracies.index(max(accuracies))
    best_model = results[best_model_idx][2]

    test_accuracy = test_model(best_model, X_test, y_test, device)

    #
    print(f"*** Total processing time for all folds across for all {args.processors} processors:\
            and total time is {total_elapsed_time:4f} seconds \
            Average validation accuracy across folds: {sum(accuracies) / len(accuracies) * 100:.4f}%")

    print(f">>> Testing accuracy using best model: {test_accuracy * 100:.4f}%")

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description="Parallel k-fold cross-validation on MNIST")
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--processors', type=int, default=2, help='Number of processors to use for parallel processing')
    parser.add_argument('--folds', type=int, default=10, help='Number of k-folds for cross-validation')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')

    args = parser.parse_args()

    # Set device to GPU if available and requested
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cifar10_dataset = datasets.CIFAR10(root='./data_cifar', train=True, transform=transform, download=True)
    # Convert the dataset to numpy arrays, but this time keep the 2D shape for convolution operations
    X = np.transpose(cifar10_dataset.data, (0, 3, 1, 2)) / 255.0
    y = np.array(cifar10_dataset.targets)
    main()
