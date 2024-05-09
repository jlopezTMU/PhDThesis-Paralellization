# This version performs n CV according to --folds for num_processors is 1
# AUTHOR. JORGE LOPEZ

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from torch.multiprocessing import Pool, set_start_method
import argparse
import time
import os
import gc
import GPUtil

from trainCV_DEBUG import train

from sklearn.model_selection import train_test_split

# Necessary for CUDA multiprocessing
set_start_method('spawn', force=True)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(nn.MaxPool2d(2)(self.conv1(x)))
        x = torch.relu(nn.MaxPool2d(2)(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
##
def chunked_data(X, y, num_chunks):
    chunk_size = len(X) // num_chunks
    for i in range(0, len(X), chunk_size):
        chunk_X = X[i:i+chunk_size]
        chunk_y = y[i:i+chunk_size]
        yield chunk_X, chunk_y

def parallel_kfold(kf):
    # NO! kf = KFold(n_splits=args.folds)

    data_chunks = [(fold, train_idx, val_idx) for fold, (train_idx, val_idx) in enumerate(chunked_data(X, y, args.processors))]

    ##print("Data chunks assigned to processors:")
    ##for processor, (fold, train_idx, val_idx) in enumerate(data_chunks):
        ##print(f"Processor {processor + 1} (PID {os.getpid()}): Fold {fold + 1}, Train indices from {train_idx[0]} to {train_idx[-1]}, Val indices from {val_idx[0]} to {val_idx[-1]}")
       ##   print(f"Processor {processor + 1} (PID {os.getpid()}): Fold {fold + 1}, Val indices from {val_idx[0]} to {val_idx[-1]}")
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

def aggregate_model_parameters(models):
    # Aggregate model parameters by averaging
    aggregated_params = {name: torch.stack([model_state[name] for model_state in models]).mean(0)
                         for name in models[0].keys()}
    return aggregated_params

def main():
    print("*FIRST THING I WILL DO IT IS TO CLEAR CUDA CACHE AND COLLECT GARBAGE*")
    torch.cuda.empty_cache()
    gc.collect()

    kf = KFold(n_splits=args.folds)
    # Split the data into training and testing (80-20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"*** The dataset has been split into 80% training with {len(y_train)} records and 20% testing with {len(y_test)} records ***")

    # Distribute data chunks among processors
    data_chunks = [(fold, train_idx, val_idx) for fold, (train_idx, val_idx) in enumerate(chunked_data(X_train, y_train, args.processors))]

    start_time = time.time()  # Start the timer

    with Pool(processes=args.processors) as pool:
        results = pool.starmap(train, [(fold, train_idx, val_idx, X_train, y_train, device, args) for fold, (train_idx, val_idx) in enumerate(kf.split(X_train))])

    end_time = time.time()  # Stop the timer after the processing finishes

    total_elapsed_time = end_time - start_time  # Calculate the total elapsed time for all processing across all processors

    accuracies = [result[1] for result in results]

    # Extract models and aggregate model parameters
    models = [result[2] for result in results]
    aggregated_params = aggregate_model_parameters(models)

    # Update global model with aggregated parameters
    global_model.load_state_dict(aggregated_params)

    # Test the best model on the test dataset
    test_accuracy = test_model(global_model, X_test, y_test, device)

    print(f"*** Total processing time for all folds across all processors: {total_elapsed_time:.4f} seconds ***")
    print(f"*** Average validation accuracy across folds: {sum(accuracies) / len(accuracies) * 100:.4f}% ***")
    print(f">>> Testing accuracy using the best model: {test_accuracy * 100:.4f}%")
###     *here ends print
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

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Convert the dataset to numpy arrays, but this time keep the 2D shape for convolution operations
    X = mnist_dataset.data.numpy().reshape(-1, 1, 28, 28) / 255.0
    y = mnist_dataset.targets.numpy()

  # Initialize global model
    global_model = LeNet().to(device)

    main()
