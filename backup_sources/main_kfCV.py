# This version performs n CV according to --folds for num_processors is 1, it performs regular CV w/o parallelizing it
# Now it will be modified to perform only 80-20 training without parallelized CV

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

from trainCV_nfv import train

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
def parallel_kfold():
    kf = KFold(n_splits=args.folds)
    start_time = time.time()  # Start the timer before initializing the pool

    with Pool(processes=args.processors) as pool:
        results = pool.starmap(train, [(fold, train_idx, val_idx, X, y, device, args) for fold, (train_idx, val_idx) in enumerate(kf.split(X))])

    end_time = time.time()  # Stop the timer after the processing finishes

    end__time = end_time - start_time
    elapsed_time = end_time - start_time  # Calculate the total elapsed time for all processing across all processors

    torch.cuda.empty_cache()

    return results, elapsed_time

def test_model(best_model, X_test, y_test, device):
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.int64).to(device)
        outputs = best_model(X_test)
        test_accuracy = (outputs.argmax(1) == y_test).float().mean().item()
    return test_accuracy

def eighty_twenty_split(X_train, X_test, y_train, y_test, device, args):
    # Initialize the model and set it to the appropriate device
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Convert dataset to PyTorch tensors and transfer to the appropriate device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64).to(device)

    # Training process
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=args.batch_size,
        shuffle=True
    )

    start_time_train = time.time()  # Start timing the training process
    for epoch in range(args.epochs):
        model.train()  # Ensure the model is in training mode
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(batch_X)  # Forward pass
            loss = criterion(outputs, batch_y)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

    elapsed_time_train = time.time() - start_time_train  # Calculate training time

    # Evaluate training accuracy
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        correct_preds = (train_outputs.argmax(1) == y_train_tensor).type(torch.float)
        train_accuracy = correct_preds.mean().item()

    # Testing process
    start_time_test = time.time()  # Start timing the testing process
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        correct_preds = (test_outputs.argmax(1) == y_test_tensor).type(torch.float)
        test_accuracy = correct_preds.mean().item()
    elapsed_time_test = time.time() - start_time_test  # Calculate testing time

    torch.cuda.empty_cache()

    # Print results
    print(f"Training time: {elapsed_time_train:.4f} seconds")
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing accuracy: {test_accuracy * 100:.2f}%")

def main():
    print("*FIRST THING I WILL DO IT IS TO CLEAR CUDA CACHE AND COLLECT GARBAGE*")
    torch.cuda.empty_cache()
    gc.collect()
    # Split the data into training and testing (80-20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"*** The dataset has been split in 80% training with {len(y_train)} records and 20% testing with {len(y_test)} records ***")

    ### results, total_elapsed_time = parallel_kfold()  # Receive the total elapsed time

    if args.folds == 0:
        torch.cuda.empty_cache()
        eighty_twenty_split(X_train, X_test, y_train, y_test, device, args)
    elif args.processors >= 2:
        kf = KFold(n_splits=args.processors)  # change this line to set the number of splits dynamically

        torch.cuda.empty_cache()
        results, total_elapsed_time = parallel_kfold()  # Receive the total elapsed time
        torch.cuda.empty_cache()

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
    else:
###     *here starts block
        # Perform the regular testing process
        model = LeNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        # Declare the best_model and best_train_accuracy here
        best_model = None
        best_train_accuracy = 0.0

        # Convert tensors and move to device
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.int64).to(device)

        start_time_train = time.time()
        kf = KFold(n_splits=args.folds)
        train_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):

            X_train_fold, X_val_fold = X_train[train_idx].clone().detach().to(device), X_train[val_idx].clone().detach().to(device)
            y_train_fold, y_val_fold = y_train[train_idx].clone().detach().to(device), y_train[val_idx].clone().detach().to(device)

            train_dataset = torch.utils.data.TensorDataset(X_train_fold, y_train_fold)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            for epoch in range(args.epochs):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                train_outputs = model(X_train_fold)
                train_accuracy = (train_outputs.argmax(1) == y_train_fold).float().mean().item()
                train_accuracies.append(train_accuracy)

                if train_accuracy > best_train_accuracy:
                    best_train_accuracy = train_accuracy
                    best_model = model.state_dict()

        end_time_train = time.time()
        elapsed_time_train = end_time_train - start_time_train

        start_time_test = time.time()

        # Load the best model and then test
        model.load_state_dict(best_model)
        test_accuracy = test_model(model, X_test, y_test, device)
        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)

        print(f">>> (1 processor with {args.folds} fCV) Training accuracy using the model: {avg_train_accuracy * 100:.4f}%")

        end_time_test = time.time()
        elapsed_time_test = end_time_test - start_time_test

        print(f"(1 processor) Total elapsed time for training: {elapsed_time_train:.4f} seconds")
        print(f"(1 processor) Total elapsed time for testing: {elapsed_time_test:.4f} seconds")
        print(f">>> Testing accuracy using best average cross-validation model: {test_accuracy * 100:.4f}%")

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
    main()
