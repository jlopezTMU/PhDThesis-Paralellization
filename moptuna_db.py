# This version performs n CV according to --folds for num_processors is 1
# This version integrates OPTUNA to find the best hyperparameters, including folds number

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from torch.multiprocessing import Pool, set_start_method
import argparse
import time
import os

from toptuna_nfv import train

from sklearn.model_selection import train_test_split

## imports for Optuna
import os
import optuna
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend which is GUI-less
import matplotlib.pyplot as plt
import numpy as np

from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_contour
#

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

# Function to create a study and optimize hyperparameters
def optimize_hyperparameters(trial, X, y, device, args):
    # Define the hyperparameters to be tuned
    batch_size = trial.suggest_categorical('batch_size', [ 32, 64, 128, 256, 512 ])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    epochs = trial.suggest_int('epochs', 10, 50)
    folds = trial.suggest_int('folds', 3, 10)  # You can adjust the range as needed
    processors = trial.suggest_categorical('processors', [2, 4, 6, 8])  # Adjust this list based on your requirements
    # Update args with the suggested values
    args.batch_size = batch_size
    args.lr = lr
    args.epochs = epochs
    args.folds = folds  # Updating the folds
    args.processors = processors  # Updating the processors

    # Use the updated args in the training function
    results, _ = parallel_kfold(X, y, device, args)

    # Calculate the average accuracy across folds
    accuracies = [result[1] for result in results]
    average_accuracy = sum(accuracies) / len(accuracies)

    return average_accuracy

def parallel_kfold(X, y, device, args):
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

# Optuna study for hyperparameter optimization
    ##study = optuna.create_study(direction='maximize')
    study = optuna.create_study(
    direction='maximize',
    study_name='myresearch_study',  # Name your study
    storage='postgresql://jlopez:dbpass@localhost/optuna_study',  # Use your PostgreSQL credentials and database name
    load_if_exists=True)

    n_trials = args.trials ## set num of trials
    study.optimize(lambda trial: optimize_hyperparameters(trial, X_train, y_train, device, args), n_trials) ## SET NUMBER OF TRIALS!

    # Best hyperparameters
    print("Best hyperparameters: ", study.best_params)

    # Ensure you have a headless backend for Matplotlib if you're on a server without GUI
    plt.switch_backend('agg')

    # Plotting optimization history
    ax = plot_optimization_history(study)
    fig = ax.figure
    fig.set_size_inches(8, 6)  # Set the dimensions after getting the figure
    fig.savefig("optimization_history.png")

    # Plotting parameter importances
    ax = plot_param_importances(study)
    fig = ax.figure
    fig.set_size_inches(8, 6)  # Set the dimensions after getting the figure
    fig.savefig("param_importances.png")

    # Plotting contour plot
    axes_array = plot_contour(study)
    fig = axes_array.ravel()[0].figure  # Access the figure from one of the subplots
    fig.set_size_inches(8, 6)  # Set the dimensions after getting the figure
    fig.savefig("plot_contour.png")

    #NO GUI optuna.visualization.plot_optimization_history(study)

    #NO GUI optuna.visualization.plot_param_importances(study)

    #NO GUI optuna.visualization.plot_contour(study)

    # Train and test the model with the best hyperparameters
    args.batch_size = study.best_params['batch_size']
    args.lr = study.best_params['lr']
    args.epochs = study.best_params['epochs']
    args.folds = study.best_params['folds']  # Update folds
    args.processors = study.best_params['processors']  # Update processors
    #== OPTUNA ENDS

    ### results, total_elapsed_time = parallel_kfold(X_train, y_train, device, args)  # Receive the total elapsed time
    if args.processors >= 2:

        kf = KFold(n_splits=args.processors)  # change this line to set the number of splits dynamically
        results, total_elapsed_time = parallel_kfold(X_train, y_train, device, args)  # Pass the necessary arguments  # Receive the total elapsed time

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
###     just be careful here, there is no support when folds=1!
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
    parser.add_argument('--trials', type=int, default=10, help='Number of trials Optuna will perform')
    parser.add_argument('--processors', type=int, default=2, help='Number of processors to use for parallel processing')
    parser.add_argument('--folds', type=int, default=10, help='Number of k-folds for cross-validation')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')

    args = parser.parse_args()

    # Set device to GPU if available and requested
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    ## n_trials = args.trials ## set num of trials

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Convert the dataset to numpy arrays, but this time keep the 2D shape for convolution operations
    X = mnist_dataset.data.numpy().reshape(-1, 1, 28, 28) / 255.0
    y = mnist_dataset.targets.numpy()
    main()

