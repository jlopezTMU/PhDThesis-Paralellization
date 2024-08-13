import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import argparse
import time
import os

from trainSimCNN import train_simulated

# Necessary for CUDA multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)

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
###     
def simulate_kfold(original_training_size):
    kf = KFold(n_splits=args.processors)
    cumulative_time = 0  # To accumulate the processing time for all nodes

    losses = []
    accuracies = []
    models = []

    num_samples_per_fold = original_training_size // args.processors

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(original_training_size))):
        start_sample = fold * num_samples_per_fold
        end_sample = start_sample + num_samples_per_fold - 1
        if fold == args.processors - 1:  # Ensure the last fold includes all remaining samples
            end_sample = original_training_size - 1

        # Start timing for this node
        fold_start_time = time.time()
        print(f"*** Starting processing for node {fold + 1}/{args.processors} on samples {start_sample} to {end_sample} ***")

        fold_loss, fold_accuracy, model = train_simulated(fold, train_idx, val_idx, X, y, device, args, original_training_size)

        # End timing for this node
        fold_end_time = time.time()
        fold_time = fold_end_time - fold_start_time
        cumulative_time += fold_time  # Accumulate time

        losses.append(fold_loss)
        accuracies.append(fold_accuracy)
        models.append(model)

        # Display partial results for this node
        print(f"--- Node {fold + 1}/{args.processors} completed in {fold_time:.4f} seconds ---")
        print(f"--- Cumulative processing time after node {fold + 1}: {cumulative_time:.4f} seconds ---")
        print(f"--- Fold {fold + 1} Loss: {fold_loss:.4f}, Accuracy: {fold_accuracy * 100:.2f}% ---")

    # Calculate the average processing time per node
    average_time_per_node = cumulative_time / args.processors

    print(f"*** Total average processing time for all nodes: {average_time_per_node:.4f} seconds ***")
    
    return losses, accuracies, models, average_time_per_node


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
    training_size = len(y_train)
    print(f"*** The dataset has been split into 80% training with {training_size} records and 20% testing with {len(y_test)} records ***")

    if args.processors >= 2:
        # Pass the training size to the training function
        losses, accuracies, models, total_elapsed_time = simulate_kfold(training_size)  # Perform the simulated data parallelization

        best_model_idx = accuracies.index(max(accuracies))
        best_model = models[best_model_idx]

        test_accuracy = test_model(best_model, X_test, y_test, device)

        print(f"*** Total processing time for all simulated nodes: {total_elapsed_time:.4f} seconds")
        print(f"*** Average validation accuracy across folds: {sum(accuracies) / len(accuracies) * 100:.4f}%")
        print(f">>> Testing accuracy using the best model: {test_accuracy * 100:.4f}%")
    else:
        # Single processor training (regular training)
        model = LeNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.int64).to(device)

        start_time_train = time.time()
        kf = KFold(n_splits=args.folds)
        train_accuracies = []

        best_model = None
        best_train_accuracy = 0.0

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

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
        model.load_state_dict(best_model)
        test_accuracy = test_model(model, X_test, y_test, device)
        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)

        print(f">>> (1 processor with {args.folds} fCV) Training accuracy using the model: {avg_train_accuracy * 100:.4f}%")
        end_time_test = time.time()
        elapsed_time_test = end_time_test - start_time_test

        print(f"(1 processor) Total elapsed time for training: {elapsed_time_train:.4f} seconds")
        print(f"(1 processor) Total elapsed time for testing: {elapsed_time_test:.4f} seconds")
        print(f">>> Testing accuracy using best average cross-validation model: {test_accuracy * 100:.4f}%")

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description="Simulated parallel k-fold cross-validation on MNIST")
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--processors', type=int, default=4, help='Number of simulated processors (nodes) to use')
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
