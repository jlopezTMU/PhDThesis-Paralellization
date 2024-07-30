import logging
import os
import sys  # Import sys for argument checking
import argparse
import time
import multiprocessing  # Import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from torch.multiprocessing import Pool, set_start_method
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from trainCV_nfv import train  # Ensure this module is accessible

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Necessary for CUDA multiprocessing
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass

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

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x = self.transform(x)
        return x, y

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel k-fold cross-validation on MNIST")
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--processors', type=int, default=2, help='Number of processors to use for parallel processing (default=2)')
    parser.add_argument('--folds', type=int, default=3, help='Number of k-folds for cross-validation (default=3)')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for training (default=2048)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (default=5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    
    args, unknown = parser.parse_known_args()
    
    if unknown:
        logger.warning('Unrecognized arguments: %s', unknown)
    
    return args

def parallel_kfold(X, y, device, args):
    kf = KFold(n_splits=args.folds)
    start_time = time.time()

    with Pool(processes=args.processors) as pool:
        results = pool.starmap(train, [(fold, train_idx, val_idx, X, y, device, args) for fold, (train_idx, val_idx) in enumerate(kf.split(X))])

    end_time = time.time()
    elapsed_time = end_time - start_time
    return results, elapsed_time

def test_model(best_model, dataloader, device):
    with torch.no_grad():
        correct = 0
        total = 0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device).to(torch.float32)
            y_batch = y_batch.to(device)
            outputs = best_model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        test_accuracy = correct / total
    return test_accuracy

def main(args):
    logger.info('Starting main function')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    X = mnist_dataset.data.numpy().reshape(-1, 1, 28, 28).astype('float32') / 255.0
    y = mnist_dataset.targets.numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Dataset split: {len(y_train)} training records, {len(y_test)} testing records")

    train_dataset = CustomDataset(X_train, y_train, transform=None)
    test_dataset = CustomDataset(X_test, y_test, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logger.info(f'Using device: {device}')

    if args.processors >= 2:
        results, total_elapsed_time = parallel_kfold(X_train, y_train, device, args)

        losses = [result[0] for result in results]
        accuracies = [result[1] for result in results]
        best_model_idx = accuracies.index(max(accuracies))
        best_model = results[best_model_idx][2]

        test_accuracy = test_model(best_model, test_loader, device)

        logger.info(f"Total processing time: {total_elapsed_time:.4f} seconds")
        logger.info(f"Average validation accuracy: {sum(accuracies) / len(accuracies) * 100:.4f}%")
        logger.info(f"Testing accuracy: {test_accuracy * 100:.4f}%")
    else:
        model = LeNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        best_model = None
        best_train_accuracy = 0.0

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        start_time_train = time.time()
        kf = KFold(n_splits=args.folds)
        train_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            fold_train_dataset = CustomDataset(X_train_fold, y_train_fold, transform=None)
            fold_train_loader = DataLoader(fold_train_dataset, batch_size=args.batch_size, shuffle=True)

            for epoch in range(args.epochs):
                for X_batch, y_batch in fold_train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    X_batch = X_batch.view(-1, 1, 28, 28).to(torch.float32)
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                train_outputs = model(torch.tensor(X_train_fold, dtype=torch.float32).to(device))
                train_accuracy = (train_outputs.argmax(1) == torch.tensor(y_train_fold, dtype=torch.int64).to(device)).float().mean().item()
                train_accuracies.append(train_accuracy)

                if train_accuracy > best_train_accuracy:
                    best_train_accuracy = train_accuracy
                    best_model = model.state_dict()

        end_time_train = time.time()
        elapsed_time_train = end_time_train - start_time_train

        start_time_test = time.time()

        model.load_state_dict(best_model)
        test_accuracy = test_model(model, test_loader, device)
        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)

        logger.info(f"Training accuracy: {avg_train_accuracy * 100:.4f}%")
        end_time_test = time.time()
        elapsed_time_test = end_time_test - start_time_test

        logger.info(f"Total training time: {elapsed_time_train:.4f} seconds")
        logger.info(f"Total testing time: {elapsed_time_test:.4f} seconds")
        logger.info(f"Testing accuracy: {test_accuracy * 100:.4f}%")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Add this line
    if '--multiprocessing-fork' in sys.argv:
        sys.exit(0)
    args = parse_args()
    logger.info('Hyperparameters:')
    logger.info(f'Processors: {args.processors}')
    logger.info(f'Folds: {args.folds}')
    logger.info(f'Batch size: {args.batch_size}')
    logger.info(f'Epochs: {args.epochs}')
    logger.info(f'Learning rate: {args.lr}')
    if args.gpu:
        logger.info('Using GPU')
    main(args)
