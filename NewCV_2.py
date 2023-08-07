import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from torch.multiprocessing import Pool, set_start_method
import argparse
import time

# Necessary for CUDA multiprocessing
set_start_method('spawn', force=True)

# Command line arguments
parser = argparse.ArgumentParser(description="Parallel k-fold cross-validation on MNIST")
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--processors', type=int, default=2, help='Number of processors to use for parallel processing')
parser.add_argument('--folds', type=int, default=10, help='Number of k-folds for cross-validation')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
args = parser.parse_args()

# Set device to GPU if available and requested
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Convert the dataset to numpy arrays, but this time keep the 2D shape for convolution operations
X = mnist_dataset.data.numpy().reshape(-1, 1, 28, 28) / 255.0
y = mnist_dataset.targets.numpy()

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

def train(fold, train_idx, val_idx):
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    X_train, X_val = torch.tensor(X[train_idx], dtype=torch.float32).to(device), torch.tensor(X[val_idx], dtype=torch.float32).to(device)
    y_train, y_val = torch.tensor(y[train_idx], dtype=torch.int64).to(device), torch.tensor(y[val_idx], dtype=torch.int64).to(device)

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()
        val_accuracy = (val_outputs.argmax(1) == y_val).float().mean().item()

    # Progress message
    print(f"Processor {fold+1}: Processed data from {train_idx[0]} to {train_idx[-1]}, Fold {fold+1}, Accuracy: {val_accuracy*100:.2f}%")

    return val_loss, val_accuracy

def parallel_kfold():
    kf = KFold(n_splits=args.folds)
    start_time = time.time()
    with Pool(processes=args.processors) as pool:
        results = pool.starmap(train, [(fold, train_idx, val_idx) for fold, (train_idx, val_idx) in enumerate(kf.split(X))])
    end_time = time.time()
    return results, end_time - start_time

def main():
    results, elapsed_time = parallel_kfold()

    losses = [result[0] for result in results]
    accuracies = [result[1] for result in results]

    print(f"Cross-validation took {elapsed_time:.2f} seconds")
    print(f"Average validation accuracy across folds: {sum(accuracies) / len(accuracies) * 100:.2f}%")

if __name__ == '__main__':
    main()
