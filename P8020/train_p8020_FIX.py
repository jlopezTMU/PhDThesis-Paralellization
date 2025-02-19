import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed
from torchvision.models import vgg11

###############################################################################
# 1) LeNet for MNIST (unchanged)
###############################################################################
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

###############################################################################
# 2) Modified train_8args(...) function with weight synchronization
###############################################################################
def train_8args(fold, train_idx, val_idx, X, y, device, ds, args):
    print(f"Process {os.getpid()} is using device: {device}")

    # Select the model
    if ds.upper() == "MNIST":
        model = LeNet().to(device)
    elif ds.upper() == "CIFAR10":
        model = vgg11(num_classes=10).to(device)
    else:
        raise ValueError("Unsupported dataset. Must be MNIST or CIFAR10.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Prepare training data (chunk only)
    X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    y_train = torch.tensor(y[train_idx], dtype=torch.int64).to(device)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    start_time = time.time()
    # Training loop for the chunk
    for epoch in range(args.epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Weight synchronization after each epoch:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for param in model.parameters():
                torch.distributed.all_reduce(param.data, op=torch.distributed.ReduceOp.SUM)
                param.data /= torch.distributed.get_world_size()

    elapsed_time = time.time() - start_time

    # Calculate the number of correctly classified entries for the chunk
    with torch.no_grad():
        outputs = model(X_train)
        correctly_classified_chunk = (outputs.argmax(1) == y_train).sum().item()
        chunk_size = len(y_train)

    print(
        f"Process {os.getpid()} -> Partition {fold+1}, "
        f"Correctly Classified: {correctly_classified_chunk}/{chunk_size}, "
        f"Elapsed: {elapsed_time:.4f}s"
    )
    return correctly_classified_chunk, chunk_size, model
