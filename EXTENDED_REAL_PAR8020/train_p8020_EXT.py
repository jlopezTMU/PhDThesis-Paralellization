import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

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

def train(fold, train_idx, val_idx, X, y, device, args):
    print(f"Process {os.getpid()} is using device: {device}")  # Debug: Show assigned GPU
    model = LeNet().to(device)  # Load the model to the assigned GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    y_train = torch.tensor(y[train_idx], dtype=torch.int64).to(device)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
    y_val = torch.tensor(y[val_idx], dtype=torch.int64).to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    for epoch in range(args.epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()
        val_accuracy = (val_outputs.argmax(1) == y_val).float().mean().item()

    elapsed_time = time.time() - start_time  # Elapsed time for this fold

    print(f"Process {os.getpid()} processed Fold {fold+1}, Accuracy: {val_accuracy*100:.2f}%, Elapsed time: {elapsed_time:.4f} seconds")

    return val_loss, val_accuracy, model
