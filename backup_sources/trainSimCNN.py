import torch
import torch.nn as nn
import torch.optim as optim

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

def train_simulated(fold, train_idx, val_idx, X, y, device, args, original_training_size):
    # Access the original training size
    print(f"Original training data size: {original_training_size}")
    
    # Split the data into training and validation sets for the fold
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    # Initialize the model, loss function, and optimizer
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Create DataLoader for the training data
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32).to(device), 
                                                   torch.tensor(y_train_fold, dtype=torch.int64).to(device))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop
    for epoch in range(args.epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Validation after training
    with torch.no_grad():
        val_outputs = model(torch.tensor(X_val_fold, dtype=torch.float32).to(device))
        val_accuracy = (val_outputs.argmax(1) == torch.tensor(y_val_fold, dtype=torch.int64).to(device)).float().mean().item()
        fold_loss = criterion(val_outputs, torch.tensor(y_val_fold, dtype=torch.int64).to(device)).item()

    return fold_loss, val_accuracy, model
