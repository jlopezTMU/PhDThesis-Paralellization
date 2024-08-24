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
    # Debug: Print sizes of the training and validation sets
    print(f"Training with {len(train_idx)} examples, validating with {len(val_idx)} examples")

    # Split the data into training and validation sets for the fold
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    # Convert numpy arrays to PyTorch tensors and move to the specified device
    X_train_fold = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
    X_val_fold = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
    y_train_fold = torch.tensor(y_train_fold, dtype=torch.int64).to(device)
    y_val_fold = torch.tensor(y_val_fold, dtype=torch.int64).to(device)

    # Initialize the model, loss function, and optimizer
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Create DataLoader for the training data
    train_dataset = torch.utils.data.TensorDataset(X_train_fold, y_train_fold)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop
    for epoch in range(args.epochs):
        model.train()  # Set model to training mode
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # Ensure batch data is on the correct device
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Validation after training
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        X_val_fold = X_val_fold.to(device)  # Ensure validation data is on the correct device
        val_outputs = model(X_val_fold)
        val_accuracy = (val_outputs.argmax(1) == y_val_fold).float().mean().item()
        fold_loss = criterion(val_outputs, y_val_fold).item()

        # Debug: Print validation results
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
        print(f"Validation Loss: {fold_loss:.4f}")

        # Calculate the number of correctly classified examples in the training set
        train_outputs = model(X_train_fold)
        correct_classifications_train = (train_outputs.argmax(1) == y_train_fold).sum().item()

        # Debug: Print the number of correctly classified training examples
        print(f"Correctly classified training examples: {correct_classifications_train}/{original_training_size}")

    # Return metrics
    return fold_loss, val_accuracy, model, correct_classifications_train
