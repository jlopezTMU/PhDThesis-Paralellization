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

def train_simulated(fold, train_idx, val_idx, X, y, device, args, original_training_size, sync_callback):
    # Debug: Print sizes of the training and validation sets
    print(f"Training with {len(train_idx)} examples, validating with {len(val_idx)} examples")

    # Use the full validation set for each node
    X_val_fold = X[val_idx]
    y_val_fold = y[val_idx]

    # Convert numpy arrays to PyTorch tensors and move to the specified device
    X_train_fold = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    X_val_fold = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
    y_train_fold = torch.tensor(y[train_idx], dtype=torch.int64).to(device)
    y_val_fold = torch.tensor(y_val_fold, dtype=torch.int64).to(device)

    # Initialize the model, loss function, and optimizer
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Early stopping variables
    best_val_loss = float('inf')
    patience = args.patience  # Use the patience value from the arguments
    batches_without_improvement = 0

    # Create DataLoader for the training data
    train_dataset = torch.utils.data.TensorDataset(X_train_fold, y_train_fold)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    total_batches = len(train_loader)
    total_correct_train = 0
    total_examples_train = 0

    # Training loop (synchronization and early stopping after each mini-batch)
    for epoch in range(args.epochs):
        model.train()  # Set model to training mode
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # Ensure batch data is on the correct device
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            # Synchronize weights after each mini-batch
            sync_callback(model.state_dict())

            # Calculate training accuracy for the batch
            correct_train = (outputs.argmax(1) == batch_y).sum().item()
            total_correct_train += correct_train
            total_examples_train += batch_y.size(0)

            # Validation after each mini-batch
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                val_outputs = model(X_val_fold)
                val_loss = criterion(val_outputs, y_val_fold).item()
                correct_val = (val_outputs.argmax(1) == y_val_fold).sum().item()
                validation_accuracy = (correct_val / len(y_val_fold)) * 100

                # Debug: Print training and validation results
                batch_training_accuracy = (correct_train / batch_y.size(0)) * 100
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{total_batches}, Training Accuracy: {batch_training_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}%")

                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    batches_without_improvement = 0  # Reset counter if there's improvement
                else:
                    batches_without_improvement += 1

                if batches_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch+1}, batch {batch_idx+1}")
                    # Calculate the overall training accuracy up to this point
                    overall_training_accuracy = (total_correct_train / total_examples_train) * 100
                    print(f"Overall Training Accuracy: {total_correct_train}/{total_examples_train} = {overall_training_accuracy:.2f}%")
                    # Return metrics
                    return loss.item(), validation_accuracy, model, total_correct_train

            model.train()  # Set back to training mode for the next batch

        # Optionally, reset training accuracy counters after each epoch
        # total_correct_train = 0
        # total_examples_train = 0

    # Calculate the number of correctly classified examples in the training set
    train_outputs = model(X_train_fold)
    correct_classifications_train = (train_outputs.argmax(1) == y_train_fold).sum().item()

    # Return metrics
    return loss.item(), validation_accuracy, model, correct_classifications_train
