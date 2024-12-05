import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np  # Import NumPy to handle array-based indexing
import time  # For tracking processing time

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # 28x28 -> 28x28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)  # 14x14 -> 10x10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 10x10 -> 5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Flattened feature maps
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_lenet5_model(num_classes=10):
    """Return an instance of the LeNet-5 model."""
    return LeNet5(num_classes=num_classes)

def train_simulated(fold, train_idx, val_idx, X, y, device, args, original_training_size, sync_callback):
    print(f"Training with {len(train_idx)} examples, validating with {len(val_idx)} examples")

    # Convert data to NumPy arrays if they are not already
    if isinstance(X, list):  # If X is a list, convert it to a NumPy array
        X = np.array(X)
    if isinstance(y, list):  # If y is a list, convert it to a NumPy array
        y = np.array(y)

    # Prepare data for LeNet-5 (grayscale: single channel)
    X_train_fold = torch.tensor(X[train_idx], dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dimension
    y_train_fold = torch.tensor(y[train_idx], dtype=torch.int64).to(device)

    # Split the testing dataset into p parts based on the number of processors
    p = args.processors
    split_X_test = np.array_split(X[val_idx], p)  # Split the test dataset into p parts
    split_y_test = np.array_split(y[val_idx], p)

    # Assign the test portion for the current node (fold)
    X_val_fold = torch.tensor(split_X_test[fold], dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dimension
    y_val_fold = torch.tensor(split_y_test[fold], dtype=torch.int64).to(device)

    # Initialize LeNet-5 model
    model = get_lenet5_model(num_classes=10).to(device)  # MNIST has 10 classes
    criterion = nn.CrossEntropyLoss().to(device)

    # Use Adam optimizer with a lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    patience = args.patience
    epochs_without_improvement = 0

    # Create DataLoader for the training data
    train_dataset = torch.utils.data.TensorDataset(X_train_fold, y_train_fold)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    processing_times = []  # Track processing times per epoch

    for epoch in range(args.epochs):
        model.train()  # Set model to training mode
        correct_train = 0
        total_train = 0
        epoch_start_time = time.time()  # Start tracking epoch processing time

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # Ensure batch data is on the correct device
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            correct_train += (outputs.argmax(1) == batch_y).sum().item()  # Accumulate correct predictions
            total_train += batch_y.size(0)  # Accumulate total examples

        epoch_end_time = time.time()  # End tracking processing time
        epoch_processing_time = epoch_end_time - epoch_start_time  # Calculate processing time for the epoch
        processing_times.append(epoch_processing_time)  # Add to processing times list

        # Synchronize weights after each epoch
        sync_callback(model.state_dict())

        # Validation per node using its portion of the test set
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            val_outputs = model(X_val_fold)
            val_loss = criterion(val_outputs, y_val_fold).item()
            correct_val = (val_outputs.argmax(1) == y_val_fold).sum().item()  # Correct predictions in validation
            validation_accuracy = (correct_val / len(y_val_fold)) * 100  # Validation accuracy as percentage

            training_accuracy = (correct_train / total_train) * 100  # Training accuracy for the epoch

            # Report the requested format: correct_train/total_train for training and node-specific validation results
            print(f"Epoch {epoch+1}, Node {fold+1} Validation Loss: {val_loss:.4f}, "
                  f"Training Accuracy: {training_accuracy:.2f}%, "
                  f"Validation Accuracy: {correct_val}/{len(y_val_fold)} = {validation_accuracy:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Report the slowest processing time across all epochs for this node
    slowest_processing_time = max(processing_times)
    print(f"Node {fold+1} Processing Time: {slowest_processing_time:.4f} seconds")

    train_outputs = model(X_train_fold)
    correct_classifications_train = (train_outputs.argmax(1) == y_train_fold).sum().item()

    return loss.item(), validation_accuracy, model, correct_classifications_train, correct_val, len(y_val_fold), slowest_processing_time
