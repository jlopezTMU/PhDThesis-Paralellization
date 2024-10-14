import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg11
import numpy as np  # Import NumPy to handle array-based indexing
import time  # For tracking processing time

def get_vgg11_model(num_classes=10):
    # Load VGG-11 model
    model = vgg11(weights=None)  # Use weights=None to initialize with random weights
    model.classifier[6] = nn.Linear(4096, num_classes)  # Adjust the final layer for CIFAR-10 (10 classes)
    return model

def train_simulated(fold, train_idx, val_idx, X, y, device, args, original_training_size, sync_callback):
    print(f"Training with {len(train_idx)} examples, validating with {len(val_idx)} examples")

    # Convert data to NumPy arrays if they are not already
    if isinstance(X, list):  # If X is a list, convert it to a NumPy array
        X = np.array(X)
    if isinstance(y, list):  # If y is a list, convert it to a NumPy array
        y = np.array(y)

    # VGG-11 expects images to have 3 channels (RGB)
    X_train_fold = torch.tensor(X[train_idx], dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # Convert to [N, C, H, W]
    y_train_fold = torch.tensor(y[train_idx], dtype=torch.int64).to(device)

    # Split the testing dataset into p parts based on the number of processors
    p = args.processors
    split_X_test = np.array_split(X[val_idx], p)  # Split the test dataset into p parts
    split_y_test = np.array_split(y[val_idx], p)

    # Assign the test portion for the current node (fold)
    X_val_fold = torch.tensor(split_X_test[fold], dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # Convert to [N, C, H, W]
    y_val_fold = torch.tensor(split_y_test[fold], dtype=torch.int64).to(device)

    # Initialize VGG-11 model
    model = get_vgg11_model(num_classes=10).to(device)  # CIFAR-10 has 10 classes
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

