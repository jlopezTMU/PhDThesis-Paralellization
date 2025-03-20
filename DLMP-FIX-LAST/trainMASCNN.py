# trainMASCNN.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg11
import numpy as np  # Import NumPy to handle array-based indexing
import time  # For tracking processing time

def get_model(arch, num_classes=10):
    if arch == 'LeNet5':
        class LeNet5(nn.Module):
            def __init__(self, num_classes=10, activation=nn.ReLU()):
                super(LeNet5, self).__init__()
                self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # 28x28 -> 28x28
                self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
                self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)  # 14x14 -> 10x10
                self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 10x10 -> 5x5
                self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Flattened feature maps
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, num_classes)
                self.activation = activation

            def forward(self, feature_map):
                feature_map = self.activation(self.conv1(feature_map))
                feature_map = self.pool1(feature_map)
                feature_map = self.activation(self.conv2(feature_map))
                feature_map = self.pool2(feature_map)
                feature_map = feature_map.view(feature_map.size(0), -1)
                feature_map = self.activation(self.fc1(feature_map))
                feature_map = self.activation(self.fc2(feature_map))
                feature_map = self.fc3(feature_map)
                return feature_map

        return LeNet5(num_classes=num_classes)
    elif arch == 'VGG11':
        model = vgg11(weights=None)  # Use VGG-11 from torchvision
        model.classifier[6] = nn.Linear(4096, num_classes)  # Adjust the final layer for the number of classes
        return model

def get_vgg11_model(num_classes=10):
    return get_model('VGG11', num_classes=num_classes)

def train_simulated(fold, train_idx, val_idx, X, y, device, args, original_training_size, sync_callback):
    print(f"Training with {len(train_idx)} examples, validating with {len(val_idx)} examples")

    # Convert data to NumPy arrays if they are not already
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(y, list):
        y = np.array(y)

    # Use a default architecture of 'LeNet5' if args.arch is not provided
    arch = getattr(args, 'arch', 'LeNet5')

    # Data preparation
    if arch == 'LeNet5':  # For MNIST
        X_train_fold = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
        # Only unsqueeze if the data is 3D (N, H, W); if already 4D, do nothing.
        if X_train_fold.dim() == 3:
            X_train_fold = X_train_fold.unsqueeze(1)
        X_val_fold = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
        if X_val_fold.dim() == 3:
            X_val_fold = X_val_fold.unsqueeze(1)
    elif arch == 'VGG11':  # For CIFAR-10
        X_train_fold = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
        X_val_fold = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
        if len(X_train_fold.shape) == 4 and X_train_fold.shape[1] == 3:
            print("CIFAR-10 data is in the correct shape.")
        else:
            raise ValueError(f"Unexpected CIFAR-10 shape: {X_train_fold.shape}. Expected (batch_size, 3, height, width)")

    y_train_fold = torch.tensor(y[train_idx], dtype=torch.int64).to(device)
    y_val_fold = torch.tensor(y[val_idx], dtype=torch.int64).to(device)

    model = get_model(arch, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    patience = args.patience
    epochs_without_improvement = 0

    train_dataset = torch.utils.data.TensorDataset(X_train_fold, y_train_fold)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    processing_times = []

    for epoch in range(args.epochs):
        model.train()
        correct_train = 0
        total_train = 0
        epoch_start_time = time.time()

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            correct_train += (outputs.argmax(1) == batch_y).sum().item()
            total_train += batch_y.size(0)

        # Synchronize weights after each epoch using the provided callback.
        sync_callback(model.state_dict())

        epoch_end_time = time.time()
        epoch_processing_time = epoch_end_time - epoch_start_time
        processing_times.append(epoch_processing_time)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_fold)
            val_loss = criterion(val_outputs, y_val_fold).item()
            correct_val = (val_outputs.argmax(1) == y_val_fold).sum().item()
            validation_accuracy = (correct_val / len(y_val_fold)) * 100
            training_accuracy = (correct_train / total_train) * 100

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

    slowest_processing_time = max(processing_times)
    print(f"Node {fold+1} Processing Time: {slowest_processing_time:.4f} seconds")
    train_outputs = model(X_train_fold)
    correct_classifications_train = (train_outputs.argmax(1) == y_train_fold).sum().item()

    return loss.item(), validation_accuracy, model, correct_classifications_train, correct_val, len(y_val_fold), slowest_processing_time

