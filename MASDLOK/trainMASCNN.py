import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np  # Import NumPy to handle array-based indexing
import time  # For tracking processing time

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
        self.activation = activation  # Added line

    def forward(self, feature_map):
        feature_map = self.activation(self.conv1(feature_map))  # Modified line
        feature_map = self.pool1(feature_map)
        feature_map = self.activation(self.conv2(feature_map))  # Modified line
        feature_map = self.pool2(feature_map)
        feature_map = feature_map.view(feature_map.size(0), -1)  # Flatten
        feature_map = self.activation(self.fc1(feature_map))    # Modified line
        feature_map = self.activation(self.fc2(feature_map))    # Modified line
        feature_map = self.fc3(feature_map)
        return feature_map

def get_lenet5_model(num_classes=10, activation=nn.ReLU()):
    """Return an instance of the LeNet-5 model."""
    return LeNet5(num_classes=num_classes, activation=activation)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def train_simulated(fold_idx, train_idx, val_idx, Full_ds, Full_ds_lbls, device, args, original_training_size, sync_callback):
    print(f"Training with {len(train_idx)} examples, validating with {len(val_idx)} examples")

    # Convert data to NumPy arrays if they are not already
    if isinstance(Full_ds, list):
        Full_ds = np.array(Full_ds)
    if isinstance(Full_ds_lbls, list):
        Full_ds_lbls = np.array(Full_ds_lbls)

    # Prepare data for LeNet-5 (grayscale: single channel)
    Training_ds_fold_idx = torch.tensor(Full_ds[train_idx], dtype=torch.float32).unsqueeze(1).to(device)
    Training_lbls_fold_idx = torch.tensor(Full_ds_lbls[train_idx], dtype=torch.int64).to(device)

    # Split the testing dataset into p parts based on the number of processors
    p = args.processors
    split_Testing_ds = np.array_split(Full_ds[val_idx], p)
    split_Testing_lbls = np.array_split(Full_ds_lbls[val_idx], p)

    # Assign the test portion for the current node (fold_idx)
    X_val_fold_idx = torch.tensor(split_Testing_ds[fold_idx], dtype=torch.float32).unsqueeze(1).to(device)
    y_val_fold_idx = torch.tensor(split_Testing_lbls[fold_idx], dtype=torch.int64).to(device)

    # Choose activation function based on args.activation
    if args.activation == 'RELU':
        activation_fn = nn.ReLU()
    elif args.activation == 'LEAKY_RELU':
        activation_fn = nn.LeakyReLU()
    elif args.activation == 'ELU':
        activation_fn = nn.ELU()
    elif args.activation == 'SELU':
        activation_fn = nn.SELU()
    elif args.activation == 'GELU':
        activation_fn = nn.GELU()
    elif args.activation == 'MISH':
        activation_fn = nn.Mish()

    # Initialize LeNet-5 model with chosen activation
    model = get_lenet5_model(num_classes=10, activation=activation_fn).to(device)

    # Set criterion based on args.loss
    if args.loss == 'CE':
        criterion = nn.CrossEntropyLoss().to(device)
    elif args.loss == 'LSCE':
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    elif args.loss == 'FC':
        criterion = FocalLoss().to(device)
    elif args.loss == 'WCE':
        weight = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device)
        criterion = nn.CrossEntropyLoss(weight=weight).to(device)

    # Set optimizer based on args.optimizer
    if args.optimizer == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif args.optimizer == 'ADAMW':
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
    elif args.optimizer == 'SGDM':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif args.optimizer == 'RMSP':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    patience = args.patience
    epochs_without_improvement = 0

    train_dataset = torch.utils.data.TensorDataset(Training_ds_fold_idx, Training_lbls_fold_idx)
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

        epoch_end_time = time.time()
        epoch_processing_time = epoch_end_time - epoch_start_time
        processing_times.append(epoch_processing_time)

        # Synchronize weights after each epoch
        sync_callback(model.state_dict())

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_fold_idx)
            val_loss = criterion(val_outputs, y_val_fold_idx).item()
            correct_val = (val_outputs.argmax(1) == y_val_fold_idx).sum().item()
            validation_accuracy = (correct_val / len(y_val_fold_idx)) * 100
            training_accuracy = (correct_train / total_train) * 100

            print(f"Epoch {epoch+1}, Node {fold_idx+1} Validation Loss: {val_loss:.4f}, "
                  f"Training Accuracy: {training_accuracy:.2f}%, "
                  f"Validation Accuracy: {correct_val}/{len(y_val_fold_idx)} = {validation_accuracy:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    slowest_processing_time = max(processing_times)
    print(f"Node {fold_idx+1} Processing Time: {slowest_processing_time:.4f} seconds")

    train_outputs = model(Training_ds_fold_idx)
    correct_classifications_train = (train_outputs.argmax(1) == Training_lbls_fold_idx).sum().item()

    return loss.item(), validation_accuracy, model, correct_classifications_train, correct_val, len(y_val_fold_idx), slowest_processing_time
