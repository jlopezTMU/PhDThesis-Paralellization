import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# For LeNet5
class LeNet5(nn.Module):
    def __init__(self, num_classes=10, activation=nn.ReLU()):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
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

def get_lenet5_model(num_classes=10, activation=nn.ReLU()):
    return LeNet5(num_classes=num_classes, activation=activation)

# For FocalLoss
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

# We'll import torchvision.models for VGG11
import torchvision.models as models

def train_simulated(fold_idx, train_idx, val_idx,
                    Full_ds, Full_ds_lbls,
                    device, args,
                    original_training_size,
                    sync_callback):

    print(f"Training with {len(train_idx)} examples, validating with {len(val_idx)} examples")

    # Ensure arrays are NumPy if needed
    if isinstance(Full_ds, list):
        Full_ds = np.array(Full_ds)
    if isinstance(Full_ds_lbls, list):
        Full_ds_lbls = np.array(Full_ds_lbls)

    # Prepare data depending on which dataset we are using
    if args.ds == 'MNIST':
        # MNIST is single-channel; keep the existing approach
        Training_ds_fold_idx = torch.tensor(Full_ds[train_idx], dtype=torch.float32).unsqueeze(1).to(device)
        Training_lbls_fold_idx = torch.tensor(Full_ds_lbls[train_idx], dtype=torch.int64).to(device)

        # For validation
        p = args.processors
        split_Testing_ds = np.array_split(Full_ds[val_idx], p)
        split_Testing_lbls = np.array_split(Full_ds_lbls[val_idx], p)

        X_val_fold_idx = torch.tensor(split_Testing_ds[fold_idx], dtype=torch.float32).unsqueeze(1).to(device)
        y_val_fold_idx = torch.tensor(split_Testing_lbls[fold_idx], dtype=torch.int64).to(device)

    else:  # CIFAR10
        # CIFAR-10 is RGB with shape Nx3x32x32
        Training_ds_fold_idx = torch.tensor(Full_ds[train_idx], dtype=torch.float32).to(device)
        Training_lbls_fold_idx = torch.tensor(Full_ds_lbls[train_idx], dtype=torch.int64).to(device)

        # For validation
        p = args.processors
        split_Testing_ds = np.array_split(Full_ds[val_idx], p)
        split_Testing_lbls = np.array_split(Full_ds_lbls[val_idx], p)

        X_val_fold_idx = torch.tensor(split_Testing_ds[fold_idx], dtype=torch.float32).to(device)
        y_val_fold_idx = torch.tensor(split_Testing_lbls[fold_idx], dtype=torch.int64).to(device)

    # Choose activation function
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

    # Pick model based on dataset
    if args.ds == 'MNIST':
        model = get_lenet5_model(num_classes=10, activation=activation_fn).to(device)
    else:  # CIFAR10 => use VGG-11
        model = models.vgg11(weights=None)  # Start with no pretraining
        # Modify the final classifier layer to output 10 classes
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
        model = model.to(device)

    # Pick the criterion
    if args.loss == 'CE':
        criterion = nn.CrossEntropyLoss().to(device)
    elif args.loss == 'LSCE':
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    elif args.loss == 'FC':
        criterion = FocalLoss().to(device)
    elif args.loss == 'WCE':
        weight = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device)
        criterion = nn.CrossEntropyLoss(weight=weight).to(device)

    # Pick the optimizer
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

    # Build DataLoader
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

        # Synchronize after each epoch
        sync_callback(model.state_dict())

        # Validation step
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

    # Evaluate final training correctness
    train_outputs = model(Training_ds_fold_idx)
    correct_classifications_train = (train_outputs.argmax(1) == Training_lbls_fold_idx).sum().item()

    return (loss.item(),
            validation_accuracy,
            model,
            correct_classifications_train,
            correct_val,
            len(y_val_fold_idx),
            slowest_processing_time)
