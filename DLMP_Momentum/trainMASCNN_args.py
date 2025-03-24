import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

# get_model is assumed to be used for creating the network.
# For MNIST, we use LeNet5 and for CIFAR-10, we use VGG11.
def get_model(arch, num_classes=10, activation=None):
    if arch == 'LeNet5':
        # Define LeNet5 with a configurable activation function.
        class LeNet5(nn.Module):
            def __init__(self, activation):
                super(LeNet5, self).__init__()
                self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
                self.pool = nn.AvgPool2d(2, stride=2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                # Assuming input of 28x28 for MNIST, after conv/pooling:
                # 28 -> 14 -> 10 -> 5 (if no padding on conv2) so, 16*5*5 features.
                self.fc1 = nn.Linear(16*5*5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, num_classes)
                self.activation = activation

            def forward(self, x):
                x = self.activation(self.conv1(x))
                x = self.pool(x)
                x = self.activation(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.activation(self.fc1(x))
                x = self.activation(self.fc2(x))
                x = self.fc3(x)
                return x
        if activation is None:
            activation = nn.ReLU()
        return LeNet5(activation)
    elif arch == 'VGG11':
        from torchvision.models import vgg11
        return vgg11(weights=None, num_classes=num_classes)
    else:
        raise ValueError("Unsupported architecture")

def train_simulated(fold, train_idx, val_idx, X, y, device, args, original_training_size, sync_callback):
    print(f"Training with {len(train_idx)} examples, validating with {len(val_idx)} examples")
    # Ensure data is in NumPy format
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(y, list):
        y = np.array(y)

    # Data preparation
    if args.arch == 'LeNet5':
        X_train_fold = torch.tensor(X[train_idx], dtype=torch.float32).unsqueeze(1).to(device)
        X_val_fold = torch.tensor(X[val_idx], dtype=torch.float32).unsqueeze(1).to(device)
    elif args.arch == 'VGG11':
        X_train_fold = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
        X_val_fold = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
        if len(X_train_fold.shape) == 4 and X_train_fold.shape[1] == 3:
            print("CIFAR-10 data is in the correct shape.")
        else:
            raise ValueError(f"Unexpected CIFAR-10 shape: {X_train_fold.shape}. Expected (batch_size, 3, height, width)")
    y_train_fold = torch.tensor(y[train_idx], dtype=torch.int64).to(device)
    y_val_fold = torch.tensor(y[val_idx], dtype=torch.int64).to(device)

    # Create the model with appropriate activation if using LeNet5
    if args.arch == 'LeNet5':
        act_str = args.activation.upper()
        if act_str == "RELU":
            activation = nn.ReLU()
        elif act_str == "LEAKY_RELU":
            activation = nn.LeakyReLU()
        elif act_str == "ELU":
            activation = nn.ELU()
        elif act_str == "SELU":
            activation = nn.SELU()
        elif act_str == "GELU":
            activation = nn.GELU()
        elif act_str == "MISH":
            activation = nn.Mish()
        else:
            activation = nn.ReLU()
        model = get_model(args.arch, num_classes=10, activation=activation).to(device)
    else:
        model = get_model(args.arch, num_classes=10).to(device)

    # Select loss function based on argument
    loss_str = args.loss.upper()
    if loss_str == "CE":
        criterion = nn.CrossEntropyLoss().to(device)
    elif loss_str == "LSCE":
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    elif loss_str == "FC":
        raise NotImplementedError("Focal loss not implemented")
    elif loss_str == "WCE":
        raise NotImplementedError("Weighted Cross-Entropy not implemented")
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    # Select optimizer based on argument
    opt_str = args.optimizer.upper()
    if opt_str == "SGDM":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.92)
    elif opt_str == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif opt_str == "ADAMW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif opt_str == "RMSP":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = torch.utils.data.TensorDataset(X_train_fold, y_train_fold)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    best_val_loss = float('inf')
    patience = args.patience
    epochs_without_improvement = 0
    processing_times = []

    for epoch in range(args.epochs):
        model.train()
        correct_train = 0
        total_train = 0
        epoch_start_time = time.time()

        for batch_X, batch_y in train_loader:
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

        sync_callback(model.state_dict())

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

    # Return results: last epoch loss, validation accuracy, model state dict (moved to CPU), correct train count, correct val count, total val examples, slowest epoch time
    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    return loss.item(), validation_accuracy, cpu_state_dict, correct_classifications_train, correct_val, len(y_val_fold), slowest_processing_time

