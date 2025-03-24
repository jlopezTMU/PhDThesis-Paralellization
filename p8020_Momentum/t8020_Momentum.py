import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed
from torchvision.models import vgg11

###############################################################################
# 1) LeNet for MNIST 
###############################################################################
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(nn.MaxPool2d(2)(self.conv1(x)))
        x = torch.relu(nn.MaxPool2d(2)(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

###############################################################################
# Helper function to compute model size in MB
###############################################################################
def get_model_size(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb

###############################################################################
# 2) Modified train_8args(...) function with weight synchronization and
#    validation evaluation (using momentum).
###############################################################################
def train_8args(fold, train_idx, val_idx, X, y, device, ds, args):
    print(f"Process {os.getpid()} is using device: {device}", flush=True)
    if ds.upper() == "MNIST":
        model = LeNet().to(device)
    elif ds.upper() == "CIFAR10":
        model = vgg11(num_classes=10)
        # Adjust the model for CIFAR10 input size (32x32 images)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10)
        )
        model = model.to(device)
    else:
        raise ValueError("Unsupported dataset. Must be MNIST or CIFAR10.")

    # Print the model size right after instantiating it.
    print(f"******* Model size: {get_model_size(model):.2f} MB *********", flush=True)

    criterion = nn.CrossEntropyLoss()
    # Use momentum from args
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    y_train = torch.tensor(y[train_idx], dtype=torch.int64).to(device)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Prepare validation data from val_idx.
    X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
    y_val = torch.tensor(y[val_idx], dtype=torch.int64).to(device)

    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Weight synchronization after each epoch:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for param in model.parameters():
                torch.distributed.all_reduce(param.data, op=torch.distributed.ReduceOp.SUM)
                param.data /= torch.distributed.get_world_size()

        # Evaluate on validation data:
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val)
            val_loss = criterion(outputs_val, y_val).item()
            preds_val = outputs_val.argmax(dim=1)
            val_correct = (preds_val == y_val).sum().item()
            total_val = len(y_val)
            val_acc_percent = (val_correct / total_val) * 100.0
        print(f"Epoch {epoch+1}, Process {os.getpid()} Partition {fold+1} - Validation Loss: {val_loss:.4f}, "
              f"Validation Accuracy: {val_correct}/{total_val} = {val_acc_percent:.2f}%", flush=True)

    elapsed_time = time.time() - start_time
    with torch.no_grad():
        outputs = model(X_val)
        correctly_classified_chunk = (outputs.argmax(dim=1) == y_val).sum().item()
        chunk_size = len(y_val)

    print(f"Process {os.getpid()} -> Partition {fold+1}, Correctly Classified: {correctly_classified_chunk}/{chunk_size}, "
          f"Elapsed: {elapsed_time:.4f}s", flush=True)
    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    return correctly_classified_chunk, chunk_size, cpu_state_dict

