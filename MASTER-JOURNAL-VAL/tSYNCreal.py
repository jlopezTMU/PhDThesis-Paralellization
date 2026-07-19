import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed
from torchvision.models import vgg11


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = torch.relu(nn.MaxPool2d(2)(self.conv1(x)))
        x = torch.relu(nn.MaxPool2d(2)(self.conv2(x)))
        x = x.view(-1, 50 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model_size(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def _build_model(ds: str, device: str) -> nn.Module:
    if ds.upper() == "MNIST":
        return LeNet().to(device)
    if ds.upper() == "CIFAR10":
        model = vgg11(num_classes=10)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )
        return model.to(device)
    raise ValueError("Unsupported dataset. Must be MNIST or CIFAR10.")


def train_8args(fold, train_idx, val_idx, X_train_global, y_train_global, X_val_global, y_val_global, device, ds, args):
    """
    REAL SYNC aligned to DLMP SYNC semantics as closely as possible without changing the CLI:
    - global 80/20 split is done outside and passed in
    - each rank gets a shard of the global train and global validation sets
    - one local epoch then synchronous weight averaging
    - optimizer is re-created each epoch to match DLMP train_simulated() behavior
    - communication accounting matches DLMP SYNC: 2 * (n-1) * model_size per node per epoch
    """
    print(f"Process {os.getpid()} is using device: {device}", flush=True)

    model = _build_model(ds, device)
    print(f"******* Model size: {get_model_size(model):.2f} MB *********", flush=True)

    criterion = nn.CrossEntropyLoss()

    X_train = torch.tensor(X_train_global[train_idx], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_global[train_idx], dtype=torch.int64).to(device)
    if ds.upper() == "MNIST" and X_train.ndim == 3:
        X_train = X_train.unsqueeze(1)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    X_val = torch.tensor(X_val_global[val_idx], dtype=torch.float32).to(device)
    if ds.upper() == "MNIST" and X_val.ndim == 3:
        X_val = X_val.unsqueeze(1)
    y_val = torch.tensor(y_val_global[val_idx], dtype=torch.int64).to(device)

    cumulative_cost = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        # Match DLMP SYNC: optimizer state does NOT persist across global epochs.
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        epoch_comm_cost = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            for param in model.parameters():
                param_bytes = param.nelement() * param.element_size()
                epoch_comm_cost += 2 * (world_size - 1) * param_bytes
                torch.distributed.all_reduce(param.data, op=torch.distributed.ReduceOp.SUM)
                param.data /= world_size

        cumulative_cost += epoch_comm_cost
        print(f"Process {os.getpid()} - Epoch {epoch + 1} communication cost: {epoch_comm_cost} bytes", flush=True)
        print(f"Process {os.getpid()} - Cumulative communication cost: {cumulative_cost} bytes", flush=True)

        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val)
            val_loss = criterion(outputs_val, y_val).item()
            preds_val = outputs_val.argmax(dim=1)
            val_correct = (preds_val == y_val).sum().item()
            total_val = len(y_val)
            val_acc_percent = (val_correct / total_val) * 100.0 if total_val > 0 else 0.0

        print(
            f"Epoch {epoch + 1}, Process {os.getpid()} Partition {fold + 1} - Validation Loss: {val_loss:.4f}, "
            f"Validation Accuracy: {val_correct}/{total_val} = {val_acc_percent:.2f}%",
            flush=True,
        )

    elapsed_time = time.time() - start_time
    with torch.no_grad():
        outputs = model(X_val)
        correctly_classified_chunk = (outputs.argmax(dim=1) == y_val).sum().item()
        chunk_size = len(y_val)

    print(
        f"Process {os.getpid()} -> Partition {fold + 1}, Correctly Classified: {correctly_classified_chunk}/{chunk_size}, "
        f"Elapsed: {elapsed_time:.4f}s",
        flush=True,
    )
    print(f"Process {os.getpid()} - Grand Total Communication Cost: {cumulative_cost} bytes", flush=True)
    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    return correctly_classified_chunk, chunk_size, cumulative_cost, cpu_state_dict
