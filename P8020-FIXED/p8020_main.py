import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import argparse
import time
import numpy as np
import os
import subprocess

import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method
import torch.distributed as dist
from p8020_train import train_8args, LeNet

###############################################################################
# 1) LeNet for MNIST
###############################################################################
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

###############################################################################
# 2) Build model for MNIST or CIFAR10
###############################################################################
def build_model(ds: str) -> nn.Module:
    from torchvision.models import vgg11
    if ds.upper() == "MNIST":
        return LeNet()
    elif ds.upper() == "CIFAR10":
        return vgg11(num_classes=10)
    else:
        raise ValueError("Unsupported dataset. Must be MNIST or CIFAR10.")

###############################################################################
# 3) Validate function for chunk-level validation
###############################################################################
def validate(model, X_val, y_val, device, criterion):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_val, dtype=torch.float32).to(device)
        targets = torch.tensor(y_val, dtype=torch.int64).to(device)
        outputs = model(inputs)
        val_loss = criterion(outputs, targets).item()
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).sum().item()
        total = len(y_val)
    return val_loss, correct, total

###############################################################################
# 4) Single node training function:
#    Called in parallel for chunk i. Logs the requested info each epoch.
###############################################################################
def single_node_train(
    node_id: int,
    X_chunk: np.ndarray,
    y_chunk: np.ndarray,
    device_str: str,
    ds: str,
    args
):
    """
    node_id: which chunk / node index (1-based)
    X_chunk, y_chunk: the data for this node
    device_str: e.g. 'cuda:0' or 'cpu'
    ds: dataset name
    args: from parser
    """
    device = torch.device(device_str)
    print(f"*** Using device: {device_str} for Node {node_id} ***")

    # Split the node's chunk into training and validation sets
    cX_train, cX_val, cy_train, cy_val = train_test_split(X_chunk, y_chunk, test_size=0.2, random_state=42)

    # Build model and set up optimizer and loss
    model = build_model(ds).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Create DataLoader for the training portion of this chunk
    X_train_t = torch.tensor(cX_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(cy_train, dtype=torch.int64).to(device)
    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    start_time = time.time()
    print(f"Training with {len(cX_train)} examples, validating with {len(cX_val)} examples")

    for epoch in range(1, args.epochs + 1):
        model.train()
        correct_train = 0
        total_train = len(cX_train)

        # Training loop over mini-batches
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            # Weight synchronization after each mini-batch
            if dist.is_available() and dist.is_initialized():
                for param in model.parameters():
                    dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                    param.data /= dist.get_world_size()

            # Track training accuracy for this mini-batch
            preds = outputs.argmax(dim=1)
            correct_train += (preds == batch_y).sum().item()

        # Validation at the end of the epoch
        val_loss, val_correct, val_total = validate(model, cX_val, cy_val, device, criterion)
        train_acc_percent = (correct_train / total_train) * 100.0
        val_acc_percent = (val_correct / val_total) * 100.0

        print(f"Epoch {epoch}, Node {node_id} Validation Loss: {val_loss:.4f}, "
              f"Training Accuracy: {train_acc_percent:.2f}%, "
              f"Validation Accuracy: {val_correct}/{val_total} = {val_acc_percent:.2f}%")
        # (No latency simulation here)

    end_time = time.time()
    node_time = end_time - start_time

    # Final validation results
    val_loss_final, val_correct_final, val_total_final = validate(model, cX_val, cy_val, device, criterion)
    val_acc_final = (val_correct_final / val_total_final) * 100.0

    print(f"Node {node_id} Processing Time: {node_time:.4f} seconds")
    print(f"--- Node {node_id} completed in {node_time:.4f} seconds ---")
    print(f"--- Node {node_id} Accuracy: {val_correct_final}/{val_total_final} = {val_acc_final:.2f}% ---")
    return node_id, val_correct_final, val_total_final, node_time, model

###############################################################################
# 5) test_model to check final test set after all nodes done (optional)
###############################################################################
def test_model(model, X_test, y_test, device):
    model.eval()
    device_ = torch.device(device)
    model = model.to(device_)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device_)
    y_test_t = torch.tensor(y_test, dtype=torch.int64).to(device_)
    with torch.no_grad():
        outputs = model(X_test_t)
        preds = outputs.argmax(dim=1)
        correct = (preds == y_test_t).sum().item()
        total = len(y_test_t)
        return correct, total

###############################################################################
# 6) get_physical_gpu_count
###############################################################################
def get_physical_gpu_count():
    try:
        cmd = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return torch.cuda.device_count()
        lines = result.stdout.strip().split("\n")
        return len(lines)
    except:
        return torch.cuda.device_count()

###############################################################################
# 7) Main
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Detailed multi-processor training with logs.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--processors", type=int, default=1, help="Number of processors (1,2,4,...)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--ds", type=str, default="MNIST", help="MNIST or CIFAR10")

    args = parser.parse_args()

    set_start_method('spawn', force=True)

    # Enforce GPU constraints
    MAX_GPUS = 4
    physical_gpus = get_physical_gpu_count()
    if physical_gpus > MAX_GPUS:
        physical_gpus = MAX_GPUS

    if args.gpu:
        if args.processors > physical_gpus:
            import sys
            sys.exit(
                f"ERROR: You requested {args.processors} GPU processors, "
                f"but only {physical_gpus} GPUs are allowed."
            )
    else:
        available_cores = os.cpu_count()
        if args.processors > available_cores:
            import sys
            sys.exit(
                f"ERROR: You requested {args.processors} CPU processors, "
                f"but only {available_cores} cores are available."
            )

    # Decide device(s)
    if args.gpu and torch.cuda.is_available():
        device_list = [f"cuda:{i}" for i in range(args.processors)]
    else:
        device_list = ["cpu"] * args.processors

    # Load dataset
    if args.ds.upper() == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_set = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        X_full = train_set.data.numpy().reshape(-1, 1, 28, 28) / 255.0
        y_full = train_set.targets.numpy()

        test_set = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
        X_test = test_set.data.numpy().reshape(-1, 1, 28, 28) / 255.0
        y_test = test_set.targets.numpy()

    elif args.ds.upper() == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
        train_set = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
        X_full = train_set.data.transpose((0, 3, 1, 2)) / 255.0
        y_full = np.array(train_set.targets)

        test_set = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
        X_test = test_set.data.transpose((0, 3, 1, 2)) / 255.0
        y_test = np.array(test_set.targets)
    else:
        raise ValueError("Use --ds MNIST or --ds CIFAR10.")

    # Partition X_full among processors
    total_n = len(X_full)
    chunk_size = total_n // args.processors
    remainder = total_n % args.processors

    index_chunks = []
    start_idx = 0
    for i in range(args.processors):
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        index_chunks.append(list(range(start_idx, end_idx)))
        start_idx = end_idx

    total_start = time.time()

    # Single processor (in-line) execution
    if args.processors == 1:
        node_id = 1
        X_chunk = X_full[index_chunks[0]]
        y_chunk = y_full[index_chunks[0]]
        single_node_train(
            node_id,
            X_chunk,
            y_chunk,
            device_list[0],
            args.ds,
            args
        )
        total_end = time.time()
        print(f"Total Time Across Nodes: {total_end - total_start:.4f} seconds.")

    else:
        # Multi-processor path: spawn processes using a Pool
        print(f"*** Running with {args.processors} processors ***")
        param_list = []
        for node_id in range(1, args.processors + 1):
            idx = node_id - 1
            X_chunk = X_full[index_chunks[idx]]
            y_chunk = y_full[index_chunks[idx]]
            param_list.append((node_id, X_chunk, y_chunk, device_list[idx], args.ds, args))

        with mp.Pool(processes=args.processors) as pool:
            results = pool.starmap(single_node_train, param_list)

        total_end = time.time()
        total_time = total_end - total_start
        sum_correct = 0
        sum_total = 0
        sum_time = 0.0
        for r in results:
            sum_correct += r[1]
            sum_total += r[2]
            sum_time += r[3]
        overall_val_acc = 100.0 * sum_correct / sum_total if sum_total > 0 else 0.0

        print(f"--- Combined Node Validation Accuracy: {sum_correct}/{sum_total} = {overall_val_acc:.2f}% ---")
        print(f"Total Time Across Nodes: {total_time:.4f} seconds.")

if __name__ == "__main__":
    set_start_method('spawn', force=True)
    main()
