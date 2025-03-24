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

# Import the distributed training function from t8020_Momentum
from t8020_Momentum import train_8args, LeNet

import optuna

###############################################################################
# Local "build_model" to choose MNIST or CIFAR10
###############################################################################
def build_model(ds: str) -> nn.Module:
    from torchvision.models import vgg11
    if ds.upper() == "MNIST":
        return LeNet()
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
        return model
    else:
        raise ValueError("Unsupported dataset. Must be MNIST or CIFAR10.")

###############################################################################
# Simple validation function for single-node training
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
# Single-node function when processors=1
###############################################################################
def single_node_train(node_id: int, X_chunk: np.ndarray, y_chunk: np.ndarray, device_str: str, ds: str, args):
    device = torch.device(device_str)
    print(f"*** Using device: {device_str} for Node {node_id} ***")
    cX_train, cX_val, cy_train, cy_val = train_test_split(X_chunk, y_chunk, test_size=0.2, random_state=42)
    model = build_model(ds).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

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
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            preds = outputs.argmax(dim=1)
            correct_train += (preds == batch_y).sum().item()

        val_loss, val_correct, val_total = validate(model, cX_val, cy_val, device, criterion)
        train_acc_percent = (correct_train / total_train) * 100.0
        val_acc_percent = (val_correct / val_total) * 100.0

        print(
            f"Epoch {epoch}, Node {node_id} - Validation Loss: {val_loss:.4f}, "
            f"Training Accuracy: {train_acc_percent:.2f}%, "
            f"Validation Accuracy: {val_correct}/{val_total} = {val_acc_percent:.2f}%"
        )

    end_time = time.time()
    node_time = end_time - start_time
    val_loss_final, val_correct_final, val_total_final = validate(model, cX_val, cy_val, device, criterion)
    val_acc_final = (val_correct_final / val_total_final) * 100.0

    print(f"Node {node_id} Processing Time: {node_time:.4f} seconds")
    print(f"--- Node {node_id} completed in {node_time:.4f} seconds ---")
    print(f"--- Node {node_id} Accuracy: {val_correct_final}/{val_total_final} = {val_acc_final:.2f}% ---")
    return node_id, val_correct_final, val_total_final, node_time, model

###############################################################################
# For multi-processor: each worker calls train_8args in t8020_Momentum.py
###############################################################################
def distributed_train_worker(rank, args, X_full, y_full, index_chunks, device_list, return_dict):
    import torch.distributed as dist
    world_size = args.processors

    # Decide backend automatically based on GPU usage
    if args.gpu and torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank
    )
    indices = index_chunks[rank]

    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    print(f"Process {os.getpid()} (Rank {rank}) training on indices {min(indices)}-{max(indices)} with device {device_list[rank]}")

    result = train_8args(rank, train_idx, val_idx, X_full, y_full, device_list[rank], args.ds, args)

    dist.destroy_process_group()
    return_dict[rank] = result

###############################################################################
# Helper to detect GPUs physically available
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
# Main simulation to dispatch single-node or distributed training
###############################################################################
def run_simulation(args):
    MAX_GPUS = 4
    physical_gpus = get_physical_gpu_count()
    if physical_gpus > MAX_GPUS:
        physical_gpus = MAX_GPUS

    # Basic checks
    if args.gpu:
        if args.processors > physical_gpus:
            import sys
            sys.exit(
                f"ERROR: You requested {args.processors} GPU processors, but only {physical_gpus} GPU(s) available."
            )
    else:
        available_cores = os.cpu_count()
        if args.processors > available_cores:
            import sys
            sys.exit(
                f"ERROR: You requested {args.processors} CPU processors, but only {available_cores} cores available."
            )

    # Build device list
    if args.gpu and torch.cuda.is_available():
        device_list = [f"cuda:{i}" for i in range(args.processors)]
    else:
        device_list = ["cpu"] * args.processors

    # Load data
    if args.ds.upper() == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_set = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        X_full = train_set.data.numpy().reshape(-1, 1, 28, 28) / 255.0
        y_full = train_set.targets.numpy()
    elif args.ds.upper() == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
        X_full = train_set.data.transpose((0, 3, 1, 2)) / 255.0
        y_full = np.array(train_set.targets)
    else:
        raise ValueError("Use --ds MNIST or --ds CIFAR10.")

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
    if args.processors == 1:
        node_id = 1
        X_chunk = X_full[index_chunks[0]]
        y_chunk = y_full[index_chunks[0]]
        single_node_train(node_id, X_chunk, y_chunk, device_list[0], args.ds, args)
        total_end = time.time()
        overall_time = total_end - total_start
        overall_val_acc = None
    else:
        print(f"*** Running with {args.processors} processors ***")
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        for rank in range(args.processors):
            p = mp.Process(
                target=distributed_train_worker,
                args=(rank, args, X_full, y_full, index_chunks, device_list, return_dict)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        total_end = time.time()
        overall_time = total_end - total_start

        # Combine correctness from each partition
        sum_correct = 0
        sum_total = 0
        for r in return_dict.values():
            sum_correct += r[0]
            sum_total += r[1]
        overall_val_acc = 100.0 * sum_correct / sum_total if sum_total > 0 else 0.0

        print(f"--- Combined Node Validation Accuracy: {sum_correct}/{sum_total} = {overall_val_acc:.2f}% ---")
        print(f"Total Time Across Nodes: {overall_time:.4f} seconds.")

    return overall_val_acc

###############################################################################
# Optuna objective function for hyperparameter search
###############################################################################
def objective(trial):
    global base_args
    base_args.lr = trial.suggest_float("lr", 0.00010, 0.00100, log=True)
    base_args.momentum = trial.suggest_float("momentum", 0.8, 0.99)
    print(f"Trial {trial.number}: testing lr = {base_args.lr:.6f}, momentum = {base_args.momentum:.2f}")
    overall_val_acc = run_simulation(base_args)
    if overall_val_acc is None:
        overall_val_acc = 0.0
    print(f"Trial {trial.number}: overall validation accuracy = {overall_val_acc:.2f}%")
    return overall_val_acc

###############################################################################
# Main entry point
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="p8020_FIX with Optuna HPO integration.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--processors", type=int, default=1, help="Number of processors (1,2,4,...)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--ds", type=str, default="MNIST", help="MNIST or CIFAR10")
    parser.add_argument("--optuna", action="store_true", help="Use Optuna HPO to optimize LR and momentum")
    args = parser.parse_args()

    # For multi‑GPU, using NCCL requires these environment variables
    if args.gpu and torch.cuda.is_available() and args.processors > 1:
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["UCX_TLS"] = "tcp"
        os.environ["NCCL_P2P_DISABLE"] = "1"

    # Use spawn to avoid fork issues on some systems
    set_start_method("spawn", force=True)

    global base_args
    base_args = args

    if args.optuna:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5)
        print("Hyperparameter Optimization complete!")
        print(f"Best learning rate: {study.best_params['lr']:.6f}")
        print(f"Best momentum: {study.best_params['momentum']:.2f}")
        print(f"Best validation accuracy: {study.best_value:.2f}%")
    else:
        run_simulation(args)

if __name__ == "__main__":
    # Make sure spawn is used so that NCCL/GPU resources are not shared incorrectly.
    set_start_method("spawn", force=True)
    main()

