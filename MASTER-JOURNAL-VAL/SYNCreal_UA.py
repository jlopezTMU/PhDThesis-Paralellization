import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from PIL import Image

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from tSYNCreal import LeNet, train_8args


# -------------------------------------------------------------------------
# UA-DETRAC streaming dataset
# -------------------------------------------------------------------------
class UADetracSceneDataset(Dataset):
    """
    UA-DETRAC scene-level classification dataset.
    It streams images from disk using the CSV file generated for DLMP.
    This avoids loading the full UA-DETRAC image dataset into RAM/GPU memory.
    """
    CLASS_TO_ID = {"mild": 0, "medium": 1, "congested": 2}

    def __init__(self, csv_path, dataset_root, split, transform=None, limit=0):
        base_dir = Path(__file__).resolve().parent

        csv_p = Path(csv_path)
        self.csv_path = (base_dir / csv_p) if not csv_p.is_absolute() else csv_p
        self.csv_path = self.csv_path.resolve()

        root_p = Path(dataset_root)
        self.dataset_root = (base_dir / root_p) if not root_p.is_absolute() else root_p
        self.dataset_root = self.dataset_root.resolve()

        if self.dataset_root.is_file():
            raise ValueError(
                f"UA-DETRAC dataset_root points to a FILE, not a directory: {self.dataset_root}\n"
                f"Expected a directory like .../DETRAC_Upload"
            )
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"UA-DETRAC dataset_root not found: {self.dataset_root}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"UA-DETRAC CSV not found: {self.csv_path}")

        self.split = split
        self.transform = transform

        df = pd.read_csv(self.csv_path)
        df = df[df["split"] == split].copy()

        if limit and int(limit) > 0:
            df = df.head(int(limit)).copy()

        df["image_rel"] = df["image_rel"].astype(str).str.replace("\\", "/", regex=False)

        self.image_rels = df["image_rel"].tolist()
        self.labels = [self.CLASS_TO_ID[s] for s in df["scene_name"].astype(str).tolist()]

        assert len(self.image_rels) == len(self.labels) and len(self.image_rels) > 0, \
            f"Empty UA-DETRAC split={split} after filtering."

    def __len__(self):
        return len(self.image_rels)

    def __getitem__(self, idx):
        img_path = self.dataset_root / self.image_rels[idx]
        y = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(y, dtype=torch.long)


def build_model(ds: str) -> nn.Module:
    from torchvision.models import vgg11, resnet18, ResNet18_Weights

    if ds.upper() == "MNIST":
        return LeNet()

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
        return model

    if ds.upper() == "UA_DETRAC":
        # Match DLMP UA-DETRAC: ResNet18 with 3 scene classes.
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 3)
        return model

    raise ValueError("Unsupported dataset. Must be MNIST, CIFAR10, or UA_DETRAC.")


# -------------------------------------------------------------------------
# Existing MNIST/CIFAR validation path. Do not change.
# -------------------------------------------------------------------------
def validate(model, X_val, y_val, device, criterion):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_val, dtype=torch.float32).to(device)
        if getattr(model, 'conv1', None) is not None and inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)
        targets = torch.tensor(y_val, dtype=torch.int64).to(device)
        outputs = model(inputs)
        val_loss = criterion(outputs, targets).item()
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).sum().item()
        total = len(y_val)
    return val_loss, correct, total


def single_node_train(node_id: int, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                      device_str: str, ds: str, args):
    device = torch.device(device_str)
    print(f"*** Using device: {device_str} for Node {node_id} ***")
    model = build_model(ds).to(device)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.int64).to(device)
    if ds.upper() == "MNIST" and X_train_t.ndim == 3:
        X_train_t = X_train_t.unsqueeze(1)
    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    start_time = time.time()
    print(f"Training with {len(X_train)} examples, validating with {len(X_val)} examples")
    for epoch in range(1, args.epochs + 1):
        # Match DLMP SYNC: recreate optimizer every global epoch.
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        model.train()
        correct_train = 0
        total_train = len(X_train)
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            preds = outputs.argmax(dim=1)
            correct_train += (preds == batch_y).sum().item()

        val_loss, val_correct, val_total = validate(model, X_val, y_val, device, criterion)
        train_acc_percent = (correct_train / total_train) * 100.0 if total_train > 0 else 0.0
        val_acc_percent = (val_correct / val_total) * 100.0 if val_total > 0 else 0.0
        print(
            f"Epoch {epoch}, Node {node_id} - Validation Loss: {val_loss:.4f}, "
            f"Training Accuracy: {train_acc_percent:.2f}%, "
            f"Validation Accuracy: {val_correct}/{val_total} = {val_acc_percent:.2f}%"
        )

    end_time = time.time()
    node_time = end_time - start_time
    val_loss_final, val_correct_final, val_total_final = validate(model, X_val, y_val, device, criterion)
    val_acc_final = (val_correct_final / val_total_final) * 100.0 if val_total_final > 0 else 0.0

    print(f"Node {node_id} Processing Time: {node_time:.4f} seconds")
    print(f"--- Node {node_id} completed in {node_time:.4f} seconds ---")
    print(f"--- Node {node_id} Accuracy: {val_correct_final}/{val_total_final} = {val_acc_final:.2f}% ---")
    return node_id, val_correct_final, val_total_final, node_time, model


def distributed_train_worker(rank, args, X_train_global, y_train_global, X_val_global, y_val_global,
                             train_index_chunks, val_index_chunks, device_list, return_dict):
    import torch.distributed as dist

    world_size = args.processors
    backend = "nccl" if args.gpu and torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method="tcp://127.0.0.1:29501",
        world_size=world_size,
        rank=rank,
    )

    train_idx = train_index_chunks[rank]
    val_idx = val_index_chunks[rank]

    print(
        f"Process {os.getpid()} (Rank {rank}) training on train indices {min(train_idx)}-{max(train_idx)} "
        f"with device {device_list[rank]}",
        flush=True,
    )

    result = train_8args(
        rank,
        train_idx,
        val_idx,
        X_train_global,
        y_train_global,
        X_val_global,
        y_val_global,
        device_list[rank],
        args.ds,
        args,
    )

    dist.destroy_process_group()
    return_dict[rank] = result


# -------------------------------------------------------------------------
# UA-DETRAC real synchronous validation path
# -------------------------------------------------------------------------
def build_uadetrac_datasets(args):
    resize_n = int(args.ua_resize)

    t_list = [
        transforms.Resize((resize_n, resize_n)),
        transforms.ToTensor(),
    ]

    # DLMP forces ImageNet normalization for UA-DETRAC because it uses pretrained ResNet18.
    if args.ua_use_imagenet_norm:
        t_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    ua_transform = transforms.Compose(t_list)

    train_ds = UADetracSceneDataset(
        csv_path=args.ua_csv,
        dataset_root=args.ua_root,
        split="train",
        transform=ua_transform,
        limit=args.ua_limit,
    )

    val_ds = UADetracSceneDataset(
        csv_path=args.ua_csv,
        dataset_root=args.ua_root,
        split="val",
        transform=ua_transform,
        limit=args.ua_limit,
    )

    return train_ds, val_ds


def get_model_size_bytes(model):
    return sum(p.nelement() * p.element_size() for p in model.parameters())


def validate_loader(model, val_loader, device, criterion):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            batch_n = batch_y.size(0)
            loss_sum += loss.item() * batch_n
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_n

    avg_loss = loss_sum / total if total > 0 else 0.0
    return avg_loss, correct, total


def average_model_parameters(model):
    import torch.distributed as dist

    world_size = dist.get_world_size()
    with torch.no_grad():
        for param in model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= world_size


def train_one_epoch_loader(model, train_loader, device, optimizer, criterion):
    model.train()
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    return correct, total


def single_node_train_uadetrac(device_str: str, args):
    device = torch.device(device_str)
    print(f"*** Using device: {device_str} for UA-DETRAC single-node REAL validation ***")

    train_ds, val_ds = build_uadetrac_datasets(args)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = build_model("UA_DETRAC").to(device)
    criterion = nn.CrossEntropyLoss()

    print(f"Training with {len(train_ds)} UA-DETRAC images, validating with {len(val_ds)} images")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        # Match DLMP SYNC: recreate optimizer every global epoch.
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        train_correct, train_total = train_one_epoch_loader(model, train_loader, device, optimizer, criterion)
        val_loss, val_correct, val_total = validate_loader(model, val_loader, device, criterion)

        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0

        print(
            f"Epoch {epoch}, Node 1 - Validation Loss: {val_loss:.4f}, "
            f"Training Accuracy: {train_acc:.2f}%, "
            f"Validation Accuracy: {val_correct}/{val_total} = {val_acc:.2f}%"
        )

    node_time = time.time() - start_time
    val_loss_final, val_correct_final, val_total_final = validate_loader(model, val_loader, device, criterion)
    val_acc_final = 100.0 * val_correct_final / val_total_final if val_total_final > 0 else 0.0

    print(f"Node 1 Processing Time: {node_time:.4f} seconds")
    print(f"--- Node 1 completed in {node_time:.4f} seconds ---")
    print(f"--- Node 1 Accuracy: {val_correct_final}/{val_total_final} = {val_acc_final:.2f}% ---")

    return 1, val_correct_final, val_total_final, node_time, model


def distributed_train_worker_uadetrac(rank, args, train_index_chunks, val_index_chunks, device_list, return_dict):
    import torch.distributed as dist

    world_size = args.processors
    backend = "nccl" if args.gpu and torch.cuda.is_available() else "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="tcp://127.0.0.1:29501",
        world_size=world_size,
        rank=rank,
    )

    device = torch.device(device_list[rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)

    train_ds, val_ds = build_uadetrac_datasets(args)

    train_idx = train_index_chunks[rank]
    val_idx = val_index_chunks[rank]

    print(
        f"Process {os.getpid()} (Rank {rank}) training on UA-DETRAC train indices "
        f"{min(train_idx)}-{max(train_idx)} with device {device_list[rank]}",
        flush=True,
    )

    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(val_ds, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = build_model("UA_DETRAC").to(device)
    criterion = nn.CrossEntropyLoss()
    model_size_bytes = get_model_size_bytes(model)

    start_time = time.time()
    total_comm_cost = 0

    for epoch in range(1, args.epochs + 1):
        # Match DLMP SYNC: recreate optimizer every global epoch.
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        train_correct, train_total = train_one_epoch_loader(model, train_loader, device, optimizer, criterion)

        # REAL synchronous-central validation:
        # each rank trains locally, then all ranks average model parameters.
        average_model_parameters(model)

        # Match the DLMP/SYNC communication accounting convention:
        # each node sends and receives a full model to/from the synchronization process.
        epoch_comm_cost = (world_size - 1) * model_size_bytes * 2
        total_comm_cost += epoch_comm_cost

        val_loss, val_correct, val_total = validate_loader(model, val_loader, device, criterion)

        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0

        print(
            f"Epoch {epoch}, Rank {rank} - Validation Loss: {val_loss:.4f}, "
            f"Training Accuracy: {train_acc:.2f}%, "
            f"Validation Accuracy: {val_correct}/{val_total} = {val_acc:.2f}%, "
            f"Communication Cost: {epoch_comm_cost} bytes",
            flush=True,
        )

    node_time = time.time() - start_time
    val_loss_final, val_correct_final, val_total_final = validate_loader(model, val_loader, device, criterion)

    print(f"Rank {rank} Processing Time: {node_time:.4f} seconds", flush=True)
    print(
        f"--- Rank {rank} Accuracy: {val_correct_final}/{val_total_final} = "
        f"{(100.0 * val_correct_final / val_total_final if val_total_final else 0.0):.2f}% ---",
        flush=True,
    )

    dist.destroy_process_group()
    return_dict[rank] = (val_correct_final, val_total_final, total_comm_cost, node_time)


def get_physical_gpu_count():
    try:
        cmd = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return torch.cuda.device_count()
        return len(result.stdout.strip().split("\n")) if result.stdout.strip() else torch.cuda.device_count()
    except Exception:
        return torch.cuda.device_count()


def _split_indices_evenly(total_n: int, parts: int):
    chunk_size = total_n // parts
    remainder = total_n % parts
    chunks = []
    start_idx = 0
    for i in range(parts):
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        chunks.append(list(range(start_idx, end_idx)))
        start_idx = end_idx
    return chunks


def run_simulation(args):
    MAX_GPUS = 4
    physical_gpus = min(get_physical_gpu_count(), MAX_GPUS)

    if args.gpu:
        if args.processors > physical_gpus:
            import sys
            sys.exit(f"ERROR: You requested {args.processors} GPU processors, but only {physical_gpus} GPU(s) available.")
    else:
        available_cores = os.cpu_count()
        if args.processors > available_cores:
            import sys
            sys.exit(f"ERROR: You requested {args.processors} CPU processors, but only {available_cores} cores available.")

    device_list = [f"cuda:{i}" for i in range(args.processors)] if args.gpu and torch.cuda.is_available() else ["cpu"] * args.processors

    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

    # ------------------------------------------------------------------
    # UA-DETRAC path: streaming Dataset/DataLoader. Does not preload images.
    # ------------------------------------------------------------------
    if args.ds.upper() == "UA_DETRAC":
        # Force DLMP-compatible UA-DETRAC settings.
        args.ua_use_imagenet_norm = True

        # Build once here only to obtain lengths and fail fast if paths are wrong.
        train_ds, val_ds = build_uadetrac_datasets(args)
        train_index_chunks = _split_indices_evenly(len(train_ds), args.processors)
        val_index_chunks = _split_indices_evenly(len(val_ds), args.processors)

        total_start = time.time()

        if args.processors == 1:
            result = single_node_train_uadetrac(device_list[0], args)
            overall_val_acc = 100.0 * result[1] / result[2] if result[2] > 0 else 0.0
            print(f"--- Combined Node Validation Accuracy: {result[1]}/{result[2]} = {overall_val_acc:.2f}% ---")
            print(f"Total Time Across Nodes: {time.time() - total_start:.4f} seconds.")
            return overall_val_acc

        print(f"*** Running UA-DETRAC REAL SYNC with {args.processors} processors ***")
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []

        for rank in range(args.processors):
            p = mp.Process(
                target=distributed_train_worker_uadetrac,
                args=(rank, args, train_index_chunks, val_index_chunks, device_list, return_dict),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        for p in processes:
            if p.exitcode != 0:
                raise RuntimeError(f"UA-DETRAC worker process failed with exit code {p.exitcode}")

        sum_correct = 0
        sum_total = 0
        sum_comm_cost = 0
        max_node_time = 0.0

        for r in return_dict.values():
            sum_correct += r[0]
            sum_total += r[1]
            sum_comm_cost += r[2]
            max_node_time = max(max_node_time, r[3])

        overall_val_acc = 100.0 * sum_correct / sum_total if sum_total > 0 else 0.0
        print(f"--- Combined Node Validation Accuracy: {sum_correct}/{sum_total} = {overall_val_acc:.2f}% ---")
        print(f"--- Grand Total Communication Cost: {sum_comm_cost} bytes ---")
        print(f"Max Node Time: {max_node_time:.4f} seconds.")
        print(f"Total Time Across Nodes: {time.time() - total_start:.4f} seconds.")
        return overall_val_acc

    # ------------------------------------------------------------------
    # Existing MNIST/CIFAR10 path. Preserved.
    # ------------------------------------------------------------------
    # Match DLMP mainMASCNN loading behavior as closely as possible.
    if args.ds.upper() == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_set = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
        X_full = train_set.data.numpy().astype(np.float32)
        y_full = train_set.targets.numpy()
    elif args.ds.upper() == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_set = datasets.CIFAR10(root=data_root, train=True, transform=transform, download=True)
        X_full = train_set.data.transpose((0, 3, 1, 2)).astype(np.float32)
        y_full = np.array(train_set.targets)
    else:
        raise ValueError("Use --ds MNIST, --ds CIFAR10, or --ds UA_DETRAC.")

    # Match DLMP SYNC split order: global split first, then shard train and val separately.
    X_train_global, X_val_global, y_train_global, y_val_global = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    train_index_chunks = _split_indices_evenly(len(X_train_global), args.processors)
    val_index_chunks = _split_indices_evenly(len(X_val_global), args.processors)

    total_start = time.time()
    if args.processors == 1:
        single_node_train(1, X_train_global, y_train_global, X_val_global, y_val_global, device_list[0], args.ds, args)
        overall_val_acc = None
    else:
        print(f"*** Running with {args.processors} processors ***")
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        for rank in range(args.processors):
            p = mp.Process(
                target=distributed_train_worker,
                args=(rank, args, X_train_global, y_train_global, X_val_global, y_val_global,
                      train_index_chunks, val_index_chunks, device_list, return_dict),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        sum_correct = 0
        sum_total = 0
        sum_comm_cost = 0
        for r in return_dict.values():
            sum_correct += r[0]
            sum_total += r[1]
            sum_comm_cost += r[2]

        overall_val_acc = 100.0 * sum_correct / sum_total if sum_total > 0 else 0.0
        print(f"--- Combined Node Validation Accuracy: {sum_correct}/{sum_total} = {overall_val_acc:.2f}% ---")
        print(f"--- Grand Total Communication Cost: {sum_comm_cost} bytes ---")
        print(f"Total Time Across Nodes: {time.time() - total_start:.4f} seconds.")

    return overall_val_acc


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


def main():
    parser = argparse.ArgumentParser(description="p8020_FIX with Optuna HPO integration.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--processors", type=int, default=1, help="Number of processors (1,2,4,...)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--ds", type=str, default="MNIST", help="MNIST, CIFAR10, or UA_DETRAC")
    parser.add_argument("--optuna", action="store_true", help="Use Optuna HPO to optimize LR and momentum")

    # UA-DETRAC additions. Existing MNIST/CIFAR10 behavior is unchanged.
    parser.add_argument("--ua_resize", type=int, default=224, help="Resize UA-DETRAC frames to NxN.")
    parser.add_argument("--ua_limit", type=int, default=0, help="If >0, limit UA-DETRAC train/val samples for smoke tests.")
    parser.add_argument("--ua_use_imagenet_norm", action="store_true", help="Use ImageNet normalization for UA-DETRAC.")
    parser.add_argument(
        "--ua_csv",
        type=str,
        default="../data/UA-DETRAC/DLMP/scene_labels_traffic.csv",
        help="Path to DLMP UA-DETRAC scene_labels_traffic.csv.",
    )
    parser.add_argument(
        "--ua_root",
        type=str,
        default="../data/UA-DETRAC/dataset/UA_DETRAC_CLEAN/content/UA-DETRAC/DETRAC_Upload",
        help="Path to UA-DETRAC DETRAC_Upload image root directory.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="DataLoader workers per process. Use 0 or 1 first for UA-DETRAC stability.",
    )

    args = parser.parse_args()

    if args.ds.upper() == "UA_DETRAC":
        # DLMP uses pretrained ResNet18 and ImageNet normalization for UA-DETRAC.
        args.ua_use_imagenet_norm = True

    if args.gpu and torch.cuda.is_available() and args.processors > 1:
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["UCX_TLS"] = "tcp"
        os.environ["NCCL_P2P_DISABLE"] = "1"

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
    set_start_method("spawn", force=True)
    main()

