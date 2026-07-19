"""
REAL P2P-ring validation for UA-DETRAC.

Each process:
  1. trains one ResNet18 replica on a disjoint training shard;
  2. sends its model state to the clockwise neighbour;
  3. receives the counter-clockwise neighbour's state;
  4. averages the local and received states;
  5. evaluates on a disjoint validation shard.

Communication cost counts bytes transmitted once:
    per epoch total CC = n * M
where n is the number of processes and M is the serialized model-state size.

Example:
python3 P2Preal_UA.py \
    --ds UA_DETRAC \
    --processors 2 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --momentum 0.9 \
    --weight_decay 0.001 \
    --gpu
"""

from __future__ import annotations

import argparse
import os
import random
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18, vgg11
from tSYNCreal import LeNet


CLASS_TO_ID = {"mild": 0, "medium": 1, "congested": 2}


class UADetracSceneDataset(Dataset):
    """UA-DETRAC scene-level classification dataset."""

    def __init__(
        self,
        csv_path: str,
        dataset_root: str,
        split: str,
        transform=None,
        limit: int = 0,
    ) -> None:
        base_dir = Path(__file__).resolve().parent
        self.transform = transform

        csv_p = Path(csv_path)
        self.csv_path = (base_dir / csv_p).resolve() if not csv_p.is_absolute() else csv_p.resolve()

        root_p = Path(dataset_root)
        self.dataset_root = (
            (base_dir / root_p).resolve() if not root_p.is_absolute() else root_p.resolve()
        )

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"UA-DETRAC CSV not found: {self.csv_path}")
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"UA-DETRAC dataset root not found: {self.dataset_root}")

        df = pd.read_csv(self.csv_path)
        required = {"split", "image_rel", "scene_name"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")

        df = df[df["split"].astype(str) == split].copy()
        if limit > 0:
            df = df.head(limit).copy()

        df["image_rel"] = df["image_rel"].astype(str).str.replace("\\", "/", regex=False)
        unknown = sorted(set(df["scene_name"].astype(str)) - set(CLASS_TO_ID))
        if unknown:
            raise ValueError(f"Unknown scene labels in CSV: {unknown}")

        self.image_rels = df["image_rel"].tolist()
        self.labels = [CLASS_TO_ID[x] for x in df["scene_name"].astype(str)]

        if not self.image_rels:
            raise ValueError(f"No samples found for split={split!r}")

    def __len__(self) -> int:
        return len(self.image_rels)

    def __getitem__(self, index: int):
        image_path = self.dataset_root / self.image_rels[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, torch.tensor(self.labels[index], dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REAL UA-DETRAC P2P-ring validation")
    parser.add_argument("--ds", "--dataset", dest="ds", default="UA_DETRAC")
    parser.add_argument("--processors", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--ua_resize", type=int, default=224)
    parser.add_argument("--ua_limit", type=int, default=0)

    # Defaults follow the directory structure used by the DLMP UA-DETRAC code.
    parser.add_argument(
        "--csv_path",
        default="../data/UA-DETRAC/DLMP/scene_labels_traffic.csv",
        help="CSV containing split, image_rel, and scene_name columns",
    )
    parser.add_argument(
        "--dataset_root",
        default="../data/UA-DETRAC/dataset/UA_DETRAC_CLEAN/content/UA-DETRAC/DETRAC_Upload",
        help="UA-DETRAC image root",
    )
    parser.add_argument(
        "--master_addr",
        default="127.0.0.1",
        help="Address used by torch.distributed",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=0,
        help="Port used by torch.distributed; 0 selects a free local port",
    )
    return parser.parse_args()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def split_indices(length: int, world_size: int, rank: int) -> List[int]:
    """Contiguous, non-overlapping shard with remainder distributed safely."""
    shards = np.array_split(np.arange(length, dtype=np.int64), world_size)
    return shards[rank].tolist()

def build_array_datasets(args: argparse.Namespace):
    data_root = Path(__file__).resolve().parent.parent / "data"
    ds = args.ds.upper()

    if ds == "MNIST":
        dataset = datasets.MNIST(
            root=str(data_root),
            train=True,
            download=True,
        )
        X = dataset.data.numpy().astype(np.float32)
        y = dataset.targets.numpy().astype(np.int64)

    elif ds == "CIFAR10":
        dataset = datasets.CIFAR10(
            root=str(data_root),
            train=True,
            download=True,
        )
        X = dataset.data.transpose(0, 3, 1, 2).astype(np.float32)
        y = np.asarray(dataset.targets, dtype=np.int64)

    elif ds == "CIFAR100":
        dataset = datasets.CIFAR100(
            root=str(data_root),
            train=True,
            download=True,
        )
        X = dataset.data.transpose(0, 3, 1, 2).astype(np.float32)
        y = np.asarray(dataset.targets, dtype=np.int64)

    else:
        raise ValueError("build_array_datasets supports only MNIST, CIFAR10, and CIFAR100.")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    if ds == "MNIST":
        X_train_t = X_train_t.unsqueeze(1)
        X_val_t = X_val_t.unsqueeze(1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    validation_dataset = TensorDataset(X_val_t, y_val_t)

    return train_dataset, validation_dataset

def create_model(ds: str, device: torch.device) -> nn.Module:
    ds = ds.upper()

    if ds == "MNIST":
        return LeNet().to(device)

    if ds == "CIFAR10":
        model = vgg11(weights=None)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )
        return model.to(device)

    if ds == "CIFAR100":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 100)
        return model.to(device)

    if ds == "UA_DETRAC":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 3)
        return model.to(device)

    raise ValueError("Unsupported dataset. Use MNIST, CIFAR10, CIFAR100, or UA_DETRAC.")


def state_size_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


@torch.no_grad()
def ring_exchange_and_average(model: nn.Module, rank: int, world_size: int) -> None:
    """
    Send current state clockwise and receive from the counter-clockwise neighbour.
    Average the local and received state.

    For n=2, the clockwise and counter-clockwise neighbour are the same rank.
    batch_isend_irecv prevents send/receive deadlock.
    """
    if world_size <= 1:
        return

    send_to = (rank + 1) % world_size
    recv_from = (rank - 1 + world_size) % world_size

    state = model.state_dict()
    received: Dict[str, torch.Tensor] = {}

    for key, local_tensor in state.items():
        send_tensor = local_tensor.detach().contiguous()
        recv_tensor = torch.empty_like(send_tensor)

        operations = [
            dist.P2POp(dist.isend, send_tensor, send_to),
            dist.P2POp(dist.irecv, recv_tensor, recv_from),
        ]
        requests = dist.batch_isend_irecv(operations)
        for request in requests:
            request.wait()

        received[key] = recv_tensor

    averaged: Dict[str, torch.Tensor] = {}
    for key, local_tensor in state.items():
        peer_tensor = received[key]
        if torch.is_floating_point(local_tensor):
            averaged[key] = (local_tensor + peer_tensor) / 2.0
        else:
            # Integer buffers, such as BatchNorm counters, cannot be averaged safely.
            averaged[key] = local_tensor

    model.load_state_dict(averaged, strict=True)
    dist.barrier()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, int, int]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.size(0))

    return running_loss / max(total, 1), correct, total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, int, int]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.size(0))

    return running_loss / max(total, 1), correct, total


def worker(
    rank: int,
    args: argparse.Namespace,
    shared_results,
) -> None:
    world_size = args.processors

    use_cuda = args.gpu and torch.cuda.is_available()
    if use_cuda:
        if torch.cuda.device_count() < world_size:
            raise RuntimeError(
                f"Requested {world_size} GPU processes, but only "
                f"{torch.cuda.device_count()} CUDA devices are visible."
            )
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    if world_size > 1:
        os.environ["MASTER_ADDR"] = args.master_addr
        os.environ["MASTER_PORT"] = str(args.master_port)
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )

    if args.ds.upper() == "UA_DETRAC":
        resize = args.ua_resize
        transform = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        train_dataset = UADetracSceneDataset(
            csv_path=args.csv_path,
            dataset_root=args.dataset_root,
            split="train",
            transform=transform,
            limit=args.ua_limit,
        )

        validation_dataset = UADetracSceneDataset(
            csv_path=args.csv_path,
            dataset_root=args.dataset_root,
            split="val",
            transform=transform,
            limit=args.ua_limit,
        )

    else:
        train_dataset, validation_dataset = build_array_datasets(args)

    train_indices = split_indices(len(train_dataset), world_size, rank)
    validation_indices = split_indices(len(validation_dataset), world_size, rank)

    workers = max(0, args.num_workers)

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=use_cuda,
        persistent_workers=(workers > 0),
    )

    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=use_cuda,
        persistent_workers=(workers > 0),
    )

    model = create_model(args.ds, device)

    if world_size > 1:
        for tensor in model.state_dict().values():
            dist.broadcast(tensor, src=0)

    criterion = nn.CrossEntropyLoss()

    model_bytes = state_size_bytes(model)
    per_epoch_cc = model_bytes if world_size > 1 else 0
    cumulative_cc = 0

    if rank == 0:
        if world_size == 1:
            print(f"*** Using device: {device} for {args.ds.upper()} single-node P2P validation ***")
        else:
            print(f"*** Running {args.ds.upper()} REAL P2P-ring with {world_size} processors ***")

    print(
        f"Process {os.getpid()} (Rank {rank}) training on "
        f"{len(train_indices)} images with device {device}",
        flush=True,
    )

    start_time = time.time()
    final_correct = 0
    final_total = 0

    for epoch in range(1, args.epochs + 1):
        
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.time()
        optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
        _, train_correct, train_total = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        ring_exchange_and_average(model, rank, world_size)
        cumulative_cc += per_epoch_cc

        validation_loss, validation_correct, validation_total = evaluate(
            model, validation_loader, criterion, device
        )
        final_correct = validation_correct
        final_total = validation_total

        train_accuracy = 100.0 * train_correct / max(train_total, 1)
        validation_accuracy = 100.0 * validation_correct / max(validation_total, 1)
        
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        epoch_elapsed = time.time() - epoch_start
        cumulative_elapsed = time.time() - start_time

        label = "Node 1" if world_size == 1 else f"Rank {rank}"
        print(
            f"Epoch {epoch}, {label} - Validation Loss: {validation_loss:.4f}, "
            f"Training Accuracy: {train_accuracy:.2f}%, "
            f"Validation Accuracy: {validation_correct}/{validation_total} "
            f"= {validation_accuracy:.2f}%, "
            f"Communication Cost: {per_epoch_cc} bytes, "
            f"Epoch Elapsed Time: {epoch_elapsed:.4f} seconds, "
            f"Cumulative Elapsed Time: {cumulative_elapsed:.4f} seconds",
            flush=True,
        )

    elapsed = time.time() - start_time

    shared_results[rank] = {
        "correct": final_correct,
        "total": final_total,
        "time": elapsed,
        "cc": cumulative_cc,
    }

    label = "Node 1" if world_size == 1 else f"Rank {rank}"
    print(f"{label} Processing Time: {elapsed:.4f} seconds", flush=True)
    print(
        f"--- {label} Accuracy: {final_correct}/{final_total} "
        f"= {100.0 * final_correct / max(final_total, 1):.2f}% ---",
        flush=True,
    )
    print(
        f"{label} Grand Total Communication Cost: {cumulative_cc} bytes",
        flush=True,
    )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()

    supported_datasets = {"MNIST", "CIFAR10", "CIFAR100", "UA_DETRAC"}

    if args.ds.upper() not in supported_datasets:
        raise ValueError(
            "Supported datasets are MNIST, CIFAR10, CIFAR100, and UA_DETRAC."
        )
    if args.processors < 1:
        raise ValueError("--processors must be at least 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")

    if args.master_port == 0:
        args.master_port = find_free_port()

    total_start = time.time()

    manager = mp.Manager()
    shared_results = manager.dict()

    if args.processors == 1:
        worker(0, args, shared_results)
    else:
        mp.spawn(
            worker,
            args=(args, shared_results),
            nprocs=args.processors,
            join=True,
        )

    results = [shared_results[i] for i in range(args.processors)]
    combined_correct = sum(int(item["correct"]) for item in results)
    combined_total = sum(int(item["total"]) for item in results)
    grand_total_cc = sum(int(item["cc"]) for item in results)
    max_node_time = max(float(item["time"]) for item in results)
    wall_time = time.time() - total_start

    print(
        f"--- Combined Node Validation Accuracy: "
        f"{combined_correct}/{combined_total} "
        f"= {100.0 * combined_correct / max(combined_total, 1):.2f}% ---"
    )
    print(f"--- Grand Total Communication Cost: {grand_total_cc} bytes ---")
    if args.processors > 1:
        print(f"Max Node Time: {max_node_time:.4f} seconds.")
    print(f"Total Time Across Nodes: {wall_time:.4f} seconds.")


if __name__ == "__main__":
    # Required for torch.multiprocessing on platforms using spawn.
    mp.set_start_method("spawn", force=True)
    main()
