"""
DLMP: Deep Learning Multi-Processing Simulator

Author: Jorge A. Lopez
Affiliation: Toronto Metropolitan University

Description:
Agent-based simulation framework for studying coordination strategies
in distributed deep learning systems.

Repository:
https://github.com/DLMPsim/DLMP

License:
MIT License
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg11, resnet18, ResNet18_Weights # now includes resnet18
import time
from sklearn.model_selection import train_test_split

from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

#---------------------------------------------------------------------------
# UA-DETRAC dataset
#--------------------------------------------------------------------------
class UADetracSceneDataset(Dataset):
    """
    UA-DETRAC scene-level classification dataset (mild/medium/congested).
    Uses DLMP-generated scene_labels_traffic.csv.
    Returns (image_tensor, class_id).
    """
    CLASS_TO_ID = {"mild": 0, "medium": 1, "congested": 2}

    def __init__(self, csv_path, dataset_root, split, transform=None, limit=0):
        self.dataset_root = Path("../data/UA-DETRAC/dataset/UA_DETRAC_CLEAN/content/UA-DETRAC/DETRAC_Upload").resolve()
        self.split = split
        self.transform = transform
        
        base_dir = Path(__file__).resolve().parent
        self.csv_path = base_dir / csv_path
        df = pd.read_csv(self.csv_path)

        # Keep only the requested split
        df = df[df["split"] == split].copy()

        # Optional limit for smoke tests (limit > 0)
        if limit and int(limit) > 0:
            df = df.head(int(limit)).copy()

        # Normalize rel paths for Linux even if CSV was generated on Windows
        df["image_rel"] = df["image_rel"].astype(str).str.replace("\\", "/", regex=False)

        self.image_rels = df["image_rel"].tolist()
        self.labels = [self.CLASS_TO_ID[s] for s in df["scene_name"].astype(str).tolist()]

        assert len(self.image_rels) == len(self.labels) and len(self.image_rels) > 0, \
            f"Empty UA-DETRAC split={split} after filtering."

    def __len__(self):
        return len(self.image_rels)

    def __getitem__(self, idx):
        y = self.labels[idx]
        
        rel = Path(self.image_rels[idx])
        img_path = (self.dataset_root / rel).resolve()
        img = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(y, dtype=torch.long)

def get_model(arch, num_classes=10, pretrained=False):

    if arch == 'LeNet5':
        model = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(4 * 4 * 50, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes)
        )
        return model
    elif arch == 'VGG11':
        model = vgg11(weights=None)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        return model
    elif arch == 'ResNet18':
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    else:
        raise ValueError("Unsupported architecture. Use 'LeNet5', 'VGG11', or 'ResNet18'.")

def build_loaders(Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args):
    from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset as TorchDataset
    import torch

    dataset_mode = isinstance(Training_ds, TorchDataset) and isinstance(Testing_ds, TorchDataset)

    if dataset_mode:
        # Training_lbls and Testing_lbls are index lists in dataset mode.
        train_subset = Subset(Training_ds, list(map(int, Training_lbls)))
        test_subset = Subset(Testing_ds, list(map(int, Testing_lbls)))

        # Keep the number of workers bounded across processors.
 
        num_workers = max(1, 8 // args.processors)
        persistent_workers = (num_workers > 0)
        pin_memory = (device.type == "cuda")

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )
        return train_loader, test_loader, True, None

    # Existing MNIST/CIFAR behavior (array-mode)
    X_train = torch.tensor(Training_ds, dtype=torch.float32).to(device)
    y_train = torch.tensor(Training_lbls, dtype=torch.long).to(device)
    X_test = torch.tensor(Testing_ds, dtype=torch.float32).to(device)
    y_test = torch.tensor(Testing_lbls, dtype=torch.long).to(device)

    if args.ds.upper() == 'MNIST' and X_train.ndim == 3:
        X_train = X_train.unsqueeze(1)
        X_test = X_test.unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # In array mode, evaluation uses direct tensor-based inference.
    test_loader = None

    return train_loader, test_loader, False, (X_test, y_test)


# ------------------------------------------------------------------------- #
#  Training routine
# ------------------------------------------------------------------------- #
def train_simulated(
    unique_id,
    model,
    Training_ds,
    Training_lbls,
    Testing_ds,
    Testing_lbls,
    device,
    args,
    sync_callback,
    epochs_override=None,
    train_loader=None,
    test_loader=None,
    dataset_mode=None,
    array_eval=None
):



    start_time = time.time()

    # -----------------------------
    # Build loaders or reuse cached ones
    # -----------------------------
    if train_loader is None:
        train_loader, test_loader, dataset_mode_built, array_eval_built = build_loaders(
            Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args
        )
        dataset_mode = dataset_mode_built
        array_eval = array_eval_built
    else:
        # Reuse cached loaders provided by the caller.
        if dataset_mode is None:
            dataset_mode = (test_loader is not None)
        # In array mode, evaluation uses X_test and y_test stored in array_eval.
        pass

    # Unpack array-mode eval tensors if applicable
    if not dataset_mode:
        if array_eval is None:
            # Fall back to rebuilding loaders to preserve the original behavior.
            train_loader, test_loader, dataset_mode_built, array_eval = build_loaders(
                Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args
            )
            dataset_mode = dataset_mode_built
        X_test, y_test = array_eval


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9 if args.optimizer == 'SGDM' else 0.0)

    running_loss = 0.0
    total_samples = 0
    correct_train = 0

    # Run one epoch only per call
    model.train()
    for inputs, labels in train_loader:
        if dataset_mode:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total_samples += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()

    avg_loss = running_loss / max(1, len(train_loader))
    train_acc = 100.0 * correct_train / max(1, total_samples)

    print(f"Node {unique_id + 1}: Loss = {avg_loss:.4f}, Accuracy = {train_acc:.2f}%")

    # Synchronize weights once per call.
    sync_callback(model.state_dict())

    # -----------------------------
    # Evaluation
    # -----------------------------
    model.eval()
    with torch.no_grad():
        if dataset_mode:
            correct_test = 0
            total_test = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()

            test_acc = 100.0 * correct_test / max(1, total_test)
        else:
            outputs = model(X_test)
            _, predicted = outputs.max(1)
            total_test = y_test.size(0)
            correct_test = predicted.eq(y_test).sum().item()
            test_acc = 100.0 * correct_test / total_test if total_test else 0.0

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Node {unique_id + 1} Final Validation: Accuracy = {test_acc:.2f}%\n")

    return avg_loss, test_acc, correct_test, total_test, processing_time, correct_train, total_samples
