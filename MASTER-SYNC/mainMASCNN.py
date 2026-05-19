"""
DLMP: Deep Learning Multi-Processing simulator

Author: Jorge A. Lopez
Affiliation: Toronto Metropolitan University

Description:
Agent-based simulation framework for studying coordination strategies
in distributed deep learning systems.

Includes support for non-IID distribution

Repository:
https://github.com/DLMPsim/DLMP

License:
MIT License
"""

# --------------------------------------------------
# Entry point for the DLMP synchronous (SYNC) distributed training experiment.
# This script configures the simulation parameters, loads the selected dataset,
# initializes the agent-based training model, and runs the experiment.
# --------------------------------------------------
import torch
import argparse
from MASModelCNN import ParallelizationModel
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import os

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path


def check_latency_range(value):
    try:
        x, y = map(float, value.split(','))
        if x < 0 or y < 0:
            raise argparse.ArgumentTypeError("Both x and y must be positive integers.")
        if x > y:
            raise argparse.ArgumentTypeError("x must be less than y.")
        return x, y
    except ValueError:
        raise argparse.ArgumentTypeError("Latency range must be in the format 'x,y' where x and y are integers.")

def check_capacity_max(value):
    try:
        v = float(value)
        if v < 1.0:
            raise argparse.ArgumentTypeError("capacity_max must be ≥ 1.0 (min is fixed at 1.0).")
        return v
    except ValueError:
        raise argparse.ArgumentTypeError("capacity_max must be a number (e.g., 2.0).")

# --------------------------------------------------
# Parse command-line arguments that define the experiment configuration
# (dataset, number of processors, training epochs, batch size, etc.).
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Simulated parallel processing using MAS")
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--processors', type=int, default=4, help='Number of simulated processors (nodes) to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='Learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping (default: 5)')
    parser.add_argument('--latency', type=check_latency_range, default="1,10", help="Latency range in ms as 'x,y'")

    parser.add_argument('--capacity_max', type=check_capacity_max, default=2.0,
                    help='Upper bound for uniform compute capacity in [1.0, capacity_max] (default: 2.0)')


    parser.add_argument('--net_bw_mbps', type=float, default=100.0,
                    help='Uniform link bandwidth used to convert CC bytes→seconds (default: 100 Mbps)')
    parser.add_argument('--min_gpu_mem_gb', type=float, default=2.0,
                        help='Minimum available GPU memory (in GiB) required to use GPU (default: 2.0)')

    parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'ADAMW', 'SGDM', 'RMSP'],
                        help='Optimizer to use (default: ADAM)')
    parser.add_argument('--loss', type=str, default='CE', choices=['CE', 'LSCE', 'FC', 'WCE'],
                        help='Loss function to use (default: CE)')
    parser.add_argument('--activation', type=str, default='RELU', choices=['RELU', 'LEAKY_RELU', 'ELU', 'SELU', 'GELU', 'MISH'],
                        help='Activation function to use (default: RELU)')
###
    parser.add_argument("--dataset", type=str, default="MNIST",
                    choices=["MNIST", "CIFAR10", "CIFAR100", "UA_DETRAC"],
                    help="Dataset to use")
    parser.add_argument("--ua_resize", type=int, default=224,
                    help="Resize UA-DETRAC frames to NxN (e.g., 224).")
    parser.add_argument("--ua_limit", type=int, default=0,
                    help="If >0, limit UA-DETRAC samples for smoke tests (e.g., 2000).")
 ###
    parser.add_argument("--ua_use_imagenet_norm", action="store_true",
                    help="Use ImageNet normalization (recommended if using pretrained models).")
    parser.add_argument("--partition", type=str, default="iid",
                        choices=["iid", "nonIID_cifar10"],
                        help="Data partition mode: iid or nonIID_cifar10.")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5,
                        help="Dirichlet alpha for non-IID CIFAR-10 partitioning.")
    parser.add_argument("--partition_seed", type=int, default=42,
                        help="Random seed for non-IID partitioning.")

    args = parser.parse_args()
    args.ds = args.dataset

    if args.partition == "nonIID_cifar10":
        if args.ds != "CIFAR10":
            raise ValueError("--partition nonIID_cifar10 is only supported with --dataset CIFAR10.")

        if args.processors != 4:
            raise ValueError("--partition nonIID_cifar10 is only supported with --processors 4.")

        if args.dirichlet_alpha <= 0:
            raise ValueError("--dirichlet_alpha must be > 0.")
 ###

    args.imagenet_pretrained = False
    if args.ds == "UA_DETRAC":
        args.imagenet_pretrained = True

        args.ua_use_imagenet_norm = True
#------------------------------------------------------ 
# Assumption: uniform network bandwidth used to convert
# communication traffic (bytes) into transmission time.
#------------------------------------------------------ 
    print(f"Assumption: uniform bandwidth = {args.net_bw_mbps} Mbps")
    DATA_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))

    use_gpu = False
    device = torch.device("cpu")
    selected_gpu_idx = None
# --------------------------------------------------
# Select a GPU with sufficient free memory if available;
# otherwise fall back to CPU execution.
# --------------------------------------------------
    if args.gpu and torch.cuda.is_available():
        try:
            num_gpus = torch.cuda.device_count()
            found = False
            for gpu_idx in range(num_gpus):
                mem_info = None
                if hasattr(torch.cuda, 'mem_get_info'):
                    mem_info = torch.cuda.mem_get_info(gpu_idx)
                if mem_info is not None:
                    free_mem_gb = mem_info[0] / 1024**3
                else:
                    import subprocess
                    out = subprocess.check_output(
                        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader']
                    )
                    free_mem_list = out.decode().strip().split('\n')
                    free_mem_gb = float(free_mem_list[gpu_idx]) / 1024

                if free_mem_gb >= args.min_gpu_mem_gb:
                    device = torch.device(f"cuda:{gpu_idx}")
                    use_gpu = True
                    selected_gpu_idx = gpu_idx
                    print(f"***Using device: GPU:{gpu_idx} (free {free_mem_gb:.2f} GiB)***")
                    found = True
                    break
                else:
                    print(f"GPU:{gpu_idx} available but only {free_mem_gb:.2f} GiB free, less than required {args.min_gpu_mem_gb} GiB.")
            if not found:
                print(f"***No available GPU with >= {args.min_gpu_mem_gb} GiB free, using CPU.***")
        except Exception as e:
            print(f"***Unable to determine GPU memory, using CPU. Reason: {e}***")
    else:
        print("***Using device: CPU***")

# --------------------------------------------------
# Dataset loading and preprocessing
# --------------------------------------------------
    if args.ds == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(root=DATA_ROOT, train=True, transform=transform, download=True)
        model_arch = 'LeNet5'
        num_classes = 10
        X = dataset.data.numpy()
        y = dataset.targets.numpy()
    elif args.ds == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, transform=transform, download=True)
        model_arch = 'VGG11'
        num_classes = 10
        X = dataset.data.transpose((0, 3, 1, 2))
        y = dataset.targets
    elif args.ds == 'CIFAR100':  
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Common for CIFAR100, can be changed
        ])
        dataset = datasets.CIFAR100(root=DATA_ROOT, train=True, transform=transform, download=True)
        model_arch = 'ResNet18'
        num_classes = 100
        X = dataset.data.transpose((0, 3, 1, 2))
        y = dataset.targets
    ###
    elif args.ds == 'UA_DETRAC':

        from trainMASCNN import UADetracSceneDataset  

        
        ua_csv  = "../data/UA-DETRAC/DLMP/scene_labels_traffic.csv"
        ua_root = "../data/UA-DETRAC/dataset/UA_DETRAC_CLEAN/content/UA-DETRAC/DETRAC_Upload"

        resize_n = int(args.ua_resize)
        t_list = [transforms.Resize((resize_n, resize_n)), transforms.ToTensor()]
        if args.ua_use_imagenet_norm:
            t_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
        ua_transform = transforms.Compose(t_list)

        # Build streaming datasets (train/val are already defined by CSV split column)

        train_ds = UADetracSceneDataset(
            csv_path=ua_csv,
            dataset_root=ua_root,
            split="train",
            transform=ua_transform,
            limit=args.ua_limit
        )

        val_ds = UADetracSceneDataset(
            csv_path=ua_csv,
            dataset_root=ua_root,
            split="val",
            transform=ua_transform,
            limit=args.ua_limit
        )   

        Training_ds, Training_lbls = train_ds, None
        Testing_ds, Testing_lbls = val_ds, None

        model_arch, num_classes = 'ResNet18', 3

    
    # Split data
    print("Using dataset:", args.ds)
    if args.ds != 'UA_DETRAC':
        Training_ds, Testing_ds, Training_lbls, Testing_lbls = \
            train_test_split(X, y, test_size=0.2, random_state=42)

    args.arch = model_arch
    args.num_classes = num_classes
#---------------------------------------------------------------------
# Initialize the multi-agent model that simulates distributed training
# across the specified number of processors.
#---------------------------------------------------------------------
    simulation_model = ParallelizationModel(Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args)
    simulation_model.run_model(args.epochs)

if __name__ == '__main__':
    main()

