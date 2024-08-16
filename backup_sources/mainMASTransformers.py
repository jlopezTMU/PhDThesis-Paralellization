import torch
import argparse
from MASModelTransformers import ParallelizationModel
from datasets import load_dataset
import os
import warnings

# Suppress all non-critical warnings globally
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Simulated parallel training with Transformers on STS-B using MAS")
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--processors', type=int, default=4, help='Number of simulated processors (nodes) to use')
    parser.add_argument('--folds', type=int, default=10, help='Number of k-folds for cross-validation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')  # Reduced default batch size
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, metavar='LR', help='Learning rate (default: 2e-5)')

    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dynamically determine the cache directory based on the current working directory
    cache_dir = os.path.join(os.getcwd(), 'data')

    # Load the STS-B dataset, specifying the cache directory
    dataset = load_dataset('glue', 'stsb', split='train', cache_dir=cache_dir)

    X = [f"{row['sentence1']} [SEP] {row['sentence2']}" for row in dataset]
    y = dataset['label']

    model = ParallelizationModel(X, y, device, args)

    model.step()

if __name__ == '__main__':
    main()
