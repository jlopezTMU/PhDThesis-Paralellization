import torch
import argparse
from MASAModelCNN import ParallelizationModel
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

def check_latency_range(value):
    """
    Validates the latency range format and values.
    Parameters:
        value (str): A string formatted as 'x,y' where x and y represent milliseconds.
    Returns:
        tuple: A tuple (x, y) if valid.
    Raises:
        argparse.ArgumentTypeError: If the format is incorrect or values are invalid.
    """
    try:
        x, y = map(float, value.split(','))
        if x < 0 or y < 0:
            raise argparse.ArgumentTypeError("Both x and y must be positive integers.")
        if x > y:
            raise argparse.ArgumentTypeError("x must be less than y.")
        return x, y
    except ValueError:
        raise argparse.ArgumentTypeError("Latency range must be in the format 'x,y' where x and y are integers.")

def main():
    parser = argparse.ArgumentParser(description="Simulated parallel processing using MAS")
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--processors', type=int, default=4, help='Number of simulated processors (nodes) to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='Learning rate (default: 0.01)')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping (default: 5)')
    parser.add_argument('--latency', type=check_latency_range, default="1,10", help="Latency range in ms as 'x,y'")

    # Add missing argument for GPU memory requirement
    parser.add_argument('--min_gpu_mem_gb', type=float, default=2.0,
                        help='Minimum available GPU memory (in GiB) required to use GPU (default: 2.0)')

    parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'ADAMW', 'SGDM', 'RMSP'],
                        help='Optimizer to use (default: ADAM)')
    parser.add_argument('--loss', type=str, default='CE', choices=['CE', 'LSCE', 'FC', 'WCE'],
                        help='Loss function to use (default: CE)')
    parser.add_argument('--activation', type=str, default='RELU', choices=['RELU', 'LEAKY_RELU', 'ELU', 'SELU', 'GELU', 'MISH'],
                        help='Activation function to use (default: RELU)')
    parser.add_argument('--ds', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'],
                        help="Dataset to process (default: MNIST)")
    args = parser.parse_args()

    # --- Dynamic GPU selection with minimum memory requirement ---
    use_gpu = False
    if args.gpu and torch.cuda.is_available():
        try:
            gpu_idx = torch.cuda.current_device()
            mem_info = torch.cuda.mem_get_info(gpu_idx) if hasattr(torch.cuda, 'mem_get_info') else None
            if mem_info is not None:
                free_mem_gb = mem_info[0] / 1024**3
            else:
                # Fallback: use nvidia-smi if available
                import subprocess
                out = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader']
                )
                free_mem_gb = float(out.decode().split('\n')[gpu_idx]) / 1024
            if free_mem_gb >= args.min_gpu_mem_gb:
                device = torch.device("cuda")
                use_gpu = True
                print(f"***Using device: GPU (free {free_mem_gb:.2f} GiB)***")
            else:
                device = torch.device("cpu")
                print(f"***GPU available but only {free_mem_gb:.2f} GiB free, less than required {args.min_gpu_mem_gb} GiB. Using CPU.***")
        except Exception as e:
            device = torch.device("cpu")
            print(f"***Unable to determine GPU memory, using CPU. Reason: {e}***")
    else:
        device = torch.device("cpu")
        print("***Using device: CPU***")

    # Load the specified dataset and apply transformations.
    if args.ds == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        model_arch = 'LeNet5'
        X = dataset.data.numpy()
        y = dataset.targets.numpy()
    elif args.ds == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        model_arch = 'VGG11'
        X = dataset.data.transpose((0, 3, 1, 2))
        y = dataset.targets

    # Split the dataset into training and testing sets (80% training, 20% testing).
    Training_ds, Testing_ds, Training_lbls, Testing_lbls = train_test_split(X, y, test_size=0.2, random_state=42)
    args.arch = model_arch

    simulation_model = ParallelizationModel(Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args)
    simulation_model.step()

if __name__ == '__main__':
    main()
