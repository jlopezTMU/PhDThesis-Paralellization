import torch
import argparse
from MASModelCNN import ParallelizationModel
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

# ---------------------------------------------
# GPU SELECTION SECTION (added by request)
# ---------------------------------------------
import os

def select_first_gpu(min_free_mem_gb=16):
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mem_gb = meminfo.free / (1024 ** 3)
            if free_mem_gb >= min_free_mem_gb:
                pynvml.nvmlShutdown()
                return str(i)
        pynvml.nvmlShutdown()
        return None
    except ImportError:
        raise RuntimeError("pynvml is required for GPU selection. Please install with: pip install pynvml")
    except Exception as e:
        raise RuntimeError(f"Error selecting GPU: {e}")

# --------------------------------------------------------------------------------
# Function to check the latency range input provided via command-line arguments.
# Expects a string of the form 'x,y', where x and y are non-negative floats.
# --------------------------------------------------------------------------------
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
        # Ensure both latency bounds are non-negative.
        if x < 0 or y < 0:
            raise argparse.ArgumentTypeError("Both x and y must be positive integers.")
        # Make sure the lower bound is not greater than the upper bound.
        if x > y:
            raise argparse.ArgumentTypeError("x must be less than y.")
        return x, y
    except ValueError:
        raise argparse.ArgumentTypeError("Latency range must be in the format 'x,y' where x and y are integers.")

# --------------------------------------------------------------------------------
# Main function for setting up the simulation.
# --------------------------------------------------------------------------------
def main():
    # Set up command-line arguments.
    parser = argparse.ArgumentParser(description="Simulated parallel processing using MAS")
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--processors', type=int, default=4, help='Number of simulated processors (nodes) to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='Learning rate (default: 0.01)')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping (default: 5)')
    parser.add_argument('--latency', type=check_latency_range, default="1,10", help="Latency range in ms as 'x,y'")

    # Additional command-line arguments for optimizer, loss function, and activation.
    parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'ADAMW', 'SGDM', 'RMSP'],
                        help='Optimizer to use (default: ADAM)')
    parser.add_argument('--loss', type=str, default='CE', choices=['CE', 'LSCE', 'FC', 'WCE'],
                        help='Loss function to use (default: CE)')
    parser.add_argument('--activation', type=str, default='RELU', choices=['RELU', 'LEAKY_RELU', 'ELU', 'SELU', 'GELU', 'MISH'],
                        help='Activation function to use (default: RELU)')

    parser.add_argument('--ds', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'],
                        help="Dataset to process (default: MNIST)")
    parser.add_argument('--min_gpu_mem_gb', type=int, default=16,
                        help='Minimum free GPU memory per GPU (GB) for selection (default: 16)')
    args = parser.parse_args()

    # --------------------- GPU selection logic (ADDED) ---------------------------
    if args.gpu and torch.cuda.is_available():
        try:
            selected_gpu = select_first_gpu(args.min_gpu_mem_gb)
            if selected_gpu is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu
                print(f"Selected GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
            else:
                print(f"WARNING: No GPU with at least {args.min_gpu_mem_gb} GB free found. Using CPU.")
                args.gpu = False
        except RuntimeError as e:
            print(f"GPU selection error: {e}")
            print("***Using device: CPU***")
            args.gpu = False
    # ------------------------------------------------------------------------------

    # Determine the device – GPU if available and requested, else CPU.
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("***Using device: GPU***")
    else:
        device = torch.device("cpu")
        print("***Using device: CPU***")

    # Load the specified dataset and apply transformations.
    if args.ds == 'MNIST':
        # MNIST dataset normalization.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        model_arch = 'LeNet5'
        X = dataset.data.numpy()
        y = dataset.targets.numpy()
    elif args.ds == 'CIFAR10':
        # CIFAR-10 dataset normalization.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        model_arch = 'VGG11'
        # Adjust CIFAR-10 channel ordering for PyTorch.
        X = dataset.data.transpose((0, 3, 1, 2))
        y = dataset.targets

    # Split the dataset into training and testing sets (80% training, 20% testing).
    Training_ds, Testing_ds, Training_lbls, Testing_lbls = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose the model architecture based on the dataset.
    args.arch = model_arch

    # Create the simulation model and execute one simulation step (one epoch).
    simulation_model = ParallelizationModel(Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args)
    simulation_model.step()

# Entry point of the script.
if __name__ == '__main__':
    main()
