# This program trains a CNN using LeNet5 architecture,
# note that train module remains as trainCPUac
# GPU version

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms

from BtrainPARCV import train, test


from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import time

from sklearn.model_selection import KFold

# Set the random seed
seed = 123
torch.manual_seed(seed)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--n-folds', type=int, default=5, metavar='N',
                    help='number of folds for cross-validation (default: 5)') #consider that I may want to take this argument automatically from the n process, for now for testing is better having it separate

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    mp.set_start_method('spawn')

    init_time = time.time()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device used is: use_cuda", use_cuda)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size

    dataset1, dataset2 = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    print('the dataset has been split in training of {} and testing of {} examples'.format(train_size,test_size))

    dataloader_kwargs = {'batch_size': args.batch_size,'shuffle': False}
    val_dataloader_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}

    kfold = KFold(n_splits=args.n_folds)

    folds = list(kfold.split(dataset1))

    if use_cuda:
        dataloader_kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                      })

    model = LeNet5().to(device)



    model.share_memory() # gradients are allocated lazily, so they are not shared here

    processes = []
    init_time_train = time.time()
    cpu_model = model.cpu()  # Create a CPU version of the model
    cpu_model.share_memory()  # gradients are allocated lazily, so they are not shared here
    manager = mp.Manager()
    results = manager.list()
    model_states = manager.list()
    for rank in range(args.num_processes):
        train_index, val_index = folds[rank]
        train_dataset = torch.utils.data.Subset(dataset1, train_index)
        val_dataset = torch.utils.data.Subset(dataset1, val_index)
        start_time = time.time()
        p = mp.Process(target=train, args=(rank, args, cpu_model, device, train_dataset, val_dataset, dataloader_kwargs, val_dataloader_kwargs, results, start_time, model_states))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    model.to(device)

    best_accuracy = 0.0
    best_fold = -1

    for i in range(args.num_processes):
        print("Fold: {}, Accuracy: {:.2f}%, Training Time: {:.2f} seconds".format(i, results[i][0], results[i][1]))
        if results[i][0] > best_accuracy:
            best_accuracy = results[i][0]
            best_fold = i
    print("The best performing model is with the fold {} with an accuracy of {:.2f}%".format(best_fold, best_accuracy))

    #print("--- Total TRAINING  time is %s seconds ---" % (init_time_train - time.time()))

    # Once training is complete, we can test the model
    model.load_state_dict(model_states[best_fold])
    test(args, model, device, dataset2, dataloader_kwargs)


    print("--- Total procesing time is %s seconds ---" % (init_time - time.time()))
