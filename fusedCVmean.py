# This program trains a CNN using LeNet5 architecture,
# 
# This includes the model train in one place for convenience
# This version selects the best Average Accuracy


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms

##NO! from trainCV import train, test


from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import time

from sklearn.model_selection import KFold

# Set the random seed
seed = 123
torch.manual_seed(seed)

#################################################
# train and test within the main for convenience#
#################################################

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler # support multGPU
import time

def compute_validation_accuracy(model, device, data_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    ##print('** Overall Training Accuracy is {}/{}=: {:.2f}%'.format(correct, total, accuracy))
    return accuracy

#train
def train(rank, args, model, device, train_dataset, dataloader_kwargs, val_dataloader_kwargs, results, start_time, model_states):
    accuracies_per_fold = [] # To save accuracies for each fold
    model = model.to(device)
    kfold = KFold(n_splits=10)
    folds = list(kfold.split(train_dataset))

    for fold, (train_index, val_index) in enumerate(folds):
        train_sub_dataset = torch.utils.data.Subset(train_dataset, train_index)
        val_sub_dataset = torch.utils.data.Subset(train_dataset, val_index)
        train_loader = torch.utils.data.DataLoader(train_sub_dataset, **dataloader_kwargs)
        val_loader = torch.utils.data.DataLoader(val_sub_dataset, **val_dataloader_kwargs)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        for epoch in range(1, args.epochs + 1):
            train_epoch(epoch, args, model, train_loader=train_loader, optimizer=optimizer, device=device)

        accuracy = compute_validation_accuracy(model, device, val_loader)
        accuracies_per_fold.append(accuracy)
        print('>>> For fold {} within processor {} the accuracy is {:.2f}%'.format(fold, rank, accuracy))

    average_accuracy = sum(accuracies_per_fold) / len(accuracies_per_fold)
    closest_fold_accuracy = min(accuracies_per_fold, key=lambda x: abs(x - average_accuracy))
    best_fold = accuracies_per_fold.index(closest_fold_accuracy)

    print('>>> For processor {} the average accuracy is {:.2f}%'.format(rank, average_accuracy))

    results.append((closest_fold_accuracy, best_fold, average_accuracy, time.time() - start_time))
    model_states.append({k: v.cpu() for k, v in model.cpu().state_dict().items()})



def test(args, model, device, test_dataset, dataloader_kwargs):
    torch.manual_seed(args.seed)
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_kwargs)
    accuracy = test_epoch(model, device, test_loader)
    return accuracy

def train_epoch(epoch, args, model, train_loader, optimizer, device):
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(train_loader):  # Fixed this line
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # The following lines were deleted to sppedup the printing 23-7-30
        # print some debug info
        if batch_idx % args.log_interval == 0:
            print('PID: {}, Batch index: {}, Data shape: {}'.format(pid, batch_idx, data.shape))

        #if batch_idx % args.log_interval == 0:
        #    print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                ###pid, epoch, batch_idx * len(data), len(train_loader.dataset) * len(data),
                ###pid, epoch, batch_idx * len(data), len(data),
        #        pid, epoch, batch_idx * len(data), len(data),                
        #        100. * batch_idx * len(data) / len(train_loader.dataset), loss.item()))

        #    if args.dry_run:
        #        break

def test_epoch(model, device, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy


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
        # not used anywhere val_dataset = torch.utils.data.Subset(dataset1, val_index)
        start_time = time.time()
        p = mp.Process(target=train, args=(rank, args, cpu_model, device, train_dataset, dataloader_kwargs, val_dataloader_kwargs, results, start_time, model_states))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    model.to(device)

    best_accuracy = 0.0
    best_processor = -1
    best_fold = -1

    for i, (accuracy, fold, training_time) in enumerate(results):
        print("Processor: {}, Fold: {}, Accuracy: {:.2f}%, Training Time: {:.2f} seconds".format(i, fold, accuracy, training_time))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_processor = i
            best_fold = fold
     
    #NObest_global_accuracy = max(results, key=lambda x: x[0])[0] # Select the best global accuracy
    #NObest_processor = [i for i, res in enumerate(results) if res[0] == best_global_accuracy][0] # Processor where best accuracy was reached
    ## USED? best_model_state = model_states[best_processor] # State dictionary of the best model
    
    # Calculate global average accuracy
    global_avg_accuracy = sum(result[2] for result in results) / len(results)
    best_global_accuracy = min(results, key=lambda x: abs(x[2] - global_avg_accuracy))
    best_processor, best_fold = best_global_accuracy[1], best_global_accuracy[2]

    # Apply the best global model to the testing dataset
    model.load_state_dict(model_states[best_processor])
    test_accuracy = test(args, model, device, dataset2, dataloader_kwargs)
    print("*** The accuracy with the processor {} and fold {} is {:.2f}%.".format(best_processor, best_fold, test_accuracy))
    print("The processor with the closest accuracy to the average accuracy is {:.2f}%".format(best_global_accuracy[0]))

    print("--- Total processing time is %s seconds ---" % (time.time() - init_time))
