###############
# trainCV3.pt #
###############

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

#
def train(rank, args, model, device, train_dataset, val_dataset, dataloader_kwargs, val_dataloader_kwargs, results, start_time, model_states):
    torch.manual_seed(args.seed + rank)

    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_dataloader_kwargs)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, train_loader, optimizer, device)

    # Once training is done, validate the model on the validation set
    accuracy = compute_validation_accuracy(model, device, val_loader)

    end_time = time.time()  # Get the end time after the training is done
    training_time = end_time - start_time  # Calculate the training time
    results.append((accuracy, time.time() - start_time))
    # If this model performs better than all previous folds, save its state
    if accuracy > max([res[0] for res in results]):
        model_states.append(model.state_dict())
    else:
        model_states.append({k: v.cpu() for k, v in model.cpu().state_dict().items()})
    print(f"Fold {rank} took {training_time} seconds to process.") ###


def test(args, model, device, test_dataset, dataloader_kwargs):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_kwargs)

    test_epoch(model, device, test_loader)


def train_epoch(epoch, args, model, data_loader, optimizer, device):
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # print some debug info
        if batch_idx % args.log_interval == 0:
            print('PID: {}, Batch index: {}, Data shape: {}'.format(pid, batch_idx, data.shape))

        if batch_idx % args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
            if args.dry_run:
                break

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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
