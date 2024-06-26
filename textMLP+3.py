import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
import os
import warnings

# Hardcoded path for the dataset
DATA_PATH = os.path.expanduser('~/data/AGNews')

# MLP Model Definition
class TextMLP(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, hidden_units):
        super(TextMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embed_size)
        x = x.mean(dim=1)  # Averaging embeddings (batch_size, embed_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        logits = self.fc4(x)  # (batch_size, num_classes)
        return logits

def collate_batch(batch, vocab, tokenizer):
    label_list, text_list = [], []
    for _label, _text in batch:
        label_list.append(int(_label) - 1)
        processed_text = torch.tensor(vocab(tokenizer(_text)), dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return label_list, text_list

def load_data(batch_size, percent, tokenizer, rank, world_size):
    # Ensure dataset directory exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    # Download dataset if it doesn't exist
    train_iter, test_iter = AG_NEWS(root=DATA_PATH, split=('train', 'test'))

    # Build vocabulary based on the percentage of the dataset
    def yield_tokens(data_iter, total_len):
        for i, (_, text) in enumerate(data_iter):
            if i >= total_len:
                break
            yield tokenizer(text)

    train_dataset = list(train_iter)
    total_len = int(len(train_dataset) * (percent / 100))
    vocab = build_vocab_from_iterator(yield_tokens(train_dataset, total_len), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    # Adjust dataset size based on the percent argument
    train_dataset = train_dataset[:total_len]

    train_len = int(0.8 * len(train_dataset))
    valid_len = len(train_dataset) - train_len
    train_dataset, valid_dataset = random_split(train_dataset, [train_len, valid_len])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=lambda x: collate_batch(x, vocab, tokenizer))
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, collate_fn=lambda x: collate_batch(x, vocab, tokenizer))
    return train_dataloader, valid_dataloader, len(vocab)

def train_model(rank, world_size, args, results):
    print(f"Process {rank} started")

    dist.init_process_group(backend='nccl' if args.use_gpu else 'gloo', init_method='file:///tmp/ddp_example', world_size=world_size, rank=rank)

    torch.manual_seed(0)
    if args.use_gpu:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    model = TextMLP(args.vocab_size, 64, 4, 128).to(device)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank] if args.use_gpu else None)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dataloader, valid_dataloader, _ = load_data(args.batch_size, args.percent, args.tokenizer, rank, world_size)

    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        for labels, text in train_dataloader:
            labels, text = labels.to(device), text.to(device)
            optimizer.zero_grad()
            outputs = model(text)
            loss = criterion(outputs, labels)
            loss.backward()  # Gradient synchronization happens here
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for labels, text in valid_dataloader:
            labels, text = labels.to(device), text.to(device)
            outputs = model(text)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    training_time = time.time() - start_time
    accuracy = 100 * correct / total

    # Store results in the shared dictionary
    results[rank] = {'training_time': training_time, 'accuracy': accuracy}
    print(f'Rank {rank} - TRAINING TIME {training_time:.2f} seconds, VALIDATION ACCURACY {accuracy:.2f}%')

    dist.destroy_process_group()
    print(f"Process {rank} finished")

def custom_tokenizer(text):
    stop_words = set(["a", "an", "the", "and", "or", "but"])
    basic_tokenizer = get_tokenizer("basic_english")
    tokens = basic_tokenizer(text)
    tokens = [token for token in tokens if token not in stop_words and not token.isdigit()]
    return tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='use_gpu', action='store_true')
    parser.add_argument('--cpu', dest='use_gpu', action='store_false')
    parser.set_defaults(use_gpu=False)
    parser.add_argument('--processors', type=int, default=1, help='number of GPUs or CPU cores used for parallelization')
    parser.add_argument('-bs', '--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.099, help='learning rate')
    parser.add_argument('--percent', type=int, default=100, help='percentage of rows to process')

    args = parser.parse_args()

    if args.percent <= 0 or args.percent > 100:
        raise ValueError('The --percent argument must be a positive integer less than or equal to 100.')

    if args.percent < 100:
        warnings.warn(f'The program will process only {args.percent}% of the total rows of the dataset.')

    args.tokenizer = custom_tokenizer

    os.makedirs(DATA_PATH, exist_ok=True)

    # Dummy initialization to get vocab size
    _, _, vocab_size = load_data(args.batch_size, args.percent, args.tokenizer, rank=0, world_size=args.processors)
    args.vocab_size = vocab_size

    print(f'Total number of words in the vocabulary: {vocab_size}')

    world_size = args.processors
    manager = mp.Manager()
    results = manager.dict()

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=train_model, args=(rank, world_size, args, results))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Print aggregated results
    if results:
        total_training_time = max(result['training_time'] for result in results.values())
        avg_accuracy = sum(result['accuracy'] for result in results.values()) / world_size

        for rank, result in results.items():
            print(f'Rank {rank} - TRAINING TIME {result["training_time"]:.2f} seconds, VALIDATION ACCURACY {result["accuracy"]:.2f}%')

        print(f'TOTAL TRAINING TIME {total_training_time:.2f} seconds, AVERAGE VALIDATION ACCURACY {avg_accuracy:.2f}%')

if __name__ == "__main__":
    main()
