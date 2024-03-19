import argparse
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.parallel import DataParallel
from torchtext.data.functional import to_map_style_dataset

# Parsing command-line arguments
parser = argparse.ArgumentParser(description='Transformer on AG NEWS')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--processors', type=int, default=1, help='Number of processors to use')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
args = parser.parse_args()

# Setup device
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, 4)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output.mean(dim=0))
        return output

def load_ag_news_dataset():
    tokenizer = get_tokenizer('basic_english')
    train_iter, test_iter = AG_NEWS()
    vocab = build_vocab_from_iterator(map(lambda x: tokenizer(x[1]), train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    def data_process(vocab, data_iter):
        processed_text = [torch.tensor([vocab[token] for token in tokenizer(item[1])], dtype=torch.long) for item in data_iter]
        processed_label = [torch.tensor(int(item[0]) - 1, dtype=torch.long) for item in data_iter]
        return list(zip(processed_text, processed_label))

    train_data = data_process(vocab, to_map_style_dataset(train_iter))
    test_data = data_process(vocab, to_map_style_dataset(test_iter))

    return train_data, test_data, vocab

class AGNewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    labels = torch.tensor(label_list, dtype=torch.int64)
    texts = nn.utils.rnn.pad_sequence(text_list, padding_value=3.0)
    return labels.to(device), texts.to(device)

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.
    for labels, texts in dataloader:
        texts = texts.transpose(0, 1) # Transformer expects [seq_len, batch_size]
        optimizer.zero_grad()
        output = model(texts)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.
    correct = 0
    with torch.no_grad():
        for labels, texts in dataloader:
            texts = texts.transpose(0, 1) # Transformer expects [seq_len, batch_size]
            output = model(texts)
            loss = criterion(output, labels)
            total_loss += loss.item()
            correct += (output.argmax(1) == labels).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

def main():
    # Load the dataset
    train_data, test_data, vocab = load_ag_news_dataset()
    train_dataset = AGNewsDataset(train_data)
    test_dataset = AGNewsDataset(test_data)

    # Split the dataset
    train_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    # Initialize the model
    ntokens = len(vocab) # the size of vocabulary
    model = TransformerModel(ntokens, ninp=200, nhead=2, nhid=200, nlayers=2, dropout=0.5).to(device)
    if args.gpu and torch.cuda.device_count() > 1 and args.processors > 1:
        model = DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Track total training time
    start_time = time.time()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(model, train_dataloader, optimizer, criterion)
        val_loss, val_accuracy = evaluate(model, val_dataloader, criterion)
        print(f'Epoch: {epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Testing
    test_loss, test_accuracy = evaluate(model, test_dataloader, criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # Total processing time
    end_time = time.time()
    print(f'Total processing time: {(end_time - start_time):.2f} seconds')

if __name__ == '__main__':
    main()
