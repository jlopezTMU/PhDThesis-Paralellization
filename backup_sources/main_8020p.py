# Define the LeNet architecture
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(nn.MaxPool2d(2)(self.conv1(x)))
        x = torch.relu(nn.MaxPool2d(2)(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
######
def train_and_test_model(model, train_loader, test_loader, device, epochs, lr):
    print("@train_and_test")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Initialize variables to track training performance
    total_train_loss = 0
    correct_train = 0
    total_train_samples = 0

    # Training
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Accumulate loss and accuracy
            total_train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_train += pred.eq(target.view_as(pred)).sum().item()
            total_train_samples += data.size(0)

    average_train_loss = total_train_loss / len(train_loader.dataset)
    train_accuracy = 100. * correct_train / total_train_samples

    # Testing
    model.eval()
    test_loss = 0
    correct_test = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_test += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct_test / len(test_loader.dataset)

    return average_train_loss, train_accuracy, test_loss, test_accuracy
######
def main():
    print("@main")
    # Command line arguments
    parser = argparse.ArgumentParser(description='Simple Training and Testing with Data Parallelism')
    parser.add_argument('--processors', type=int, default=2, help='Number of processors/GPUs to use for parallel processing')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)  # Large batch size for testing for efficiency

    # Initialize model and apply DataParallel
    model = LeNet().to(device)

    if torch.cuda.device_count() > 1 and args.processors > 1:
        model = nn.DataParallel(model, device_ids=list(range(args.processors)))

    start_time = time.time()
    train_loss, train_accuracy, test_loss, test_accuracy = train_and_test_model(model, train_loader, test_loader, device, args.epochs, args.lr)
    elapsed_time = time.time() - start_time

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    print(f"Total processing time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()
