import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg11, resnet18  # now includes resnet18
import time
from sklearn.model_selection import train_test_split

def get_model(arch, num_classes=10):
    if arch == 'LeNet5':
        model = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(4 * 4 * 50, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes)
        )
        return model
    elif arch == 'VGG11':
        model = vgg11(weights=None)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        return model
    elif arch == 'ResNet18':
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError("Unsupported architecture. Use 'LeNet5', 'VGG11', or 'ResNet18'.")

def train_simulated(unique_id, model, Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args, sync_callback):
    start_time = time.time()

    X_train = torch.tensor(Training_ds, dtype=torch.float32).to(device)
    y_train = torch.tensor(Training_lbls, dtype=torch.long).to(device)
    X_test = torch.tensor(Testing_ds, dtype=torch.float32).to(device)
    y_test = torch.tensor(Testing_lbls, dtype=torch.long).to(device)

    if args.ds.upper() == 'MNIST' and X_train.ndim == 3:
        X_train = X_train.unsqueeze(1)
        X_test = X_test.unsqueeze(1)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9 if args.optimizer == 'SGDM' else 0.0)

    best_val_loss = float('inf')
    patience_counter = 0
    total_samples = 0
    running_loss = 0.0
    correct_train = 0

    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(labels).sum().item()

        avg_loss = running_loss / total_samples
        train_acc = 100.0 * correct_train / total_samples

        # REMOVED: sync_callback(model.state_dict())

        print(f"Node {unique_id + 1} Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {train_acc:.2f}%")

        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            val_loss = criterion(outputs, y_test).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Node {unique_id + 1}: Early stopping triggered at epoch {epoch + 1}")
                break

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = outputs.max(1)
        total_test = y_test.size(0)
        correct_test = predicted.eq(y_test).sum().item()
        test_acc = 100.0 * correct_test / total_test

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Node {unique_id + 1} Final Validation: Accuracy = {test_acc:.2f}%\n")

    return best_val_loss, test_acc, correct_test, total_test, processing_time
