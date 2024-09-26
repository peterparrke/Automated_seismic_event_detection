import os
import csv
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check and create directory for saving weights
if not os.path.exists("./weights_ANN"):
    os.makedirs("./weights_ANN")

# Check the device and use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} for training.")

# Image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.9078, 0.7846, 0.9741], [0.1788, 0.2800, 0.0500])
])

# Load the dataset
dataset = datasets.ImageFolder(root='./train', transform=transform)
num_classes = 6
print("Number of classes:", num_classes)


class CustomANN(nn.Module):
    def __init__(self):
        super(CustomANN, self).__init__()
        # Define the fully connected layers
        self.fc1 = nn.Linear(3 * 100 * 100, 1024)  # Assuming input images are 224x224x3
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.view(-1, 3 * 100 * 100)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set up 10-fold cross-validation
k_folds = 10
kfold = KFold(n_splits=k_folds,shuffle=True, random_state=42)

# Use context manager to handle the CSV file
with open('New_ANN.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Fold', 'Epoch', 'Train Loss', 'Val Loss', 'Train Accu', 'Val Accuracy'])

    start_time = time.time()

    # Training and validation model
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold+1}/{k_folds}')
        train_loader = DataLoader(Subset(dataset, train_ids), batch_size=50, shuffle=True, num_workers=8, pin_memory=True)
        test_loader = DataLoader(Subset(dataset, test_ids), batch_size=50, shuffle=False, num_workers=8, pin_memory=True)

        # Initialize the model
        model = CustomANN().to(device)

        # Load pre-trained weights
        pretrained_dict = torch.load('./weights/pretrained_ANN.pth')
        model_dict = model.state_dict()

        # Filter out unnecessary keys (the last layer)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc3' not in k}

        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # Load the new state dict
        model.load_state_dict(model_dict)

        # Freeze all layers except the last one
        for name, param in model.named_parameters():
            if 'fc3' not in name:
                param.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)


        for epoch in range(30):  # Set number of epochs
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for images, labels in tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{30}, Training...'):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_acc = 100 * train_correct / train_total

            # Validation
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in tqdm.tqdm(test_loader, desc=f'Epoch {epoch+1}/{30}, Validating...'):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = 100 * val_correct / val_total

            print(
                f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, Validation Loss: {val_loss / len(test_loader)}, Train Accuracy: {train_acc}, Validation Accuracy: {val_acc}')

            csv_writer.writerow([fold+1, epoch+1, train_loss / len(train_loader), val_loss / len(test_loader), train_acc, val_acc])
            csv_file.flush()  # Flush after each write

            torch.save(model.state_dict(), "./weights_ANN/model_fold{}_{:02d}.pth".format(fold, epoch))

# 记录训练的结束时间
end_time = time.time()
total_time = end_time - start_time
print(f"Training completed. Total training time: {total_time:.2f} seconds.")

# No need to close the file as the context manager handles it
print("Training completed and results saved to CSV.")
