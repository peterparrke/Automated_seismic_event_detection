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
if not os.path.exists("./weights"):
    os.makedirs("./weights")

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

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # 定义第一层卷积
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=41, stride=1, padding=20)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义第二层卷积
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=30, stride=1, padding=10)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义全连接层
        self.fc1 = nn.Linear(20 * 20 * 20, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一层卷积+激活
        x = self.pool1(x)  # 第一层池化
        x = F.relu(self.conv2(x))  # 第二层卷积+激活
        x = self.pool2(x)  # 第二层池化
        x = x.view(-1, 20 * 20 * 20)  # Flatten
        x = self.fc1(x)  # 全连接层
        return x  # Softmax层

pretrained_weights_path = 'weights/pretrained_CNN.pth'  # 更新为你的路径
model = CustomCNN().to(device)

# Load pretrained weights and update selectively
pretrained_weights = torch.load(pretrained_weights_path)
model.conv1.weight.data = pretrained_weights['conv1.weight']
model.conv1.bias.data = pretrained_weights['conv1.bias']
model.conv2.weight.data = pretrained_weights['conv2.weight']
model.conv2.bias.data = pretrained_weights['conv2.bias']

# Set up 10-fold cross-validation
k_folds = 10
kfold = KFold(n_splits=k_folds,shuffle=True, random_state=42)

# Use context manager to handle the CSV file
with open('New_CNN.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Fold', 'Epoch', 'Train Loss', 'Val Loss', 'Train Accu', 'Val Accuracy'])

    # 记录训练的起始时间
    start_time = time.time()

    # Training and validation model
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold+1}/{k_folds}')
        train_loader = DataLoader(Subset(dataset, train_ids), batch_size=50, shuffle=True, num_workers=8, pin_memory=True)
        test_loader = DataLoader(Subset(dataset, test_ids), batch_size=50, shuffle=False, num_workers=8, pin_memory=True)

        # Initialize the model
        model = CustomCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00001)

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

            torch.save(model.state_dict(), "./weights/model_fold{}_{:02d}.pth".format(fold, epoch))

# 记录训练的结束时间
end_time = time.time()
total_time = end_time - start_time
print(f"Training completed. Total training time: {total_time:.2f} seconds.")

# No need to close the file as the context manager handles it
print("Training completed and results saved to CSV.")
