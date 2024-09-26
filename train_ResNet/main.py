import torch
import torchvision
import torchvision.models
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

train_data = torchvision.datasets.ImageFolder(root = "./data/train" ,   transform = data_transform["train"])

traindata = DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=0)

test_data = torchvision.datasets.ImageFolder(root = "./data/val" , transform = data_transform["val"])

train_size = len(train_data)
test_size = len(test_data)
print(train_size)
print(test_size)
testdata = DataLoader(dataset=test_data, batch_size=128, shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

