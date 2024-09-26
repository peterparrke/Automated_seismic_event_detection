import os
import sys
import json
# import random
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
# import matplotlib.pyplot as plt
from model import resnet34
from sklearn.model_selection import StratifiedKFold


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.9078, 0.7846, 0.9741], [0.1788, 0.2800, 0.0500])
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.9078, 0.7846, 0.9741], [0.1788, 0.2800, 0.0500])
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./data_2"))
    image_path = os.path.join(data_root)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    train_num = len(train_dataset)

    data_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in data_list.items())
    json_str = json.dumps(cla_dict, indent=6)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 50
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers for each process'.format(nw))

    k = 10
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    folds = skf.split(train_dataset.imgs, train_dataset.targets)

    acc_list = []
    train_acc_list = []
    loss1_list = []
    loss2_list = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        print("Fold {}/{}".format(fold + 1, k))

        net = resnet34()
        model_weight_path = "./resnet34-pre.pth"
        assert os.path.exists(model_weight_path), "File {} does not exist.".format(model_weight_path)
        net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, 6)
        net.to(device)

        loss_function = nn.CrossEntropyLoss()
        params = [p for p in net.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=0.0001)

        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader_fold = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                                        num_workers=nw)
        validate_loader_fold = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler,
                                                           num_workers=nw)

        train_steps = len(train_loader_fold)
        val_steps = len(validate_loader_fold)

        epochs = 50
        for epoch in range(epochs):
            net.train()
            running_loss = 0.0
            correct = 0
            total = 0
            train_bar = tqdm(train_loader_fold, file=sys.stdout)

            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels.to(device)).sum().item()

                train_bar.desc = "Train Epoch [{}/{}] Loss: {:.3f}".format(epoch + 1, epochs, loss)

            train_accuracy = 100.0 * correct / total
            train_acc_list.append(train_accuracy)

            net.eval()
            acc = 0.0
            valrunning_loss = 0.0
            with torch.no_grad():
                val_bar = tqdm(validate_loader_fold, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))
                    loss2 = loss_function(outputs, val_labels.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                    val_bar.desc = "Validation Epoch [{}/{}] Loss: {:.3f}".format(epoch + 1, epochs, loss2)
                    valrunning_loss += loss2.item()

            val_accurate = acc / len(val_idx)
            acc_list.append(val_accurate)
            print(
                '[Fold {}/{}][Epoch {}] Train Loss: {:.3f} Validation Loss: {:.3f} Train Accuracy: {:.3f} Validation Accuracy: {:.3f}'.format(
                    fold + 1, k, epoch + 1, running_loss / train_steps, valrunning_loss / val_steps, train_accuracy,
                    val_accurate))
            loss1_list.append(running_loss / train_steps)
            loss2_list.append(valrunning_loss / val_steps)

            # torch.save(net.state_dict(), "resnet34_fold{}_{}.pth".format(fold + 1, epoch + 1))

    with open('without_TL_34_50.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Fold', 'Epoch', 'Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy'])
        for fold in range(k):
            for epoch in range(epochs):
                train_loss = loss1_list[fold * epochs + epoch]
                val_loss = loss2_list[fold * epochs + epoch]
                train_accuracy = train_acc_list[fold * epochs + epoch]
                val_accuracy = acc_list[fold * epochs + epoch]
                writer.writerow([fold + 1, epoch + 1, train_loss, val_loss, train_accuracy, val_accuracy*100])


if __name__ == '__main__':
    main()
