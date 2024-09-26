import os
import json
import torch
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from prettytable import PrettyTable
import torch.nn as nn
from vit_model import vit_base_patch16_224_in21k as create_model

class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("Model accuracy: ", acc)

        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        plt.imshow(matrix, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion Matrix')

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.9078, 0.7846, 0.9741], [0.1788, 0.2800, 0.0500])])
    ])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./xiazhang_1/"))
    image_path = os.path.abspath(os.path.join(os.getcwd(), "./xiazhang_1/"))
    assert os.path.exists(image_path), "Data path {} does not exist.".format(image_path)

    test_dataset = datasets.ImageFolder(root=os.path.join(image_path), transform=data_transform)

    batch_size = 150
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    model = create_model(num_classes=6)

    model_weight_path = "./weights/model-24.pth"
    assert os.path.exists(model_weight_path), "Cannot find {} file".format(model_weight_path)

    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)

    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "Cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=6, labels=labels)

    model.eval()
    with torch.no_grad():
        for test_data in tqdm(test_loader):
            test_images, test_labels = test_data
            outputs = model(test_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), test_labels.to("cpu").numpy())

    confusion.plot()
    confusion.summary()
