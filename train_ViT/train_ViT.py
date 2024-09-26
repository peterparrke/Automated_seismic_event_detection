import os
import math
import argparse
import torch
import csv
import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from my_dataset import MyDataSet
from vit_model import vit_large_patch32_224_in21k as create_model
from utils import read_split_data, train_one_epoch, evaluate

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    acc_list = []
    train_acc_list = []
    loss1_list = []
    loss2_list = []

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    print(f"Total samples available: {len(train_images_path)}")

    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.9078, 0.7846, 0.9741], [0.1788, 0.2800, 0.0500])]),
        "val": transforms.Compose([transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.9078, 0.7846, 0.9741], [0.1788, 0.2800, 0.0500])])
    }

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_seed)
    fold = 1

    for train_idx, val_idx in skf.split(train_images_path, train_images_label):
        print(f"Training on {len(train_idx)} samples, validating on {len(val_idx)} samples")

        print("Fold {}/{}".format(fold, args.k_folds))
        train_images_fold = [train_images_path[i] for i in train_idx]
        train_labels_fold = [train_images_label[i] for i in train_idx]
        val_images_fold = [train_images_path[i] for i in val_idx]
        val_labels_fold = [train_images_label[i] for i in val_idx]

        train_dataset = MyDataSet(images_path=train_images_fold,
                                  images_class=train_labels_fold,
                                  transform=data_transform["train"])

        val_dataset = MyDataSet(images_path=val_images_fold,
                                images_class=val_labels_fold,
                                transform=data_transform["val"])

        batch_size = args.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers for each process'.format(nw))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=nw,
                                                   collate_fn=train_dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=nw,
                                                 collate_fn=val_dataset.collate_fn)

        model = create_model(num_classes=args.num_classes).to(device)

        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' does not exist.".format(args.weights)
            weights_dict = torch.load(args.weights, map_location=device)
            del_keys = ['head.weight', 'head.bias'] if model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        # if args.freeze_layers:
        #     for name, param in model.named_parameters():
        #         if "head" not in name and "pre_logits" not in name:
        #             param.requires_grad_(False)
        #         else:
        #             print("Training {}".format(name))

        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(pg, lr=0.0001)
        # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
        # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        for epoch in range(args.epochs):
            print("Epoch {}/{}".format(epoch+1, args.epochs))

            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch)

            # scheduler.step()

            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch)

            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            loss1_list.append(train_loss)
            loss2_list.append(val_loss)
            train_acc_list.append(train_acc)
            acc_list.append(val_acc)

            # torch.save(model.state_dict(), "./weights2/model_fold{}_{:02d}.pth".format(fold, epoch))

        fold += 1

    with open('L_32withouTL.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Fold', 'Epoch', 'Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy'])
        for fold in range(args.k_folds):
            for epoch in range(args.epochs):
                train_loss = loss1_list[fold * args.epochs + epoch]
                val_loss = loss2_list[fold * args.epochs + epoch]
                train_accuracy = train_acc_list[fold * args.epochs + epoch]
                val_accuracy = acc_list[fold * args.epochs + epoch]
                writer.writerow([fold + 1, epoch + 1, train_loss, val_loss, train_accuracy*100, val_accuracy * 100])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--k_folds', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--data-path', type=str, default="./data_2/train/")
    parser.add_argument('--model-name', default='', help='create model name')
    # parser.add_argument('--weights', type=str, default="./vit_base_patch16_224_in21k.pth", help='initial weights path')
    # parser.add_argument('--freeze-layers', action='store_true', default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (e.g. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
