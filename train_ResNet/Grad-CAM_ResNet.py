import os
import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision import models
#from utils import GradCAM, show_cam_on_image, center_crop_img
from pytorch_grad_cam.utils.image import show_cam_on_image

from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise
import torch
from matplotlib import pyplot as plt
from torch import nn

from torchvision.transforms import transforms
def main():
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                   kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                   kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channel)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            out += identity
            out = self.relu(out)

            return out


    class Bottleneck(nn.Module):

        expansion = 4

        def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                     groups=1, width_per_group=64):
            super(Bottleneck, self).__init__()

            width = int(out_channel * (width_per_group / 64.)) * groups

            self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                                   kernel_size=1, stride=1, bias=False)  # squeeze channels
            self.bn1 = nn.BatchNorm2d(width)
            # -----------------------------------------
            self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                                   kernel_size=3, stride=stride, bias=False, padding=1)
            self.bn2 = nn.BatchNorm2d(width)
            # -----------------------------------------
            self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                                   kernel_size=1, stride=1, bias=False)  # unsqueeze channels
            self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            out += identity
            out = self.relu(out)

            return out


    class ResNet(nn.Module):

        def __init__(self,
                     block,
                     blocks_num,
                     num_classes=7,
                     include_top=True,
                     groups=1,
                     width_per_group=64):
            super(ResNet, self).__init__()
            self.include_top = include_top
            self.in_channel = 64

            self.groups = groups
            self.width_per_group = width_per_group

            self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                                   padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.in_channel)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, blocks_num[0])
            self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
            self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
            self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
            if self.include_top:
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
                self.fc = nn.Linear(512 * block.expansion, num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        def _make_layer(self, block, channel, block_num, stride=1):
            downsample = None
            if stride != 1 or self.in_channel != channel * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(channel * block.expansion))

            layers = []
            layers.append(block(self.in_channel,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
            self.in_channel = channel * block.expansion

            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    width_per_group=self.width_per_group))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            if self.include_top:
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)

            return x


    def resnet34(num_classes=6, include_top=True):
        # https://download.pytorch.org/models/resnet34-333f7ec4.pth
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


    def resnet50(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnet50-19c8e357.pth
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


    def resnet101(num_classes=7, include_top=True):
        # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


    def resnext50_32x4d(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
        groups = 32
        width_per_group = 4
        return ResNet(Bottleneck, [3, 4, 6, 3],
                      num_classes=num_classes,
                      include_top=include_top,
                      groups=groups,
                      width_per_group=width_per_group)


    def resnext101_32x8d(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
        groups = 32
        width_per_group = 8
        return ResNet(Bottleneck, [3, 4, 23, 3],
                      num_classes=num_classes,
                      include_top=include_top,
                      groups=groups,
                      width_per_group=width_per_group)


    net=resnet34()
    device = torch.device("cpu")
    net.load_state_dict(
        torch.load("./resnet34.24.pth", map_location=device))  # 载入训练的resnet模型权重，你将训练的模型权重放到当前文件夹下即可

    target_layers = [net.layer3[-1]]
    print(target_layers)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        ])

    imgs_root = "./data_2/gradcam/"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.repeat(3,1,1)),
    ])
    for ids in range(0, len(img_path_list)):
        img_list = []
        for img_path in img_path_list[ids:(ids + 1)]:
            assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
            name = img_path
            # name2 = name[10:-4]
            name2 = name[17:-4]
            img = Image.open(img_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)
            #img = center_crop_img(img, image_size)

            # [C, H, W]
            img_tensor = data_transform(img)
            # expand batch dimension
            # [C, H, W] -> [N, C, H, W]
            input_tensor = torch.unsqueeze(img_tensor, dim=0)  # 增加一个batch维度
            cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)
            grayscale_cam = cam(input_tensor=input_tensor)

            grayscale_cam = grayscale_cam[0, :]
            # grayscale_cam = np.maximum(grayscale_cam, 0)
            visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                              grayscale_cam,
                                              use_rgb=True)
            plt.axis('off')
            plt.imshow(visualization)
            plt.savefig('./data_2/gradcam_out_2/'+str(name2)+'_3[-1].jpg', bbox_inches='tight', pad_inches=0)  # 将热力图的结果保存到本地当前文件夹

            plt.show()

if __name__ == '__main__':
    main()
