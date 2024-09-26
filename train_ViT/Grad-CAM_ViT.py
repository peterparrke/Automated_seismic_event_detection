import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils_gradcam import GradCAM, show_cam_on_image, center_crop_img
from vit_model import vit_base_patch16_224_in21k
class ReshapeTransform:
    def __init__(self, model):
        input_size = model.patch_embed.img_size
        patch_size = model.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


def main():
    imgs_root = "./data_2/gradcam/"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]

    model = vit_base_patch16_224_in21k(num_classes=6)
    weights_path = "./weights/model-24.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    # Since the final classification is done on the class token computed in the last attention block,
    # the output will not be affected by the 14x14 channels in the last layer.
    # The gradient of the output with respect to them, will be 0!
    # We should chose any layer before the final attention block.
    target_layers = [model.blocks[-2].norm2]
    # target_layers

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.9078, 0.7846, 0.9741], [0.1788, 0.2800, 0.0500])])
    # load image
    for ids in range(0, len(img_path_list)):
        for img_path in img_path_list[ids:(ids + 1)]:
            assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
            name = img_path
            # name2 = name[10:-4]
            name2 = name[16:-4]
            img = Image.open(img_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)
            img = center_crop_img(img, 224)
            # [C, H, W]
            img_tensor = data_transform(img)
            # expand batch dimension
            # [C, H, W] -> [N, C, H, W]
            input_tensor = torch.unsqueeze(img_tensor, dim=0)

            cam = GradCAM(model=model,
                          target_layers=target_layers,
                          use_cuda=False,
                          reshape_transform=ReshapeTransform(model))
            # target_category = 2  # tabby, tabby cat
            # target_category = 254  # pug, pug-dog

            grayscale_cam = cam(input_tensor=input_tensor)

            grayscale_cam = grayscale_cam[0, :]
            # grayscale_cam = np.maximum(grayscale_cam, 0)
            visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
            plt.axis('off')
            plt.imshow(visualization)
            plt.savefig('./data_2/gradcam_out_2/' + str(name2) + '_norm2[-2].jpg', bbox_inches='tight', pad_inches=0)
            plt.show()


if __name__ == '__main__':
    main()
