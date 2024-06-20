import cv2
import os
import torch
import scipy.io
import numpy as np
import torchvision.transforms as T
from PIL import Image
import trimesh
import pytorch3d
import pytorch3d.structures as p3dstr
import torchvision.transforms.functional as F
import pickle

import sys
image_size=224

class Resize_with_pad:
    def __init__(self, w=224, h=224):
        self.w = w
        self.h = h

    def __call__(self, image):

        _, w_1, h_1 = image.size()
        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1


        # check if the original and final aspect ratios are the same within a margin
        if round(ratio_1, 2) != round(ratio_f, 2):

            # padding to preserve aspect ratio
            hp = int(w_1/ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                hp = hp // 2
                image = F.pad(image, (hp, 0, hp, 0), 0, "constant")
                return F.resize(image, (self.h, self.w))

            elif hp < 0 and wp > 0:
                wp = wp // 2
                image = F.pad(image, (0, wp, 0, wp), 0, "constant")
                return F.resize(image, (self.h, self.w))

        else:
            return F.resize(image, (self.h, self.w))

transform = T.Resize((image_size, image_size))

def load_images_v2(path):
    path = path.strip()
    img = Image.open(path).convert('RGB')
    img = transform(img)
    img = torch.Tensor(np.array(img))
    img = torch.permute(img, (2, 0, 1))

    return img

def load_masks(path):
    path = path.strip()
    img = np.load(path)
    img = np.reshape(img, (1, img.shape[0], img.shape[1]))
    img = torch.Tensor(img)
    rwp = Resize_with_pad()
    img = rwp(img)

    return img
    
path="/root/Prim3D/d3dhoi_video_data/microwave/b004-0001/frames/images-0001.jpg"
mask_path="/root/Prim3D/d3dhoi_video_data/microwave/b004-0001/gt_mask/0001_object_mask.npy"


import matplotlib.pyplot as plt
# debug
# rgb= load_images_v2(path)
# o_mask = load_masks(mask_path)

# o_mask_3ch=o_mask.repeat(3,1,1) #torch.Size([3, 224, 224])
# object_image = rgb * o_mask
# object_image_np = object_image.permute(1,2,0).numpy()
# # object_image_np = np.clip(object_image_np, 0, 1)
# object_image_np = np.clip(object_image_np, 0, 1) * 255
# object_image_np = object_image_np.astype(np.uint8)

# output_image = Image.fromarray(object_image_np)
# output_image.save('/root/Prim3D/object_image_rendered.png')


import torch
import numpy as np
from PIL import Image

# 示例图片 (3, 224, 224) 和 object_mask (1, 224, 224)
# 这里我们用随机数据生成假设的图像和掩码
image = load_images_v2(path)  # 假设这是你的图片 Tensor，值在 [0, 1] 之间
object_mask = load_masks(mask_path)  # 假设这是你的 object_mask Tensor，值为 0 或 1

# # 将 object_mask 复制到与图片相同的通道数
# object_mask_3ch = object_mask.repeat(3, 1, 1)

# # 使用 object_mask 进行逐元素相乘
# object_image = image * object_mask_3ch

# # 将 Tensor 转换为 NumPy 数组以便保存为图片
# object_image_np = object_image.permute(1, 2, 0).numpy()

# # 确保数据在 [0, 1] 范围内，并缩放到 [0, 255] 范围内
# object_image_np = np.clip(object_image_np, 0, 1) * 255
# object_image_np = object_image_np.astype(np.uint8)

# # 保存结果图片到本地
# output_image = Image.fromarray(object_image_np)
# output_image.save('object_image_rendered.png')

# print("图片已保存到 'object_image_rendered.png'")

import torch
from PIL import Image

# 假设 image_tensor 是你的彩色图片tensor，形状为 (3, 224, 224)
# object_mask 是你的物体遮罩tensor，形状为 (1, 224, 224)
# 确保两者都在 CPU 上且数据类型为 float32
image_tensor  =  load_images_v2(path) 
object_mask = load_masks(mask_path) 
# 遮罩可能需要扩展至与图像相同的通道数
mask_expanded = object_mask.expand_as(image_tensor)

# 使用遮罩提取物体部分
object_img_tensor = image_tensor * mask_expanded

# 将 tensor 数据类型转换为 uint8，取值范围在0到255之间，如果未在0-255范围内，需要先进行缩放
object_img_tensor = (object_img_tensor * 255).to(torch.uint8)

# Permute张量维度，从 (C, H, W) 变成 (H, W, C) 以符合PIL的要求
object_img_tensor = object_img_tensor.permute(1, 2, 0)

# 将 tensor 转换为 PIL 图像
object_img_pil = Image.fromarray(object_img_tensor.numpy())

# 保存图像到本地
object_img_pil.save('object_img.png')
