import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from patchify import patchify  #Only to handle large images
from datasets import Dataset
import random


images_folder_path='data/2021LoveDA/Train/Urban/images_L_png' #读取输入的训练数据
masks_folder_path='data/2021LoveDA/Train/Urban/masks_png'
save_path = "data/dataset/all_images"  #保存数据集的路径
max_files = 500  # 设置计数器上限

#遍历文件夹内所有图片
images = []
masks = []
# 遍历前n个图像文件
image_count = 0
for root, dirs, files in os.walk(images_folder_path):
    for file in files:
        if file.endswith('.png'):
            image_path = os.path.join(root, file)
            img = Image.open(image_path)  # 使用 PIL 加载图片
            images.append(np.array(img))  # 转换为 NumPy 数组并添加到列表
            image_count += 1
            if image_count >= max_files:
                break
    if image_count >= max_files:
        break

mask_count = 0
for root, dirs, files in os.walk(masks_folder_path):
    for file in files:
        if file.endswith('.png'):
            mask_path = os.path.join(root, file)
            mask = Image.open(mask_path)  # 使用 PIL 加载图片
            masks.append(np.array(mask))
            mask_count += 1
            if mask_count >= max_files:
                break
    if mask_count >= max_files:
        break
# 将图片和掩码列表转换为 NumPy 数组
images = np.array(images)
masks = np.array(masks)
print('images:',images.shape)
print('masks',masks.shape)
print(images[0])

#将大图像和mask分割成小图像和mask
patch_size=256
step=256
all_img_patches = []
all_mask_patches = []
for img in range(images.shape[0]):
    large_image = images[img]
    patches_img=patchify(large_image, (patch_size, patch_size), step=step)  #分割成小图像
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            all_img_patches.append(patches_img[i,j,:,:])
images=np.array(all_img_patches)
for mask in range(masks.shape[0]):
    large_mask = masks[mask]
    patches_mask=patchify(large_mask, (patch_size, patch_size), step=step)  #分割成小mask
    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            all_mask_patches.append(patches_mask[i,j,:,:])
masks=np.array(all_mask_patches)
print('images:',images.shape)
print('masks',masks.shape)

# 创建一个列表存储非空掩码的索引
valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]

# 筛选出非空掩码对应的图片对
filtered_images = images[valid_indices]
filtered_masks = masks[valid_indices]

# 输出筛选后图片和掩码的形状
print("Image shape:", filtered_images.shape)  # 形如 (num_valid_frames, height, width)
print("Mask shape:", filtered_masks.shape)

# Convert the NumPy arrays to Pillow images and store them in a dictionary
dataset_dict = {
    "image": [Image.fromarray(img) for img in filtered_images],
    "label": [Image.fromarray(mask) for mask in filtered_masks],
}

# Create the dataset using the datasets.Dataset class
dataset = Dataset.from_dict(dataset_dict)
print(dataset)
# 导出数据集到指定路径
dataset.save_to_disk(save_path)
print('数据集已保存到指定路径!!!')