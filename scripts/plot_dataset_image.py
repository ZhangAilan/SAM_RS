'''
@zyh 2024/10/31
绘制数据集中的图像和掩码
'''
import matplotlib.pyplot as plt
from datasets import load_from_disk
import numpy as np

dataset = load_from_disk("data/dataset/500imags")  #加载数据集
idx=15

#随机绘制输入数据以及mask（分割后）
example_image = dataset[idx]["image"]
example_mask = dataset[idx]["label"]
#将mask保存为txt
np.savetxt("results/dataset_mask.txt", example_mask, fmt='%d', delimiter=',')
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Assuming the first image is grayscale
axes[0].set_title("Image")
# Plot the second image on the right
axes[1].imshow(example_mask, cmap='gray')  # Assuming the second image is grayscale
axes[1].set_title("Mask")
# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
# 使用 plt.savefig() 将图像保存到文件中
plt.savefig("results/dataset_image.png")
print("Image saved as output_image.png")