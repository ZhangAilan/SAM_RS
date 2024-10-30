'''
@zyh 2024/10/31
检查模型效果：原始模型和经遥感数据训练后的模型，分别绘制两种模型的分割结果
'''

from transformers import SamModel, SamConfig, SamProcessor
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from prompt_use import get_bounding_box,get_min_region

#加载图片
test_image=np.array(Image.open('data/2021LoveDA/Val/Urban/images_png/3566.png'))
ground_truth_mask = np.array(Image.open('data/2021LoveDA/Val/Urban/masks_png/3566.png'))
print("ground_truth_mask:\n",ground_truth_mask)
model_path='models/sam_model.pth'
save_path='results'  #保存结果的路径

if test_image.ndim == 2:
    test_image = np.stack((test_image,) * 3, axis=-1)  #将 (height, width) 转换为 (height, width, 1) 三通道灰度图

#根据真实分割标签生成边界框提示，用于指导模型分割
prompt = get_bounding_box(ground_truth_mask)  
# prompt=get_min_region(ground_truth_mask)
print("Prompt:",prompt)

# Load the model configuration
model_config = SamConfig.from_pretrained("facebook/sam-vit-base")  #预加载模型配置
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the model architecture with the loaded configuration
my_rs_model = SamModel(config=model_config)
#Update the model by loading the weights from saved file.
my_rs_model.load_state_dict(torch.load(model_path))  

# set the device to cuda if available, otherwise use cpu
# 选择GPU或者CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
my_rs_model.to(device)
print("device:",device)

# prepare image + box prompt for the model
inputs = processor(test_image, input_boxes=[[prompt]], return_tensors="pt")

# Move the input tensor to the GPU if it's not already there
inputs = {k: v.to(device) for k, v in inputs.items()}

my_rs_model.eval()  #设置为评估模式

with torch.no_grad():
    outputs = my_rs_model(**inputs,multimask_output=False)  #模型推理

# apply sigmoid
RSsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
RSsam_seg_prob = RSsam_seg_prob.cpu().numpy().squeeze()
RSsam_seg = (RSsam_seg_prob > 0.5).astype(np.uint8)

fig1, axes = plt.subplots(1, 4, figsize=(15, 5))

# Plot the first image on the left
axes[0].imshow(np.array(test_image), cmap='gray')  # Assuming the first image is grayscale
axes[0].set_title("Raw Image")
axes[1].imshow(ground_truth_mask, cmap='gray')
axes[1].set_title("Ground Truth Mask")
# Plot the second image on the right
axes[2].imshow(RSsam_seg, cmap='gray')  # Assuming the second image is grayscale
axes[2].set_title("Trained Mask")

# Plot the second image on the right
axes[3].imshow(RSsam_seg_prob)  # Assuming the second image is grayscale
axes[3].set_title("Trained Probability Map")
plt.colorbar(axes[3].imshow(RSsam_seg_prob), ax=axes[3])

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.savefig(f"{save_path}/RSmodel_use_to_image.png")
print("RSmodel_use_to_image.png has been saved.")

#使用原始模型，作为对比
raw_sam_model=SamModel(config=model_config)
raw_sam_model.to(device)
raw_sam_model.eval()
with torch.no_grad():
    outputs_raw = my_rs_model(**inputs, multimask_output=False)  # 模型推理
# apply sigmoid
raw_seg_prob = torch.sigmoid(outputs_raw.pred_masks.squeeze(1))
# convert soft mask to hard mask
raw_seg_prob = raw_seg_prob.cpu().numpy().squeeze()
raw_seg = (raw_seg_prob > 0.5).astype(np.uint8)

fig2,axes=plt.subplots(1,4,figsize=(15,5))
axes[0].imshow(np.array(test_image), cmap='gray')  
axes[0].set_title("Raw Image")
axes[1].imshow(ground_truth_mask, cmap='gray')
axes[1].set_title("Ground Truth Mask")
axes[2].imshow(raw_seg, cmap='gray') 
axes[2].set_title("rawSAM Mask")
axes[3].imshow(raw_seg_prob)  
axes[3].set_title("rawSAM Probability Map")
plt.colorbar(axes[3].imshow(raw_seg_prob), ax=axes[3])
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
plt.savefig(f"{save_path}/rawSAM_use_to_image.png")
print("rawSAM_use_to_image.png has been saved.")
print("Done!")