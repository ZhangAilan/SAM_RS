'''
@zyh 2024/10/31
使用原始模型进行单张图片的预测
测试不同prompt下的生成效果
'''
from transformers import SamModel, SamConfig, SamProcessor
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from prompt_use import get_bounding_box, generate_grid_points

# 设置参数
image_path='data/2021LoveDA/Val/Urban/images_png/3548.png'  
mask_path='data/2021LoveDA/Val/Urban/masks_png/3548.png'
save_path='results'  #保存结果的路径
prompt_flag='2' #prompt类型 1:bounding box 2:grid points
grid_size=100    #将图像分割成grid_size*grid_size的网格

# 加载模型和处理器
test_image=np.array(Image.open(image_path))
ground_truth_mask = np.array(Image.open(mask_path))
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:",device)
if test_image.ndim == 2:
    test_image = np.stack((test_image,) * 3, axis=-1)

# 将图像和提示转换为模型输入格式
if prompt_flag=='1':
    prompt=get_bounding_box(ground_truth_mask)
    inputs = processor(test_image, input_boxes=[[prompt]], return_tensors="pt")
elif prompt_flag=='2':
    prompt=generate_grid_points(ground_truth_mask, grid_size)
    inputs = processor(test_image, input_points=prompt, return_tensors="pt")

# 使用模型进行预测
inputs = {k: v.to(device) for k, v in inputs.items()}
model = SamModel.from_pretrained("facebook/sam-vit-base")
model.to(device)
model.eval()
with torch.no_grad():
    outputs = model(**inputs, multimask_output=False)
seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
seg_prob = seg_prob.cpu().numpy().squeeze()
seg = (seg_prob > 0.5).astype(np.uint8)

# 保存预测结果
fig,axes=plt.subplots(1,4,figsize=(15,5))
axes[0].imshow(test_image)
axes[0].set_title("Original Image")
axes[1].imshow(ground_truth_mask)
axes[1].set_title("Ground Truth Mask")
axes[2].imshow(seg) 
axes[2].set_title("rawSAM Mask")
axes[3].imshow(seg_prob)  
axes[3].set_title("rawSAM Probability Map")
plt.colorbar(axes[3].imshow(seg_prob), ax=axes[3])
plt.savefig(f"{save_path}/single_image_prediction(raw model).png")
print("Prediction saved successfully!")