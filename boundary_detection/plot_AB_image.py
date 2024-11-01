'''
@zyh 2024/10/31
使用原始模型进行AB时相图片的预测，mask为深度学习预测的变化结果
测试不同prompt下的生成效果
'''
from transformers import SamModel, SamProcessor
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from prompt_use import generate_grid_points,get_connectedComponent_boxes,get_centroid_points
from image_operation import union_image,intersection_image

# 设置参数
image_name='baoding1_768_2048'
# image_name='baoding1_2816_768'
# image_name='baoding1_4096_768'
save_path='boundary_detection/results'  #保存结果的路径
prompt_flag='3'                         #prompt类型 1:connectedComponent-box 2:grid-points 3:centroid-points
seg_value=0.5                           #阈值
grid_size=100                           #将图像分割成grid_size*grid_size的网格
min_area=500                            #连通组件的最小面积

A_image_path='data/ABdata/A/{}.png'.format(image_name)  
B_image_path='data/ABdata/B/{}.png'.format(image_name)
mask_path='data/ABdata/predict_orignial/{}.png'.format(image_name)
label_path='data/ABdata/label/{}.png'.format(image_name)

# 读取图片
A_image=np.array(Image.open(A_image_path))
B_image=np.array(Image.open(B_image_path))
ground_truth_mask = np.array(Image.open(mask_path))
label_image=np.array(Image.open(label_path))
if A_image.ndim == 2:
    A_image = np.stack((A_image,) * 3, axis=-1)  #将灰度图转换,使其具有三个通道
if B_image.ndim == 2:
    B_image = np.stack((B_image,) * 3, axis=-1)
print("A_image.shape:",A_image.shape)
print("B_image.shape:",B_image.shape)
if np.max(ground_truth_mask)>0: #判断是否有大于0的值
    print("ground_truth_mask.shape",ground_truth_mask.shape)
else:
    print("ground_truth_mask is empty!!!\n")

#加载模型
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:",device)

# 将图像和提示转换为模型输入格式
if prompt_flag=='1':
    prompt_type='connectedComponent-box'
    prompt=get_connectedComponent_boxes(ground_truth_mask,min_area)
    print("prompt:",prompt)
    A_inputs = processor(A_image, input_boxes=[prompt], return_tensors="pt")
    B_inputs = processor(B_image, input_boxes=[prompt], return_tensors="pt")
elif prompt_flag=='2':
    prompt_type='grid-points'
    prompt=generate_grid_points(ground_truth_mask, grid_size)
    print("prompt:",prompt)
    A_inputs = processor(A_image, input_points=prompt, return_tensors="pt")
    B_inputs = processor(B_image, input_points=prompt, return_tensors="pt")
elif prompt_flag=='3':
    prompt_type='centroid-points'
    prompt,labels=get_centroid_points(ground_truth_mask,min_area)
    print("prompt:",prompt)
    print("labels:",labels)
    A_inputs = processor(A_image, input_points=prompt,input_labels=labels,return_tensors="pt")
    B_inputs = processor(B_image, input_points=prompt,input_labels=labels,return_tensors="pt")

# 使用模型进行预测，对两幅图片进行预测
A_inputs = {k: v.to(device) for k, v in A_inputs.items()}
B_inputs = {k: v.to(device) for k, v in B_inputs.items()}
model = SamModel.from_pretrained("facebook/sam-vit-base")
model.to(device)
model.eval()
with torch.no_grad():
    A_outputs = model(**A_inputs,multimask_output=False)
    B_outputs = model(**B_inputs, multimask_output=False)
A_seg_prob = torch.sigmoid(A_outputs.pred_masks.squeeze(1))
B_seg_prob = torch.sigmoid(B_outputs.pred_masks.squeeze(1))
A_seg_prob = A_seg_prob.cpu().numpy().squeeze()
B_seg_prob = B_seg_prob.cpu().numpy().squeeze()
A_seg = (A_seg_prob > seg_value).astype(np.uint8)
B_seg = (B_seg_prob > seg_value).astype(np.uint8)
if A_seg.ndim > 2:
    A_seg = np.max(A_seg, axis=0)
if B_seg.ndim > 2:
    B_seg = np.max(B_seg, axis=0)  # 若有多个mask，合并
if A_seg_prob.ndim > 2:
    A_seg_prob = np.max(A_seg_prob, axis=0)
if B_seg_prob.ndim > 2:
    B_seg_prob = np.max(B_seg_prob, axis=0)

# 保存预测结果
fig,axes=plt.subplots(2,5,figsize=(10,5))
axes[0][0].imshow(ground_truth_mask,cmap='gray')
axes[0][0].set_title("Input Mask")
axes[0][1].imshow(A_image)
axes[0][1].set_title("A Image")
axes[0][2].imshow(A_seg,cmap='gray')
axes[0][2].set_title("A Segment")
axes[0][3].imshow(A_seg_prob)
axes[0][3].set_title("A Probability Map")
axes[0][4].imshow(union_image(A_seg,B_seg),cmap='gray')
axes[0][4].set_title("Union Mask")

axes[1][0].imshow(label_image,cmap='gray')
axes[1][0].set_title("True Mask")
axes[1][1].imshow(B_image)
axes[1][1].set_title("B Image")
axes[1][2].imshow(B_seg,cmap='gray')
axes[1][2].set_title("B Segment")
axes[1][3].imshow(B_seg_prob)
axes[1][3].set_title("B Probability Map")
axes[1][4].imshow(intersection_image(A_seg,B_seg),cmap='gray')
axes[1][4].set_title("Inter Mask")
fig.suptitle(f"Prompt Type: {prompt_type},  Threshold: {seg_value}")
plt.tight_layout()
plt.savefig(f"{save_path}/{image_name}_{prompt_type}.png")
print("Prediction saved successfully!")