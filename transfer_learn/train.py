from datasets import load_from_disk
import numpy as np
from torch.utils.data import Dataset
from transformers import SamProcessor
from torch.utils.data import DataLoader
from transformers import SamModel
import torch
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
from statistics import mean
from prompt_train import get_bounding_box

#路径
dataset = load_from_disk("data/dataset/500imags")  #加载数据集
save_path = "models"  #保存模型的路径
loop_times = 1  #训练次数
batch_size = 1  #批处理大小

#检查加载的数据集
print(dataset)
print('number of samples:',len(dataset))

class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = np.array(item["image"])  # 将 PIL 图像转换为 NumPy 数组
    ground_truth_mask = np.array(item["label"])

    # 检查图像维度，如果是灰度图像则添加通道维度
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)  # 将 (height, width) 转换为 (height, width, 3) 三通道灰度图

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)
    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}
    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask
    return inputs

# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the SAMDataset
train_dataset = SAMDataset(dataset=dataset, processor=processor)
example = train_dataset[0]
print("train_dataset第一个样本：")
for k,v in example.items():
  print(k,v.shape)

# Create a DataLoader instance for the training dataset
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
batch = next(iter(train_dataloader))
print("每次批处理的样本:")
for k,v in batch.items():
  print(k,v.shape)

# Load the model
model = SamModel.from_pretrained("facebook/sam-vit-base")
# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)

# Initialize the optimizer and the loss function
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
# 这里选择交叉熵损失（适用于多类分割任务）
seg_loss = nn.CrossEntropyLoss()

#Training loop
num_epochs = loop_times
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Training on {device}")

model.train()
print("Start training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")
    epoch_losses = []
    for batch in tqdm(train_dataloader):
      # forward pass
      outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

      # compute loss
      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')

# Save the model's state dictionary to a file
torch.save(model.state_dict(), save_path + "/sam_model.pth")
print("Model saved successfully!")