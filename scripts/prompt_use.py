def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  import numpy as np
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]
  return bbox

def generate_grid_points(image, grid_size):
    """
    生成图像中均匀分布的网格点坐标，并返回符合模型输入格式的张量。
    
    参数:
    - image (np.ndarray): 输入图像，形状为 (H, W)。
    - grid_size (int): 每个维度中点的数量，将图像分割为 grid_size x grid_size 的网格。
    
    返回:
    - input_points (torch.Tensor): 形状为 (1, 1, grid_size*grid_size, 2) 的张量，
      每个点为图像中网格的 (x, y) 坐标。
    """
    import numpy as np
    import torch
    height, width = image.shape[:2]
    
    # 生成 x 和 y 方向上等间距的网格点
    x = np.linspace(0, width - 1, grid_size)
    y = np.linspace(0, height - 1, grid_size)   
    # 生成网格坐标
    xv, yv = np.meshgrid(x, y) 
    # 将网格坐标转为嵌套列表格式
    input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv.tolist(), yv.tolist())]
    # 转换为 PyTorch 张量并调整形状
    input_points = torch.tensor(input_points).view(1, 1, grid_size * grid_size, 2)   
    return input_points
