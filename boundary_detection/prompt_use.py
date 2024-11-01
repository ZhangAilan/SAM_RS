def generate_grid_points(image, grid_size):
    """
    @zyh 2024/11/1
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


def get_connectedComponent_boxes(ground_truth_mask , min_area=500):
    """
    @xh
    从指定路径的掩码图像中提取符合最小面积条件的连通组件边界框。
    
    参数：
    - mask_path (str): 掩码图像文件路径。
    - min_area (int): 最小面积阈值，默认值为500。仅保留面积大于或等于该值的组件。
    
    返回：
    - list 或 list[list[int]]: 如果检测到一个组件，返回单个框 [x_min, y_min, x_max, y_max]；
      如果检测到多个组件，返回多个框 [[x_min, y_min, x_max, y_max], ...]。
    """
    import numpy as np
    import torch
    import cv2   

    def hwtobox(box):
      '''
      @xh
      '''
      x1, y1 = box[0] + box[2], box[1] + box[3]
      box[2] = x1
      box[3] = y1
      return box

    # 转换为 uint8 类型并执行连通组件分析
    component_masks = (ground_truth_mask > 0).astype(np.uint8)  # 将掩码转换为二值图像
    _, labeled_masks, stats, _ = cv2.connectedComponentsWithStats(component_masks, connectivity=8)
    
    # 筛选符合面积要求的组件
    stats_pre = np.delete(stats[:, :], [0], axis=0)  # 去掉背景组件
    stats_pre = [n for n in stats_pre if n[-1] >= min_area]
    
    # 存储边界框
    boxes = []
    for x in stats_pre:
        box = hwtobox(x[:4])  # 将组件的 [x, y, width, height] 转换为 [x_min, y_min, x_max, y_max]
        boxes.append(box)
    
    # 根据检测到的组件数量返回单个框或多个框
    if len(boxes) == 1:
        return [boxes[0].tolist()]  # 返回单个框的列表 [x_min, y_min, x_max, y_max]，并转换为列表
    else:
        return [b.tolist() for b in boxes]  # 返回多个框的列表 [[x_min, y_min, x_max, y_max], ...]，并转换为列表


def get_centroid_points(component_masks, min_area=500):
    """
    @xh
    计算每个符合面积要求的连通组件的质心坐标，并生成符合指定格式的 PyTorch 张量。
    
    参数：
    - component_masks (numpy.ndarray): 二值化后的掩膜图像。
    - min_area (int): 最小面积阈值，默认值为500。仅保留面积大于或等于该值的组件。
    
    返回：
    - input_points (torch.Tensor): 形状为 (1, 1, num_points, 2) 的张量，其中包含所有符合条件的质心点坐标。
    - input_label (list): 点标签列表，形状为 (num_points,)，值为1。
    """
    import numpy as np
    import cv2
    import torch
    # 计算连通组件及其质心
    _, _, stats, centroids = cv2.connectedComponentsWithStats(component_masks, connectivity=8)
    
    # 筛选符合面积要求的组件质心
    stats_pre = np.delete(stats, 0, axis=0)  # 删除背景组件
    centroids = np.delete(centroids, 0, axis=0)  # 删除背景的质心

    # 保留面积大于等于 min_area 的质心
    filtered_centroids = [centroids[i] for i, stat in enumerate(stats_pre) if stat[-1] >= min_area]
    
    # 转换为指定形状的 PyTorch 张量
    point_cor = np.array(filtered_centroids).astype(int)  # 转换为整数类型
    num_points = len(point_cor)
    input_points = torch.tensor(point_cor).view(1, 1, num_points, 2)
    
    # 生成点标签列表
    input_label = [[1] * len(point_cor)]
    
    return input_points, input_label
