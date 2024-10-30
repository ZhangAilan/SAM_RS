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

def get_min_region(image):
    import numpy as np
    region_size = 30
    
    # 确保图像是 NumPy 数组
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # 获取最小值的索引
    y_min, x_min = np.unravel_index(np.argmin(image), image.shape)
    
    # 计算区域的边界
    half_size = region_size // 2
    x_min_bound = max(0, x_min - half_size)
    x_max_bound = min(image.shape[1] - 1, x_min + half_size)
    y_min_bound = max(0, y_min - half_size)
    y_max_bound = min(image.shape[0] - 1, y_min + half_size)
    
    return [x_min_bound, y_min_bound, x_max_bound, y_max_bound]
