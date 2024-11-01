# 变化区域边界检测
### prompt_use.py（prompt的不同选择方法）
```
generate_grid_points：生成图像中均匀分布的网格点坐标
get_connectedComponent_boxes：提取符合最小面积条件的连通组件边界框
get_centroid_points：计算每个符合面积要求的连通组件的质心坐标
```