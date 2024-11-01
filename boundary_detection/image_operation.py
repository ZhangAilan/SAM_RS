'''
@zyh 2024/11/1
一些图像操作函数
'''

def union_image(img1, img2):
    '''
    @zyh 2024/11/1
    取两幅图像的逐像素并集
    ''' 
    import numpy as np
    union_img = np.logical_or(img1, img2).astype(int)
    return union_img

def intersection_image(img1, img2):
    '''
    @zyh 2024/11/1
    取两幅图像的逐像素交集
    ''' 
    import numpy as np
    intersection_img = np.logical_and(img1, img2).astype(int)
    return intersection_img