'''
@zyh 2024/10/30
将一个文件夹内所有png图片转为灰度图，并保存到新的文件夹
'''
import os
from PIL import Image

def rgb_to_L(src_path, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    for file in os.listdir(src_path):
        img = Image.open(os.path.join(src_path, file))
        img = img.convert('L')
        img.save(os.path.join(dst_path, file))
    print('Done!')

if __name__ == '__main__':
    src_path = 'data/2021LoveDA/Train/Urban/images_png'
    dst_path = 'data/2021LoveDA/Train/Urban/images_L_png'
    rgb_to_L(src_path, dst_path)