from PIL import Image
import os
from tqdm import tqdm
from pdb import set_trace as stx
import time



# 图像文件夹路径和输出 GIF 文件名
# img_folder = '/home/ycai51/Medical_NeRF/3D_vis/leg/elevation_30_sigma_0.6_alpha_0.5'
# category = 'leg'

img_folder = '/home/ycai51/Medical_NeRF/3D_vis/box/elevation_20_sigma_0.6_alpha_0.3_axisoff'
category = 'box'


proj_num = 120
angle_interval = 360 / proj_num

# 获取所有图像文件名
img_files = []


for i in range(proj_num):
    angle = angle_interval * i
    img_files.append(os.path.join(img_folder, f'angle_{angle}.png'))

# dx_1 = 100
# dy_1 = 200
# dx_2 = 100
# dy_2 = -150

box = (300+150, 300+200, 1700-50, 1700-150)

# 打开第一张图像
img = Image.open(img_files[0])

# 创建 GIF 对象，将第一张图像作为基准帧
gif_frames = [img.crop(box)]

# 逐一添加图像帧
for filename in tqdm(img_files[1:]):
    img = Image.open(filename)
    
    # stx()
    gif_frames.append(img.crop(box))

start = time.time()

fps = 45
duration = 1000 / fps

gif_filename = f'rotate_{category}_fps_{fps}_frame_{proj_num}.gif'

# 保存 GIF 动画
gif_frames[0].save(gif_filename, save_all=True, append_images=gif_frames[1:], duration=duration, loop=0)

print(f'used time for saving gif: {time.time() - start} s')

