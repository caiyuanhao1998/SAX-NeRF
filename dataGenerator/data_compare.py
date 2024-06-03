'''
    将两个原始数据加载并打印出来看看
'''

import scipy.io as io
import numpy as np
from pdb import set_trace as stx
import cv2
import imageio.v2 as iio

chest_mat_path = '/data2/cyh23/NAF_raw_data/chest/img.mat'          # 胸腔
jaw_mat_path = '/data2/cyh23/NAF_raw_data/jaw/img.mat'              # 下巴
foot_mat_path = '/data2/cyh23/NAF_raw_data/foot/img.mat'            # 脚
abdomen_mat_path = '/data2/cyh23/NAF_raw_data/abdomen/img.mat'      # 腹部

raw_path = '/data2/cyh23/NAF_raw_data/vis_male_128x256x256_uint8.raw'

'''
    mat_data 读取出来后是一个字典, 有如下的关键字:
    __header__  : b'MATLAB 5.0 MAT-file Platform: posix, Created on: Fri Oct  8 01:53:35 2021'
    __version__ : '1.0'
    __globals__ : []
    img         : 一个 【128, 128, 128】 的numpy, 就是 CT slide, 就是数据存放的地方, 
                    里面的数据类型是 numpy.ndarry, 然后每一位都是 32-bit 浮点数

                chest:      [128, 128, 128]
                jaw:        [256, 256, 256]
                foot:       [256, 256, 256]
                abdomen:    [512, 512, 463]
'''

chest_mat_data   =  io.loadmat(chest_mat_path) 
jaw_mat_data     =  io.loadmat(jaw_mat_path)
foot_mat_data    =  io.loadmat(foot_mat_path)
abdomen_mat_data =  io.loadmat(abdomen_mat_path)
# stx()

'''
    example: head

    unit8 : unsigned int 无符号 8-bit 整型数, 
        取值范围是 2^8 - 1, 0 ~ 255 matlab默认的存图方式
    
    对于 .raw 文件, 本质上是以二进制存储的一个大文件, 
        所以数据的读取要注意位宽, 位宽不对, 数据就有可能读错。
    
    是一个shape为 [256*256*128] 的 numpy, 数据类型为 0 - 255 的整数
'''

raw_data = np.fromfile(raw_path, dtype='uint8')
raw_data_vis = raw_data.reshape(256, 256, 128)
# raw_data_vis = raw_data.reshape(256, 128, 256)

show_slice = 10
show_step = raw_data_vis.shape[-1]//show_slice
show_image = raw_data_vis[...,::show_step]
# show_image = np.concatenate(show_image, axis=0)


for i in range(show_slice+1):
    iio.imwrite('CT_vis/head_CT_raw_'+str(i)+'.png', show_image[...,i])

stx()

img_data = (np.float32(raw_data) / 255.0).reshape(256, 256, 128)

dict_data = dict()
dict_data['img'] = img_data
stx()

io.savemat('head/img.mat', dict_data)

