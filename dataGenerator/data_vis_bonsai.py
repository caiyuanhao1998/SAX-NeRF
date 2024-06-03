'''
    将两个原始数据加载并打印出来看看
'''

import scipy.io as io
import numpy as np
from pdb import set_trace as stx
import cv2
import imageio.v2 as iio
import os

category = 'bonsai'

raw_path = '/data2/cyh23/NAF_raw_data/bonsai_256x256x256_uint8.raw'

raw_data = np.fromfile(raw_path, dtype='uint8')
# stx()
# raw_data_vis = raw_data.reshape(256, 256, 256)
# raw_data_vis = np.swapaxes(np.swapaxes(raw_data.reshape(256, 256, 256), 0, 1), 1, 2)
raw_data_vis = np.swapaxes(raw_data.reshape(256, 256, 256), 1, 2)
# raw_data_vis = np.swapaxes(np.swapaxes(raw_data.reshape(56, 301, 324), 0, 1), 1, 2)
# raw_data_vis = np.swapaxes(np.swapaxes(raw_data.reshape(56, 324, 301), 0, 1), 1, 2)
# raw_data_vis = np.swapaxes(np.swapaxes(raw_data.reshape(93, 341, 341), 0, 1), 1, 2)
# raw_data_vis = raw_data.reshape(341, 93, 341)

show_slice = 50
show_step = raw_data_vis.shape[-1]//show_slice
show_image = raw_data_vis[...,::show_step]

vis_dir = 'CT_vis/' + category + '/'
os.makedirs(vis_dir, exist_ok=True)

for i in range(show_slice+1):
    iio.imwrite(vis_dir + 'CT_raw_'+str(i) + '.png', show_image[...,i])



stx()

img_data = (np.float32(raw_data_vis) / 255.0)
dict_data = dict()
dict_data['img'] = img_data
# stx()
save_dir = 'raw_data/' + category + '/'
os.makedirs(save_dir, exist_ok=True)
io.savemat(save_dir+'img.mat', dict_data)

