# export LD_LIBRARY_PATH=/home/ycai51/anaconda3/envs/naf/lib
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import pickle

# from src.dataset import TIGREDataset as Dataset

from pdb import set_trace as stx
import argparse

start = time.time()

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default="0", help="gpu to use")
    return parser

parser = config_parser()
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


category = 'backpack'

path = f'../data/{category}_50.pickle'

with open(path, "rb") as handle:
            data = pickle.load(handle)

CT_image = np.swapaxes(data["image"], 0, 1)
# CT_image = CT_image[...,::-1]
# CT_image = CT_image[...,::-1]
# CT_image = CT_image[::-1,...]

# stx()

verts, faces, _, _ = measure.marching_cubes(CT_image)

# stx()

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
alpha = 0.30
mesh = Poly3DCollection(verts[faces], alpha=alpha)
face_color = [0.5, 0.5, 0.5]
mesh.set_facecolor(face_color)
ax.add_collection3d(mesh)

ax.set_xlim(0, CT_image.shape[0])
ax.set_ylim(0, CT_image.shape[1])
ax.set_zlim(0, CT_image.shape[2])

alpha_axis = 0.01
ax.set_alpha(alpha_axis)

# plt.savefig(f'3d_chest_{alpha}.png')
plt.savefig(f'3d_{category}_{alpha}_{alpha_axis}.png')

print(f"used time: {time.time()-start} s")