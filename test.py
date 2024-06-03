import os
import os.path as osp
import torch
import imageio.v2 as iio
import numpy as np
import argparse
from tqdm import tqdm
import time

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default="6", help="gpu to use")
    parser.add_argument("--method", default=f"Lineformer", help="name of the tested method")
    parser.add_argument("--category", default=f"chest", help="category of the tested scene")
    parser.add_argument("--config", default=f"config/Lineformer/chest_50.yaml", help="path to configs file")
    parser.add_argument("--weights", default=f"pretrained/chest.tar", help="path to the experiments")
    parser.add_argument("--output_path", default=f"output", help="path to the output folder")
    parser.add_argument("--vis", default="True", help="visualization or not?")
    return parser

parser = config_parser()
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


from src.network import get_network
from src.encoder import get_encoder
from src.dataset import TIGREDataset as Dataset
from src.config.configloading import load_config
from src.render import render, run_network
from src.utils import get_psnr, get_ssim, get_psnr_3d, get_ssim_3d, cast_to_image
from pdb import set_trace as stx


def eval_step(eval_dset, model, model_fine, cfg):
    """
    Evaluation step
    """
    # Evaluate projection    渲染投射的 RGB 图
    # select_ind = np.random.choice(len(eval_dset))       # 13, 一个数字
    # stx()
    projs = eval_dset.projs                 # [256, 256] -> [50, 256, 256]
    rays = eval_dset.rays.reshape(-1, 8)    # [65536,8]  -> [3276800, 8]
    # stx()
    N, H, W = projs.shape
    projs_pred = []
    n_rays = cfg["train"]["n_rays"]
    netchunk = cfg["render"]["netchunk"]
    print("Start rendering projection")
    proj_start_time = time.time()
    for i in tqdm(range(0, rays.shape[0], n_rays)):     # 每一簇射线是 n_rays ，每隔这么多射线渲染一次
        projs_pred.append(render(rays[i:i+n_rays], model, model_fine, **cfg["render"])["acc"])
    proj_end_time = time.time()
    print(f"Time of rendering projection: {proj_end_time - proj_start_time} s")
    # stx()
    projs_pred = torch.cat(projs_pred, 0).reshape(N, H, W) # 3200 length, 1024, 在第0纬度上 concate 起来

    # Evaluate density      渲染3D图像
    image = eval_dset.image
    print("Start reconstructing CT")
    ct_start_time = time.time()
    image_pred = run_network(eval_dset.voxels, model_fine if model_fine is not None else model, netchunk)
    ct_end_time = time.time()
    print(f"Time of reconstructing CT: {ct_end_time - ct_start_time} s")
    # stx()
    image_pred = image_pred.squeeze()

    print("Evaluating performance...")
    loss = {
        "proj_psnr": get_psnr(projs_pred, projs),
        "proj_ssim": get_ssim(projs_pred, projs),
        "psnr_3d": get_psnr_3d(image_pred, image),
        "ssim_3d": get_ssim_3d(image_pred, image),
    }

    resdir = os.path.join(args.output_path, args.method, args.category)

    # Save
    # 保存各种视图

    proj_pred_dir = osp.join(resdir, "proj_pred")
    proj_gt_dir = osp.join(resdir, "proj_gt")

    ct_pred_dir_H = osp.join(resdir, "CT", "H", "ct_pred")
    ct_gt_dir_H = osp.join(resdir, "CT", "H", "ct_gt")
    ct_pred_dir_W = osp.join(resdir, "CT", "W", "ct_pred")
    ct_gt_dir_W = osp.join(resdir, "CT", "W", "ct_gt")
    ct_pred_dir_L = osp.join(resdir, "CT", "L", "ct_pred")
    ct_gt_dir_L = osp.join(resdir, "CT", "L", "ct_gt")

    # os.makedirs(eval_save_dir, exist_ok=True)

    H, W, L = image_pred.shape
    print(image_pred.shape)

    os.makedirs(proj_pred_dir, exist_ok=True)
    os.makedirs(proj_gt_dir, exist_ok=True)
    os.makedirs(ct_pred_dir_H, exist_ok=True)
    os.makedirs(ct_gt_dir_H, exist_ok=True)
    os.makedirs(ct_pred_dir_W, exist_ok=True)
    os.makedirs(ct_gt_dir_W, exist_ok=True)
    os.makedirs(ct_pred_dir_L, exist_ok=True)
    os.makedirs(ct_gt_dir_L, exist_ok=True)

    for i in tqdm(range(N)):
        iio.imwrite(osp.join(proj_pred_dir, f"proj_pred_{str(i)}.png"), ((1-cast_to_image(projs_pred[i]))*255).astype(np.uint8))
        iio.imwrite(osp.join(proj_gt_dir, f"proj_gt_{str(i)}.png"), ((1-cast_to_image(projs[i]))*255).astype(np.uint8))
    
    for i in tqdm(range(H)):
        iio.imwrite(osp.join(ct_pred_dir_H, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[i,...])*255).astype(np.uint8))
        iio.imwrite(osp.join(ct_gt_dir_H, f"ct_gt_{str(i)}.png"), (cast_to_image(image[i,...])*255).astype(np.uint8))

    for i in tqdm(range(W)):
        iio.imwrite(osp.join(ct_pred_dir_W, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[:,i,:])*255).astype(np.uint8))
        iio.imwrite(osp.join(ct_gt_dir_W, f"ct_gt_{str(i)}.png"), (cast_to_image(image[:,i,:])*255).astype(np.uint8))

    for i in tqdm(range(L)):
        iio.imwrite(osp.join(ct_pred_dir_L, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[...,i])*255).astype(np.uint8))
        iio.imwrite(osp.join(ct_gt_dir_L, f"ct_gt_{str(i)}.png"), (cast_to_image(image[...,i])*255).astype(np.uint8))
    
    print(loss)

    return




cfg = load_config(args.config)

device = torch.device("cuda")


# 先读数据
eval_dset = Dataset(cfg["exp"]["datadir"], cfg["train"]["n_rays"], "val", device) if cfg["log"]["i_eval"] > 0 else None
voxels = eval_dset.voxels if cfg["log"]["i_eval"] > 0 else None

# 根据cfg文件来进行 model 的实例化
network = get_network(cfg["network"]["net_type"])
cfg["network"].pop("net_type", None)
encoder = get_encoder(**cfg["encoder"])
model = network(encoder, **cfg["network"]).to(device)
model_fine = None
n_fine = cfg["render"]["n_fine"]

if n_fine > 0:
    model_fine = network(encoder, **cfg["network"]).to(device)

ckpt = torch.load(args.weights)
print(ckpt["epoch"])
model.load_state_dict(ckpt["network"])

if n_fine > 0:
    # stx()
    model_fine.load_state_dict(ckpt["network_fine"])

# 对 model 进行inference
model.eval()

if n_fine > 0:
    model_fine.eval()

with torch.no_grad():
    eval_step(eval_dset, model, model_fine, cfg)