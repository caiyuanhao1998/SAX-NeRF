import torch
import pickle
import os
import sys
import numpy as np

from torch.utils.data import DataLoader, Dataset
from pdb import set_trace as stx


# 这里的各项参数代表的物理含义可以在哪查到呢？
class ConeGeometry(object):
    """
    Cone beam CT geometry. Note that we convert to meter from millimeter. 1 m = 1000 mm
    """
    def __init__(self, data):

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data["DSD"]/1000 # Distance Source to Detector      (m) x射线发射源到x射线接收器之间的距离
        self.DSO = data["DSO"]/1000  # Distance Source Origin        (m) 发射源到起点之间的距离

        # Detector parameters
        self.nDetector = np.array(data["nDetector"])  # number of pixels              (px)
        self.dDetector = np.array(data["dDetector"])/1000  # size of each pixel            (m)
        self.sDetector = self.nDetector * self.dDetector  # total size of the detector    (m)
        
        # Image parameters
        self.nVoxel = np.array(data["nVoxel"])  # number of voxels              (vx)
        self.dVoxel = np.array(data["dVoxel"])/1000  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # Offsets
        self.offOrigin = np.array(data["offOrigin"])/1000  # Offset of image from origin   (m)
        self.offDetector = np.array(data["offDetector"])/1000  # Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data["accuracy"]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data["mode"]  # parallel, cone                ...
        self.filter = data["filter"]


# dataloader，把数据做成 TIGRE 数据类型
class TIGREDataset(Dataset):
    """
    TIGRE dataset.
    """
    def __init__(self, path, n_rays=1024, type="train", device="cuda"):    
        super().__init__()

        with open(path, "rb") as handle:
            data = pickle.load(handle)
            # stx()
        
        self.geo = ConeGeometry(data) # 把数据处理成ConeGeometry
        self.type = type
        self.n_rays = n_rays
        self.near, self.far = self.get_near_far(self.geo)

        if type == "train":
            self.projs = torch.tensor(data["train"]["projections"], dtype=torch.float32, device=device) # [50, 256, 256]
            angles = data["train"]["angles"]                    # [50]
            rays = self.get_rays(angles, self.geo, device)      # [50, 256, 256, 6] 在每一个角度下获取射线的原点和方向
            # stx()
            '''
                self.rays的变化: [50, 256, 256, 6] -> [50, 256, 256, 8]
                self.near 和 self.far 分别是两个数值 [1]
                给 rays concate 了一个近平面 self.near * ones([50, 256, 256, 1]) 和一个远平面 self.far * ones([50, 256, 256, 1])
            '''
            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1) 
            self.n_samples = data["numTrain"]
            # (256, 256, 2)
            coords = torch.stack(torch.meshgrid(torch.linspace(0, self.geo.nDetector[1] - 1, self.geo.nDetector[1], device=device),
                                                torch.linspace(0, self.geo.nDetector[0] - 1, self.geo.nDetector[0], device=device), indexing="ij"),
                                 -1)
            # (256, 256, 2) -> (256 x 256, 2)
            self.coords = torch.reshape(coords, [-1, 2])
            # 也有 raw，但是和 rays 并不对应
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)                 # [128, 128, 128]
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)    # [128, 128, 128, 3]
        elif type == "val":
            self.projs = torch.tensor(data["val"]["projections"], dtype=torch.float32, device=device)
            angles = data["val"]["angles"]
            rays = self.get_rays(angles, self.geo, device)
            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1)
            self.n_samples = data["numVal"]
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)
            # stx()
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.type == "train":
            # stx()
            '''
                index = 0 时, self.projs[0]: [256, 256], 
                (self.projs[index]>0) 变成一个布尔值矩阵 [256, 256], 然后是 projs_valid [65536]
                self.projs 是 data 里面的投影数据
                如果不flatten, 直接用projs_valid 来取rays和projs会怎么样呢？
            '''
            projs_valid = (self.projs[index]>0).flatten()   
            coords_valid = self.coords[projs_valid] # [65536, 2] -> [40653, 2], 将布尔值矩阵当做索引，可能是因为并不是所有的
            select_inds = np.random.choice(coords_valid.shape[0], size=[self.n_rays], replace=False) # 从 0 ~ 40653-1 中选取 1024 个值
            select_coords = coords_valid[select_inds].long()                    # 根据选取的索引值来取坐标
            rays = self.rays[index, select_coords[:, 0], select_coords[:, 1]]   # self.rays: [50, 256, 256, 6], index 决定了取哪一个角度或样例，后两项决定了横纵坐标
            projs = self.projs[index, select_coords[:, 0], select_coords[:, 1]] # 
            out = {
                "projs":projs,
                "rays":rays,
            }
        elif self.type == "val":
            rays = self.rays[index]
            projs = self.projs[index]
            out = {
                "projs":projs,
                "rays":rays,
            }
        return out

    # 此处的 geo: ConeGeometry 表示什么？圆锥形几何
    # 冒号是类型建议符，告诉程序员希望传入的实参的类型
    def get_voxels(self, geo: ConeGeometry):
        """
        Get the voxels.
        """
        n1, n2, n3 = geo.nVoxel
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2  #这个参数是什么意思？

        '''
        # Image parameters
        self.nVoxel = np.array(data["nVoxel"])  # number of voxels              (vx)
        self.dVoxel = np.array(data["dVoxel"])/1000  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)
        '''
        # xyz 是一个多层嵌套list，外部是3, 256, 256, 178 
        xyz = np.meshgrid(np.linspace(-s1, s1, n1),
                        np.linspace(-s2, s2, n2),
                        np.linspace(-s3, s3, n3), indexing="ij")
        # voxel 就是一个 numpy.ndarray, 形状为 (256, 256, 178, 3)
        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])
        # stx()
        return voxel
    
    # 从哪可以看
    # H, W 信息融合在了 ConeGeometry 内部
    '''
        一般而言, 由一张RGB图片找到渲染其的一簇rays的流程是
        (1) 像素坐标系转相机坐标系
        (2) 相机坐标系转世界坐标系，分别确定射线的源点(rays_o)和方向(rays_d)
    '''
    def get_rays(self, angles, geo: ConeGeometry, device):
        """
        Get rays given one angle and x-ray machine geometry.
        """

        W, H = geo.nDetector
        DSD = geo.DSD
        rays = []
        
        for angle in angles:
            pose = torch.Tensor(self.angle2pose(geo.DSO, angle)).to(device)
            rays_o, rays_d = None, None
            if geo.mode == "cone":
                # 构造像素坐标系, 创建grid
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                    torch.linspace(0, H - 1, H, device=device), indexing="ij")  # pytorch"s meshgrid has indexing="ij"
                # 以中心点为圆心的像素坐标系
                '''
                    geo.dDetector    ——  size of each pixel
                    geo.offDetector  ——  Offset of Detector from origin, 一般等于 DSD - DSO
                    一般会以 origin 作为圆心建立xy轴, 然后垂直纸面朝外为z轴
                '''
                # 把像素坐标转换成对origin的横纵偏移
                '''
                    如果xy和uv对齐, 那么下面这两条式子就可以说通了
                    看着像是像素坐标系转成相机坐标系
                '''
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]

                # 为啥要除以DSD呢？不清楚，xy方向确定了，z方向为何stack个1呢？
                # 除以normalize是为了normalize吗？
                dirs = torch.stack([uu / DSD, vv / DSD, torch.ones_like(uu)], -1) # 由像素上的一些坐标确定射线簇的方向

                # source to origin 或者 origin to source
                rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] * 
                rays_o = pose[:3, -1].expand(rays_d.shape) #相机偏移量为射线的源点

            elif geo.mode == "parallel":
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                        torch.linspace(0, H - 1, H, device=device), indexing="ij")  # pytorch"s meshgrid has indexing="ij"
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)], -1) # 与cone geometry的区别在于
                rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] * 
                rays_o = torch.sum(torch.matmul(pose[:3,:3], torch.stack([uu,vv,torch.zeros_like(uu)],-1)[..., None]).to(device), -1) + pose[:3, -1].expand(rays_d.shape)

                # import open3d as o3d
                # from src.util.draw_util import plot_rays, plot_cube, plot_camera_pose
                # cube1 = plot_cube(np.zeros((3,1)), geo.sVoxel[...,np.newaxis])
                # cube2 = plot_cube(np.zeros((3,1)), np.ones((3,1))*geo.DSO*2)
                # rays1 = plot_rays(rays_d.cpu().detach().numpy(), rays_o.cpu().detach().numpy(), 2)
                # poseray = plot_camera_pose(pose.cpu().detach().numpy())
                # o3d.visualization.draw_geometries([cube1, cube2, rays1, poseray])
            
            else:
                raise NotImplementedError("Unknown CT scanner type!")
            rays.append(torch.concat([rays_o, rays_d], dim=-1))

        return torch.stack(rays, dim=0)

    # world to camera transer ？
    # 相机外参矩阵
    # 绕x轴旋转90度，绕z轴旋转90度后再旋转angle
    # 前两次旋转为对齐
    # 此处为 source_to_origin 的pose
    def angle2pose(self, DSO, angle):

        # 绕x轴逆时针转了-90度
        phi1 = -np.pi / 2
        R1 = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(phi1), -np.sin(phi1)],
                    [0.0, np.sin(phi1), np.cos(phi1)]])

        # 绕z轴，逆时针转了90度
        phi2 = np.pi / 2
        R2 = np.array([[np.cos(phi2), -np.sin(phi2), 0.0],
                    [np.sin(phi2), np.cos(phi2), 0.0],
                    [0.0, 0.0, 1.0]])
        
        # 绕z轴，逆时针转了angle
        R3 = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0]])
        rot = np.dot(np.dot(R3, R2), R1)

        # source 对应的偏移而非相机 (detector) 对应的偏移
        # 不沿着z轴运动，所以z轴偏移量为0
        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0]) # DSO 投影
        T = np.eye(4)
        T[:-1, :-1] = rot
        T[:-1, -1] = trans  # 偏移量参数
        return T

    # 根据 CT 的锥形几何计算近端和远端
    # near 和 far 是由什么决定的呢？很奇怪
    # tolerance是允许的误差范围
    '''
        Numpy中的 linalg 模块包含线性代数中的函数方法，用于求解矩阵的逆矩阵、求特征值、解线性方程组以及求行列式等
    '''
    def get_near_far(self, geo: ConeGeometry, tolerance=0.005):
        """
            Compute the near and far threshold.
            self.offOrigin = np.array(data["offOrigin"])/1000       #   Offset of image from origin
            self.sVoxel = self.nVoxel * self.dVoxel                 #   
        """
        dist1 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist2 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        dist3 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist4 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        dist_max = np.max([dist1, dist2, dist3, dist4])
        near = np.max([0, geo.DSO - dist_max - tolerance])
        far = np.min([geo.DSO * 2, geo.DSO + dist_max + tolerance])
        return near, far
