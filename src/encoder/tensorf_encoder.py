import torch
import torch.nn as nn
import torch.nn.functional as F


class TensorfEncoder(torch.nn.Module):
    def __init__(self, num_levels, density_n_comp=8, app_dim=32, device='cpu', **kwargs):
        super().__init__()
        
        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.density_n_comp = density_n_comp
        self.output_dim = app_dim
        self.app_dim = app_dim
        
        self.init_svd_volume(num_levels, device)

    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line = self.init_one_svd([self.density_n_comp] * 3, [res] * 3, 0.1, device)
        self.basis_mat = torch.nn.Linear(self.density_n_comp * 3, self.app_dim, bias=False).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
    
    def forward(self, inputs, size=1):
        assert not (inputs.min().item() < -size or inputs.max().item() > size)
        inputs = inputs / size   # map to [-1, 1]
        outputs = self.compute_densityfeature(inputs)
        return outputs

    def compute_densityfeature(self, xyz_sampled):
        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        
        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point.append(F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.density_plane[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

        return self.basis_mat((plane_coef_point * line_coef_point).T)