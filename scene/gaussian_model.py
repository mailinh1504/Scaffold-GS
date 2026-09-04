#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding

# ============================================================================================
# Net version
class FiLMNet(nn.Module):
    """
    Simple FiLM-style network: processes an anchor feature vector and modulates
    intermediate activations with scale/shift parameters computed from a
    conditioning vector (e.g. relative viewing distance + direction).

    forward(feature, cond, extra=None) -> output
    - feature: (B, feat_dim)
    - cond: (B, cond_dim)
    - extra: optional tensor concatenated before final layer (e.g. appearance)
    """
    def __init__(self, X_dim, Y_dim, out_dim, hidden_dim=None, activation=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = X_dim

        self.X_norm = nn.LayerNorm(X_dim)

        self.X_embed = nn.Sequential(
            nn.Linear(X_dim, hidden_dim*2),
            nn.ReLU(True),
            nn.Linear(hidden_dim*2, hidden_dim),
        )

        self.Y_embed = nn.Sequential(
            nn.Linear(Y_dim, hidden_dim*2),
            nn.ReLU(True),
            nn.Linear(hidden_dim*2, hidden_dim*2),
        )
        self.cond_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.last_act = activation

    def forward(self, X, Y):
        # X_embed = Linear(ReLU(Linear(X)))
        # gamma, beta = Linear(ReLU(Linear(Y))).chunk(2)
        # FXY = gamma * X_embed + beta
        # out = Linear(FXY)

        # feature: (B, feat_dim)
        # cond: (B, cond_dim)
        X_embed = self.X_embed(self.X_norm(X))
        # produce gamma and beta for FiLM
        gamma, beta = self.Y_embed(Y).chunk(2, dim=-1)
        FXY = gamma * X_embed + beta
        out = self.out(self.cond_norm(FXY))
        if self.last_act is not None:
            out = self.last_act(out)
        return out

# FPS: 37.18651
# SSIM :    0.6103338
# PSNR :   18.5071049
# LPIPS:    0.3529382

class LiteFiLMNet(nn.Module):
    def __init__(self, X_dim, Y_dim, out_dim, hidden_dim=None, activation = None):
        super().__init__()
        # 1. Embed X
        if hidden_dim is None:
            hidden_dim = X_dim

        self.X_embed = nn.Sequential(
            nn.Linear(X_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.Y_embed = nn.Linear(Y_dim, hidden_dim * 2)

        self.out_net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, out_dim)
        )
        self.last_act = activation

    def forward(self, X, Y):
        X_embed = self.X_embed(X)

        # Generate gamma and beta
        gamma_beta = self.Y_embed(Y)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        # Apply FiLM
        FXY = (gamma * X_embed) + beta
        out = self.out_net(FXY)
        if self.last_act is not None:
            out = self.last_act(out)

        return out

class GatedMLP(nn.Module):
    def __init__(self, X_dim, Y_dim, out_dim, hidden_dim=None, activation=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = X_dim
        self.X_embed = nn.Sequential(
            nn.Linear(X_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Y generates a "gate" of the same size as X_embed
        self.Y_gate = nn.Linear(Y_dim, hidden_dim, bias=False)

        self.out_net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, out_dim)
        )

        self.last_act = activation

    def forward(self, X, Y):
        X_embed = self.X_embed(X)

        # Use sigmoid to make the gate values 0-1
        gate = torch.sigmoid(self.Y_gate(Y))

        # Apply gate
        FXY = X_embed * gate
        out = self.out_net(FXY)
        if self.last_act is not None:
            out = self.last_act(out)

        return out

class AdditiveNet(nn.Module):
    def __init__(self, X_dim, Y_dim, out_dim, hidden_dim=None, activation=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = X_dim
        # Both must embed to the same dimension
        self.X_embed = nn.Linear(X_dim, hidden_dim)
        self.Y_embed = nn.Linear(Y_dim, hidden_dim)

        self.out_net = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(hidden_dim, out_dim)
        )
        self.last_act = activation

    def forward(self, X, Y):
        X_embed = self.X_embed(X)
        Y_embed = self.Y_embed(Y)

        # Add them together
        FXY = X_embed + Y_embed
        out = self.out_net(FXY)
        if self.last_act is not None:
            out = self.last_act(out)

        return out

class MultiplicativeNet(nn.Module):
    def __init__(self, X_dim, Y_dim, out_dim, hidden_dim=None, activation=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = X_dim
        # Both must embed to the same dimension
        self.X_embed = nn.Linear(X_dim, hidden_dim)
        self.Y_embed = nn.Linear(Y_dim, hidden_dim)

        self.out_net = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(hidden_dim, out_dim)
        )
        self.last_act = activation

    def forward(self, X, Y):
        X_embed = self.X_embed(X)
        Y_embed = self.Y_embed(Y)

        # Element-wise multiplication
        FXY = X_embed * Y_embed
        out = self.out_net(FXY)
        if self.last_act is not None:
            out = self.last_act(out)

        return out

class SymmetricalMLP(nn.Module):
    def __init__(self, X_dim, Y_dim, out_dim, hidden_dim=None, activation=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = X_dim
        # Separate embedding layers
        self.X_embed = nn.Linear(X_dim, hidden_dim)

        y_hidden = Y_dim*4

        self.Y_embed = nn.Linear(Y_dim, y_hidden)

        self.out_net = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(hidden_dim + y_hidden, out_dim)
        )
        self.last_act = activation

    def forward(self, X, Y):
        X_embed = self.X_embed(X)
        Y_embed = self.Y_embed(Y)

        # Concatenate the *embeddings*
        FXY = torch.cat([X_embed, Y_embed], dim=-1)
        out = self.out_net(FXY)
        if self.last_act is not None:
            out = self.last_act(out)
        return out

class GRuBased(nn.Module):
    def __init__(self, feat_dim_1, feat_dim_2, out_dim, hidden_dim=None, activation=None):
        super(GRuBased, self).__init__()

        feat_dim = feat_dim_1 + feat_dim_2
        if hidden_dim is None:
            hidden_dim = feat_dim

        out_dim = out_dim // 5

        self.first_encoder = nn.Linear(feat_dim, hidden_dim).cuda()

        self.gru = nn.GRUCell(input_size=out_dim, hidden_size=hidden_dim).cuda()

        self.fc = nn.Linear(hidden_dim, out_dim).cuda()

        self.last_act = activation


    def forward(self, X, Y, k=5):
        batch_size = X.shape[0]

        X = torch.cat([X, Y], dim=1)

        h_t = self.first_encoder(X)

        o_t = torch.zeros(batch_size, 1).cuda()

        out = []

        for i in range(k):
            h_t = self.gru(o_t, h_t)

            o_t = self.fc(h_t)
            out.append(o_t)

            # o_t = o_t_.detach()


        out = torch.cat(out, dim=1)

        if self.last_act is not None:
            out = self.last_act(out)

        return out

Used_net = FiLMNet

# ============================================================================================


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self,
                 feat_dim: int=32,
                 n_offsets: int=5,
                 voxel_size: float=0.01,
                 update_depth: int=3,
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 use_film_net : bool = False,
                 ):

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist
        self.use_film_net = use_film_net

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)

        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.offset_gradient_accum = torch.empty(0)
        self.opacity_gradient_accum = torch.empty(0)
        self.opacity_gradient_denom = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()
        if self.use_film_net:
            self.mlp_opacity = Used_net(feat_dim, 3+self.opacity_dist_dim, n_offsets, activation=nn.Tanh()).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()
        if self.use_film_net:
            self.mlp_cov = Used_net(feat_dim, 3+self.cov_dist_dim, 7*self.n_offsets).cuda()

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()
        if self.use_film_net:
            self.mlp_color = Used_net(feat_dim+self.appearance_dim, 3+self.color_dist_dim, 3*self.n_offsets, activation=nn.Sigmoid()).cuda()



    def eval(self) -> None:
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._anchor,
        self._offset,
        self._local,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        return self._anchor

    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size

        return data

    def _voxelize_sample(self, data=None, voxel_size=0.01, avg_max_capacity = 10, decay_rate = 0.5):
        np.random.shuffle(data)

        grid_coords = np.round(data / voxel_size)

        unique_coords, counts = np.unique(grid_coords, axis=0, return_counts=True)

        if counts.mean() > avg_max_capacity:

            while counts.mean() > avg_max_capacity:
                voxel_size *= decay_rate
                grid_coords = np.round(data / voxel_size)
                unique_coords, counts = np.unique(grid_coords, axis=0, return_counts=True)

        # update voxel size
        self.voxel_size = voxel_size

        return unique_coords * voxel_size

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')


        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.reset_densification_stats()



        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr

    # FiLM-MP: reset buffers at phase boundaries so old statistics do not leak.
    def reset_densification_stats(self):
        anchor_count = self.get_anchor.shape[0]
        offset_count = anchor_count * self.n_offsets
        self.opacity_accum = torch.zeros((anchor_count, 1), device="cuda")
        self.offset_gradient_accum = torch.zeros((offset_count, 1), device="cuda")
        self.opacity_gradient_accum = torch.zeros((offset_count, 1), device="cuda")
        self.opacity_gradient_denom = torch.zeros((offset_count, 1), device="cuda")
        self.offset_denom = torch.zeros((offset_count, 1), device="cuda")
        self.anchor_demon = torch.zeros((anchor_count, 1), device="cuda")

    # FiLM-MP: fixed-anchor refinement does not need densification buffers.
    def release_densification_stats(self):
        for name in (
            "opacity_accum",
            "offset_gradient_accum",
            "opacity_gradient_accum",
            "opacity_gradient_denom",
            "offset_denom",
            "anchor_demon",
        ):
            if hasattr(self, name):
                delattr(self, name)


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # Accumulate statistics for Scaffold-GS densification.
    def training_statis(
            self,
            viewspace_point_tensor,
            opacity,
            update_filter,
            offset_selection_mask,
            anchor_visible_mask,
            use_grad=True,
            use_opacity_grad=False):
        # Opacity values are used by the original pruning rule.
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0

        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)

        # Count how often each visible anchor has been observed.
        self.anchor_demon[anchor_visible_mask] += 1
        if not use_grad:
            return

        # Map rendered Gaussians back to their source anchor-offset pairs.
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        # Position cue: g_pos = ||dL / d mu_2D||_2.
        position_grad = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += position_grad

        # FiLM-MP fine cue: g_opa = |dL / d alpha|.
        # It is accumulated separately and used only as a confirmation signal.
        if use_opacity_grad and opacity.grad is not None:
            opacity_grad_visible = opacity.grad.detach().abs().view(-1, 1)
            opacity_grad = opacity_grad_visible[offset_selection_mask][update_filter]
            self.opacity_gradient_accum[combined_mask] += opacity_grad
            self.opacity_gradient_denom[combined_mask] += 1
        self.offset_denom[combined_mask] += 1




    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]


        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


    def anchor_growing(
            self,
            grads,
            threshold,
            offset_mask,
            grow_phase="base",
            opacity_grads=None,
            opacity_confirm_ratio=1.0):
        init_length = self.get_anchor.shape[0]*self.n_offsets
        if grow_phase == "coarse":
            levels = range(1)
        elif grow_phase == "fine":
            levels = range(1, self.update_depth)
        else:
            levels = range(self.update_depth)

        opacity_threshold = None
        if grow_phase == "fine" and opacity_grads is not None:
            valid_opacity = opacity_grads[offset_mask]
            valid_opacity = valid_opacity[valid_opacity > 0]
            if valid_opacity.numel() > 0:
                opacity_threshold = valid_opacity.mean().clamp_min(1e-12) * float(opacity_confirm_ratio)

        for level in levels:
            cur_threshold = threshold*((self.update_hierachy_factor//2)**level)
            candidate_mask = torch.logical_and(grads >= cur_threshold, offset_mask)

            # Coarse/base keep Scaffold-GS sampling; fine is confirmed by opacity grad.
            if grow_phase != "fine":
                rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(level+1))
                candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if level > 0:
                    continue
            else:
                candidate_mask = torch.cat([
                    candidate_mask,
                    torch.zeros(length_inc, dtype=torch.bool, device=candidate_mask.device),
                ], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)

            # Scaffold-GS voxel hierarchy: coarse levels use larger voxels.
            size_factor = self.update_init_factor // (self.update_hierachy_factor**level)
            cur_size = self.voxel_size*size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            if selected_xyz.shape[0] == 0:
                continue
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for chunk_idx in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates

            # Fine phase: position proposes voxels, opacity gradient confirms them.
            if grow_phase == "fine":
                if opacity_threshold is None or opacity_grads is None:
                    remove_duplicates = torch.zeros_like(remove_duplicates)
                else:
                    opacity_values = opacity_grads
                    if candidate_mask.shape[0] > opacity_values.shape[0]:
                        padding = torch.zeros(
                            candidate_mask.shape[0] - opacity_values.shape[0],
                            dtype=opacity_values.dtype,
                            device=opacity_values.device,
                        )
                        opacity_values = torch.cat([opacity_values, padding], dim=0)
                    selected_opacity = opacity_values[candidate_mask].view(-1, 1)
                    voxel_opacity = scatter_max(
                        selected_opacity,
                        inverse_indices.unsqueeze(1).expand(-1, 1),
                        dim=0,
                    )[0].squeeze(1)
                    remove_duplicates = torch.logical_and(remove_duplicates, voxel_opacity > opacity_threshold)

            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }


                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]



    def _reset_grow_stats(self, offset_mask):
        target_count = self.get_anchor.shape[0]*self.n_offsets

        self.offset_denom[offset_mask] = 0
        pad_offset_denom = torch.zeros([target_count - self.offset_denom.shape[0], 1],
                                           dtype=self.offset_denom.dtype,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, pad_offset_denom], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([target_count - self.offset_gradient_accum.shape[0], 1],
                                           dtype=self.offset_gradient_accum.dtype,
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        self.opacity_gradient_accum[offset_mask] = 0
        padding_opacity_gradient_accum = torch.zeros([target_count - self.opacity_gradient_accum.shape[0], 1],
                                           dtype=self.opacity_gradient_accum.dtype,
                                           device=self.opacity_gradient_accum.device)
        self.opacity_gradient_accum = torch.cat([self.opacity_gradient_accum, padding_opacity_gradient_accum], dim=0)

        self.opacity_gradient_denom[offset_mask] = 0
        padding_opacity_gradient_denom = torch.zeros([target_count - self.opacity_gradient_denom.shape[0], 1],
                                           dtype=self.opacity_gradient_denom.dtype,
                                           device=self.opacity_gradient_denom.device)
        self.opacity_gradient_denom = torch.cat([self.opacity_gradient_denom, padding_opacity_gradient_denom], dim=0)

    def prune_low_opacity(self, check_interval=100, success_threshold=0.8, min_opacity=0.005):
        # Original Scaffold-GS pruning: remove anchors with weak accumulated opacity.
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        opacity_gradient_accum = self.opacity_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        opacity_gradient_accum = opacity_gradient_accum.view([-1, 1])
        del self.opacity_gradient_accum
        self.opacity_gradient_accum = opacity_gradient_accum

        opacity_gradient_denom = self.opacity_gradient_denom.view([-1, self.n_offsets])[~prune_mask]
        opacity_gradient_denom = opacity_gradient_denom.view([-1, 1])
        del self.opacity_gradient_denom
        self.opacity_gradient_denom = opacity_gradient_denom

        # update opacity accum
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.any():
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def adjust_anchor(
            self,
            check_interval=100,
            success_threshold=0.8,
            grad_threshold=0.0002,
            min_opacity=0.005,
            grow_phase="base",
            opacity_confirm_ratio=1.0,
            allow_grow=True,
            allow_prune=True):
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        if allow_grow:
            pos_grads = self.offset_gradient_accum / self.offset_denom.clamp_min(1.0)
            pos_grads[pos_grads.isnan()] = 0.0
            pos_grads = torch.norm(pos_grads, dim=-1)

            opa_grads = self.opacity_gradient_accum / self.opacity_gradient_denom.clamp_min(1.0)
            opa_grads[opa_grads.isnan()] = 0.0
            opa_grads = opa_grads.squeeze(dim=-1)

            self.anchor_growing(
                pos_grads,
                grad_threshold,
                offset_mask,
                grow_phase=grow_phase,
                opacity_grads=opa_grads,
                opacity_confirm_ratio=opacity_confirm_ratio,
            )
            self._reset_grow_stats(offset_mask)

        if allow_prune:
            self.prune_low_opacity(check_interval, success_threshold, min_opacity)
        else:
            self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode = 'unite'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            if self.use_film_net:
                opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim).cuda(), torch.rand(1, 3+self.opacity_dist_dim).cuda()))
            else:
                opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda(),))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            if self.use_film_net:
                cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim).cuda(), torch.rand(1, 3+self.cov_dist_dim).cuda()))
            else:
                cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda(),))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            if self.use_film_net:
                color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+self.appearance_dim).cuda(), torch.rand(1, 3+self.color_dist_dim).cuda()))
            else:
                color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda(),))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'unite'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError
