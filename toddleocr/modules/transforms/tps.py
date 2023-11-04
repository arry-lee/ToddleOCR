"""
This code is refer from:
https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/modules/transformation.py
"""


import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from toddleocr.ops import ConvBNLayer


class LocalizationNetwork(nn.Module):
    def __init__(self, in_channels, num_fiducial, loc_lr, model_name):
        super().__init__()
        self.F = num_fiducial
        F = num_fiducial
        if model_name == "large":
            num_filters_list = [64, 128, 256, 512]
            fc_dim = 256
        else:
            num_filters_list = [16, 32, 64, 128]
            fc_dim = 64

        self.block_list = []
        for fno in range(0, len(num_filters_list)):
            num_filters = num_filters_list[fno]
            name = "loc_conv%d" % fno
            conv = self.add_module(
                name,
                ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=3,
                    act="relu",
                    name=name,
                ),
            )
            self.block_list.append(conv)
            if fno == len(num_filters_list) - 1:
                pool = nn.AdaptiveAvgPool2d(1)
            else:
                pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            in_channels = num_filters
            self.block_list.append(pool)
        name = "loc_fc1"
        stdv = 1.0 / math.sqrt(num_filters_list[-1] * 1.0)
        self.fc1 = nn.Linear(in_channels, fc_dim)

        # Init fc2 in LocalizationNetwork
        initial_bias = self.get_initial_fiducials()
        initial_bias = initial_bias.reshape(-1)
        name = "loc_fc2"

        self.fc2 = nn.Linear(fc_dim, F * 2, bias=True)
        self.out_channels = F * 2

    def forward(self, x):
        """
        Estimating parameters of geometric transformation
        Args:
            image: input
        Return:
            batch_C_prime: the matrix of the geometric transformation
        """
        B = x.shape[0]
        i = 0
        for block in self.block_list:
            x = block(x)
        x = x.squeeze(axis=2).squeeze(axis=2)
        x = self.fc1(x)

        x = F.relu(x)
        x = self.fc2(x)
        x = x.reshape(shape=[-1, self.F, 2])
        return x

    def get_initial_fiducials(self):
        """see RARE paper Fig. 6 (a)"""
        F = self.F
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return initial_bias


class GridGenerator(nn.Module):
    def __init__(self, in_channels, num_fiducial):
        super().__init__()
        self.eps = 1e-6
        self.F = num_fiducial

        name = "ex_fc"
        # initializer = nn.init.Constant(value=0.0)

        self.fc = nn.Linear(in_channels, 6, bias=True)

    def forward(self, batch_C_prime, I_r_size):
        """
        Generate the grid for the grid_sampler.
        Args:
            batch_C_prime: the matrix of the geometric transformation
            I_r_size: the shape of the input image
        Return:
            batch_P_prime: the grid for the grid_sampler
        """
        C = self.build_C_paddle()
        P = self.build_P_paddle(I_r_size)

        inv_delta_C_tensor = self.build_inv_delta_C_paddle(C).astype("float32")
        P_hat_tensor = self.build_P_hat_paddle(C, torch.Tensor(P)).astype("float32")

        inv_delta_C_tensor.stop_gradient = True
        P_hat_tensor.stop_gradient = True

        batch_C_ex_part_tensor = self.get_expand_tensor(batch_C_prime)

        batch_C_ex_part_tensor.stop_gradient = True

        batch_C_prime_with_zeros = torch.concat(
            [batch_C_prime, batch_C_ex_part_tensor], dim=1
        )
        batch_T = torch.matmul(inv_delta_C_tensor, batch_C_prime_with_zeros)
        batch_P_prime = torch.matmul(P_hat_tensor, batch_T)
        return batch_P_prime

    def build_C_paddle(self):
        """Return coordinates of fiducial points in I_r; C"""
        F = self.F
        ctrl_pts_x = torch.linspace(-1.0, 1.0, int(F / 2), dtype=torch.float64)
        ctrl_pts_y_top = -1 * torch.ones([int(F / 2)], dtype=torch.float64)
        ctrl_pts_y_bottom = torch.ones([int(F / 2)], dtype=torch.float64)
        ctrl_pts_top = torch.stack([ctrl_pts_x, ctrl_pts_y_top], dim=1)
        ctrl_pts_bottom = torch.stack([ctrl_pts_x, ctrl_pts_y_bottom], dim=1)
        C = torch.concat([ctrl_pts_top, ctrl_pts_bottom], dim=0)
        return C  # F x 2

    def build_P_paddle(self, I_r_size):
        I_r_height, I_r_width = I_r_size
        I_r_grid_x = (
            torch.arange(-I_r_width, I_r_width, 2, dtype=torch.float64) + 1.0
        ) / torch.Tensor(np.array([I_r_width]))

        I_r_grid_y = (
            torch.arange(-I_r_height, I_r_height, 2, dtype=torch.float64) + 1.0
        ) / torch.Tensor(np.array([I_r_height]))

        # P: self.I_r_width x self.I_r_height x 2
        P = torch.stack(torch.meshgrid(I_r_grid_x, I_r_grid_y), dim=2)
        P = P.permute(1, 0, 2)
        # n (= self.I_r_width x self.I_r_height) x 2
        return P.reshape([-1, 2])

    def build_inv_delta_C_paddle(self, C):
        """Return inv_delta_C which is needed to calculate T"""
        F = self.F
        hat_eye = torch.eye(F, dtype=torch.float64)  # F x F
        hat_C = torch.norm(C.reshape([1, F, 2]) - C.reshape([F, 1, 2]), dim=2) + hat_eye
        hat_C = (hat_C**2) * torch.log(hat_C)
        delta_C = torch.concat(  # F+3 x F+3
            [
                torch.concat(
                    [torch.ones((F, 1), dtype=torch.float64), C, hat_C], dim=1
                ),  # F x F+3
                torch.concat(
                    [torch.zeros((2, 3), dtype=torch.float64), C.permute(1, 0)], dim=1
                ),  # 2 x F+3
                torch.concat(
                    [
                        torch.zeros((1, 3), dtype="float64"),
                        torch.ones((1, F), dtype=torch.float64),
                    ],
                    dim=1,
                ),  # 1 x F+3
            ],
            axis=0,
        )
        inv_delta_C = torch.inverse(delta_C)
        return inv_delta_C  # F+3 x F+3

    def build_P_hat_paddle(self, C, P):
        F = self.F
        eps = self.eps
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        # P_tile: n x 2 -> n x 1 x 2 -> n x F x 2
        P_tile = torch.tile(torch.unsqueeze(P, dim=1), (1, F, 1))
        C_tile = torch.unsqueeze(C, dim=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        # rbf_norm: n x F
        rbf_norm = torch.norm(P_diff, p=2, dim=2, keepdim=False)

        # rbf: n x F
        rbf = torch.multiply(torch.square(rbf_norm), torch.log(rbf_norm + eps))
        P_hat = torch.concat([torch.ones((n, 1), dtype=torch.float64), P, rbf], dim=1)
        return P_hat  # n x F+3

    def get_expand_tensor(self, batch_C_prime):
        B, H, C = batch_C_prime.shape
        batch_C_prime = batch_C_prime.reshape([B, H * C])
        batch_C_ex_part_tensor = self.fc(batch_C_prime)
        batch_C_ex_part_tensor = batch_C_ex_part_tensor.reshape([-1, 3, 2])
        return batch_C_ex_part_tensor


class TPS(nn.Module):
    def __init__(self, in_channels, num_fiducial, loc_lr, model_name):
        super().__init__()
        self.loc_net = LocalizationNetwork(
            in_channels, num_fiducial, loc_lr, model_name
        )
        self.grid_generator = GridGenerator(self.loc_net.out_channels, num_fiducial)
        self.out_channels = in_channels

    def forward(self, image):
        image.stop_gradient = False
        batch_C_prime = self.loc_net(image)
        batch_P_prime = self.grid_generator(batch_C_prime, image.shape[2:])
        batch_P_prime = batch_P_prime.reshape([-1, image.shape[2], image.shape[3], 2])
        batch_I_r = F.grid_sample(image, grid=batch_P_prime)
        return batch_I_r
