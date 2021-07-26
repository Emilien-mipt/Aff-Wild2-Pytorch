import math

import numpy as np
import torch
import torch.nn as nn


class CCCLoss(nn.Module):
    """Class for Lin’s Concordance Correlation Coefficient."""

    def __init__(self, device, eps=1e-4):
        super().__init__()
        self.mean = torch.mean
        self.var = torch.var
        self.sum = torch.sum
        self.sqrt = torch.sqrt
        self.std = torch.std
        self.device = device
        self.eps = eps

    def forward(self, prediction, ground_truth):
        gt_shape = ground_truth.shape
        random_gt = self.eps * torch.rand(gt_shape, device=self.device)
        ground_truth = ground_truth + random_gt
        mean_gt = self.mean(ground_truth)
        mean_pred = self.mean(prediction)
        var_gt = self.var(ground_truth)
        var_pred = self.var(prediction)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        cor = self.sum(v_pred * v_gt) / (self.sqrt(self.sum(v_pred ** 2)) * self.sqrt(self.sum(v_gt ** 2)))
        sd_gt = self.std(ground_truth)
        sd_pred = self.std(prediction)
        numerator = 2 * cor * sd_gt * sd_pred
        denominator = var_gt + var_pred + (mean_gt - mean_pred) ** 2
        ccc = numerator / denominator
        return 1 - ccc


def ccc_score(x, y, eps=1e-4):
    y_shape = y.shape
    random_y = eps * np.random.rand(y_shape[0], y_shape[1])
    y = y + random_y
    x_m = np.mean(x)
    y_m = np.mean(y)
    vx = x - x_m
    vy = y - y_m
    rho = (np.sum(vx * vy)) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2 * rho * x_s * y_s / (x_s ** 2 + y_s ** 2 + (x_m - y_m) ** 2)
    return ccc


def mse_score(x, y):
    return (np.square(x - y)).mean()
