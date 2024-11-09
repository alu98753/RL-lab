import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class RNDModel(nn.Module):
    def __init__(self, input_shape, output_size):
        super(RNDModel, self).__init__()
        # 目标网络（固定参数）
        self.target = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, output_size)
        )
        # 预测网络（需要训练）
        self.predictor = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, output_size)
        )
        # 初始化目标网络参数，不参与训练
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        target_output = self.target(x)
        predictor_output = self.predictor(x)
        return predictor_output, target_output
