""" model.py
"""

import torch
import torch.nn as nn

class NetG(nn.Module):
    """ Generative Network
    """
    def __init__(self, conf):
        super(NetG, self).__init__()
        self.z = conf.z_num
        self.n = conf.n_num
        self.main = nn.Sequential(
            # 1x1
            nn.ConvTranspose2d(self.z, self.n*6, 4, 1, 0),
            nn.BatchNorm2d(self.n*6),
            nn.ReLU(True),
            # 4x4
            nn.ConvTranspose2d(self.n*6, self.n*5, 4, 2, 1),
            nn.BatchNorm2d(self.n*5),
            nn.ReLU(True),
            # 8x8
            nn.ConvTranspose2d(self.n*5, self.n*4, 4, 2, 1),
            nn.BatchNorm2d(self.n*4),
            nn.ReLU(True),
            # 16x16
            nn.ConvTranspose2d(self.n*4, self.n*3, 4, 2, 1),
            nn.BatchNorm2d(self.n*3),
            nn.ReLU(True),
            # 32x32
            nn.ConvTranspose2d(self.n*3, self.n*2, 4, 2, 1),
            nn.BatchNorm2d(self.n*2),
            nn.ReLU(True),
            # 64x64
            nn.ConvTranspose2d(self.n*2, self.n*1, 4, 2, 1),
            nn.BatchNorm2d(self.n*1),
            nn.ReLU(True),
            # 128*128
            nn.Conv2d(self.n*1, 3, 3, 1, 1),
        )

    def forward(self, x):
        x = self.main(x.view(x.size(0), -1, 1, 1))
        return x

class NetD(nn.Module):
    """ Discriminative Network
    """
    def __init__(self, config):
        super(NetD, self).__init__()
        self.main = nn.Sequential(
            # 128x128
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
            # 1x1
        )

    def forward(self, x):
        x = self.main(x).view(x.size(0), -1)
        return x
