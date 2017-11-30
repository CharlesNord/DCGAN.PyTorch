""" model.py
"""

import torch.nn as nn
from models.layers import Upsample, Norm2d

class NetG(nn.Module):
    """ Generative Network """
    def __init__(self, conf):
        super(NetG, self).__init__()
        nzf = conf.z_num
        ngf = conf.n_num
        self.main = nn.Sequential(
            # 1x1
            nn.ConvTranspose2d(nzf, ngf*16, 8, 1, 0),
            Norm2d(ngf*16, mode=conf.norm_mode),
            nn.ReLU(True),
            # 8x8
            Upsample(ngf*16, ngf*8, mode=conf.up_mode),
            Norm2d(ngf*8, mode=conf.norm_mode),
            nn.ReLU(True),
            # 16x16
            Upsample(ngf*8, ngf*4, mode=conf.up_mode),
            Norm2d(ngf*4, mode=conf.norm_mode),
            nn.ReLU(True),
            # 32x32
            Upsample(ngf*4, ngf*2, mode=conf.up_mode),
            Norm2d(ngf*2, mode=conf.norm_mode),
            nn.ReLU(True),
            # 64x64
            Upsample(ngf*2, ngf*1, mode=conf.up_mode),
            Norm2d(ngf*1, mode=conf.norm_mode),
            nn.ReLU(True),
            # 128*128
            nn.Conv2d(ngf*1, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.main(x.view(x.size(0), -1, 1, 1))
        return x

class NetD(nn.Module):
    """ Discriminative Network """
    def __init__(self, conf):
        super(NetD, self).__init__()
        ndf = conf.n_num
        self.main = nn.Sequential(
            # 128x128
            nn.Conv2d(3, ndf*1, 4, 2, 1),
            Norm2d(ndf*1, mode=conf.norm_mode),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64
            nn.Conv2d(ndf*1, ndf*2, 4, 2, 1),
            Norm2d(ndf*2, mode=conf.norm_mode),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            Norm2d(ndf*4, mode=conf.norm_mode),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1),
            Norm2d(ndf*8, mode=conf.norm_mode),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            nn.Conv2d(ndf*8, ndf*16, 4, 2, 1),
            Norm2d(ndf*16, mode=conf.norm_mode),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4
            nn.Conv2d(ndf*16, 1, 3, 1, 1),
            nn.Sigmoid()
            # 4x4
        )

    def forward(self, x):
        x = self.main(x).view(x.size(0), -1)
        return x

