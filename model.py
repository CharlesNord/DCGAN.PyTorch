""" model.py
"""

import torch
import torch.nn as nn

class NetG(nn.Module):
    """ Generative Network
    """
    def __init__(self, config):
        super(NetG, self).__init__()
        self.z = config.z_num
        self.n = config.n_num

        self.fc = nn.Linear(self.z, 512 * 6 * 6)
        self.main = nn.Sequential(
            nn.BatchNorm2d(self.n*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.n*8, self.n*4, 4, 2, 1),
            nn.BatchNorm2d(self.n*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.n*4, self.n*2, 4, 2, 1),
            nn.BatchNorm2d(self.n*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.n*2, self.n, 4, 2, 1),
            nn.BatchNorm2d(self.n),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.n, 3, 4, 2, 1),
        )

    def forward(self, x):
        x = self.fc(x).view(x.size(0), 512, 6, 6)
        x = self.main(x)
        return x

class NetD(nn.Module):
    """ Discriminative Network
    """
    def __init__(self, config):
        super(NetD, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ELU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ELU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ELU(256),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ELU(256),
        )
        self.fc = nn.Sequential(
            nn.Linear(512*6*6, 2),
            nn.LogSoftmax(x)
        )

    def forward(self, x):
        x = self.main(x).view(x.size(0), -1)
        x = self.fc(x)
        return x
