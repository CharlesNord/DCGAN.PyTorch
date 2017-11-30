""" layers.py
"""
import torch
import torch.nn as nn

class Upsample(nn.Module):
    """ Upsample """
    def __init__(self, inplanes, planes, mode='deconv'):
        super(Upsample, self).__init__()
        self.mode = mode
        if mode == 'deconv':
            self.main = nn.Sequential(
                nn.ConvTranspose2d(inplanes, planes, 4, 2, 1)
            )
        elif mode == 'nearest':
            self.main = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(inplanes, planes, 3, 1, 1),
            )
        elif mode == 'blinear':
            self.main = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='blinear'),
                nn.Conv2d(inplanes, planes, 3, 1, 1),
            )
        elif mode == 'mixed':
            self.main = nn.ModuleList([
                nn.Upsample(scale_factor=2, mode='nearset'),
                nn.Upsample(scale_factor=2, mode='blinear'),
                nn.ConvTranspose2d(inplanes, inplanes, 4, 2, 1)
            ])
            self.merge = nn.Conv2d(inplanes*3, planes, 3, 1, 1)

    def forward(self, x):
        if self.mode == 'mixed':
            out = [i(x) for i in self.main]
            out = torch.cat(out, dim=1)
            out = self.merge(out)
        else:
            out = self.main(x)
        return out

class Norm2d(nn.Module):
    """ Norm2d """
    def __init__(self, inplanes, mode='batch'):
        super(Norm2d, self).__init__()
        if mode == 'batch':
            self.main = nn.BatchNorm2d(inplanes)
        elif mode == 'instance':
            self.main = nn.InstanceNorm2d(inplanes)
        
    def forward(self, x):
        out = self.main(x)
        return out