""" main.py
"""

import numpy as np
import torch

from torch.autograd import Variable
from torch.optim import Adam
from utils import get_loader, make_summary, make_path
from config import get_config
from model import NetG, NetD

def train(config):
    """ train
    """
    # data_loader
    data_loader = get_loader(config)
    loader = iter(data_loader)

    # network
    net_g = NetG(config)
    net_d = NetD(config)

    # tensor
    z_g = torch.FloatTensor(config.batch_size, config.z_num)
    z_f = torch.FloatTensor(config.batch_size, config.z_num)
    real = torch.FloatTensor(config.batch_size, 3, config.image_size, config.image_size)

    # criterion
    loss = torch.nn.BCELoss()

    # variable
    if config.use_cuda:
        z_g = z_g.cuda()
        z_f = z_f.cuda()
        real = real.cuda()
        net_d = net_d.cuda()
        net_g = net_g.cuda()
        loss = loss.cuda()
    z_g = Variable(z_g)
    z_f = Variable(z_f, volatile=True)
    z_f.data.normal_(0.0, 1.0)
    real = Variable(real)

    # optimizer
    betas = (config.beta1, config.beta2)
    opt_g = Adam(net_g.parameters(), lr=config.lr, betas=betas, weight_decay=0.00001)
    opt_d = Adam(net_d.parameters(), lr=config.lr, betas=betas, weight_decay=0.00001)


    for step in range(config.start_step, config.final_step):
        try:
            image = next(loader)[0]
        except:
            loader = iter(data_loader)
            image = next(loader)
        # update net d
        net_d.zero_grad()
        real.data.resize(image.size()).copy_(image)
        z_d.data.normal_(0.0, 1.0)
        z_g.data.normal_(0.0, 1.0)

        fake = net_g(z_g)
        fake_d = net_d(fake.detach())
        real_d = net_d(real)
        cost_d = loss(real_d, Variable(config.batch_size).one_()) + loss(fake_d, Variable(config.batch_size).zero_())
        cost_d.backward()
        cost_d.step()

        # update net g
        net_g.zero_grad()
        net_d.zero_grad()
        fake_g = net_d(fake)
        cost_g = loss(fake_d, Variable(config.batch_size).one_())
        cost_g.backward()
        cost_g.step()


        if step % config.save_step == 0:
            save_model(net_g, "{0}/G_{1}.pth".format(config.model_path, step))
            save_model(net_d, "{0}/D_{1}.pth".format(config.model_path, step))

        if step % config.log_step == 0:

            if config.use_tensorboard:
                info = {
                    'loss/loss_d': cost_d.data[0],
                    'loss/loss_g': cost_g.data[0],
                    'misc/lr': config.lr,
                }
                for k, v in info.items():
                    make_summary(summary_writer, k, v, step)

                fake_f = net_g(z_f)
                utils.make_summary(summary_writer, "fake_fixed", fake_f.data.cpu().numpy(), step)

def test(config):
    pass

if __name__ == '__main__':
    config, _ = get_config()

    make_path(config)

    if config.is_train:
        train(config)
    else:
        test(config)
