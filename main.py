""" main.py
"""
import tensorflow as tf
import numpy as np
import torch

from torch.autograd import Variable
from torch.optim import Adam
from utils import get_loader, make_summary, make_path, save_model, save_image
from config import get_config
from model import NetG, NetD
from tqdm import tqdm

def train(config):
    """ train
    """
    torch.manual_seed(config.random_seed)
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
    real_label = torch.Tensor(config.batch_size)
    fake_label = torch.Tensor(config.batch_size)
    # criterion
    bce = torch.nn.BCELoss()

    # variable
    if config.use_cuda:
        z_g = z_g.cuda()
        z_f = z_f.cuda()
        real = real.cuda()
        net_d = net_d.cuda()
        net_g = net_g.cuda()
        bce = bce.cuda()
        real_label = real_label.cuda()
        fake_label = fake_label.cuda()
        torch.cuda.manual_seed(config.random_seed)

    z_g = Variable(z_g)
    z_f = Variable(z_f, volatile=True)
    z_f.data.normal_(0.0, 1.0)
    real = Variable(real)
    real_label = Variable(real_label)
    fake_label = Variable(fake_label)
    # optimizer
    betas = (config.beta1, config.beta2)
    opt_g = Adam(net_g.parameters(), lr=config.lr, betas=betas, weight_decay=0.00001)
    opt_d = Adam(net_d.parameters(), lr=config.lr, betas=betas, weight_decay=0.00001)

    summary_writer = tf.summary.FileWriter(config.log_path)
    epoch = 1
    for step in tqdm(range(config.start_step, config.final_step)):
        try:
            image = next(loader)[0]
        except:
            loader = iter(data_loader)
            image = next(loader)[0]
            epoch += 1

        # update net d
        net_d.zero_grad()
        net_g.zero_grad()
        real.data.resize_(image.size()).copy_(image)
        z_g.data.resize_(image.size(0),config.z_num).normal_(0.0, 1.0)

        real_label.data.resize_(image.size(0)).fill_(1)
        fake_label.data.resize_(image.size(0)).fill_(0)

        fake = net_g(z_g)
        fake_d = net_d(fake.detach())
        real_d = net_d(real)
        cost_d = bce(real_d, real_label) + bce(fake_d, fake_label)
        cost_d.backward()
        opt_d.step()

        # update net g
        fake_g = net_d(fake)
        cost_g = bce(fake_g, real_label.detach())
        cost_g.backward()
        opt_g.step()

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
                make_summary(summary_writer, "fake_fixed", fake_f.data.cpu().numpy(), step)
                make_summary(summary_writer, "fake", fake.data.cpu().numpy(), step)

            if step % config.save_step == 0:
                save_model(net_g, "{0}/G_{1}.pth".format(config.model_path, step))
                save_model(net_d, "{0}/D_{1}.pth".format(config.model_path, step))
                save_image(fake_f.data, 'images/fake_fixed_{}_{}.png'.format(epoch, step))
                save_image(fake.data, 'images/fake_{}_{}.png'.format(epoch, step))

def test(config):
    pass

if __name__ == '__main__':
    config, _ = get_config()

    make_path(config)

    if config.is_train:
        train(config)
    else:
        test(config)
