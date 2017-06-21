from __future__ import print_function, division
import torch
from Modules.Config import cfg
from torch import nn
import math
from discriminator import discriminator
from generator import lr_generator, hr_generator

class CondGAN(nn.Module):
    def __init__(self, lr_imsize, hr_lr_ratio):
        super(CondGAN, self).__init__()
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.network_type = cfg.GAN.NETWORK_TYPE
        self.hr_lr_ratio = hr_lr_ratio
        self.gf_dim = cfg.GAN.GF_DIM
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.s = lr_imsize
        print('lr_imsize: ', lr_imsize)
        self.s2, self.s4, self.s8, self.s16 = \
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)
        # the input must be 16 * k, otherwise it's not tf padding
        self.lr_disc = discriminator(False, lr_imsize)
        self.hr_disc = discriminator(True, lr_imsize)
        self.lr_generator = lr_generator(lr_imsize, z_size)
        self.hr_generator = hr_generator(lr_imsize)

    def forward(self):
        pass