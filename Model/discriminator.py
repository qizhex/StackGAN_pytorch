from __future__ import print_function, division
import torch
from Modules.Config import cfg
from torch import nn
import math
from layers import custom_con2d

class d_image_encoder(nn.Module):
    def __init__(self, imsize):
        super(d_image_encoder, self).__init__()
        self.s = imsize
        kernel_0 = (4, 4)
        self.s2, self.s4, self.s8, self.s16 = \
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)

        self.node_d_0 = nn.Sequential(
            custom_con2d((self.s, self.s), 3, self.df_dim, kernel_0),
            nn.LeakyReLU(negative_slope=0.2),
            custom_con2d((self.s2, self.s2), self.df_dim, self.df_dim * 2, kernel_0),
            nn.BatchNorm2d(self.df_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            custom_con2d((self.s4, self.s4), self.df_dim * 2, self.df_dim * 4, kernel_0),
            nn.BatchNorm2d(self.df_dim * 4),
            custom_con2d((self.s8, self.s8), self.df_dim * 4, self.df_dim * 8, kernel_0),
            nn.BatchNorm2d(self.df_dim * 8),
        )
        self.node_d_1 = nn.Sequential(
            custom_con2d((self.s16, self.s16), self.df_dim * 8, self.df_dim * 2, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.df_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            custom_con2d((self.s16, self.s16), self.df_dim * 2, self.df_dim * 2, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.df_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            custom_con2d((self.s16, self.s16), self.df_dim * 2, self.df_dim * 8, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.df_dim * 8),
        )
        self.activ = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, img):
        n0 = self.node_d_0(img)
        n1 = self.node_d_1(n0)
        out = self.activ(n0 + n1)
        return out

class hr_d_image_encoder(nn.Module):
    def __init__(self, imsize): #ori image size, *4
        super(hr_d_image_encoder, self).__init__()
        self.s = imsize
        kernel_0 = (4, 4)
        self.node_d_0 = nn.Sequential(
            custom_con2d((self.s * 4, self.s * 4), 3, self.df_dim, kernel_0),
            nn.LeakyReLU(negative_slope=0.2),
            custom_con2d((self.s * 2, self.s * 2), self.df_dim, self.df_dim * 2, kernel_0),
            nn.BatchNorm2d(self.df_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            custom_con2d((self.s, self.s), self.df_dim * 2, self.df_dim * 4, kernel_0),
            nn.BatchNorm2d(self.df_dim * 4),
            nn.LeakyReLU(negative_slope=0.2),
            custom_con2d((self.s2, self.s2), self.df_dim * 4, self.df_dim * 8, kernel_0),
            nn.BatchNorm2d(self.df_dim * 8),
            nn.LeakyReLU(negative_slope=0.2),
            custom_con2d((self.s4, self.s4), self.df_dim * 8, self.df_dim * 16, kernel_0),
            nn.BatchNorm2d(self.df_dim * 16),
            nn.LeakyReLU(negative_slope=0.2),
            custom_con2d((self.s8, self.s8), self.df_dim * 16, self.df_dim * 32, kernel_0),
            nn.BatchNorm2d(self.df_dim * 32),
            nn.LeakyReLU(negative_slope=0.2),
            custom_con2d((self.s16, self.s16), self.df_dim * 32, self.df_dim * 16, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.df_dim * 16),
            nn.LeakyReLU(negative_slope=0.2),
            custom_con2d((self.s16, self.s16), self.df_dim * 16, self.df_dim * 8, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.df_dim * 8),
        )
        self.node_d_1 = nn.Sequential(
            custom_con2d((self.s16, self.s16), self.df_dim * 8, self.df_dim * 2, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.df_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            custom_con2d((self.s16, self.s16), self.df_dim * 2, self.df_dim * 2, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.df_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            custom_con2d((self.s16, self.s16), self.df_dim * 2, self.df_dim * 8, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.df_dim * 8),
        )
        self.activ = nn.LeakyReLU(negative_slope=0.2)
        self.init_weight()

    #TODO initialize all
    def init_weight(self, std_dev=0.02):
        for p in self.parameters():
            p.data.normal_(0, std_dev)

    def forward(self, img):
        n0 = self.node_d_0(img)
        n1 = self.node_d_1(n0)
        out = self.activ(n0 + n1)
        return out

class discriminator(nn.Module):
    def __init__(self, high_res_model, lr_imsize):
        super(discriminator, self).__init__()
        context_input_size =
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.d_context_template = nn.Sequential(
            nn.Linear(context_input_size, self.ef_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.s = lr_imsize
        self.s2, self.s4, self.s8, self.s16 = \
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)
        #channel should be put in the second dimension (the first is mini-batch)
        if high_res_model:
            self.d_image_template = d_image_encoder(self.s)
        else:
            self.d_image_template = hr_d_image_encoder(self.s)
        assert self.ef_dim == self.df_dim
        self.discriminator_combine = nn.Sequential(
            custom_con2d((self.s16, self.s16), self.df_dim * 9, self.df_dim * 8, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.df_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            custom_con2d((self.s16, self.s16), self.df_dim * 8, 1, (self.s16, self.s16), (self.s16, self.s16)),
        )
        self.logSigmoid = torch.nn.LogSigmoid()

    def forward(self, x_var, c_var):
        x_rep = self.d_image_template(x_var)
        c_rep = self.d_context_template(c_var)
        c_rep = c_rep.view((c_rep.size(0), 1, 1, c_rep.size(1)))
        c_rep = c_rep.expand(c_rep.size(0), self.s16, self.s16, c_rep.size(3))
        x_c_rep = torch.cat([x_rep, c_rep], 3)
        logits = self.discriminator_combine(x_c_rep).view(-1)
        return self.logSigmoid(logits)