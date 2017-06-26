import torch
from Modules.Config import cfg
from torch import nn
import math
from layers import custom_con2d


class lr_generator(nn.Module):
    def __init__(self, imsize, z_size, c_size):
        super(lr_generator, self).__init__()
        self.s = imsize
        self.s2, self.s4, self.s8, self.s16 = \
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)
        self.gf_dim = cfg.GAN.GF_DIM

        self.node1_0 = nn.Sequential(
            nn.Linear(z_size + c_size, self.s16 * self.s16 * self.gf_dim * 8),
            nn.BatchNorm1d(self.s16 * self.s16 * self.gf_dim * 8)
        )
        self.node1_1 = nn.Sequential(
            custom_con2d((self.s16, self.s16), self.gf_dim * 8, self.gf_dim * 2, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.gf_dim * 2),
            nn.ReLU(),
            custom_con2d((self.s16, self.s16), self.gf_dim * 2, self.gf_dim * 2, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.gf_dim * 2),
            nn.ReLU(),
            custom_con2d((self.s16, self.s16), self.gf_dim * 2, self.gf_dim * 8, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.gf_dim * 8)
        )
        self.node2_0 = nn.Sequential(
            nn.UpsamplingNearest2d((self.s8, self.s8)),
            custom_con2d((self.s8, self.s8), self.gf_dim * 8, self.gf_dim * 4, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.gf_dim * 4)
        )
        self.node2_1 = nn.Sequential(
            custom_con2d((self.s8, self.s8), self.gf_dim * 4, self.gf_dim * 1, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.gf_dim),
            nn.ReLU(),
            custom_con2d((self.s8, self.s8), self.gf_dim * 1, self.gf_dim * 1, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.gf_dim),
            nn.ReLU(),
            custom_con2d((self.s8, self.s8), self.gf_dim, self.gf_dim * 4, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.gf_dim * 4),
        )
        self.node3 = nn.Sequential(
            nn.UpsamplingNearest2d((self.s4, self.s4)),
            custom_con2d((self.s4, self.s4), self.gf_dim * 4, self.gf_dim * 2, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.gf_dim * 2),
            nn.ReLU(),
            nn.UpsamplingNearest2d((self.s2, self.s2)),
            custom_con2d((self.s2, self.s2), self.gf_dim * 2, self.gf_dim, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.gf_dim),
            nn.ReLU(),
            nn.UpsamplingNearest2d((self.s, self.s)),
            custom_con2d((self.s, self.s), self.gf_dim, 3, (3, 3), (1, 1)),
            nn.Tanh(),
        )
        self.activ = nn.ReLU()

    def forward(self, z):
        out1_0 = self.node1_0(z).view(-1, self.gf_dim * 8, self.s16, self.s16)
        out1 = self.activ(out1_0 + self.node1_1(out1_0))
        out2_0 = self.node2_0(out1)
        out2 = self.activ(out2_0 + self.node2_1(out2_0))
        out = self.node3(out2)
        return out

class residual_block(nn.Module):
    def __init__(self, imsize):
        super(residual_block, self).__init__()
        self.s = imsize
        self.gf_dim = cfg.GAN.GF_DIM
        self.s2, self.s4, self.s8, self.s16 = \
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)
        self.node = nn.Sequential(
            custom_con2d((self.s4, self.s4), self.gf_dim * 4, self.gf_dim * 4, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.gf_dim * 4),
            nn.ReLU(),
            custom_con2d((self.s4, self.s4), self.gf_dim * 4, self.gf_dim * 4, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.gf_dim * 4)
        )
        self.activ = nn.ReLU()

    def forward(self, x_c_code):
        return self.activ(self.node(x_c_code) + x_c_code)

class hr_generator(nn.Module):
    def __init__(self, imsize):
        super(hr_generator, self).__init__()
        self.s = imsize
        self.s2, self.s4, self.s8, self.s16 = \
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)
        self.gf_dim = cfg.GAN.GF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.encode_image = nn.Sequential(
            custom_con2d((self.s, self.s), 3, self.gf_dim, (3, 3), (1, 1)),
            nn.ReLU(),
            custom_con2d((self.s2, self.s2), self.gf_dim, self.gf_dim * 2, (4, 4)),
            nn.BatchNorm2d(self.gf_dim * 2),
            nn.ReLU(),
            custom_con2d((self.s4, self.s4), self.gf_dim * 2, self.gf_dim * 4, (4, 4)),
            nn.BatchNorm2d(self.gf_dim * 4),
            nn.ReLU(),
        )
        self.node0 = nn.Sequential(
            custom_con2d((self.s4, self.s4), self.ef_dim + self.gf_dim * 4, self.gf_dim * 4, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.gf_dim * 4),
            nn.ReLU(),
        )
        res_list = []
        for i in range(4):
            res_list += [residual_block(imsize)]
        self.node1 = nn.Sequential(*res_list)
        self.node2 = nn.Sequential(
            nn.UpsamplingNearest2d((self.s2, self.s2)),
            custom_con2d((self.s2, self.s2), self.gf_dim * 4, self.gf_dim * 2, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.gf_dim * 2),
            nn.ReLU(),
            nn.UpsamplingNearest2d((self.s, self.s)),
            custom_con2d((self.s, self.s), self.gf_dim * 2, self.gf_dim, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.gf_dim),
            nn.ReLU(),
            nn.UpsamplingNearest2d((self.s * 2, self.s * 2)),
            custom_con2d((self.s * 2, self.s * 2), self.gf_dim, self.gf_dim // 2, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.gf_dim // 2),
            nn.ReLU(),
            nn.UpsamplingNearest2d((self.s * 4, self.s * 4)),
            custom_con2d((self.s * 4, self.s * 4), self.gf_dim // 2, self.gf_dim // 4, (3, 3), (1, 1)),
            nn.BatchNorm2d(self.gf_dim // 4),
            nn.ReLU(),
            custom_con2d((self.s * 4, self.s * 4), self.gf_dim // 4, 3, (3, 3), (1, 1)),
            nn.Tanh(),
        )

    # TODO  c directly input
    def forward(self, x, c):
        x_rep = self.encode_image(x)
        c_rep = c.view(c.size(0), c.size(1), 1, 1)
        c_rep = c_rep.expand(c_rep.size(0), c_rep.size(1), self.s4, self.s4)
        x_c_rep = torch.cat([x_rep, c_rep], 1)
        out = self.node0(x_c_rep)
        out = self.node1(out)
        out = self.node2(out)
        return out

class context_encoder_g(nn.Module):
    def __init__(self, c_var_dim):
        super(context_encoder_g, self).__init__()
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.node = nn.Sequential(
            nn.Linear(c_var_dim, self.ef_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
        )
        #TODO linear last layer?

    def forward(self, input):
        out = self.node(input)
        mean = out[:, : self.ef_dim]
        log_sigma = out[:, self.ef_dim :]
        epsilon = torch.FloatTensor(mean.size(0), mean.size(1)).normal_(0, 1)
        epsilon = torch.autograd.Variable(epsilon)
        if cfg.GPU_ID != -1:
            epsilon = epsilon.cuda()
        out = mean + torch.exp(log_sigma) * epsilon
        kl_loss = -log_sigma + 0.5 * (-1 + torch.exp(2. * log_sigma) + torch.pow(mean, 2.))
        kl_loss = kl_loss.mean(1)
        return out, kl_loss