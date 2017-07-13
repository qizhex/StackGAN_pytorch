import torch
from Modules.Config import cfg
from torch import nn
import math
from discriminator import discriminator
from generator import lr_generator, hr_generator, context_encoder_g
from torch.autograd import Variable
from Modules.Utils import wrap_Variable

class StackGAN(nn.Module):
    def __init__(self, lr_imsize, hr_lr_ratio):
        super(StackGAN, self).__init__()
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.network_type = cfg.GAN.NETWORK_TYPE
        self.hr_lr_ratio = hr_lr_ratio
        self.gf_dim = cfg.GAN.GF_DIM
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.z_dim = cfg.Z_DIM
        self.s = lr_imsize
        print('lr_imsize: ', lr_imsize)
        self.s2, self.s4, self.s8, self.s16 = \
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)
        # the input must be 16 * k, otherwise it's not tf padding
        c_var_dim = 1024
        self.lr_disc = discriminator(False, lr_imsize, c_var_dim)
        self.hr_disc = discriminator(True, lr_imsize, c_var_dim)
        self.lr_generator = lr_generator(lr_imsize, self.z_dim, self.ef_dim, c_var_dim)
        self.hr_generator = hr_generator(lr_imsize, c_var_dim)

    def get_stage_d_parameters(self, stage):
        if stage == 1:
            for p in self.lr_disc.parameters():
                yield p
        elif stage == 2:
            for p in self.hr_disc.parameters():
                yield p
        else:
            assert False

    def get_stage_g_parameters(self, stage):
        if stage == 1:
            for p in self.lr_generator.parameters():
                yield p
        elif stage == 2:
            for p in self.hr_generator.parameters():
                yield p
        else:
            assert False

    def forward(self, embeddings, stage):
        batch_size = embeddings.size(0)
        fake_images, _ = self.lr_generator(batch_size, embeddings)
        if stage == 2:
            hr_fake_images, _ = self.hr_generator(fake_images, embeddings)
        else:
            hr_fake_images = None
        return fake_images, hr_fake_images

    #def forward_backward(self, embeddings, hr_images, images, hr_wrong_images, wrong_images, stage, criterion):


class StableBCELoss(nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.sum() #size average=False

class criterion():
    def __init__(self):
        self.criterion_node = StableBCELoss()
        if cfg.GPU_ID != -1:
            self.criterion_node = self.criterion_node.cuda()
        self.sigmoid = nn.Sigmoid()

    def evaluate_d_cost(self, fake_d_out, real_d_out, wrong_d_out):
        ones = Variable(torch.ones(fake_d_out.size(0)))
        zeros = Variable(torch.zeros(fake_d_out.size(0)))
        if cfg.GPU_ID != -1:
            ones = ones.cuda()
            zeros = zeros.cuda()
        real_d_loss = self.criterion_node(real_d_out, ones)
        wrong_d_loss = self.criterion_node(wrong_d_out, zeros)
        fake_d_loss = self.criterion_node(fake_d_out, zeros)
        real_output = self.sigmoid(real_d_out)
        fake_output = self.sigmoid(fake_d_out)
        wrong_output = self.sigmoid(wrong_d_out)
        if cfg.TRAIN.B_WRONG:
            discriminator_loss = real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
        else:
            discriminator_loss = real_d_loss + fake_d_loss
        return discriminator_loss, real_output.data.mean(), fake_output.data.mean(), wrong_output.data.mean()

    def evaluate_cost(self, fake_d_out):
        ones = Variable(torch.ones(fake_d_out.size(0)))
        if cfg.GPU_ID != -1:
            ones = ones.cuda()
        generator_loss = self.criterion_node(fake_d_out, ones)
        fake_output = self.sigmoid(fake_d_out)
        return generator_loss, fake_output.data.mean()
