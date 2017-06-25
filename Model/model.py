import torch
from Modules.Config import cfg
from torch import nn
import math
from discriminator import discriminator
from generator import lr_generator, hr_generator, context_encoder_g
from torch.autograd import Variable

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
        self.lr_generator = lr_generator(lr_imsize, self.z_dim, self.ef_dim)
        self.hr_generator = hr_generator(lr_imsize)
        self.lr_context_encoder = context_encoder_g(c_var_dim)
        self.hr_context_encoder = context_encoder_g(c_var_dim)

    def get_stage_parameters(self, stage):
        if stage == 1:
            for i in [self.lr_disc, self.lr_generator, self.lr_context_encoder]:
                for p in i.parameters():
                    yield p
        elif stage == 2:
            for i in [self.hr_disc, self.hr_generator, self.hr_context_encoder]:
                for p in i.parameters():
                    yield p
        else:
            assert False

    def forward(self, embeddings, hr_images, images, hr_wrong_images, wrong_images, stage):
        z = torch.FloatTensor(self.batch_size, self.z_dim).normal_(0, 1)
        z = Variable(z)
        if cfg.GPU_ID != -1:
            z = z.cuda()
        c, kl_loss = self.lr_context_encoder(embeddings)
        kl_loss = kl_loss * cfg.TRAIN.COEFF.KL
        fake_images = self.lr_generator(torch.cat([z, c], 1))
        if stage == 1:
            fake_d_out = self.lr_disc(fake_images, embeddings)
            real_d_out = self.lr_disc(images, embeddings)
            wrong_d_out = self.lr_disc(wrong_images, embeddings)
            hr_fake_images = None
        elif stage == 2:
            hr_c, hr_kl_loss = self.hr_context_encoder(embeddings)
            hr_kl_loss *= cfg.TRAIN.COEFF.KL
            hr_fake_images = self.hr_generator(fake_images, hr_c)
            fake_d_out = self.hr_disc(hr_fake_images, embeddings)
            real_d_out = self.hr_disc(hr_images, embeddings)
            wrong_d_out = self.hr_disc(hr_wrong_images, embeddings)
            kl_loss = hr_kl_loss
        else:
            assert False
        return fake_d_out, real_d_out, wrong_d_out, kl_loss, fake_images, hr_fake_images

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

    def evaluate_cost(self, fake_d_out, real_d_out, wrong_d_out):
        ones = Variable(torch.ones(fake_d_out.size(0)))
        zeros = Variable(torch.zeros(fake_d_out.size(0)))
        if cfg.GPU_ID != -1:
            ones = ones.cuda()
            zeros = zeros.cuda()
        real_d_loss = self.criterion_node(real_d_out, ones)
        wrong_d_loss = self.criterion_node(wrong_d_out, zeros)
        fake_d_loss = self.criterion_node(fake_d_out, zeros)
        if cfg.TRAIN.B_WRONG:
            discriminator_loss = real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
        else:
            discriminator_loss = real_d_loss + fake_d_loss
        generator_loss = self.criterion_node(fake_d_out, ones)
        return discriminator_loss, generator_loss