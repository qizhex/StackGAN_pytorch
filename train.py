import dateutil
import dateutil.tz
import datetime
import argparse
import pprint

from Modules.Optim import Optim
from Modules.Datasets import TextDataset
from Model.model import StackGAN, criterion
from Modules.Utils import mkdir_p, save_super_images, wrap_Variable
from Modules.Config import cfg, cfg_from_file
import torch
from torch import cuda
import time
import numpy as np

class CondGANTrainer(object):
    def __init__(self,
                 model,
                 dataset=None,
                 exp_name="model",
                 ckt_logs_dir="ckt_logs",
                 stage=None,
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.stage = stage
        self.dataset = dataset
        self.exp_name = exp_name
        self.log_dir = ckt_logs_dir
        self.checkpoint_dir = ckt_logs_dir

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.model_path = cfg.TRAIN.PRETRAINED_MODEL

        self.log_vars = []

        self.hr_image_shape = self.dataset.image_shape
        ratio = self.dataset.hr_lr_ratio
        self.lr_image_shape = [int(self.hr_image_shape[0] / ratio),
                               int(self.hr_image_shape[1] / ratio),
                               self.hr_image_shape[2]]
        self.criterion = criterion()
        print('hr_image_shape', self.hr_image_shape)
        print('lr_image_shape', self.lr_image_shape)

    def sample_super_image(self, stage, update):
        print "sampling image"
        num_caption = 1
        hr_images, lr_images, embeddings_batchs, savenames, captions_batchs = \
            self.dataset.test.next_batch_test(self.batch_size, 0, num_caption)
        numSamples = min(16, cfg.TRAIN.NUM_COPY)
        samples_batchs = []
        hr_samples_batchs = []
        for i in range(num_caption):
            for j in range(numSamples):
                samples, hr_samples = self.model(embeddings_batchs[i], stage)
                hr_samples_batchs.append(hr_samples)
                samples_batchs.append(samples)
            save_super_images(hr_images, samples_batchs,
                              hr_samples_batchs,
                              savenames, captions_batchs,
                              i, self.log_dir, "test")

    def train(self):
        if len(self.model_path) > 0:
            print("Reading model parameters from %s" % self.model_path)
            #assert False
            self.model = torch.load(self.model_path)
        model = self.model
        '''for p in model.parameters():
            size = p.data.size()
            u1 = torch.rand(size) * (1 - np.exp(-2)) + np.exp(-2)
            # sample u2:
            u2 = torch.rand(size)
            # sample the truncated gaussian ~TN(0,1,[-2,2]):
            z = torch.sqrt(-2*torch.log(u1)) * torch.cos(2*np.pi*u2)
            p.data.copy_(z * 0.02)
            #p.data.normal_(0, 0.02)'''
        update_count = 0
        for stage in [self.stage]:#[1, 2]:
            print "stage", stage
            cfg_from_file(args.cfg_file + "stage%d.yml" % stage)
            lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
            stage_d_optim = Optim(
                model.get_stage_d_parameters(stage), "adam", cfg.TRAIN.DISCRIMINATOR_LR, 10000000,
                lr_decay=0.5,
                start_decay_at=0,
                lr_decay_freq =lr_decay_step,
            )
            stage_g_optim = Optim(
                model.get_stage_g_parameters(stage), "adam", cfg.TRAIN.DISCRIMINATOR_LR, 10000000,
                lr_decay=0.5,
                start_decay_at=0,
                lr_decay_freq =lr_decay_step,
            )
            for epoch in range(cfg.TRAIN.MAX_EPOCH):
                for hr_images, lr_images, hr_wrong_images, lr_wrong_images, embeddings in self.dataset.train:
                    #print lr_images.size()
                    hr_images = wrap_Variable(hr_images)
                    lr_images = wrap_Variable(lr_images)
                    hr_wrong_images = wrap_Variable(hr_wrong_images)
                    lr_wrong_images = wrap_Variable(lr_wrong_images)
                    embeddings = wrap_Variable(embeddings)
                    batch_size = embeddings.size(0)
                    fake_images, kl_loss = self.model.lr_generator(batch_size, embeddings)
                    #print fake_images
                    if stage == 1:
                        self.model.lr_disc.zero_grad()
                        fake_d_out = self.model.lr_disc(fake_images.detach(), embeddings)
                        real_d_out = self.model.lr_disc(lr_images, embeddings)
                        wrong_d_out = self.model.lr_disc(lr_wrong_images, embeddings)
                        hr_fake_images = None
                    elif stage == 2:
                        self.model.hr_disc.zero_grad()
                        hr_fake_images, kl_loss = self.model.hr_generator(fake_images, embeddings)
                        fake_d_out = self.model.hr_disc(hr_fake_images.detach(), embeddings)
                        real_d_out = self.model.hr_disc(hr_images, embeddings)
                        wrong_d_out = self.model.hr_disc(hr_wrong_images, embeddings)
                    else:
                        assert False
                    disc_loss, p_real, p_fake, p_wrong = self.criterion.evaluate_d_cost(fake_d_out, real_d_out, wrong_d_out)
                    disc_loss.div(batch_size).backward()
                    stage_d_optim.step()
                    if stage == 1:
                        self.model.lr_generator.zero_grad()
                        fake_d_out = self.model.lr_disc(fake_images, embeddings)
                    else:
                        self.model.hr_generator.zero_grad()
                        fake_d_out = self.model.hr_disc(hr_fake_images, embeddings)

                    gen_loss, p_fake_new = self.criterion.evaluate_g_cost(fake_d_out)
                    gen_loss = gen_loss * 2
                    gen_loss += kl_loss.sum()
                    gen_loss.div(batch_size).backward()
                    stage_g_optim.step()
                    update_count += 1

                    if update_count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                        print "stage: %d epoch: %d total update: %d, gen_loss: %.5f, disc_loss: %.5f, real prob: %.5f, fake prob: %.5f, new fake prob(smaller): %.5f, wrong prob: %.5f" % (stage, epoch, update_count, gen_loss.data[0], disc_loss.data[0], p_real, p_fake, p_fake_new, p_wrong)
                        self.sample_super_image(stage, update_count)
                        torch.save(model, "%s/%s.ckpt" % (self.checkpoint_dir, self.exp_name))


    def evaluate(self):
        if self.model_path.find('ckpt') != -1:
            print("Reading model parameters from %s" % self.model_path)
            test_data = self.dataset.test
            print('num_examples:', test_data._num_examples)
            count = 0
            while count < test_data._num_examples:
                start = count % test_data._num_examples
                images, embeddings_batchs, savenames, captions_batchs = \
                    test_data.next_batch_test(self.batch_size, start, 1)

                print('count = ', count, 'start = ', start)
                # the i-th sentence/caption
                for i in range(len(embeddings_batchs)):
                    samples_batchs = []
                    hr_samples_batchs = []
                    numSamples = min(16, cfg.TRAIN.NUM_COPY)
                    for j in range(numSamples):
                        samples_batchs.append(samples)
                        hr_samples_batchs.append(hr_samples)
                    self.save_super_images(images, samples_batchs,
                                           hr_samples_batchs,
                                           savenames, captions_batchs,
                                           i, save_dir, subset)

                count += self.batch_size
        else:
            print("Input a valid model path.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=-1, type=int)
    parser.add_argument('--stage', type=int, default=1)
    # if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file + "stage%d.yml" % args.stage)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
        cuda.set_device(cfg.GPU_ID)
    print('Using config:')
    pprint.pprint(cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    datadir = '../data/%s' % cfg.DATASET_NAME
    dataset = TextDataset(datadir,  cfg.EMBEDDING_TYPE, 4)
    filename_test = '%s/test' % (datadir)
    dataset.test = dataset.get_data(filename_test)
    if cfg.TRAIN.FLAG:
        filename_train = '%s/train' % (datadir)
        dataset.train = dataset.get_multi_process_data(filename_train, cfg.TRAIN.NUM_EMBEDDING, cfg.TRAIN.BATCH_SIZE)
        log_dir = "../ckt_logs/%s/%s_%s" % \
                  (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        mkdir_p(log_dir)
    else:
        s_tmp = cfg.TRAIN.PRETRAINED_MODEL
        log_dir = s_tmp[:s_tmp.find('.ckpt')]

    model = StackGAN(
        lr_imsize=int(dataset.image_shape[0] / dataset.hr_lr_ratio),
        hr_lr_ratio=dataset.hr_lr_ratio
    )
    if cfg.GPU_ID != -1:
        model.cuda()
    algo = CondGANTrainer(
        model=model,
        dataset=dataset,
        ckt_logs_dir=log_dir,
        stage=args.stage,
    )

    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        algo.evaluate()
