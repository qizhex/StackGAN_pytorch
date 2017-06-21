from __future__ import division
from __future__ import print_function

import dateutil
import dateutil.tz
import datetime
import argparse
import pprint

from Modules.Datasets import TextDataset
from Model.Stage2 import CondGAN
from Modules.Utils import mkdir_p
from Modules.Config import cfg, cfg_from_file

class CondGANTrainer(object):
    def __init__(self,
                 model,
                 dataset=None,
                 exp_name="model",
                 ckt_logs_dir="ckt_logs",
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
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
        print('hr_image_shape', self.hr_image_shape)
        print('lr_image_shape', self.lr_image_shape)

    def train(self):

        if len(self.model_path) > 0:
            print("Reading model parameters from %s" % self.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=-1, type=int)
    # if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    print('Using config:')
    pprint.pprint(cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    datadir = 'Data/%s' % cfg.DATASET_NAME
    dataset = TextDataset(datadir,  cfg.EMBEDDING_TYPE, 4)
    filename_test = '%s/test' % (datadir)
    dataset.test = dataset.get_data(filename_test)
    if cfg.TRAIN.FLAG:
        filename_train = '%s/train' % (datadir)
        dataset.train = dataset.get_data(filename_train)
        log_dir = "ckt_logs/%s/%s_%s" % \
                  (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        mkdir_p(log_dir)
    else:
        s_tmp = cfg.TRAIN.PRETRAINED_MODEL
        log_dir = s_tmp[:s_tmp.find('.ckpt')]

    model = CondGAN(
        lr_imsize=int(dataset.image_shape[0] / dataset.hr_lr_ratio),
        hr_lr_ratio=dataset.hr_lr_ratio
    )

    algo = CondGANTrainer(
        model=model,
        dataset=dataset,
        ckt_logs_dir=log_dir
    )

    if cfg.TRAIN.FLAG:
        algo.train()
    else:

        algo.evaluate()
