import dateutil
import dateutil.tz
import datetime
import argparse
import pprint

from Modules.Optim import Optim
from Modules.Datasets import TextDataset
from Model.model import StackGAN
from Modules.Utils import mkdir_p, save_super_images
from Modules.Config import cfg, cfg_from_file
import path

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
        for p in model.parameters():
            p.data.normal_(0, 0.02)
        images, embeddings_batchs, savenames, captions_batchs = \
            self.dataset.train.next_batch_test(self.batch_size, 0, 1)
        #embeddings_batchs list of 1 batch * 1024
        print len(savenames)
        print len(captions_batchs), len(captions_batchs[0])
        print len(embeddings_batchs)
        print embeddings_batchs[0].shape #
        numSamples = min(16, cfg.TRAIN.NUM_COPY)
        samples_batchs = []
        hr_samples_batchs = []

        for i in range(1):
            for j in range(numSamples):
                hr_samples, samples = images, images
                hr_samples_batchs.append(hr_samples)
                samples_batchs.append(samples)
            save_super_images(images, samples_batchs,
                                   hr_samples_batchs,
                                   savenames, captions_batchs,
                                   i, self.log_dir, "test")

        for stage in [1, 2]:
            cfg_from_file(args.cfg_file + "stage%d.yml" % stage)
            lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
            stage_optim = Optim(
                model.get_stage_parameters(stage), "adam", cfg.TRAIN.DISCRIMINATOR_LR, 10000000,
                lr_decay=0.5,
                start_decay_at=0,
                lr_decay_freq =lr_decay_step,
            )
            number_example = self.dataset.train._num_examples
            updates_per_epoch = int(number_example / self.batch_size)
            for epoch in range(600):
                for i in range(updates_per_epoch):
                    hr_images, lr_images, hr_wrong_images, lr_wrong_images, embeddings, _, _ = \
                        self.dataset.train.next_batch(self.batch_size,
                                                      cfg.TRAIN.NUM_EMBEDDING)
                    self.model(embeddings, hr_images, lr_images, hr_wrong_images, lr_wrong_images, stage)

    def evaluate(self):
        if self.model_path.find('.ckpt') != -1:
            self.init_opt()
            print("Reading model parameters from %s" % self.model_path)
            saver = tf.train.Saver(tf.all_variables())
            saver.restore(sess, self.model_path)
            # self.eval_one_dataset(sess, self.dataset.train,
            #                       self.log_dir, subset='train')
            count = 0
            print('num_examples:', dataset._num_examples)
            while count < dataset._num_examples:
                start = count % dataset._num_examples
                images, embeddings_batchs, savenames, captions_batchs = \
                    dataset.next_batch_test(self.batch_size, start, 1)

                print('count = ', count, 'start = ', start)
                # the i-th sentence/caption
                for i in range(len(embeddings_batchs)):
                    samples_batchs = []
                    hr_samples_batchs = []
                    # Generate up to 16 images for each sentence,
                    # with randomness from noise z and conditioning augmentation.
                    numSamples = np.minimum(16, cfg.TRAIN.NUM_COPY)
                    for j in range(numSamples):
                        hr_samples, samples = \
                            sess.run([self.hr_fake_images, self.fake_images],
                                     {self.embeddings: embeddings_batchs[i]})
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
    # if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file + "stage1.yml")
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
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
        dataset.train = dataset.get_data(filename_train)
        log_dir = "ckt_logs/%s/%s_%s" % \
                  (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        mkdir_p(log_dir)
    else:
        s_tmp = cfg.TRAIN.PRETRAINED_MODEL
        log_dir = s_tmp[:s_tmp.find('.ckpt')]

    model = StackGAN(
        lr_imsize=int(dataset.image_shape[0] / dataset.hr_lr_ratio),
        hr_lr_ratio=dataset.hr_lr_ratio
    )
    model.cuda()
    algo = CondGANTrainer(
        model=model,
        dataset=dataset,
        ckt_logs_dir=log_dir
    )

    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        algo.evaluate()