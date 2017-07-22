import numpy as np
import pickle
import random
import torch
from torch.autograd import Variable
from Utils import wrap_Variable
import torchvision
import time

class SimpleLoader(torch.utils.data.Dataset):
    def __init__(self, images, imsize, embeddings, hr_lr_ratio, class_id, sample_emb_num):
        self._images = []
        self.toPIL = torchvision.transforms.ToPILImage()
        for i in images:
            self._images += [self.toPIL(i)]
        self.embeddings = embeddings
        self.toTensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self._imsize = imsize
        self.hr_lr_ratio = hr_lr_ratio
        self.resize_lr = torchvision.transforms.Scale(self._imsize // self.hr_lr_ratio)
        self.random_crop = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(self._imsize),
            torchvision.transforms.RandomHorizontalFlip(),
        ])
        self._class_id = class_id
        self.sample_emb_num = sample_emb_num
        self.size = len(self._images)
        self.embedding_num = embeddings.size(1)

    def __getitem__(self, index):
        hr_img, lr_img = self.transform(self._images[index])
        wrong_index = int(random.random() * self.size)
        while self._class_id[wrong_index] != self._class_id[index]:
            wrong_index = int(random.random() * self.size)
        wrong_hr_img, wrong_lr_img = self.transform(self._images[wrong_index])
        randix = torch.rand(self.sample_emb_num) * self.embedding_num
        randix = randix.long()
        e_sample = self.embeddings[index].index_select(0, randix)
        e_mean = torch.mean(e_sample, 0).squeeze()
        return hr_img, lr_img, wrong_hr_img, wrong_lr_img, e_mean

    def __len__(self):
        return self.size

    def transform(self, img):
        current_image = img
        current_image = self.random_crop(current_image)
        lr_image = self.toTensor(self.resize_lr(current_image))
        current_image = self.toTensor(current_image)
        return current_image, lr_image

class Dataset(object):
    def __init__(self, images, imsize, embeddings=None,
                 filenames=None, workdir=None,
                 labels=None, aug_flag=True,
                 class_id=None, class_range=None, hr_lr_ratio=None):
        #self._images = images
        self._images = []
        self.toPIL = torchvision.transforms.ToPILImage()
        for i in images:
            self._images += [self.toPIL(i)]
        self._embeddings = embeddings
        self._filenames = filenames
        self.workdir = workdir
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = len(images)
        self._saveIDs = self.saveIDs()
        self.hr_lr_ratio = hr_lr_ratio

        # shuffle on first run
        self._index_in_epoch = self._num_examples
        self._aug_flag = aug_flag
        self._class_id = np.array(class_id)
        self._class_range = class_range
        self._imsize = imsize
        self._perm = None
        self.toTensor = torchvision.transforms.ToTensor()
        self.resize_lr = torchvision.transforms.Scale(self._imsize // self.hr_lr_ratio)
        self.random_crop = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(self._imsize),
            torchvision.transforms.RandomHorizontalFlip(),
        ])

    @property
    def images(self):
        return self._images

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def filenames(self):
        return self._filenames

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def saveIDs(self):
        #self._saveIDs = np.arange(self._num_examples)
        #np.random.shuffle(self._saveIDs)
        #self._saveIDs = torch.randperm(self._num_examples)
        self._saveIDs = torch.arange(0, self._num_examples)
        return self._saveIDs

    def readCaptions(self, filenames, class_id):
        name = filenames
        if name.find('jpg/') != -1:  # flowers dataset
            class_name = 'class_%05d/' % class_id
            name = name.replace('jpg/', class_name)
        cap_path = '%s/text_c10/%s.txt' %\
                   (self.workdir, name)
        with open(cap_path, "r") as f:
            captions = f.read().split('\n')
        captions = [cap for cap in captions if len(cap) > 0]
        return captions

    def transform(self, images):
        if self._aug_flag:
            transformed_images =\
                torch.zeros(len(images), 3, self._imsize, self._imsize)
            lr_images = torch.zeros(len(images), 3, self._imsize // self.hr_lr_ratio, self._imsize // self.hr_lr_ratio)
            start_time = time.time()
            for i in range(len(images)):
                current_image = images[i]
                current_image = self.random_crop(current_image)
                lr_image = self.toTensor(self.resize_lr(current_image)) * 2. - 1
                current_image = self.toTensor(current_image) * 2. - 1
                transformed_images[i] = current_image
                lr_images[i] = lr_image
            return wrap_Variable(transformed_images), wrap_Variable(lr_images)
        else:
            assert False
            return wrap_Variable(torch.FloatTensor(images.tolist()))

    def sample_embeddings(self, embeddings, current_idx, filenames, class_id, sample_num):
        if embeddings.dim == 2 or embeddings.dim == 1:
            assert False
            return np.squeeze(embeddings)
        else:
            batch_size = current_idx.shape[0]
            embedding_num = embeddings.size()[1]
            embedding_size = embeddings.size()[2]
            # Take every sample_num captions to compute the mean vector
            sampled_embeddings = torch.zeros(batch_size, embedding_size)
            sampled_captions = []
            for i in range(batch_size):
                #randix = np.random.choice(embedding_num,
                #                          sample_num, replace=False)
                randix = torch.rand(sample_num) * embedding_num
                randix = randix.long()
                if sample_num == 1:
                    assert False
                    randix = int(randix)
                    captions = self.readCaptions(filenames[i],
                                                 class_id[i])
                    sampled_captions.append(captions[randix])
                    sampled_embeddings.append(embeddings[current_idx[i], randix, :])
                else:
                    e_sample = embeddings[current_idx[i]].index_select(0, randix)
                    #e_mean = np.mean(e_sample, axis=0)
                    e_mean = torch.mean(e_sample, 0)
                    #sampled_embeddings.append(e_mean)
                    sampled_embeddings[i] = e_mean
            #sampled_embeddings_array = np.array(sampled_embeddings)
            #return np.squeeze(sampled_embeddings_array), sampled_captions
            return sampled_embeddings, sampled_captions

    def next_batch(self, batch_size, window):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            self._perm = np.arange(self._num_examples)
            np.random.shuffle(self._perm)

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        current_ids = self._perm[start:end]
        fake_ids = np.random.randint(self._num_examples, size=batch_size)
        #fake_ids = torch.rand((batch_size, )) * self._num_examples
        #fake_ids = fake_ids.long()
        collision_flag = (self._class_id[current_ids] == self._class_id[fake_ids])
        fake_ids[collision_flag] =\
            (fake_ids[collision_flag] + random.randrange(100, 200)) % self._num_examples
        sampled_images = []
        for i in current_ids:
            sampled_images += [self._images[i]]
        sampled_wrong_images = []
        for i in fake_ids:
            sampled_wrong_images += [self._images[i]]
        #sampled_wrong_images = self._images[fake_ids, :, :, :]
        #sampled_images = sampled_images.astype(np.float32)
        #sampled_wrong_images = sampled_wrong_images.astype(np.float32)
        #sampled_images = sampled_images * (2. / 255) - 1.
        #sampled_wrong_images = sampled_wrong_images * (2. / 255) - 1.

        sampled_images, sampled_lr_images = self.transform(sampled_images)
        sampled_wrong_images, sampled_lr_wrong_images = self.transform(sampled_wrong_images)
        ret_list = [sampled_images, sampled_lr_images, sampled_wrong_images, sampled_lr_wrong_images]

        if self._embeddings is not None:
            filenames = [self._filenames[i] for i in current_ids]
            class_id = [self._class_id[i] for i in current_ids]
            sampled_embeddings, sampled_captions = \
                self.sample_embeddings(self._embeddings, current_ids,
                                       filenames, class_id, window)
            sampled_embeddings = wrap_Variable(sampled_embeddings)
            ret_list.append(sampled_embeddings)
            ret_list.append(sampled_captions)
        else:
            ret_list.append(None)
            ret_list.append(None)
        if self._labels is not None:
            ret_list.append(self._labels[current_ids])
        else:
            ret_list.append(None)

        return ret_list

    def next_batch_test(self, batch_size, start, max_captions):
        """Return the next `batch_size` examples from this data set."""
        if (start + batch_size) > self._num_examples:
            end = self._num_examples
            start = end - batch_size
        else:
            end = start + batch_size

        sampled_images = self._images[start:end]
        #sampled_images = sampled_images.astype(np.float32)
        # from [0, 255] to [-1.0, 1.0]
        #sampled_images = sampled_images * (2. / 255) - 1.
        sampled_images, lr_images = self.transform(sampled_images)

        sampled_embeddings = self._embeddings[start:end]
        embedding_num = sampled_embeddings.size()[1]
        sampled_embeddings_batchs = []

        sampled_captions = []
        sampled_filenames = self._filenames[start:end]
        sampled_class_id = self._class_id[start:end]
        for i in range(len(sampled_filenames)):
            captions = self.readCaptions(sampled_filenames[i],
                                         sampled_class_id[i])
            # print(captions)
            sampled_captions.append(captions)

        for i in range(np.minimum(max_captions, embedding_num)):
            batch = sampled_embeddings[:, i, :]
            sampled_embeddings_batchs.append(wrap_Variable(torch.FloatTensor(np.squeeze(batch))))

        return [sampled_images, lr_images, sampled_embeddings_batchs,
                self._saveIDs[start:end], sampled_captions]

class TextDataset(object):
    def __init__(self, workdir, embedding_type, hr_lr_ratio):
        lr_imsize = 64
        self.hr_lr_ratio = hr_lr_ratio
        if self.hr_lr_ratio == 1:
            self.image_filename = '/76images.pt'
        elif self.hr_lr_ratio == 4:
            self.image_filename = '/304images.pt'

        self.image_shape = [lr_imsize * self.hr_lr_ratio,
                            lr_imsize * self.hr_lr_ratio, 3]
        self.image_dim = self.image_shape[0] * self.image_shape[1] * 3
        self.embedding_shape = None
        self.train = None
        self.test = None
        self.workdir = workdir
        if embedding_type == 'cnn-rnn':
            self.embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            self.embedding_filename = '/skip-thought-embeddings.pickle'

    def get_information(self, pickle_path):

        with open(pickle_path + self.image_filename, 'rb') as f:
            images = torch.load(f)
            #images = np.array(images)

        with open(pickle_path + self.embedding_filename, 'rb') as f:
            embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
            embeddings = torch.FloatTensor(embeddings)
            #self.embedding_shape = [embeddings.shape[-1]]
            self.embedding_shape = embeddings.size()[-1]
            #print('embeddings: ', embeddings.shape)
            print('embeddings: ', embeddings.size())
        with open(pickle_path + '/filenames.pickle', 'rb') as f:
            list_filenames = pickle.load(f)
            print('list_filenames: ', len(list_filenames), list_filenames[0])
        with open(pickle_path + '/class_info.pickle', 'rb') as f:
            class_id = pickle.load(f)
        return images, embeddings, list_filenames, class_id

    def get_data(self, pickle_path, aug_flag=True):
        images, embeddings, list_filenames, class_id = self.get_information(pickle_path)
        return Dataset(images, self.image_shape[0], embeddings,
                   list_filenames, self.workdir, None,
                   aug_flag, class_id, None, self.hr_lr_ratio)

    def get_multi_process_data(self, pickle_path, sample_emb_num, batch_size):
        images, embeddings, list_filenames, class_id = self.get_information(pickle_path)
        simple_loader = SimpleLoader(images, self.image_shape[0], embeddings, self.hr_lr_ratio, class_id, sample_emb_num)
        return torch.utils.data.DataLoader(simple_loader, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
