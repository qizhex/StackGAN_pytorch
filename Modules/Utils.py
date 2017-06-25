"""
Some codes from
https://github.com/openai/improved-gan/blob/master/imagenet/utils.py
"""
import numpy as np
import scipy.misc
import os
import errno
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
import torchvision
import torch

def wrap_Variable(data):
    v = Variable(data)
    if True:
        v = v.cuda()
    return v

def get_image(image_path, image_size, is_crop=False, bbox=None):
    global index
    out = transform(imread(image_path), image_size, is_crop, bbox)
    return out


def save_super_images(images, sample_batchs, hr_sample_batchs,
                      savenames, captions_batchs,
                      sentenceID, save_dir, subset):
    # batch_size samples for each embedding
    # Up to 16 samples for each text embedding/sentence
    numSamples = len(sample_batchs)
    images = images.data.cpu()
    resize = torchvision.transforms.Compose(
        [torchvision.transforms.Lambda(lambda x: (x + 1) * 255. / 2),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Scale((images[0].size(1), images[0].size(2))),
         torchvision.transforms.ToTensor()])
    sample_batchs = [[resize(li[i].data.cpu()) for i in range(li.size(0))] for li in sample_batchs]
    hr_sample_batchs = [[resize(li[i].data.cpu()) for i in range(li.size(0))] for li in hr_sample_batchs]
    images = [resize(images[s]) for s in range(images.size(0))]
    for j in range(len(savenames)):
        s_tmp = '%s-1real-%dsamples/%s/%s' % \
                (save_dir, numSamples, subset, savenames[j])
        hr_images_to_show = []
        lr_images_to_show = []
        folder = s_tmp[:s_tmp.rfind('/')]
        if not os.path.isdir(folder):
            print('Make a new folder: ', folder)
            mkdir_p(folder)

        real_img = images[j]

        hr_images_to_show.append(real_img)
        lr_images_to_show.append(real_img)
        for i in range(len(sample_batchs)):
            lr_img = sample_batchs[i][j]
            hr_img = hr_sample_batchs[i][j]
            lr_images_to_show.append(lr_img)
            hr_images_to_show.append(hr_img)
        captions = captions_batchs[j][sentenceID]
        fullpath = '%s_sentence%d.jpg' % (s_tmp, sentenceID)
        two_row_image = hr_images_to_show + lr_images_to_show
        two_row_image_tensor = torch.zeros(len(two_row_image), *two_row_image[0].size())
        for i in range(len(two_row_image)):
            two_row_image_tensor[i] = two_row_image[i]
        torchvision.utils.save_image(two_row_image_tensor, fullpath, nrow=1+len(sample_batchs))

def custom_crop(img, bbox):
    # bbox = [x-left, y-top, width, height]
    imsiz = img.shape  # [height, width, channel]
    # if box[0] + box[2] >= imsiz[1] or\
    #     box[1] + box[3] >= imsiz[0] or\
    #     box[0] <= 0 or\
    #     box[1] <= 0:
    #     box[0] = np.maximum(0, box[0])
    #     box[1] = np.maximum(0, box[1])
    #     box[2] = np.minimum(imsiz[1] - box[0] - 1, box[2])
    #     box[3] = np.minimum(imsiz[0] - box[1] - 1, box[3])
    center_x = int((2 * bbox[0] + bbox[2]) / 2)
    center_y = int((2 * bbox[1] + bbox[3]) / 2)
    R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
    y1 = np.maximum(0, center_y - R)
    y2 = np.minimum(imsiz[0], center_y + R)
    x1 = np.maximum(0, center_x - R)
    x2 = np.minimum(imsiz[1], center_x + R)
    img_cropped = img[y1:y2, x1:x2, :]
    return img_cropped


def transform(image, image_size, is_crop, bbox):
    image = colorize(image)
    if is_crop:
        image = custom_crop(image, bbox)
    #
    transformed_image =\
        scipy.misc.imresize(image, [image_size, image_size], 'bicubic')
    return np.array(transformed_image)


def imread(path):
    img = scipy.misc.imread(path)
    if len(img.shape) == 0:
        raise ValueError(path + " got loaded as a dimensionless array!")
    return img.astype(np.float)


def colorize(img):
    if img.ndim == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
        img = np.concatenate([img, img, img], axis=2)
    if img.shape[2] == 4:
        img = img[:, :, 0:3]
    return img


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
