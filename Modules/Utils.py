"""
Some codes from
https://github.com/openai/improved-gan/blob/master/imagenet/utils.py
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
import os
import errno
from PIL import Image, ImageDraw, ImageFont


def get_image(image_path, image_size, is_crop=False, bbox=None):
    global index
    out = transform(imread(image_path), image_size, is_crop, bbox)
    return out

def visualization(self, n):
    fake_sum_train, superimage_train = \
        self.visualize_one_superimage(self.fake_images[:n * n],
                                      self.images[:n * n],
                                      n, "train")
    fake_sum_test, superimage_test = \
        self.visualize_one_superimage(self.fake_images[n * n:2 * n * n],
                                      self.images[n * n:2 * n * n],
                                      n, "test")
    self.superimages = tf.concat(0, [superimage_train, superimage_test])
    self.image_summary = tf.merge_summary([fake_sum_train, fake_sum_test])

    hr_fake_sum_train, hr_superimage_train = \
        self.visualize_one_superimage(self.hr_fake_images[:n * n],
                                      self.hr_images[:n * n, :, :, :],
                                      n, "hr_train")
    hr_fake_sum_test, hr_superimage_test = \
        self.visualize_one_superimage(self.hr_fake_images[n * n:2 * n * n],
                                      self.hr_images[n * n:2 * n * n],
                                      n, "hr_test")
    self.hr_superimages = \
        tf.concat(0, [hr_superimage_train, hr_superimage_test])
    self.hr_image_summary = \
        tf.merge_summary([hr_fake_sum_train, hr_fake_sum_test])

def save_super_images(self, images, sample_batchs, hr_sample_batchs,
                      savenames, captions_batchs,
                      sentenceID, save_dir, subset):
    # batch_size samples for each embedding
    # Up to 16 samples for each text embedding/sentence
    numSamples = len(sample_batchs)
    for j in range(len(savenames)):
        s_tmp = '%s-1real-%dsamples/%s/%s' % \
                (save_dir, numSamples, subset, savenames[j])
        folder = s_tmp[:s_tmp.rfind('/')]
        if not os.path.isdir(folder):
            print('Make a new folder: ', folder)
            mkdir_p(folder)

        # First row with up to 8 samples
        real_img = (images[j] + 1.0) * 127.5
        img_shape = real_img.shape
        padding0 = np.zeros(img_shape)
        padding = np.zeros((img_shape[0], 20, 3))

        row1 = [padding0, real_img, padding]
        row2 = [padding0, real_img, padding]
        for i in range(np.minimum(8, numSamples)):
            lr_img = sample_batchs[i][j]
            hr_img = hr_sample_batchs[i][j]
            hr_img = (hr_img + 1.0) * 127.5
            re_sample = scipy.misc.imresize(lr_img, hr_img.shape[:2])
            row1.append(re_sample)
            row2.append(hr_img)
        row1 = np.concatenate(row1, axis=1)
        row2 = np.concatenate(row2, axis=1)
        superimage = np.concatenate([row1, row2], axis=0)

        # Second 8 samples with up to 8 samples
        if len(sample_batchs) > 8:
            row1 = [padding0, real_img, padding]
            row2 = [padding0, real_img, padding]
            for i in range(8, len(sample_batchs)):
                lr_img = sample_batchs[i][j]
                hr_img = hr_sample_batchs[i][j]
                hr_img = (hr_img + 1.0) * 127.5
                re_sample = scipy.misc.imresize(lr_img, hr_img.shape[:2])
                row1.append(re_sample)
                row2.append(hr_img)
            row1 = np.concatenate(row1, axis=1)
            row2 = np.concatenate(row2, axis=1)
            super_row = np.concatenate([row1, row2], axis=0)
            superimage2 = np.zeros_like(superimage)
            superimage2[:super_row.shape[0],
            :super_row.shape[1],
            :super_row.shape[2]] = super_row
            mid_padding = np.zeros((64, superimage.shape[1], 3))
            superimage = np.concatenate([superimage, mid_padding,
                                         superimage2], axis=0)

        top_padding = np.zeros((128, superimage.shape[1], 3))
        superimage = \
            np.concatenate([top_padding, superimage], axis=0)

        captions = captions_batchs[j][sentenceID]
        fullpath = '%s_sentence%d.jpg' % (s_tmp, sentenceID)
        superimage = self.drawCaption(np.uint8(superimage), captions)
        scipy.misc.imsave(fullpath, superimage)

def drawCaption(self, img, caption):
    img_txt = Image.fromarray(img)
    # get a font
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    # get a drawing context
    d = ImageDraw.Draw(img_txt)

    # draw text, half opacity
    d.text((10, 256), 'Stage-I', font=fnt, fill=(255, 255, 255, 255))
    d.text((10, 512), 'Stage-II', font=fnt, fill=(255, 255, 255, 255))
    if img.shape[0] > 832:
        d.text((10, 832), 'Stage-I', font=fnt, fill=(255, 255, 255, 255))
        d.text((10, 1088), 'Stage-II', font=fnt, fill=(255, 255, 255, 255))

    idx = caption.find(' ', 60)
    if idx == -1:
        d.text((256, 10), caption, font=fnt, fill=(255, 255, 255, 255))
    else:
        cap1 = caption[:idx]
        cap2 = caption[idx+1:]
        d.text((256, 10), cap1, font=fnt, fill=(255, 255, 255, 255))
        d.text((256, 60), cap2, font=fnt, fill=(255, 255, 255, 255))

    return img_txt

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
