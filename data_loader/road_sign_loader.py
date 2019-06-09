# encoding:utf-8
import collections
import json
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as m
import torch
import torchvision
import torchvision.transforms as tfs
from PIL import Image
from torch.utils import data
from tqdm import tqdm


class RoadSignLoader(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=1024, augmentations=None, random_crop=False):
        self.root = root
        self.split = split
        self.random_crop = random_crop
        self.is_transform = is_transform
        self.n_classes = 2
        self.augmentations = augmentations
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.crop_w, self.crop_h = self.img_size
        self.mean = np.array([109.54834218, 114.86824715, 102.69644417])
        self.files = collections.defaultdict(list)
        for split in ["train", "test"]:
            with open('/home/ubuntu/data/roadSign/dataset/' + split + '.txt') as f:
                file_list = f.read().split()
                self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        root_dir = '/home/ubuntu/data/roadSign'
        image_dir = os.path.join(root_dir, 'JPEGImages')
        gt_dir = os.path.join(root_dir, 'annotations')

        img = Image.open(os.path.join(image_dir, img_name + '.jpeg'))
        img = np.array(img, dtype=np.float64)

        gt, boundary = self.build_mask(os.path.join(gt_dir, img_name + '.txt'))

        if self.random_crop:
            img_h, img_w, _ = img.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)

            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0,
                                             pad_w, cv2.BORDER_CONSTANT,
                                             value=(0.0, 0.0, 0.0))
            else:
                img_pad = img

            img_h, img_w, _ = img_pad.shape

            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)

            img = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
            gt = np.asarray(gt[h_off:h_off + self.crop_h, w_off: w_off + self.crop_w], np.uint8)
            boundary = np.asarray(boundary[h_off:h_off + self.crop_h, w_off: w_off + self.crop_w], np.uint8)

        if self.augmentations is not None:
            img, lbl, boundary = self.augmentations(img, gt, boundary)

        if self.is_transform:
            img, gt, boundary = self.transform(img, gt, boundary)

        return img, gt, gt, boundary, img_name.split('/')[-1]

    def build_mask(self, file_dir):
        boundary = np.zeros([1020, 1920], dtype=np.uint8)
        gt = np.zeros([1020, 1920], dtype=np.uint8)
        with open(file_dir) as f:
            content = json.load(f)
            for child in content:
                points = child['points']
                cv2.polylines(boundary, np.array([points], dtype=np.int32), True, 1, 2)
                cv2.fillPoly(gt, np.array([points], dtype=np.int32), 1)
        return gt, boundary

    def train_tf(self, x):
        im_aug = tfs.Compose([
            tfs.Resize(120),
            tfs.RandomHorizontalFlip(),
            tfs.RandomCrop(96),
            tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            tfs.ToTensor(),
            tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        x = im_aug(x)
        return x

    def transform(self, img, lbl, boundary):
        # RGB 2 BGR
        img = img[:, :, ::-1]
        img -= self.mean

        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)
        lbl = lbl.astype(int)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        boundary = torch.from_numpy(boundary).long().unsqueeze(0)

        return img, lbl, boundary

    def get_pascal_labels(self):
        return np.asarray([[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]])

    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for i, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = i
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, temp, plot=False):
        label_colours = self.get_pascal_labels()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.subplot(121)
            plt.imshow(rgb)
            plt.subplot(122)
            plt.imshow(temp)
            plt.show()
        else:
            return rgb

    def setup(self, pre_encode=False):
        target_path = self.root + '/image_data/train/pre_encoded/'
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        if pre_encode:
            print("Pre-encoding segmentation masks...")
            for i in tqdm(self.files['train']):
                lbl_path = self.root + '/image_data/train/label/' + i
                lbl = self.encode_segmap(m.imread(lbl_path))
                lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
                m.imsave(target_path + i[:-4] + '.png', lbl)

    def standardize(self, x):

        x -= np.mean(x)
        x /= np.std(x)

        return x


if __name__ == '__main__':
    local_path = '/data/Potsdam/'
    dst = RoadSignLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=8)
    for i, data in enumerate(trainloader):
        imgs, labels, y_cls = data
        # rgb = dst.decode_segmap(labels.numpy()[i])
        # plt.imshow(np.array(rgb,dtype=np.uint8))
        # plt.show()
        # a = labels.numpy()[i]
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            print(type(y_cls))
            plt.imshow(img)
            plt.show()
            plt.imshow(np.array(dst.decode_segmap(labels.numpy()[i]), dtype=np.uint8))
            plt.show()
        break
