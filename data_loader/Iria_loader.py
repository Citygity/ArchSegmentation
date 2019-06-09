# encoding:utf-8
import collections
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as m
import torch
import torchvision
import torchvision.transforms as tfs
from PIL import Image
from torch.utils import data
from tqdm import tqdm

from scripts.EdgeDrawing import EdgeDrawing


class IriaLoader(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=500, augmentations=None):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 2
        self.augmentations = augmentations
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([109.54834218, 114.86824715, 102.69644417])
        # train[100.14192001933334,108.95194824577779,103.23418759955555]
        # test [105.2509683148889,120.784546056,115.8624967651111]
        self.files = collections.defaultdict(list)
        for split in ["train", "val", "test", 'test_val', 'train_aug']:
            with open('/home/ubuntu/data/AerialImageDataset/dataset/' + split + '.txt') as f:
                file_list = f.read().split()
                self.files[split] = file_list
        self.EDParam = {'ksize': 3,  # gaussian Smooth filter size if smoothed = False
                        'sigma': 1,  # gaussian smooth sigma ify smoothed = False
                        'gradientThreshold': 25,  # threshold on gradient image
                        'anchorThreshold': 10,  # threshold to determine the anchor
                        'scanIntervals': 4}  # scan interval, the smaller, the more detail
        self.ED = EdgeDrawing(self.EDParam)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = '/home/ubuntu/data/AerialImageDataset/train/JPEGImages'  # self.root + '/image_data/' + self.split+ '/' + 'image/' + img_name
        gt_path = '/home/ubuntu/data/AerialImageDataset/train/SegmentationClass'  # self.root + '/image_data/train/' + 'pre_encoded/' + img_name[:-4] + '.png'
        image = Image.open(os.path.join(img_path, img_name + '.jpeg'))
        img = np.array(image)  # cv2.imread(os.path.join(img_path,img_name+'.jpeg'))
        # _, edges_map = self.ED.EdgeDrawing(np.array(image.convert('L')).astype('uint8'), smoothed=False)
        lbl = Image.open(os.path.join(gt_path, img_name + '.png'))  # cv2.imread(os.path.join(gt_path,img_name+'.png'))
        gt = np.array(lbl, dtype=np.uint8)
        boundary = gt.copy()
        boundary.astype(np.float32)
        boundary[gt < 255] = 0.0
        boundary[gt == 255] = 1.0
        # edges_map[edges_map == 255] = 1
        # edges_map = torch.from_numpy(edges_map).float().unsqueeze(0)
        if self.augmentations is not None:
            img, lbl, boundary = self.augmentations(img, gt, boundary)
        if self.is_transform:
            img, gt, boundary = self.transform(img, gt, boundary)

        return img, gt, gt, boundary, img_name

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
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)
        # lbl[lbl == 255] = 0
        # lbl = lbl.astype(float)
        # lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
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
    dst = IriaLoader(local_path, is_transform=True)
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
