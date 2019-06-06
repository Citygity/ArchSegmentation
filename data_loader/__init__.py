import json

from data_loader.CityScape_loader import CityscapesLoader
from data_loader.Iria_loader import IriaLoader
from data_loader.road_sign_loader import RoadSignLoader


def get_loader(name):
    """get_loader

    :param name:
    """

    return {
        'Iria': IriaLoader,
        'CityScape': CityscapesLoader,
        'RoadSign': RoadSignLoader,
    }[name]


def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']


def get_dataset_mean(dataset_path):
    import os
    import numpy as np
    import cv2

    R_means = []
    G_means = []
    B_means = []
    for file_name in os.listdir(dataset_path):
        im = cv2.imread(os.path.join(dataset_path, file_name))
        # extrect value of diffient channel
        im_R = im[:, :, 0]
        im_G = im[:, :, 1]
        im_B = im[:, :, 2]
        # count mean for every channel
        im_R_mean = np.mean(im_R)
        im_G_mean = np.mean(im_G)
        im_B_mean = np.mean(im_B)
        # save single mean value to a set of means
        R_means.append(im_R_mean)
        G_means.append(im_G_mean)
        B_means.append(im_B_mean)
        print('BGR mean of {} [{},{},{}]'.format(file_name, im_R_mean, im_G_mean, im_B_mean))
    # three sets  into a large set
    a = [R_means, G_means, B_means]
    mean = [0, 0, 0]
    # count the sum of different channel means
    mean[0] = np.mean(a[0])
    mean[1] = np.mean(a[1])
    mean[2] = np.mean(a[2])
    print('the BGR mean is [{},{},{}]'.format(mean[0], mean[1], mean[2]))


import logging
from data_loader.augmentations import (
    AdjustContrast,
    AdjustGamma,
    AdjustBrightness,
    AdjustSaturation,
    AdjustHue,
    RandomCrop,
    RandomHorizontallyFlip,
    RandomVerticallyFlip,
    Scale,
    RandomSized,
    RandomSizedCrop,
    RandomRotate,
    RandomTranslate,
    CenterCrop,
    Compose,
)

logger = logging.getLogger("ptsemseg")

key2aug = {
    "gamma": AdjustGamma,
    "hue": AdjustHue,
    "brightness": AdjustBrightness,
    "saturation": AdjustSaturation,
    "contrast": AdjustContrast,
    "rcrop": RandomCrop,
    "hflip": RandomHorizontallyFlip,
    "vflip": RandomVerticallyFlip,
    "scale": Scale,
    "rsize": RandomSized,
    "rsizecrop": RandomSizedCrop,
    "rotate": RandomRotate,
    "translate": RandomTranslate,
    "ccrop": CenterCrop,
}


def get_composed_augmentations(aug_dict):
    if aug_dict is None:
        logger.info("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug[aug_key](aug_param))
        logger.info("Using {} aug with params {}".format(aug_key, aug_param))
    return Compose(augmentations)
