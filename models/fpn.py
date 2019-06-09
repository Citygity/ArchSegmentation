"""Feature Pyramid Network (FPN) on top of ResNet. Comes with task-specific heads on top of it.

See:
- https://arxiv.org/abs/1612.03144 - Feature Pyramid Networks for Object Detection
- http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf - A Unified Architecture for Instance
                                                               and Semantic Segmentation

"""

import torch
import torch.nn as nn

from models.resnet import *


class FPN(nn.Module):
    """Feature Pyramid Network (FPN): top-down architecture with lateral connections.
       Can be used as feature extractor for object detection or segmentation.
    """

    def __init__(self, num_filters=256, pretrained=True, backbone='resnet50'):
        """Creates an `FPN` instance for feature extraction.

        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super(FPN, self).__init__()
        if backbone == 'resnet18':
            self.resnet = resnet18(pretrained=pretrained, input_channel=3)
        if backbone == 'resnet34':
            self.resnet = resnet34(pretrained=pretrained, input_channel=3)
        if backbone == 'resnet50':
            self.resnet = resnet50(pretrained=pretrained, input_channel=3)
        if backbone == 'resnet101':
            self.resnet = resnet101(pretrained=pretrained, input_channel=3)
        if backbone == 'resnet152':
            self.resnet = resnet152(pretrained=pretrained, input_channel=3)
        if backbone == 'resnext50':
            self.resnet = resnext50_32x4d(pretrained=pretrained, input_channel=3)
        if backbone == 'resnext101':
            self.resnet = resnext101_32x8d(pretrained=pretrained, input_channel=3)
        # Access resnet directly in forward pass; do not store refs here due to
        # https://github.com/pytorch/pytorch/issues/8392

        self.lateral4 = Conv1x1(2048, num_filters)
        self.lateral3 = Conv1x1(1024, num_filters)
        self.lateral2 = Conv1x1(512, num_filters)
        self.lateral1 = Conv1x1(256, num_filters)

        self.smooth4 = Conv3x3(num_filters, num_filters)
        self.smooth3 = Conv3x3(num_filters, num_filters)
        self.smooth2 = Conv3x3(num_filters, num_filters)
        self.smooth1 = Conv3x3(num_filters, num_filters)

    def forward(self, x):
        # Bottom-up pathway, from ResNet

        enc0 = self.resnet.conv1(x)  # 3*500*500->64*250*250
        enc0 = self.resnet.bn1(enc0)
        enc0 = self.resnet.relu(enc0)
        enc0 = self.resnet.maxpool(enc0)

        enc1 = self.resnet.layer1(enc0)  # 64*250*250->256*125*125
        enc2 = self.resnet.layer2(enc1)  # 256*125*125->512*63*63
        enc3 = self.resnet.layer3(enc2)  # 512*63*63->1024*32*32
        enc4 = self.resnet.layer4(enc3)  # 1024*32*32->2048*16*16

        # Lateral connections

        lateral4 = self.lateral4(enc4)  #
        lateral3 = self.lateral3(enc3)  #
        lateral2 = self.lateral2(enc2)  # 256chanel
        lateral1 = self.lateral1(enc1)  #

        # Top-down pathway

        map4 = lateral4
        map3 = lateral3 + nn.Upsample(size=lateral3.size()[2:], mode="nearest")(map4)
        map2 = lateral2 + nn.Upsample(size=lateral2.size()[2:], mode="nearest")(map3)
        map1 = lateral1 + nn.Upsample(size=lateral1.size()[2:], mode="nearest")(map2)

        map4 = self.smooth4(map4)
        map3 = self.smooth3(map3)
        map2 = self.smooth2(map2)
        map1 = self.smooth1(map1)

        return [map1, map2, map3, map4], [enc1, enc2, enc3, enc4]


class fpn101(nn.Module):
    """Semantic segmentation model on top of a Feature Pyramid Network (FPN).
    """

    def __init__(self, n_classes, num_filters=128, num_filters_fpn=256, pretrained=True):
        """Creates an `FPNSegmentation` instance for feature extraction.

        Args:
          n_classes: number of classes to predict
          num_filters: the number of filters in each segmentation head pyramid level
          num_filters_fpn: the number of filters in each FPN output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super(fpn101, self).__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.

        self.fpn = FPN(num_filters=num_filters_fpn, pretrained=pretrained, backbone='resnet50')

        # The segmentation heads on top of the FPN

        self.head1 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters), Conv3x3(num_filters, num_filters))
        self.head2 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters), Conv3x3(num_filters, num_filters))
        self.head3 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters), Conv3x3(num_filters, num_filters))
        self.head4 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters), Conv3x3(num_filters, num_filters))

        self.final = nn.Conv2d(4 * num_filters, n_classes, kernel_size=3, padding=1)

    def forward(self, x):
        [map1, map2, map3, map4], _ = self.fpn(x)

        map4 = nn.Upsample(size=map1.size()[2:], mode="nearest")(self.head4(map4))
        map3 = nn.Upsample(size=map1.size()[2:], mode="nearest")(self.head3(map3))
        map2 = nn.Upsample(size=map1.size()[2:], mode="nearest")(self.head2(map2))
        map1 = self.head1(map1)

        final = self.final(torch.cat([map4, map3, map2, map1], dim=1))

        return nn.functional.upsample(final, scale_factor=4, mode="bilinear", align_corners=False)


class Conv1x1(nn.Module):
    def __init__(self, num_in, num_out):
        super(Conv1x1, self).__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        return self.block(x)


class Conv3x3(nn.Module):
    def __init__(self, num_in, num_out):
        super(Conv3x3, self).__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.block(x)
