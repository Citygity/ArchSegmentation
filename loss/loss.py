import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/marvis/pytorch-yolo2/blob/master/FocalLoss.py
# https://github.com/unsky/focal-loss
class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=[1, 5], type='softmax'):
        target = target.view(-1, 1).long()

        if type == 'sigmoid':
            if class_weight is None:
                class_weight = [1] * 2  # [0.5, 0.5]

            prob = F.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1 - prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif type == 'softmax':
            B, C, H, W = logit.size()
            if class_weight is None:
                class_weight = [1] * C  # [1/C]*C

            logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit, 1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)
        batch_loss = - class_weight * (torch.pow((1 - prob), self.gamma)) * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss


def sigmoid_cross_entropy_loss(prediction, label):
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    # print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction.float(), label.float(), weight=mask, reduce=False)
    return torch.sum(cost)


class RobustFocalLoss2d(nn.Module):
    # assume top 10% is outliers
    def __init__(self, gamma=2, size_average=True):
        super(RobustFocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()

        if type == 'sigmoid':
            if class_weight is None:
                class_weight = [1] * 2  # [0.5, 0.5]

            prob = F.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1 - prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif type == 'softmax':
            B, C, H, W = logit.size()
            if class_weight is None:
                class_weight = [1] * C  # [1/C]*C

            logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit, 1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)

        focus = torch.pow((1 - prob), self.gamma)
        # focus = torch.where(focus < 2.0, focus, torch.zeros(prob.size()).cuda())
        focus = torch.clamp(focus, 0, 2)

        batch_loss = - class_weight * focus * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss


##  http://geek.csdn.net/news/detail/126833
class PseudoBCELoss2d(nn.Module):
    def __init__(self):
        super(PseudoBCELoss2d, self).__init__()

    def forward(self, logit, truth):
        z = logit.view(-1)
        t = truth.view(-1)
        loss = z.clamp(min=0) - z * t + torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum() / len(t)  # w.sum()
        return loss


def cross_entropy_loss(prediction, label):
    # print (label,label.max(),label.min())
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    # print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduce=True)
    return cost  # torch.sum(cost)
