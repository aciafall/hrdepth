import torch
import torch.nn as nn
from itertools import combinations

class Loss(nn.Module):

    def __init__(self, ratio=1):
        super().__init__()
        self.ratio = ratio

    def forward(self, d_pred, bboxes):
        """
        Calculates loss.

        """
        losses = {}
        loss_acc, loss_std = torch.zeros(1).cuda(), torch.zeros(1).cuda()
        depth = d_pred[0][0]
        depth = torch.div(depth-depth.min(), depth.max() - depth.min())
        us = []
        t = len(bboxes)
        tmp = 1/t

        for i in range(t):
            bbox = bboxes[i].int()
            depth_box = depth[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            us.append(torch.mean(depth_box))
            
            depth_box = (depth_box - torch.min(depth_box)) / (torch.max(depth_box) - torch.min(depth_box))
            loss_std += torch.std(depth_box)
        for i, j in combinations(range(len(bboxes)), 2):
            if us[i] + (j-i)*tmp < us[j]:
                loss_acc += torch.zeros(1).cuda()
            else:
                loss_acc += us[i] - us[j] + torch.tensor((j-i)* tmp).cuda()

        # Calculate loss
        loss = loss_acc * self.ratio + loss_std

        losses['loss_acc'] = loss_acc
        losses['loss_std'] = loss_std
        losses['loss'] = loss

        return losses
