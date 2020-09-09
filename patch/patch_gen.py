import fnmatch
import math
import os
import sys
import time
from operator import itemgetter

import gc
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random
import kornia
import torchvision.utils as tvutils

from .utils import MedianPool2d
import thinplate as tps



class MinimizeObjectness_twostage(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from two-stage object detector.
    """

    def __init__(self, cls_id, num_cls):
        super(MinimizeObjectness_twostage, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls

    def forward(self, pred_objectness_logits):
        # pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (B, A, Hi, Wi).
        batch_size = pred_objectness_logits[0].size(0)
        # Convert to a list of objness scores of different scales [B, A*Hi*Wi]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]

        pred_objectness = [(obj_level.view(batch_size, -1) + 1).clamp(min=0) for obj_level in pred_objectness_logits]
        pred_objectness = [w * obj_level.mean(1).unsqueeze(1) for obj_level, w in zip(pred_objectness, weights)]
        pred_objectness = torch.cat(pred_objectness, dim=1)
        det_loss = pred_objectness ** 2
        det_loss = det_loss.mean(1)

        return det_loss, None

class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.
    Total variation loss
    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        tvcomp1 = adv_patch[:, :, 1:] - adv_patch[:, :, :-1]
        tvcomp1 = (tvcomp1 * tvcomp1 + .01).sqrt().sum()
        tvcomp2 = adv_patch[:, 1:, :] - adv_patch[:, :-1, :]
        tvcomp2 = (tvcomp2 * tvcomp2 + .01).sqrt().sum()
        tv = tvcomp1 + tvcomp2
        return tv


class SmoothTV(nn.Module):
    """TotalVariation: calculates the total variation of a patch.
    Smooth total variation loss
    """

    def __init__(self):
        super(SmoothTV, self).__init__()

    def forward(self, adv_patch):
        tvcomp1 = adv_patch[:, :, 1:] - adv_patch[:, :, :-1]
        tvcomp1 = (tvcomp1 * tvcomp1).sum()
        tvcomp2 = adv_patch[:, 1:, :] - adv_patch[:, :-1, :]
        tvcomp2 = (tvcomp2 * tvcomp2).sum()
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches
    Generate patches with different augmentations
    """

    def __init__(self, augment=True):
        super(PatchTransformer, self).__init__()
        self.augment = augment
        self.min_contrast = 0.8 if augment else 1
        self.max_contrast = 1.2 if augment else 1
        self.min_brightness = -0.1 if augment else 0
        self.max_brightness = 0.1 if augment else 0
        self.noise_factor = 0.0
        self.minangle = -5 / 180 * math.pi if augment else 0
        self.maxangle = 5 / 180 * math.pi if augment else 0
        self.medianpooler = MedianPool2d(7, same=True)
        self.distortion_max = 0.1 if augment else 0
        self.sliding = -0.05 if augment else 0

    def get_tps_thetas(self, num_images):
        c_src = np.array([
            [0.0, 0.0],
            [1., 0],
            [1, 1],
            [0, 1],
            [0.2, 0.3],
            [0.6, 0.7],
        ])

        theta_list = []
        dst_list = []
        for i in range(num_images):
            c_dst = c_src + np.random.uniform(-1, 1, (6, 2)) / 20

            theta = tps.tps_theta_from_points(c_src, c_dst)
            theta_list.append(torch.from_numpy(theta).unsqueeze(0))
            dst_list.append(torch.from_numpy(c_dst).unsqueeze(0))

        theta = torch.cat(theta_list, dim=0).float()
        dst = torch.cat(dst_list, dim=0).float()
        return theta, dst

    def get_perspective_params(self, width, height, distortion_scale):
        half_height = int(height / 2)
        half_width = int(width / 2)
        topleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(0, int(distortion_scale * half_height)))
        topright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(0, int(distortion_scale * half_height)))
        botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        botleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def forward(self, adv_patch, lab_batch, height, width, do_rotate=True, rand_loc=False, scale_factor=0.25, cls_label=1):

        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        patch_size = adv_patch.size(-1)

        # Determine size of padding
        pad_width = (width - adv_patch.size(-1)) / 2
        pad_height = (height - adv_patch.size(-2)) / 2

        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)
        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))
        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise
        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # perspective transformations
        dis_scale = (lab_batch.size(0) * lab_batch.size(1))
        distortion = torch.empty(dis_scale).uniform_(0, self.distortion_max)

        adv_height = adv_batch.size(-1)
        adv_width = adv_batch.size(-2)
        
        # tps transformation
        if self.augment:
            theta, dst = self.get_tps_thetas(dis_scale)
            img = adv_batch.clone()
            img = img.view(-1, 3, adv_width, adv_height)

            grid = tps.torch.tps_grid(theta.cuda(), dst.cuda(), (img.size(0), 1, adv_width, adv_height))
            adv_batch = F.grid_sample(img, grid.cuda(), padding_mode='border')
            adv_batch = adv_batch.view(lab_batch.size(0), lab_batch.size(1), 3, adv_width, adv_height)


        # perpective transformations with random distortion scales
        start_end = [torch.tensor(self.get_perspective_params(adv_width, adv_height, x), \
                                  dtype=torch.float).unsqueeze(0) for x in distortion]
        start_end = torch.cat(start_end, 0)
        start_points = start_end[:, 0, :, :].squeeze()
        end_points = start_end[:, 1, :, :].squeeze()

        if dis_scale == 1:
            start_points = start_points.unsqueeze(0)
            end_points = end_points.unsqueeze(0)
        try:
            M = kornia.get_perspective_transform(start_points, end_points)

            img = adv_batch.clone()
            img = img.view(-1, 3, adv_width, adv_height)
            adv_batch = kornia.warp_perspective(img, M.cuda(), dsize=(adv_width, adv_height))
            adv_batch = adv_batch.view(lab_batch.size(0), lab_batch.size(1), 3, adv_width, adv_height)
        except:
            print('hihi')

        cls_ids = torch.narrow(lab_batch, 2, 0, 1)
        cls_mask = 1.0 * (cls_ids.expand(-1, -1, 3) != cls_label )
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask.float()

        mypad = nn.ConstantPad2d((int(pad_width + 0.5), int(pad_width), int(pad_height + 0.5), int(pad_height)), 0)
        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)


        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0) * lab_batch.size(1))
        if do_rotate:
            angle = torch.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            angle = torch.FloatTensor(anglesize).fill_(0)
        angle = angle.cuda()

        # Resizes and rotates
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        lab_batch_scaled = torch.zeros(lab_batch.size())
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * width
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * height
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * width
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * height

        # roughly estimate the size and compute the scale
        target_size = scale_factor * torch.sqrt((lab_batch_scaled[:, :, 3]) ** 2 + (lab_batch_scaled[:, :, 4]) ** 2)

        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))  # (batch_size, num_objects)
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        # target_y = target_y - 0.0

        # shift a bit from the center
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))

        off_y = (torch.FloatTensor(targetoff_y.size()).uniform_(self.sliding, 0))
        off_y = targetoff_y * off_y.cuda()
        target_y = target_y + off_y

        scale = target_size / patch_size
        scale = scale.view(anglesize)
        scale = scale.cuda()

        s = adv_batch.size()
        # print(adv_batch.size())

        adv_batch = adv_batch.view(-1, 3, height, width)
        msk_batch = msk_batch.view_as(adv_batch)

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation,rescale matrix
        theta = torch.FloatTensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale
        theta = theta.cuda()

        grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch = F.grid_sample(adv_batch, grid)
        msk_batch = F.grid_sample(msk_batch, grid)

        adv_batch = adv_batch.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch = adv_batch * msk_batch

        return adv_batch
        


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1) 

        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch
