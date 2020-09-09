import glob
import os
import random
import shutil
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from . import torch_utils  # , google_utils
import math
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from tensorboardX import SummaryWriter
import warnings
matplotlib.rc('font', **{'size': 11})

# Set printoptions
torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5


def get_region_boxes_new(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
    #anchor_step = len(anchors)/num_anchors
    anchor_step = len(anchors)//num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (5+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)

    output = output.view(batch*num_anchors, 5+num_classes, h*w)
    output = output.transpose(0,1).contiguous()
    output = output.view(5+num_classes, batch*num_anchors*h*w)
    grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    output[0] = (torch.sigmoid(output[0]) + grid_x)/w
    output[1] = (torch.sigmoid(output[1]) + grid_y)/h

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
    output[2] = (torch.exp(output[2]) * anchor_w)/w
    output[3] = (torch.exp(output[3]) * anchor_h)/h

    output[4] = torch.sigmoid(output[4])
    output[5:] = torch.nn.functional.softmax(output[5:], dim=0)
    output = output.permute(1, 0)
    output = output.view(batch, num_anchors*h*w, 5+num_classes)

    return output

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)