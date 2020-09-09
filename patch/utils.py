from torch._six import container_abcs
from itertools import repeat
from torch.nn.modules.utils import _pair, _quadruple
import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_labels(tensor, max_length):
    pad_size = max_length - tensor.size(0) 
    if (pad_size > 0):
        padded_tensor = F.pad(tensor, (0, 0, 0, pad_size), value=81)
    elif pad_size ==0 :
        padded_tensor = tensor
    else:
        # print("HIHIHIH")
        perm = torch.randperm(tensor.size(0))
        idx = perm[:max_length]
        padded_tensor = tensor[idx, ...]
    return padded_tensor.unsqueeze(0)

def batchify_labels(batch_size, img_size, targets, paths=None, testing=False, cls_label=1):
    # inputs: [image, classes, x, y, h, w]
    # output: [batch, num_instances, x, y, h, w]
    labels_list = []
    labels_length = []

    for img_idx in range(batch_size):
        boxes = targets[img_idx].convert('xywh').bbox
        boxes[:, :2] += boxes[:, 2:] / 2
        boxes[:, [0, 2]] /=  img_size[1]
        boxes[:, [1, 3]] /= img_size[0]
        num_box = boxes.size(0)
        if num_box>0:
            labels = (targets[img_idx].get_field('labels')).view(num_box, -1)
            boxes = torch.cat((labels.float(), boxes), dim=1)
            boxes = boxes[boxes[:, 0]==cls_label, :]
        else:
            boxes = torch.empty([0, 5])
        labels_list.append(boxes)
        labels_length.append(boxes.size(0))

    if testing:
        limit_label = 200
    else:
        limit_label = 8
    # return torch.from_numpy(np.array(labels_length))
    max_length = min(max(labels_length), limit_label)
    max_length = max(1, max_length) # prevent empty
    labels_list = [pad_labels(lbl, max_length) for lbl in labels_list]
    labels_list = torch.cat(labels_list, dim=0)
    return labels_list


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()