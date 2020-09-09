import torch
import torch.nn as nn
from yolov2.loss import objectness_darknet
import torchvision.utils as tvutils
import torch.nn.functional as F

class Ensemble(nn.Module):
    def __init__(self, darknet, rcnn):
        super(Ensemble, self).__init__()
        self.darknet = darknet
        self.rcnn = rcnn
        self.mean = [102.9801, 115.9465, 122.7717]
        self.std = [1.0, 1.0, 1.0]
        self.mean = torch.as_tensor(self.mean,)
        self.std = torch.as_tensor(self.std,)
    
    def forward(self, batched_inputs, targets, logits_only=True, adv_patch=None):
        device = batched_inputs.tensors.device
        imgs = batched_inputs.tensors.clone()
        loss_dict = self.rcnn(batched_inputs, targets, logits_only=True, adv_patch=adv_patch)

        
        device = imgs.device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        
        # denormalize and flip as inputs for yolo
        imgs.mul_(self.std[None, :, None, None]).add_(self.mean[None, :, None, None])
        imgs = imgs.div_(255)
        imgs = torch.flip(imgs, [1])

        # workaround to get outputs from yolo
        _, _, height, width = imgs.size()
        if height < width:
            padding = int((width - height)/2)
            imgs = F.pad(imgs, (0, 0, padding, width-height-padding), value=0.5)
        else:
            padding = int((height - width)/2)
            imgs = F.pad(imgs, (padding, height-width-padding, 0, 0), value=0.5)

        imgs = F.interpolate(imgs, (416, 416))

        output_darknet = self.darknet(imgs)
        obj_darknet = objectness_darknet(output_darknet)
        loss_dict.update({
            "loss_objectness_darknet": 10*obj_darknet.mean(),
            })

        # print(loss_dict)

        return loss_dict

class Ensemble3(nn.Module):
    def __init__(self, darknet, rcnn, darknetv3):
        super(Ensemble3, self).__init__()
        self.darknet = darknet
        self.rcnn = rcnn
        self.darknetv3 = darknetv3

        self.mean = [102.9801, 115.9465, 122.7717]
        self.std = [1.0, 1.0, 1.0]
        self.mean = torch.as_tensor(self.mean,)
        self.std = torch.as_tensor(self.std,)
    
    def forward(self, batched_inputs, targets, logits_only=True, adv_patch=None):
        device = batched_inputs.tensors.device
        imgs = batched_inputs.tensors.clone()
        loss_dict = self.rcnn(batched_inputs, targets, logits_only=True, adv_patch=adv_patch)

        
        device = imgs.device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        
        # denormalize and flip as inputs for yolo
        imgs.mul_(self.std[None, :, None, None]).add_(self.mean[None, :, None, None])
        imgs = imgs.div_(255)
        imgs = torch.flip(imgs, [1])

        # workaround to get outputs from yolo
        _, _, height, width = imgs.size()
        if height < width:
            padding = int((width - height)/2)
            imgs = F.pad(imgs, (0, 0, padding, width-height-padding), value=0.5)
        else:
            padding = int((height - width)/2)
            imgs = F.pad(imgs, (padding, height-width-padding, 0, 0), value=0.5)

        imgs = F.interpolate(imgs, (416, 416))

        output_darknet = self.darknet(imgs)
        obj_darknet = objectness_darknet(output_darknet)
      

        _, output_darknetv3 = self.darknetv3(imgs)
        obj_darknetv3 = objectness_darknet(output_darknetv3)


        loss_dict.update({
        "loss_obj_d2": 10*obj_darknet.mean(),
        "loss_obj_d3": 10*obj_darknetv3.mean(), 
        })

        return loss_dict
