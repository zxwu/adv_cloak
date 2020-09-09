# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug

from patch.utils import batchify_labels
from patch.patch_gen import *
from maskrcnn_benchmark.structures.image_list import ImageList

import torchvision.utils as tvutils

def compute_on_dataset(model, data_loader, device, timer=None, patched=False, patchfile="", cls_id=1):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    patch_applier = PatchApplier().to(device)
    patch_transformer = PatchTransformer(augment=False).to(device)
    mean = [102.9801, 115.9465, 122.7717]
    std = [1.0, 1.0, 1.0]
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)

    if patched:
        if 'rand' in patchfile:
            adv_patch_cpu = torch.rand(3, 250, 150)
        elif 'white' in patchfile:
            adv_patch_cpu = torch.zeros(3, 250, 150).fill_(0.5)
        elif 'clean' in patchfile:
            adv_patch_cpu = torch.rand(3, 250, 150)
            patched = False
        else:
            adv_patch_cpu = torch.load(patchfile)
            if 'submean' in patchfile:
                adv_patch_cpu = adv_patch_cpu['model']
            else:
                adv_patch_cpu = adv_patch_cpu.detach().cpu()
                adv_patch_cpu = torch.flip(adv_patch_cpu, [0])

        adv_patch = adv_patch_cpu.to(device)
    
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        imgs = images.tensors.to(device)



        mean = mean.to(device)
        std = std.to(device)
        imgs.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        
        images = ImageList(imgs, images.image_sizes)


        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))

                if patched:
                    imgs = images.tensors.to(device)
                    imgs.mul_(std[None, :, None, None]).add_(mean[None, :, None, None])
                    
                    bsz, _, height, width = imgs.shape
                    output = [o.to(cpu_device) for o in output]
                    using_gt = True
                    if using_gt:
                        output = targets
                    lab_batch = batchify_labels(bsz, (height, width), output, image_ids, testing=True, cls_label=cls_id)

                    adv_batch = patch_transformer(adv_patch.to(device), lab_batch.to(device), height, width, \
                                rand_loc=True, scale_factor=0.22, cls_label=cls_id)

                    adv_batch = adv_batch.mul_(255)
                
                    imgs = imgs.to(device)
                    imgs = patch_applier(imgs.to(device), adv_batch.to(device))

                    imgs.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
                    images = ImageList(imgs, images.image_sizes)

                    output = model(images.to(device))
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        patched=False,
        patchfile="",
        file_name="",
        cls_id=1,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, \
        inference_timer, patched, patchfile, cls_id)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        file_name=file_name
    )

    # if "inria" in dataset_name:
    extra_args.update({"cls_id": cls_id})
 
    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
