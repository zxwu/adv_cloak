# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.structures.image_list import ImageList

from apex import amp
from patch.utils import batchify_labels
import torchvision.utils as tvutils


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    adv_patch_cpu,
    patch_transformer,
    patch_applier,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    mean = [102.9801, 115.9465, 122.7717]
    std = [1.0, 1.0, 1.0]
    mean = torch.as_tensor(mean,)
    std = torch.as_tensor(std)
    print(len(data_loader.dataset))
    
    for iteration, (images, targets, paths) in enumerate(data_loader, start_iter):
        
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        adv_patch = adv_patch_cpu.to(device)

        scheduler.step()
        imgs = images.tensors.to(device)

        bsz, _, height, width = imgs.shape

        lab_batch = batchify_labels(
            bsz, (height, width), 
            targets, paths, 
            cls_label=int(arguments["cls_id"]))
        
        adv_batch = patch_transformer(
            adv_patch.to(device),
            lab_batch.to(device), height, width, 
            rand_loc=True, scale_factor=0.22, 
            cls_label=int(arguments["cls_id"]))

        adv_batch = adv_batch.mul_(255)
       
        imgs = imgs.to(device)
        imgs = patch_applier(imgs.to(device), adv_batch.to(device))
        
        mean = mean.to(device)
        std = std.to(device)
        imgs.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        
        targets = [target.to(device) for target in targets]
        images = ImageList(imgs, images.image_sizes)

        loss_dict = model(images, targets, logits_only=True, adv_patch=adv_patch)
        losses = sum(loss for loss in loss_dict.values())
                    
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        # with amp.scale_loss(losses, optimizer) as scaled_losses:
        #     scaled_losses.backward()
        losses.backward()
        optimizer.step()
        adv_patch.clamp_(0, 1) 

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("patch_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("patch_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
