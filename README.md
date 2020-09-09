# Code for Making an Invisibility Cloak: Real World Adversarial Attacks on Object Detectors
This is the code for [Making an Invisibility Cloak: Real World Adversarial](https://arxiv.org/abs/1910.14667), which studies the transferability of adversarial attacks on object detectors. 

It is built upon [Maskrcnn-Benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).
Please refer to [Maskrcnn-Benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) for installation instructions. The patch transformation code is modified upon [Adversarial YOLO](https://gitlab.com/EAVISE/adversarial-yolo).

### For training adversarial patches on a single model
See ./train.sh
You need to download pretrained weights to begin with.

### For training an ensemble of models.
tools/train_net_ensemble.py for training an ensemble of models.

### For evaluating models
./eval_patch.sh
