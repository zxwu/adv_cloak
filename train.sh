# !/bin/bash

python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py \
--config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
OUTPUT_DIR "runs/submean_tps_2x_new_weights/" \
SOLVER.CHECKPOINT_PERIOD 10000 SOLVER.MAX_ITER 120000 \
SOLVER.STEPS "(60000, 100000)" SOLVER.BASE_LR 0.1
