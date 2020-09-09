# !/bin/bash

NGPUS=$1 #8
WEIGHTS="weights/e2e_faster_rcnn_R_50_C4_1x.pth"

PFILE=(
"./runs/submean_tps_2x_new_weights_c4_250/patch_final.pth"
"./runs/submean_tps_2x_new_weights_c4_250_grey/patch_final.pth"
)

for pfile in "${PFILE[@]}"
do
echo $pfile
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file $CFGFILE --ckpt $WEIGHTS --patched \
--patchfile $pfile  --cls_id 1
done