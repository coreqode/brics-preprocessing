#!/bin/bash

## To Optimize Parameters
python src/object_segment.py \
    --root_dir "./assets/" \
    --network "/hdd_data/common/BRICS/instant-ngp/configs/nerf/segment.json" \
    --name "yellow-camero-random_seg" \
    --static \
    --roi 0.5 0.45 0.5 \
    --n_steps 10000 \
    --downscale_factor 0.65 \
    --mask_thresh 0.2 \
    --con_comp False \
    --use_alpha False \
    --dilate_mask False \
    --optimize_params 

