#!/bin/bash

# ## To run segmentation
python src/object_segment.py \
    --root_dir "./assets" \
    --network "/hdd_data/common/BRICS/instant-ngp/configs/nerf/segment.json" \
    --name "couch09-canonical" \
    --static \
    --roi 0.5 0.45 0.5 \
    --n_steps 5000 3000 3000 3000 \
    --downscale_factor 0.65 0.65 0.65 0.65\
    --mask_thresh 0.8 0.8 0.6 0.6 \
    --con_comp True True True False \
    --use_alpha False False False True \
    --dilate_mask False False False False \
    --save_segmented_images
