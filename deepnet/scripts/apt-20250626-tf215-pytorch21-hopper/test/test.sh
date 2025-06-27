#! /bin/bash

python image_demo.py \
    demo-input.jpg \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --out-file demo-output.jpg \
    --draw-heatmap
