#!/bin/bash

conda env create -f apt_2507.env.yaml
conda activate apt_2507
mim install mmcv-full==1.7.0
pip install mmdet==2.28.2
pip install mmpose==0.29.0
