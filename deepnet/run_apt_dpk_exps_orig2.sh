#!/bin/bash
. /opt/venv/bin/activate
cd /groups/branson/home/leea30/git/apt.aldl/deepnet
# numCores2use=7 
export PYTHONPATH="/groups/branson/home/leea30/git/dpk:/groups/branson/home/leea30/git/imgaug" 

python apt_dpk_exps.py "$@"
# "--expname $1 --exptype $2


