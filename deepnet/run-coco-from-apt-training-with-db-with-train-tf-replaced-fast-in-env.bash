#! /usr/bin/bash

SCRIPT_FILE_NAME="$0"
SCRIPT_FILE_PATH=$(realpath "${SCRIPT_FILE_NAME}")
SCRIPT_FOLDER_PATH=$(dirname "${SCRIPT_FILE_PATH}")

cd "${SCRIPT_FOLDER_PATH}"
/groups/scicompsoft/home/taylora/.miniconda3-blurgh/envs/apt_20230427_tf211_pytorch113_ampere/bin/python \
    ./run-coco-from-apt-training-with-db-with-train-tf-replaced-fast.py

