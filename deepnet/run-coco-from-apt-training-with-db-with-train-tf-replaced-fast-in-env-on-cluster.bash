#! /usr/bin/bash

SCRIPT_FILE_NAME="$0"
SCRIPT_FILE_PATH=$(realpath "${SCRIPT_FILE_NAME}")
SCRIPT_FOLDER_PATH=$(dirname "${SCRIPT_FILE_PATH}")

ARG_COUNT=$#
if [ $ARG_COUNT -lt 2 ]
then
    RUN_INDEX="00"
else
    RUN_INDEX="$1"
fi
cd "${SCRIPT_FOLDER_PATH}"
SCRIPT_NAME=run-coco-from-apt-training-with-db-with-train-tf-replaced-fast-in-env.bash
BASE_NAME="${SCRIPT_NAME%.*}"
bsub -P branson -n 12 -gpu "num=1" -q gpu_a100 -oo "${BASE_NAME}-run-${RUN_INDEX}.out.txt" -eo "${BASE_NAME}-run-${RUN_INDEX}.err.txt" "./${SCRIPT_NAME}"
