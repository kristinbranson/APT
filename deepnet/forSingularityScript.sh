#!/bin/bash
. /opt/venv/bin/activate
cd /groups/branson/home/kabram/PycharmProjects/poseTF
if nvidia-smi | grep -q 'No devices were found'; then
    { echo "No GPU devices were found. quitting"; exit 1; }
fi
numCores2use=1
python forSingularity.py
