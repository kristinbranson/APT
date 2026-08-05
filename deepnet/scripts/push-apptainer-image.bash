#! /bin/bash

# Copy the Apptainer .sif into the shared image directory that APT's bsub/cluster
# backend loads it from.  Run this from the production directory.
cp apt-20260801-tf215-pytorch21-hopper.sif /groups/branson/bransonlab/apt/sif/
