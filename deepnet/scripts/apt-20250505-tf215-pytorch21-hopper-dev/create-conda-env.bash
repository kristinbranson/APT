#! /bin/bash
PIP_NO_DEPS=1 CONDA_OVERRIDE_CUDA="12.4" conda env create -f environment.yml
