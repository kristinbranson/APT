#! /bin/bash
PIP_NO_DEPS=1 CONDA_OVERRIDE_CUDA="12.8" conda env create -f environment.yaml
