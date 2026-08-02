#! /bin/bash

# Create a conda environment from the environment.yaml (or environment.yml)
# file in the given directory.  Run this from deepnet/scripts.
#
# Usage:
#   ./create-conda-env.bash <env-dir>
#
# Example:
#   ./create-conda-env.bash apt-20260730-tf215-pytorch21-hopper-dev

set -e

envdir="$1"
if [ -z "$envdir" ] ; then
  echo "Usage: $0 <env-dir>" 1>&2
  exit 1
fi

envfile="$envdir/environment.yaml"
if [ ! -f "$envfile" ] ; then
  envfile="$envdir/environment.yml"
fi
if [ ! -f "$envfile" ] ; then
  echo "No environment.yaml or environment.yml found in $envdir" 1>&2
  exit 1
fi

# Use the cuda-version pinned in the environment file as the CUDA override,
# defaulting to 12.8 if none is present.
cuda="$(grep -oE 'cuda-version[[:space:]]*[=<>!]+[[:space:]]*[0-9.]+' "$envfile" | grep -oE '[0-9.]+' | head -n 1)"
if [ -z "$cuda" ] ; then
  cuda="12.8"
fi

# CONDA_CHANNEL_PRIORITY=strict keeps the solver from exploring lower-priority
# channels, which makes the solve substantially faster (and more predictable).
PIP_NO_DEPS=1 CONDA_OVERRIDE_CUDA="$cuda" CONDA_CHANNEL_PRIORITY=strict \
  conda env create -f "$envfile"
