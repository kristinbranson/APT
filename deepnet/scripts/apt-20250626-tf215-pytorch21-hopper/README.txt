This is the final "locked" version of the
apt-20250626-tf215-pytorch21-hopper environment.  It was created from
the apt-20250626-tf215-pytorch21-hopper-dev environment on 2025-06-26.
The idea is to fix the exact versions of all the required packages, so
that it can be recreated in future.

It was created roughly like so:

cd deepnet/scripts/apt-20250626-tf215-pytorch21-hopper-dev
./create-conda-env.bash
conda activate apt-20250626-tf215-pytorch21-hopper-dev
mkdir ../apt-20250626-tf215-pytorch21-hopper
conda env export > ../apt-20250626-tf215-pytorch21-hopper/environment.yaml
conda deactivate

ALT
2025-06-26
