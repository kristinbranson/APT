This is the "development" version of the
apt-20250505-tf215-pytorch21-hopper environment.  The environment.yml
was created by some amount of trial-and-error to get a working conda
environment for APT that would work on hopper architectures GPUs.

Once it was all set and working, the final locked 




It was created from
the apt-20250505-tf215-pytorch21-hopper-dev environment on 2025-05-05.
The idea is to fix the exact versions of all the required packages, so
that it can be recreated in future.

It was created roughly like so:

cd deepnet/scripts/apt-20250505-tf215-pytorch21-hopper-dev
./create-conda-env.bash
conda activate apt-20250505-tf215-pytorch21-hopper-dev
mkdir ../apt-20250505-tf215-pytorch21-hopper
conda env export > ../apt-20250505-tf215-pytorch21-hopper/environment.yaml
conda deactivate

ALT
2025-05-05

