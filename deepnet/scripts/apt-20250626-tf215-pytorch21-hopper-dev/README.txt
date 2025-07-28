This is the "development" version of the
apt-20250626-tf215-pytorch21-hopper environment.  The environment.yml
was created by some amount of trial-and-error to get a working conda
environment for APT that would work on hopper architectures GPUs, and
that included mmpretrain and git.

Once it is all set and working, the final locked version will be created
by freezing all the versions and dropping the -dev suffix, roughly
like so:

cd deepnet/scripts/apt-20250626-tf215-pytorch21-hopper-dev
./create-conda-env.bash
conda activate apt-20250626-tf215-pytorch21-hopper-dev
mkdir ../apt-20250626-tf215-pytorch21-hopper
conda env export > ../apt-20250626-tf215-pytorch21-hopper/environment.yaml
conda deactivate


ALT
2025-06-26

