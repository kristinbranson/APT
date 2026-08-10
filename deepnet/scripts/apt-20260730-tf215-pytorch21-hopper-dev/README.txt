This is the "development" version of the
apt-20260730-tf215-pytorch21-hopper environment.  The environment.yml
was created by starting with the environment.yml for
apt-20250626-tf215-pytorch21-hopper-dev, then adding a couple of
packages that are needed for id-linking.

Once it is all set and working, the final locked version will be created
by freezing all the versions and dropping the -dev suffix, roughly
like so:

cd deepnet/scripts/apt-20260730-tf215-pytorch21-hopper-dev
./create-conda-env.bash
conda activate apt-20260730-tf215-pytorch21-hopper-dev
mkdir ../apt-20260730-tf215-pytorch21-hopper
conda env export > ../apt-20260730-tf215-pytorch21-hopper/environment.yaml
conda deactivate


ALT
2026-07-30

