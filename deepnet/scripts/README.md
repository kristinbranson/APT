# Conda

The conda environment file used to make the most recent conda
environment (as of this writing) is in
```
apt_20230427_tf211_pytorch113_ampere.env.yaml
```
and inside the file declares the environment's name to be
`apt_20230427_tf211_pytorch113_ampere`.  But the old conda environment
file is still around, in
```
old_apt_conda_environment.yml
```
and that one uses the name `APT`.  The APT front-end code defaults to
using the environment `apt_20230427_tf211_pytorch113_ampere` for new
projects.

Currently, each project stores what images it will use for the
Conda/Docker/Singularity backends, and these will not change for an
existing project until the user changes them manually.

The conda environment above is a 'pinned' environment, which specifies
the exact versions of all dependencies, both direct and transitive.

There is also a 'development' conda environment, specified in
```
apt_development.env.yaml
```
and named `apt_development`.  This dev environment specifies just the
major and minor versions of all the direct dependencies.  The idea
here is that you use the development environment when you're want to
update the dependencies, and then once you've got it all worked out
you use
```
conda env export
```
to produce (after some manual editing) the pinned version.

To create the conda environment from the environment file, run the
bash script
```
./make_conda_environment_apt_20230427_tf211_pytorch113_ampere.bash
```
or
```
./make_conda_environment_apt_development.bash
```
depending on which version you want.

## KB 20250706
These conda environments didn't work on my desktop, which has CUDA 12.2 installed on it.
I created a new conda environment.yml file which sets the cuda version (11.6 seemed to work):
```
apt_2507.env.yaml
```
which has all the dependencies except the MM dependencies. As I use openmim to install
mmcv-full, I made a script
```
make_conda_environment_apt_2507.bash
```
which creates the conda environment and then uses openmim to install mmcv-full, followed
by pip to install mmdet and mmpose.

# Docker

The docker image is specified by the dockerfile
```
apt_20230427_tf211_pytorch113_ampere.dockerfile
```
which itself uses the pinned
`apt_20230427_tf211_pytorch113_ampere.env.yaml` file to create a conda
environment inside the docker container.

You produce the docker image using the command
```
./make_docker_image_apt_20230427_tf211_pytorch113_ampere.bash
```

You can push the docker image to DockerHub using the command
```
./push_docker_image_apt_20230427_tf211_pytorch113_ampere.bash
```
An image produced in this way was pushed to DockerHub,so now the docker image lives at the URL:
```
docker://bransonlabapt/apt_docker:apt_20230427_tf211_pytorch113_ampere
```
and can be pulled using the command
```
./pull_docker_image_apt_20230427_tf211_pytorch113_ampere.bash
```

The frontend source code was changed to use this docker spec by
default.  (But you can change the docker spec in the UI now, and old
label files will keep using what they've been using.)



# Singularity

When using the JRC Cluster backend, the frontend source has been modified to default to using the Singularity image at
```
/groups/branson/bransonlab/apt/sif/apt_20230427_tf211_pytorch113_ampere.sif
```

It used to use the image at
```
/groups/branson/bransonlab/apt/sif/prod.sif
```
or the one at
```
/groups/branson/bransonlab/apt/sif/det.sif
```
depending on whether the network in use was one-phase or two-phase, and which model was being run.

You can re-make the singularity image using the command
```
./make_singularity_image_apt_20230427_tf211_pytorch113_ampere.bash
```

ALT
2023-05-25
