# Building the APT conda / Docker / Apptainer images

APT runs its deep-learning backends out of a matched set of images: a conda
environment (conda backend), a Docker image (docker backend), and an Apptainer
`.sif` image (bsub/cluster backend).  Each such set — a "complement" — is built
from a single conda environment specification and is named like

```
apt-20260801-tf215-pytorch21-hopper
```

The name encodes the build date and the major dependency/architecture targets.
That whole name is the *tag* you give the build script; it becomes the name of
the production conda environment, and the dev environment is `<tag>-dev`.

`create_production_images.py` builds (and optionally publishes) a whole
complement in one command, resumably: run it, fix whatever breaks, run it again.


## Building a complement

From this directory (`deepnet/scripts`), run the script with the tag (the tag is
the full name, including the date):

```
./create_production_images.py apt-20260801-tf215-pytorch21-hopper
```

This builds everything for `apt-20260801-tf215-pytorch21-hopper` locally but does
not publish it.  It needs `conda`, `docker`, and `apptainer`, and a working GPU
for the smoke tests.

The script runs these stages in order, **skipping any whose output already
exists**:

1.  Create the dev folder `<tag>-dev`.
2.  Seed `<tag>-dev/environment.yaml` from `dev-environment-template.yaml`.
    When this file is first created, the script **stops here** so you can review
    and edit it before the build (see "Editing the environment and iterating"
    below); re-run to continue.
3.  Build the dev conda environment, then smoke-test it.
4.  Create the prod folder `<tag>`.
5.  Freeze the dev environment into `<tag>/environment.yaml` — a
    fully-pinned `conda env export` of the (working) dev environment, with the
    `name:` de-`-dev`'d and the machine-specific `prefix:` line removed.
6.  Build the prod conda environment, then smoke-test it.
7.  Write `<tag>/Dockerfile` (templated with the name and CUDA version).
8.  Build the Docker image, then smoke-test it.
9.  Build the Apptainer `.sif` from the local Docker image (via the
    `docker-daemon://` transport, so no Docker Hub round-trip), then smoke-test
    it.
10. Push the Docker image to Docker Hub. &nbsp;&nbsp;*(only with `--publish`)*
11. Copy the `.sif` into `/groups/branson/bransonlab/apt/sif/`.
    &nbsp;&nbsp;*(only with `--publish`)*

If any stage errors, the whole script errors.

Only `environment.yaml` in the dev directory and `environment.yaml`
and `Dockerfile` in the prod directory should normally be committed to
the repo; the built `.sif` also lands there but is git-ignored (the repo-root
`.gitignore` ignores `*.sif`).


## The smoke test

Each conda environment and image is smoke-tested with `test_pose_estimation.py`,
which runs the mmpose pose-estimation demo and compares the result to
`demo-output-target.jpg`.  The comparison is a structural-similarity (SSIM)
score computed in Python with scikit-image, so it needs no tools beyond the
environment under test — the whole test, compute and compare, runs inside that
environment or image.  It needs a GPU (`--gpus all` for Docker, `--nv` for
Apptainer).

A smoke test runs only when its environment/image is (re)built in a given run; an
environment/image that already exists is trusted and not re-tested.


## Editing the environment and iterating

New dev environments are seeded from `dev-environment-template.yaml` — the
*unpinned* spec listing the direct dependencies with loose version constraints.
To change what new environments get, edit that file.

The development stage is inherently iterative: you usually want to adjust the
seeded dependencies, and a fresh set may not solve, or may fail the smoke test,
on the first try.  Because the script is resumable, the loop is:

1. Run `./create_production_images.py <tag>`.  The first time, it seeds
   `<tag>-dev/environment.yaml` from the template and **stops**, so you can
   review and edit it.
2. Edit `<tag>-dev/environment.yaml` (a later run will **not** overwrite it), then
   run the script again to build and test the environment.
3. If a build or test fails, fix the spec and re-run; the script skips the stages
   that already succeeded and retries the rest.  If the conda environment built
   but is broken, remove it with `conda env remove --name <tag>-dev` so the next
   run rebuilds it.

To iterate on just the conda environments without building the Docker and
Apptainer images, pass `--conda-only`:

```
./create_production_images.py apt-20260801-tf215-pytorch21-hopper --conda-only
```

`--conda-only` builds and tests only the dev and prod conda environments
(stages 1–6).  It cannot be combined with `--publish`.


## Publishing

By default nothing is published.  When you are ready to share the images, pass
`--publish`:

```
./create_production_images.py apt-20260801-tf215-pytorch21-hopper --publish
```

This additionally:

- pushes the Docker image to Docker Hub as
  `bransonlabapt/apt_docker:<tag>` (needed by the docker backend and the
  AWS/remote backends); and
- copies the Apptainer `.sif` into `/groups/branson/bransonlab/apt/sif/`, where
  the bsub/cluster backend loads it from.

You must be logged in to Docker Hub (`docker login`) with push access to the
`bransonlabapt` organization before publishing.  Both publish steps are
idempotent: the push is skipped if the tag already exists on Docker Hub, and the
copy is skipped if the `.sif` is already in the shared directory.


## Using a new complement in APT

Each APT project records which conda/Docker/Apptainer images it uses, and an
existing project keeps using what it was built with until changed.  Point the
APT frontend defaults (and/or the per-project backend settings in the UI) at the
new names to adopt a freshly built complement:

- conda environment: `apt-20260801-tf215-pytorch21-hopper`
- Docker image: `docker://bransonlabapt/apt_docker:apt-20260801-tf215-pytorch21-hopper`
- Apptainer image: `.../sif/apt-20260801-tf215-pytorch21-hopper.sif`
