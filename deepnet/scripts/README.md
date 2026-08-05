# Building the APT conda / Docker / Apptainer environments/images

APT runs its deep-learning backends out of a matched set of images: a conda
environment (conda backend), a Docker image (docker backend), and an Apptainer
`.sif` image (bsub/cluster backend).  Each such set is built from a single conda
environment specification and lives in its own directory here, named like

```
apt-20260801-tf215-pytorch21-hopper
```

The name encodes the build date and the major dependency/architecture targets.

Building a new complement happens in two stages: an interactive **development**
stage where you get a working environment, and an automated **production**
stage that freezes it and produces the images.


## 1. Development stage (manual, iterative)

Create a directory whose name ends in `-dev`, e.g.

```
apt-20260801-tf215-pytorch21-hopper-dev
```

and put an `environment.yaml` in it.  This is the *unpinned* spec: it lists the
direct dependencies with loose version constraints (see an existing `-dev`
directory's `environment.yml`/`environment.yaml` for a starting point).  The
`name:` inside the file should match the directory name.

Iterate on `environment.yaml` until the environment builds.  From this
directory (`deepnet/scripts`), run:

```
./create-conda-env.bash apt-20260801-tf215-pytorch21-hopper-dev
```

`create-conda-env.bash` reads the CUDA version from the environment file's
`cuda-version` pin and passes it as `CONDA_OVERRIDE_CUDA`.  Once the command
succeeds, confirm the environment actually works for APT by running the same
smoke test the production stage uses (`test.sh`, which runs the pose-estimation
demo and compares its output to `demo-output-target.jpg`).  From this directory
(`deepnet/scripts`), run it in the dev environment with `conda run`:

```
conda run --no-capture-output --name apt-20260801-tf215-pytorch21-hopper-dev bash test.sh
```

`--no-capture-output` lets the demo's progress stream to your terminal, and
running from `deepnet/scripts` is required so `test.sh` can find the demo assets
and `compare-to-target.bash`.  The test ends with a `PASS:`/`FAIL:` line; it
needs a working GPU and ImageMagick (`compare`/`identify`) on the host.  Iterate
on the environment until it passes before moving to the production stage.


## 2. Production stage (automated)

Once the dev environment builds and works, freeze it and build the images with
the Python script:

```
./create-production-complement.py apt-20260801-tf215-pytorch21-hopper-dev
```

The script (Python 3.6, standard library only) does the following:

1. Checks that `conda`, `docker`, `apptainer`, and ImageMagick
   (`compare`/`identify`, used by the smoke test) are installed in standard
   locations, erroring out early if any is missing.
2. Smoke-tests the dev environment first (before any expensive build step), so
   a broken environment fails fast.
3. Creates the production directory `apt-20260801-tf215-pytorch21-hopper` (the
   `-dev` suffix dropped) and writes into it just two files:
   - `environment.yaml` — a frozen/pinned `conda env export` of the dev
     environment, with the `name:` de-`-dev`'d and the machine-specific
     `prefix:` line removed;
   - `Dockerfile` — templated with the production name and CUDA override.
4. Creates the production conda environment locally from the pinned
   `environment.yaml`, and smoke-tests it.
5. Builds the Docker image `bransonlabapt/apt_docker:apt-20260801-tf215-pytorch21-hopper`
   and smoke-tests it.  (It is *not* pushed to Docker Hub here — see
   "Publishing the images" below.)
6. Builds the Apptainer image `apt-20260801-tf215-pytorch21-hopper.sif` from the
   just-built local Docker image (via the `docker-daemon://` transport, so no
   Docker Hub round-trip), and smoke-tests it.

Each smoke test runs the pose-estimation demo (`test.sh` / `image_demo.py`) in
the environment/image under test and compares the result to
`demo-output-target.jpg` using perceptual similarity (`compare-to-target.bash`,
which shells out to ImageMagick — a host tool, so the check does not depend on
the environment being tested).  A failing test aborts the run.  The Docker and
Apptainer tests generate their output inside the container (with the scripts
directory bind-mounted) and compare on the host, since the images themselves do
not include ImageMagick.  The demo runs on the GPU (`--gpus all` for Docker,
`--nv` for Apptainer), so a working GPU is required.

Pass `--force` to overwrite an existing production directory.

The script builds everything locally and does not push to Docker Hub, so it
needs no registry credentials to run.  (Publishing is a separate step — see
below.)

The completed `apt-20260801-tf215-pytorch21-hopper` directory is an example of
what a finished production directory looks like.  Only `environment.yaml` and
`Dockerfile` are committed to the repo; the built `.sif` also lands in this
directory after a run, but is git-ignored (the repo-root `.gitignore` ignores
`*.sif`).


## 3. Publishing the images

`create-production-complement.py` builds and tests everything locally but does
not deploy anything.  When you are ready to share the images with the world, run
`push-docker-and-apptainer-images.bash` from the production directory:

```
cd apt-20260801-tf215-pytorch21-hopper
../push-docker-and-apptainer-images.bash
```

It does two things:

- pushes the Docker image to Docker Hub (needed by the docker backend, and by
  the AWS/remote backends); and
- copies the Apptainer `.sif` into the shared image directory
  (`/groups/branson/bransonlab/apt/sif/`) that the bsub/cluster backend loads
  from.

You must be logged in to Docker Hub (`docker login`) with push access to the
`bransonlabapt` organization before running it.  The `push-*` scripts are
hard-coded to a single image name, so edit them to match the version you are
releasing.


## Using a new complement in APT

Each APT project records which conda/Docker/Apptainer images it uses, and an
existing project keeps using what it was built with until changed.  Point the
APT frontend defaults (and/or the per-project backend settings in the UI) at
the new names to adopt a freshly built complement:

- conda environment: `apt-20260801-tf215-pytorch21-hopper`
- Docker image: `docker://bransonlabapt/apt_docker:apt-20260801-tf215-pytorch21-hopper`
- Apptainer image: `.../sif/apt-20260801-tf215-pytorch21-hopper.sif`
