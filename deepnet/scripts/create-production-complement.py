#! /usr/bin/env python3
"""Turn a working "dev" conda environment into a full production complement.

Given a development environment directory (one whose name ends in ``-dev`` and
that contains an ``environment.yaml`` you have already gotten to build with
``./create-conda-env.bash <dir>``), this script produces the matching
production complement:

  1. A production conda environment directory (the ``-dev`` suffix dropped),
     containing a frozen/pinned ``environment.yaml`` and a ``Dockerfile``
     templated with the production name.
  2. The production conda environment, actually created on this machine so the
     local conda backend can use it.
  3. A Docker image, built locally (not pushed; push it afterward with
     ``push-docker-image.bash`` -- see the README).
  4. An Apptainer ``.sif`` file, built from the local Docker image via the
     ``docker-daemon://`` transport (no registry round-trip).

At each stage the corresponding environment/image is smoke-tested by running
the pose-estimation demo (``test.sh`` / ``image_demo.py``) and comparing its
output to ``demo-output-target.jpg``; a failing test aborts the run.  The dev
environment is tested first, before any of the (expensive) build steps.

Typical usage, run from deepnet/scripts:

    ./create-conda-env.bash apt-20260801-tf215-pytorch21-hopper-dev  # iterate until this works
    conda activate apt-20260801-tf215-pytorch21-hopper-dev            # sanity-check the env
    ./create-production-complement.py apt-20260801-tf215-pytorch21-hopper-dev

This script uses only the Python 3.6 standard library.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys


# Docker Hub repository that production images are tagged under.
DOCKER_IMAGE_REPOSITORY = 'bransonlabapt/apt_docker'

# Files (all living in this scripts directory) that make up the smoke test: run
# the pose-estimation demo and compare its output to a known-good target.
DEMO_SCRIPT = 'image_demo.py'
DEMO_INPUT_IMAGE = 'demo-input.jpg'
DEMO_CONFIG = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
DEMO_CHECKPOINT = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
TEST_SCRIPT = 'test.sh'
COMPARE_SCRIPT = 'compare-to-target.bash'

# Directories to search (in addition to $PATH) when looking for a required
# executable in its "standard" location.
STANDARD_BIN_DIRECTORIES = (
  '/usr/bin',
  '/usr/local/bin',
  '/bin',
  '/sbin',
  '/usr/sbin',
  '/snap/bin',
)


def die(message):
  # Print an error message to stderr and exit with a nonzero status.
  sys.stderr.write('Error: %s\n' % message)
  sys.exit(1)


def announce(message):
  # Print a prominent progress banner to stdout.
  print('')
  print('==== %s ====' % message)


def find_executable_in_standard_locations(programName):
  # Return the absolute path to programName, searching $PATH then the standard
  # bin directories.  Return None if it is not found in any of them.
  pathFromPath = shutil.which(programName)
  if pathFromPath:
    return pathFromPath
  for directory in STANDARD_BIN_DIRECTORIES:
    candidatePath = os.path.join(directory, programName)
    if os.path.isfile(candidatePath) and os.access(candidatePath, os.X_OK):
      return candidatePath
  return None


def find_conda():
  # Return the absolute path to the conda executable, or None if not found.
  # Checks $CONDA_EXE and the usual miniforge/miniconda/anaconda locations in
  # addition to $PATH and the standard bin directories.
  condaExeFromEnvironment = os.environ.get('CONDA_EXE')
  if condaExeFromEnvironment and os.path.isfile(condaExeFromEnvironment):
    return condaExeFromEnvironment
  pathFromStandardLocations = find_executable_in_standard_locations('conda')
  if pathFromStandardLocations:
    return pathFromStandardLocations
  homeDirectory = os.path.expanduser('~')
  for distributionName in ('miniforge3', 'miniconda3', 'anaconda3', 'mambaforge'):
    candidatePath = os.path.join(homeDirectory, distributionName, 'condabin', 'conda')
    if os.path.isfile(candidatePath):
      return candidatePath
  return None


def require_executables():
  # Verify that conda, docker, apptainer, and ImageMagick (compare/identify)
  # are all installed in standard locations.  Return their paths as a dict.
  # Exit with an error otherwise.  compare/identify are used by the smoke test
  # to compare demo output to the target image.
  announce('Checking for required executables')
  toolPathByName = {}

  condaPath = find_conda()
  if condaPath is None:
    die('conda was not found in $PATH or any standard location.  '
        'A conda installation is required.')
  print('Found conda at %s' % condaPath)
  toolPathByName['conda'] = condaPath

  packageHintByProgram = {'compare': 'ImageMagick', 'identify': 'ImageMagick'}
  for programName in ('docker', 'apptainer', 'compare', 'identify'):
    programPath = find_executable_in_standard_locations(programName)
    if programPath is None:
      packageHint = packageHintByProgram.get(programName)
      hintText = (' (provided by %s)' % packageHint) if packageHint else ''
      die('%s%s was not found in $PATH or any standard location (%s).  '
          'Please install it before running this script.'
          % (programName, hintText, ', '.join(STANDARD_BIN_DIRECTORIES)))
    print('Found %s at %s' % (programName, programPath))
    toolPathByName[programName] = programPath

  return toolPathByName


def run_command(command, cwd=None, environmentOverride=None):
  # Run command (a list of strings), echoing it first.  Raise on failure.
  print('+ %s' % ' '.join(command))
  childEnvironment = None
  if environmentOverride is not None:
    childEnvironment = dict(os.environ)
    childEnvironment.update(environmentOverride)
  subprocess.run(command, cwd=cwd, env=childEnvironment, check=True)


def capture_command(command):
  # Run command (a list of strings) and return its stdout as a string.
  print('+ %s' % ' '.join(command))
  completedProcess = subprocess.run(command,
                                    stdout=subprocess.PIPE,
                                    universal_newlines=True,
                                    check=True)
  return completedProcess.stdout


def read_environment_name(environmentFilePath):
  # Return the value of the top-level "name:" key in a conda environment file,
  # or None if there isn't one.
  with open(environmentFilePath, 'r') as environmentFile:
    for line in environmentFile:
      match = re.match(r'^name:\s*(\S+)\s*$', line)
      if match:
        return match.group(1)
  return None


def find_dev_environment_file(developmentDirectory):
  # Return the path to the environment file in a dev directory, preferring
  # environment.yaml over environment.yml.  Return None if neither exists.
  for baseName in ('environment.yaml', 'environment.yml'):
    candidatePath = os.path.join(developmentDirectory, baseName)
    if os.path.isfile(candidatePath):
      return candidatePath
  return None


def conda_environment_exists(condaPath, environmentName):
  # Return whether a conda environment with the given name exists.
  environmentListText = capture_command([condaPath, 'env', 'list'])
  for line in environmentListText.splitlines():
    strippedLine = line.strip()
    if not strippedLine or strippedLine.startswith('#'):
      continue
    firstToken = strippedLine.split()[0]
    if firstToken == environmentName:
      return True
  return False


def freeze_environment(condaPath, developmentEnvironmentName, productionEnvironmentName):
  # Export the (pinned) dev conda environment, rewrite its "name:" to the
  # production name, drop the machine-specific "prefix:" line, and return the
  # resulting environment.yaml text.
  exportedText = capture_command([condaPath, 'env', 'export', '--name', developmentEnvironmentName])
  outputLines = []
  didRewriteName = False
  for line in exportedText.splitlines():
    if not didRewriteName and re.match(r'^name:\s', line):
      outputLines.append('name: %s' % productionEnvironmentName)
      didRewriteName = True
      continue
    if re.match(r'^prefix:\s', line):
      # Machine-specific; omit from the portable production spec.
      continue
    outputLines.append(line)
  if not didRewriteName:
    outputLines.insert(0, 'name: %s' % productionEnvironmentName)
  return '\n'.join(outputLines) + '\n'


def cuda_version_from_environment_text(environmentText):
  # Return the CUDA version pinned in a conda environment file's text (e.g.
  # "12.8" from a "cuda-version=12.8=..." line), or None if not present.
  match = re.search(r'cuda-version\s*[=<>!]+\s*([0-9]+(?:\.[0-9]+)*)', environmentText)
  if match:
    return match.group(1)
  return None


def render_template(template, productionName, cudaOverride):
  # Substitute the production name and CUDA override into a template string.
  return template.replace('__NAME__', productionName).replace('__CUDA__', cudaOverride)


DOCKERFILE_TEMPLATE = """\
FROM ubuntu:22.04

# Copy the conda env .yaml file into the image
RUN mkdir -p /__NAME__
COPY environment.yaml /__NAME__/environment.yaml

# Want to use bash instead of sh for RUN commands
SHELL ["/bin/bash", "-c"]

# Update package lists
RUN apt -y update

# This will get us libGl.so, etc
RUN apt -y install libgl-dev libegl-dev libopengl-dev

# This will get us libjpeg.so, and a bunch of other stuff
RUN apt -y install ffmpeg

# Get miniforge, add its conda to path
RUN apt -y install curl bzip2
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3-$(uname)-$(uname -m).sh -b -p /miniforge3
ENV PATH="/miniforge3/condabin:${PATH}"

# Create the conda env from the environment .yaml file
RUN CONDA_OVERRIDE_CUDA="__CUDA__" PIP_NO_DEPS=1 \\
  conda env create --file /__NAME__/environment.yaml \\
                   --prefix /__NAME__/environment

# "Manually" activate the environment
# This seems like it might be fragile...
ENV PATH="/__NAME__/environment/bin:${PATH}"
ENV GSETTINGS_SCHEMA_DIR=/__NAME__/environment/share/glib-2.0/schemas

# Define default command.
CMD ["bash"]
"""


def write_production_directory_files(productionDirectory, productionName, frozenEnvironmentText, cudaOverride):
  # Write the frozen environment.yaml and the templated Dockerfile into the
  # production directory.
  environmentFilePath = os.path.join(productionDirectory, 'environment.yaml')
  with open(environmentFilePath, 'w') as environmentFile:
    environmentFile.write(frozenEnvironmentText)
  print('Wrote %s' % environmentFilePath)

  dockerfilePath = os.path.join(productionDirectory, 'Dockerfile')
  with open(dockerfilePath, 'w') as dockerfile:
    dockerfile.write(render_template(DOCKERFILE_TEMPLATE, productionName, cudaOverride))
  print('Wrote %s' % dockerfilePath)


def image_demo_command(outputBaseName):
  # Return the argv that runs the pose-estimation demo, writing its result to
  # outputBaseName.  Meant to run with the scripts directory as the working
  # directory (natively, or bind-mounted into a container).
  return ['python', DEMO_SCRIPT, DEMO_INPUT_IMAGE, DEMO_CONFIG, DEMO_CHECKPOINT,
          '--out-file', outputBaseName, '--draw-heatmap']


def compare_to_target(scriptsDirectory, outputBaseName):
  # Compare a demo output image to the target using the host-side ImageMagick
  # comparison script.  Raises CalledProcessError if it does not match.
  run_command(['bash', COMPARE_SCRIPT, outputBaseName], cwd=scriptsDirectory)


def test_conda_environment(condaPath, scriptsDirectory, environmentName):
  # Run the smoke test (generate + compare) inside the named conda environment.
  announce('Testing conda environment %s' % environmentName)
  run_command([condaPath, 'run', '--no-capture-output', '--name', environmentName,
               'bash', TEST_SCRIPT],
              cwd=scriptsDirectory)


def test_docker_image(dockerPath, scriptsDirectory, imageTag):
  # Run the demo inside the Docker image (scripts dir bind-mounted so the demo
  # assets are available and the output lands back on the host), then compare
  # the output to the target on the host (the image has no ImageMagick).
  announce('Testing Docker image %s' % imageTag)
  outputBaseName = 'docker-output.jpg'
  run_command([dockerPath, 'run', '--rm', '--gpus', 'all',
               '--volume', scriptsDirectory + ':/mnt',
               '--workdir', '/mnt',
               imageTag]
              + image_demo_command(outputBaseName))
  compare_to_target(scriptsDirectory, outputBaseName)


def test_apptainer_image(apptainerPath, scriptsDirectory, sifPath):
  # Run the demo inside the Apptainer image (which runs in the scripts
  # directory, so the demo assets and output are on the host), then compare the
  # output to the target on the host.
  announce('Testing Apptainer image %s' % sifPath)
  outputBaseName = 'apptainer-output.jpg'
  run_command([apptainerPath, 'exec', '--nv', sifPath]
              + image_demo_command(outputBaseName),
              cwd=scriptsDirectory)
  compare_to_target(scriptsDirectory, outputBaseName)


def main():
  # Parse arguments, run the checks, and produce the production complement.
  argumentParser = argparse.ArgumentParser(
    description='Turn a working "dev" conda environment into a production '
                'conda environment, a locally-built Docker image, and an Apptainer .sif file.')
  argumentParser.add_argument(
    'developmentDirectory',
    help='The dev environment directory, e.g. apt-20260801-tf215-pytorch21-hopper-dev.  '
         'Its name must end in "-dev".')
  argumentParser.add_argument(
    '--force',
    action='store_true',
    help='Overwrite the production directory if it already exists.')
  arguments = argumentParser.parse_args()

  # Operate relative to the directory this script lives in (deepnet/scripts),
  # so it works regardless of the current working directory.
  scriptsDirectory = os.path.dirname(os.path.abspath(__file__))

  # Resolve the dev directory (accept either a bare name or a path).
  developmentDirectoryArgument = os.path.normpath(arguments.developmentDirectory)
  if os.path.isdir(developmentDirectoryArgument):
    developmentDirectory = os.path.abspath(developmentDirectoryArgument)
  else:
    developmentDirectory = os.path.join(scriptsDirectory, os.path.basename(developmentDirectoryArgument))
  if not os.path.isdir(developmentDirectory):
    die('Development directory not found: %s' % arguments.developmentDirectory)

  developmentDirectoryName = os.path.basename(developmentDirectory)
  if not developmentDirectoryName.endswith('-dev'):
    die('Development directory name must end in "-dev", got: %s' % developmentDirectoryName)
  productionName = developmentDirectoryName[:-len('-dev')]
  productionDirectory = os.path.join(scriptsDirectory, productionName)

  # Fail fast if the tools we need are missing.
  toolPathByName = require_executables()
  condaPath = toolPathByName['conda']

  # Figure out the dev conda environment's name (prefer the name inside the
  # environment file; fall back to the directory name).
  developmentEnvironmentFile = find_dev_environment_file(developmentDirectory)
  if developmentEnvironmentFile is None:
    die('No environment.yaml or environment.yml found in %s' % developmentDirectory)
  developmentEnvironmentName = read_environment_name(developmentEnvironmentFile)
  if developmentEnvironmentName is None:
    developmentEnvironmentName = developmentDirectoryName
  print('Development conda environment: %s' % developmentEnvironmentName)
  print('Production name: %s' % productionName)

  # The dev environment must already have been built (that is the manual step
  # this script picks up from).
  if not conda_environment_exists(condaPath, developmentEnvironmentName):
    die('The dev conda environment "%s" does not exist.  Create it first with '
        './create-conda-env.bash %s' % (developmentEnvironmentName, developmentDirectoryName))

  dockerPath = toolPathByName['docker']
  apptainerPath = toolPathByName['apptainer']
  imageTag = '%s:%s' % (DOCKER_IMAGE_REPOSITORY, productionName)

  # 1. Test the dev environment before investing any effort in freezing it.
  test_conda_environment(condaPath, scriptsDirectory, developmentEnvironmentName)

  # Prepare the production directory.
  if os.path.exists(productionDirectory):
    if arguments.force:
      announce('Removing existing production directory %s' % productionDirectory)
      shutil.rmtree(productionDirectory)
    else:
      die('Production directory already exists: %s (use --force to overwrite)' % productionDirectory)
  os.mkdir(productionDirectory)

  # 2. Freeze the dev environment into the production environment.yaml.
  announce('Freezing conda environment %s' % developmentEnvironmentName)
  frozenEnvironmentText = freeze_environment(condaPath, developmentEnvironmentName, productionName)
  cudaOverride = cuda_version_from_environment_text(frozenEnvironmentText)
  if cudaOverride is None:
    cudaOverride = '12.8'
    print('No cuda-version pin found; defaulting CONDA_OVERRIDE_CUDA to %s' % cudaOverride)
  else:
    print('Using CONDA_OVERRIDE_CUDA=%s (from the pinned cuda-version)' % cudaOverride)
  write_production_directory_files(productionDirectory, productionName, frozenEnvironmentText, cudaOverride)

  # 3. Create the production conda environment locally from the frozen spec, and test it.
  announce('Creating production conda environment %s' % productionName)
  run_command([condaPath, 'env', 'create', '--file', 'environment.yaml'],
              cwd=productionDirectory,
              environmentOverride={'PIP_NO_DEPS': '1', 'CONDA_OVERRIDE_CUDA': cudaOverride})
  test_conda_environment(condaPath, scriptsDirectory, productionName)

  # 4. Build the Docker image and test it.  (It is not pushed to Docker Hub
  # here; do that afterward with push-docker-image.bash -- see the README.)
  announce('Building Docker image %s' % imageTag)
  run_command([dockerPath, 'build', '--file', 'Dockerfile', '--tag', imageTag, '.'],
              cwd=productionDirectory)
  test_docker_image(dockerPath, scriptsDirectory, imageTag)

  # 5. Build the Apptainer .sif from the local Docker image, and test it.  The
  # docker-daemon:// transport reads the just-built image straight from the
  # local Docker store, so no push/pull round-trip through Docker Hub is needed.
  announce('Creating Apptainer image %s.sif' % productionName)
  run_command([apptainerPath, 'build', productionName + '.sif', 'docker-daemon://' + imageTag],
              cwd=productionDirectory)
  sifPath = os.path.join(productionDirectory, productionName + '.sif')
  test_apptainer_image(apptainerPath, scriptsDirectory, sifPath)

  announce('Done')
  print('Production complement created in %s' % productionDirectory)
  print('  conda env:       %s' % productionName)
  print('  docker image:    %s:%s  (local; not yet pushed)' % (DOCKER_IMAGE_REPOSITORY, productionName))
  print('  apptainer image: %s' % os.path.join(productionDirectory, productionName + '.sif'))
  print('')
  print('When ready to share the images with the world, run')
  print('push-docker-and-apptainer-images.bash from the production directory (see the README).')


if __name__ == '__main__':
  try:
    main()
  except subprocess.CalledProcessError as calledProcessError:
    die('Command failed (exit %s): %s'
        % (calledProcessError.returncode, ' '.join(calledProcessError.cmd)))
  except KeyboardInterrupt:
    die('Interrupted')
