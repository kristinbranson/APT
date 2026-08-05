#! /usr/bin/env python3
"""Build and publish a full APT conda / Docker / Apptainer complement, resumably.

Given a single tag that includes the date (e.g. ``20260801-tf215-pytorch21-hopper``),
this builds the whole complement for ``apt-<tag>`` in one command, and does so
idempotently: every stage first checks whether its output already exists and, if
so, skips it and moves on.  So you can run it, fix whatever broke, and run it
again; it re-does only what is missing and always makes progress toward having a
working conda environment, Docker image, and Apptainer image.

Stages, in order (each skipped if its output already exists):

  1.  Create the dev folder ``apt-<tag>-dev``.
  2.  Seed ``apt-<tag>-dev/environment.yaml`` from ``dev-environment-template.yaml``
      (you may edit it between runs; a later run will not overwrite it).
  3.  Build the dev conda environment, then smoke-test it.
  4.  Create the prod folder ``apt-<tag>``.
  5.  Freeze the dev environment into ``apt-<tag>/environment.yaml``.
  6.  Build the prod conda environment, then smoke-test it.
  7.  Write ``apt-<tag>/Dockerfile``.
  8.  Build the Docker image, then smoke-test it.
  9.  Build the Apptainer .sif from the local Docker image, then smoke-test it.
  10. Push the Docker image to Docker Hub.            (only with --publish)
  11. Copy the Apptainer .sif into the shared image   (only with --publish)
      directory under /groups.

By default the images are built and tested locally but not published; pass
--publish to also run stages 10 and 11.  Pass --conda-only to build just the
dev and prod conda environments (stages 1-6) and skip the Docker and Apptainer
images; --conda-only and --publish cannot be combined.

The smoke test is test_pose_estimation.py; it runs inside the environment/image
under test.  Any stage that errors aborts the whole script.  A smoke test runs
only when its environment/image was (re)built in this run; an environment/image
that already exists is trusted, so to force a rebuild, delete it first.

This script uses only the Python 3.6 standard library.  It needs conda, docker,
and apptainer, and a working GPU for the smoke tests.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys


# Docker Hub repository that production images are tagged under.
DOCKER_IMAGE_REPOSITORY = 'bransonlabapt/apt_docker'

# Shared image directory that APT's bsub/cluster backend loads .sif images from.
SHARED_SIF_DIRECTORY = '/groups/branson/bransonlab/apt/sif'

# The smoke-test script (lives in this scripts directory).
TEST_SCRIPT = 'test_pose_estimation.py'

# The starting point for a new dev environment (lives in this scripts directory).
DEV_ENVIRONMENT_TEMPLATE = 'dev-environment-template.yaml'

# CUDA version to assume when an environment file does not pin one.
DEFAULT_CUDA_VERSION = '12.8'

# Directories to search (in addition to $PATH) for a required executable.
STANDARD_BIN_DIRECTORIES = (
  '/usr/bin',
  '/usr/local/bin',
  '/bin',
  '/sbin',
  '/usr/sbin',
  '/snap/bin',
)

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


def die(message):
  # Print an error to stderr and exit nonzero.
  sys.stderr.write('Error: %s\n' % message)
  sys.exit(1)


def announce(message):
  # Print a prominent progress banner.
  print('')
  print('==== %s ====' % message)


def run_command(command, cwd=None, environmentOverride=None):
  # Run command (a list of strings), echoing it first.  Raise on failure.
  print('+ %s' % ' '.join(command))
  childEnvironment = None
  if environmentOverride is not None:
    childEnvironment = dict(os.environ)
    childEnvironment.update(environmentOverride)
  subprocess.run(command, cwd=cwd, env=childEnvironment, check=True)


def capture_command(command):
  # Run command and return its stdout as a string.  Raise on failure.
  completedProcess = subprocess.run(command, stdout=subprocess.PIPE,
                                    universal_newlines=True, check=True)
  return completedProcess.stdout


def command_succeeds(command):
  # Return whether command exits zero, discarding its output.
  completedProcess = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  return completedProcess.returncode == 0


def find_executable_in_standard_locations(programName):
  # Return the absolute path to programName, searching $PATH then the standard
  # bin directories.  Return None if not found.
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
  # Verify that conda, docker, and apptainer are installed.  Return conda's path.
  announce('Checking for required executables')
  condaPath = find_conda()
  if condaPath is None:
    die('conda was not found in $PATH or any standard location.')
  print('Found conda at %s' % condaPath)
  for programName in ('docker', 'apptainer'):
    if find_executable_in_standard_locations(programName) is None:
      die('%s was not found in $PATH or any standard location (%s).'
          % (programName, ', '.join(STANDARD_BIN_DIRECTORIES)))
    print('Found %s' % programName)
  return condaPath


def conda_environment_exists(condaPath, environmentName):
  # Return whether a conda environment with the given name exists.
  environmentListText = capture_command([condaPath, 'env', 'list'])
  for line in environmentListText.splitlines():
    strippedLine = line.strip()
    if not strippedLine or strippedLine.startswith('#'):
      continue
    if strippedLine.split()[0] == environmentName:
      return True
  return False


def cuda_version_from_text(environmentText):
  # Return the CUDA version pinned in environment text (e.g. "12.8"), or the
  # default if none is present.
  match = re.search(r'cuda-version\s*[=<>!]+\s*([0-9]+(?:\.[0-9]+)*)', environmentText)
  return match.group(1) if match else DEFAULT_CUDA_VERSION


def read_file(path):
  # Return the full text of a file.
  with open(path, 'r') as f:
    return f.read()


def create_conda_environment_from_file(condaPath, environmentFilePath):
  # Create a conda environment from an environment file (which names the env),
  # using the file's pinned CUDA version as the override.
  cudaVersion = cuda_version_from_text(read_file(environmentFilePath))
  environmentOverride = {'PIP_NO_DEPS': '1',
                         'CONDA_OVERRIDE_CUDA': cudaVersion,
                         'CONDA_CHANNEL_PRIORITY': 'strict'}
  run_command([condaPath, 'env', 'create', '--file', environmentFilePath],
              environmentOverride=environmentOverride)


def smoke_test_conda_environment(condaPath, scriptsDirectory, environmentName):
  # Run the smoke test inside the named conda environment.
  announce('Smoke-testing conda environment %s' % environmentName)
  run_command([condaPath, 'run', '--no-capture-output', '--name', environmentName,
               'python', TEST_SCRIPT],
              cwd=scriptsDirectory)


def smoke_test_docker_image(scriptsDirectory, imageTag):
  # Run the smoke test inside the Docker image (scripts dir bind-mounted).
  announce('Smoke-testing Docker image %s' % imageTag)
  run_command(['docker', 'run', '--rm', '--gpus', 'all',
               '--volume', scriptsDirectory + ':/mnt',
               '--workdir', '/mnt',
               imageTag,
               'python', TEST_SCRIPT, '--output', 'docker-output.jpg'])


def smoke_test_apptainer_image(scriptsDirectory, sifPath):
  # Run the smoke test inside the Apptainer image.
  announce('Smoke-testing Apptainer image %s' % sifPath)
  run_command(['apptainer', 'exec', '--nv', sifPath,
               'python', TEST_SCRIPT, '--output', 'apptainer-output.jpg'],
              cwd=scriptsDirectory)


def rewrite_environment_name(environmentText, newName):
  # Return environmentText with the first top-level name: line rewritten, and
  # the machine-specific prefix: line (if any) dropped.
  outputLines = []
  didRewriteName = False
  for line in environmentText.splitlines():
    if not didRewriteName and re.match(r'^name:\s', line):
      outputLines.append('name: %s' % newName)
      didRewriteName = True
      continue
    if re.match(r'^prefix:\s', line):
      continue
    outputLines.append(line)
  if not didRewriteName:
    outputLines.insert(0, 'name: %s' % newName)
  return '\n'.join(outputLines) + '\n'


def write_seed_dev_environment(scriptsDirectory, devDirectory, devName):
  # Write devDirectory/environment.yaml, seeded from dev-environment-template.yaml
  # with the name: rewritten to match the new dev folder.
  templateFilePath = os.path.join(scriptsDirectory, DEV_ENVIRONMENT_TEMPLATE)
  if not os.path.isfile(templateFilePath):
    die('Dev environment template %s not found' % templateFilePath)
  targetFilePath = os.path.join(devDirectory, 'environment.yaml')
  text = rewrite_environment_name(read_file(templateFilePath), devName)
  with open(targetFilePath, 'w') as targetFile:
    targetFile.write(text)
  print('Seeded %s from %s' % (targetFilePath, DEV_ENVIRONMENT_TEMPLATE))


def write_frozen_prod_environment(condaPath, devName, prodName, prodDirectory):
  # Freeze the dev conda environment into prodDirectory/environment.yaml, with
  # the name de-`-dev`'d and the machine-specific prefix: line removed.
  exportedText = capture_command([condaPath, 'env', 'export', '--name', devName])
  frozenText = rewrite_environment_name(exportedText, prodName)
  with open(os.path.join(prodDirectory, 'environment.yaml'), 'w') as environmentFile:
    environmentFile.write(frozenText)


def write_dockerfile(prodDirectory, prodName):
  # Render the Dockerfile into the prod directory, templated with the prod name
  # and the CUDA version pinned in the prod environment.yaml.
  cudaVersion = cuda_version_from_text(read_file(os.path.join(prodDirectory, 'environment.yaml')))
  dockerfileText = DOCKERFILE_TEMPLATE.replace('__NAME__', prodName).replace('__CUDA__', cudaVersion)
  with open(os.path.join(prodDirectory, 'Dockerfile'), 'w') as dockerfile:
    dockerfile.write(dockerfileText)


def run_stage(description, isDoneFunction, actionFunction):
  # Run one idempotent stage: skip it if its output already exists, else do it.
  announce(description)
  if isDoneFunction():
    print('Already present; skipping.')
    return
  actionFunction()


def main():
  # Parse the tag and build the complement, one idempotent stage at a time.
  argumentParser = argparse.ArgumentParser(
    description='Build and publish an APT conda/Docker/Apptainer complement, resumably.')
  argumentParser.add_argument(
    'tag',
    help='The dated tag for the complement, e.g. 20260801-tf215-pytorch21-hopper.  '
         'The complement is named apt-<tag>.')
  modeGroup = argumentParser.add_mutually_exclusive_group()
  modeGroup.add_argument(
    '--publish',
    action='store_true',
    help='Also publish the images: push the Docker image to Docker Hub and copy '
         'the Apptainer .sif into the shared image directory.  Without this flag, '
         'those two steps are skipped.')
  modeGroup.add_argument(
    '--conda-only',
    action='store_true',
    help='Only create the dev and prod conda environments (stages 1-6); skip the '
         'Docker and Apptainer image builds.  Cannot be combined with --publish.')
  arguments = argumentParser.parse_args()

  tag = arguments.tag
  if tag.startswith('apt-'):
    tag = tag[len('apt-'):]
  if tag.endswith('-dev'):
    die('Give the tag without the -dev suffix, e.g. 20260801-tf215-pytorch21-hopper')

  scriptsDirectory = os.path.dirname(os.path.abspath(__file__))
  prodName = 'apt-' + tag
  devName = prodName + '-dev'
  devDirectory = os.path.join(scriptsDirectory, devName)
  prodDirectory = os.path.join(scriptsDirectory, prodName)
  devEnvironmentFile = os.path.join(devDirectory, 'environment.yaml')
  prodEnvironmentFile = os.path.join(prodDirectory, 'environment.yaml')
  dockerfilePath = os.path.join(prodDirectory, 'Dockerfile')
  imageTag = '%s:%s' % (DOCKER_IMAGE_REPOSITORY, prodName)
  sifName = prodName + '.sif'
  prodSifPath = os.path.join(prodDirectory, sifName)
  sharedSifPath = os.path.join(SHARED_SIF_DIRECTORY, sifName)

  condaPath = require_executables()

  # 1. Dev folder.
  run_stage('Create dev folder %s' % devName,
            lambda: os.path.isdir(devDirectory),
            lambda: os.mkdir(devDirectory))

  # 2. Dev environment.yaml (seeded; not overwritten if you have edited it).
  run_stage('Create dev environment.yaml',
            lambda: os.path.isfile(devEnvironmentFile),
            lambda: write_seed_dev_environment(scriptsDirectory, devDirectory, devName))

  # 3. Dev conda environment, then smoke-test it.
  def build_dev_environment():
    create_conda_environment_from_file(condaPath, devEnvironmentFile)
    smoke_test_conda_environment(condaPath, scriptsDirectory, devName)
  run_stage('Build and test dev conda environment %s' % devName,
            lambda: conda_environment_exists(condaPath, devName),
            build_dev_environment)

  # 4. Prod folder.
  run_stage('Create prod folder %s' % prodName,
            lambda: os.path.isdir(prodDirectory),
            lambda: os.mkdir(prodDirectory))

  # 5. Prod environment.yaml (frozen from the dev environment).
  run_stage('Freeze prod environment.yaml',
            lambda: os.path.isfile(prodEnvironmentFile),
            lambda: write_frozen_prod_environment(condaPath, devName, prodName, prodDirectory))

  # 6. Prod conda environment, then smoke-test it.
  def build_prod_environment():
    create_conda_environment_from_file(condaPath, prodEnvironmentFile)
    smoke_test_conda_environment(condaPath, scriptsDirectory, prodName)
  run_stage('Build and test prod conda environment %s' % prodName,
            lambda: conda_environment_exists(condaPath, prodName),
            build_prod_environment)

  if arguments.conda_only:
    announce('Done')
    print('Conda environments for apt-%s are built (--conda-only; skipped Docker and Apptainer).' % tag)
    print('  dev conda env:  %s' % devName)
    print('  prod conda env: %s' % prodName)
    return

  # 7. Prod Dockerfile.
  run_stage('Create prod Dockerfile',
            lambda: os.path.isfile(dockerfilePath),
            lambda: write_dockerfile(prodDirectory, prodName))

  # 8. Docker image, then smoke-test it.
  def build_docker_image():
    run_command(['docker', 'build', '--file', 'Dockerfile', '--tag', imageTag, '.'],
                cwd=prodDirectory)
    smoke_test_docker_image(scriptsDirectory, imageTag)
  run_stage('Build and test Docker image %s' % imageTag,
            lambda: command_succeeds(['docker', 'image', 'inspect', imageTag]),
            build_docker_image)

  # 9. Apptainer .sif, built from the local Docker image, then smoke-test it.
  def build_apptainer_image():
    run_command(['apptainer', 'build', sifName, 'docker-daemon://' + imageTag],
                cwd=prodDirectory)
    smoke_test_apptainer_image(scriptsDirectory, prodSifPath)
  run_stage('Build and test Apptainer image %s' % sifName,
            lambda: os.path.isfile(prodSifPath),
            build_apptainer_image)

  # 10-11. Publishing steps, only when --publish is passed.
  if arguments.publish:
    # 10. Publish the Docker image to Docker Hub.
    run_stage('Push Docker image %s to Docker Hub' % imageTag,
              lambda: command_succeeds(['docker', 'manifest', 'inspect', imageTag]),
              lambda: run_command(['docker', 'push', imageTag]))

    # 11. Publish the Apptainer image to the shared image directory.
    run_stage('Copy Apptainer image to %s' % SHARED_SIF_DIRECTORY,
              lambda: os.path.isfile(sharedSifPath),
              lambda: shutil.copy2(prodSifPath, sharedSifPath))
  else:
    announce('Publishing skipped (pass --publish to push the Docker image and copy the .sif)')

  announce('Done')
  if arguments.publish:
    print('Complement apt-%s is built and published.' % tag)
    print('  conda env:       %s' % prodName)
    print('  docker image:    docker://%s' % imageTag)
    print('  apptainer image: %s' % sharedSifPath)
  else:
    print('Complement apt-%s is built (not published; pass --publish to publish).' % tag)
    print('  conda env:       %s' % prodName)
    print('  docker image:    %s  (local; not pushed)' % imageTag)
    print('  apptainer image: %s  (local; not copied to %s)' % (prodSifPath, SHARED_SIF_DIRECTORY))


if __name__ == '__main__':
  try:
    main()
  except subprocess.CalledProcessError as calledProcessError:
    die('command failed (exit %s): %s'
        % (calledProcessError.returncode, ' '.join(calledProcessError.cmd)))
  except KeyboardInterrupt:
    die('Interrupted')
