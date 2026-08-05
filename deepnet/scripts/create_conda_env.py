#! /usr/bin/env python3
"""Create a conda environment from the environment file in the given directory.

Reads ``environment.yaml`` (or ``environment.yml``) from the given directory and
runs ``conda env create`` on it.  Run this from deepnet/scripts.

Usage:
    ./create_conda_env.py <env-dir>

Example:
    ./create_conda_env.py apt-20260801-tf215-pytorch21-hopper-dev

This script uses only the Python 3.6 standard library.
"""

import os
import re
import subprocess
import sys


# CUDA version to assume if the environment file does not pin one.
DEFAULT_CUDA_VERSION = '12.8'


def die(message):
  # Print an error to stderr and exit nonzero.
  sys.stderr.write('Error: %s\n' % message)
  sys.exit(1)


def find_environment_file(environmentDirectory):
  # Return the path to the environment file, preferring .yaml over .yml, or None.
  for baseName in ('environment.yaml', 'environment.yml'):
    candidatePath = os.path.join(environmentDirectory, baseName)
    if os.path.isfile(candidatePath):
      return candidatePath
  return None


def cuda_version_from_environment_file(environmentFilePath):
  # Return the CUDA version pinned in the environment file (e.g. "12.8" from a
  # "cuda-version=12.8" line), or None if not present.
  with open(environmentFilePath, 'r') as environmentFile:
    match = re.search(r'cuda-version\s*[=<>!]+\s*([0-9]+(?:\.[0-9]+)*)', environmentFile.read())
  if match:
    return match.group(1)
  return None


def main():
  # Parse arguments and create the conda environment.
  if len(sys.argv) != 2 or not sys.argv[1]:
    die('Usage: %s <env-dir>' % os.path.basename(sys.argv[0]))
  environmentDirectory = sys.argv[1]

  environmentFilePath = find_environment_file(environmentDirectory)
  if environmentFilePath is None:
    die('No environment.yaml or environment.yml found in %s' % environmentDirectory)

  # Use the cuda-version pinned in the environment file as the CUDA override,
  # defaulting if none is present.
  cudaVersion = cuda_version_from_environment_file(environmentFilePath)
  if cudaVersion is None:
    cudaVersion = DEFAULT_CUDA_VERSION

  # PIP_NO_DEPS keeps pip from pulling in its own dependency resolution on top of
  # conda's.  CONDA_CHANNEL_PRIORITY=strict keeps the solver from exploring
  # lower-priority channels, which makes the solve substantially faster (and more
  # predictable).
  childEnvironment = dict(os.environ)
  childEnvironment.update({'PIP_NO_DEPS': '1',
                           'CONDA_OVERRIDE_CUDA': cudaVersion,
                           'CONDA_CHANNEL_PRIORITY': 'strict'})
  command = ['conda', 'env', 'create', '-f', environmentFilePath]
  print('+ CONDA_OVERRIDE_CUDA=%s %s' % (cudaVersion, ' '.join(command)))
  try:
    subprocess.run(command, env=childEnvironment, check=True)
  except FileNotFoundError:
    die('conda was not found on the PATH')


if __name__ == '__main__':
  try:
    main()
  except subprocess.CalledProcessError as calledProcessError:
    die('command failed (exit %s): %s'
        % (calledProcessError.returncode, ' '.join(calledProcessError.cmd)))
