#! /usr/bin/env python3
"""Build the production Apptainer .sif by pulling the pushed Docker image.

Pulls from Docker Hub, so the Docker image must have been pushed first.  (To
build the .sif from the local Docker store instead, without a registry
round-trip, use ``apptainer build <name>.sif docker-daemon://<image>``, which is
what create-production-complement.py does.)  Run this from the production
directory.

This script uses only the Python 3.6 standard library.
"""

import subprocess
import sys


# The image to pull.  Edit these to match the version you are releasing.
IMAGE_NAME = 'apt-20260801-tf215-pytorch21-hopper'
IMAGE_TAG = 'bransonlabapt/apt_docker:' + IMAGE_NAME


def main():
  # Pull the Docker image into a local Apptainer .sif.
  command = ['apptainer', 'pull', IMAGE_NAME + '.sif', 'docker://' + IMAGE_TAG]
  print('+ %s' % ' '.join(command))
  subprocess.run(command, check=True)


if __name__ == '__main__':
  try:
    main()
  except subprocess.CalledProcessError as calledProcessError:
    sys.exit(calledProcessError.returncode)
