#! /usr/bin/env python3
"""Copy the Apptainer .sif into the shared image directory.

Copies the .sif from the current directory into the directory that APT's
bsub/cluster backend loads it from.  Run this from the production directory.

This script uses only the Python 3.6 standard library.
"""

import os
import shutil
import sys


# The .sif to copy.  Edit this to match the version you are releasing.
SIF_NAME = 'apt-20260801-tf215-pytorch21-hopper.sif'

# Shared image directory that APT's bsub/cluster backend loads .sif images from.
SHARED_IMAGE_DIRECTORY = '/groups/branson/bransonlab/apt/sif'


def main():
  # Copy the .sif into the shared image directory.
  if not os.path.isfile(SIF_NAME):
    sys.stderr.write('Error: %s not found in the current directory\n' % SIF_NAME)
    sys.exit(1)
  destinationPath = os.path.join(SHARED_IMAGE_DIRECTORY, SIF_NAME)
  print('+ cp %s %s' % (SIF_NAME, destinationPath))
  shutil.copy2(SIF_NAME, destinationPath)


if __name__ == '__main__':
  main()
