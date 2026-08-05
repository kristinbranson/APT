#! /usr/bin/env python3
"""Build the production Docker image from the Dockerfile in the current directory.

Run this from the production directory (the one containing the Dockerfile).

This script uses only the Python 3.6 standard library.
"""

import subprocess
import sys


# The image to build.  Edit this to match the version you are releasing.
IMAGE_TAG = 'bransonlabapt/apt_docker:apt-20260801-tf215-pytorch21-hopper'


def main():
  # Build the Docker image.
  command = ['docker', 'build', '--file', 'Dockerfile', '--tag', IMAGE_TAG, '.']
  print('+ %s' % ' '.join(command))
  subprocess.run(command, check=True)


if __name__ == '__main__':
  try:
    main()
  except subprocess.CalledProcessError as calledProcessError:
    sys.exit(calledProcessError.returncode)
