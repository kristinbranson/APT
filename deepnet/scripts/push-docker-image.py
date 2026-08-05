#! /usr/bin/env python3
"""Push the production Docker image to Docker Hub.

You must be logged in (``docker login``) with push access to the bransonlabapt
organization.

This script uses only the Python 3.6 standard library.
"""

import subprocess
import sys


# The image to push.  Edit this to match the version you are releasing.
IMAGE_TAG = 'bransonlabapt/apt_docker:apt-20260801-tf215-pytorch21-hopper'


def main():
  # Push the Docker image.
  command = ['docker', 'push', IMAGE_TAG]
  print('+ %s' % ' '.join(command))
  subprocess.run(command, check=True)


if __name__ == '__main__':
  try:
    main()
  except subprocess.CalledProcessError as calledProcessError:
    sys.exit(calledProcessError.returncode)
