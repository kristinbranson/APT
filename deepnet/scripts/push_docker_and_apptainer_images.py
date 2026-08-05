#! /usr/bin/env python3
"""Share the production images with the world.

Pushes the Docker image to Docker Hub (push_docker_image.py) and copies the
Apptainer .sif into the shared image directory (push_apptainer_image.py).  Run
this from the production directory, since push_apptainer_image.py copies the .sif
from the current directory.

This script uses only the Python 3.6 standard library.
"""

import os
import subprocess
import sys


def main():
  # Run both publish steps, in the current working directory.
  scriptDirectory = os.path.dirname(os.path.abspath(__file__))
  for scriptName in ('push_docker_image.py', 'push_apptainer_image.py'):
    subprocess.run([sys.executable, os.path.join(scriptDirectory, scriptName)], check=True)


if __name__ == '__main__':
  try:
    main()
  except subprocess.CalledProcessError as calledProcessError:
    sys.exit(calledProcessError.returncode)
