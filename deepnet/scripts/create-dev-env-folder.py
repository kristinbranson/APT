#! /usr/bin/env python3
"""Create a new development environment folder for a conda/Docker/Apptainer set.

Prompts for a tag (e.g. "tf215-pytorch21-hopper"), synthesizes the folder name
from today's date as ``apt-<YYYYMMDD>-<tag>-dev``, creates the folder, and seeds
an ``environment.yaml`` in it (copied from the most recent existing ``-dev``
environment, with the ``name:`` rewritten to match the new folder).  Run this
from deepnet/scripts.

Usage:
    ./create-dev-env-folder.py [tag]

If the tag is not given on the command line, you will be prompted for it.

This script uses only the Python 3.6 standard library.
"""

import datetime
import os
import re
import sys


# Minimal environment.yaml to write when there is no existing -dev environment
# to seed from.
SKELETON_ENVIRONMENT_TEMPLATE = """\
name: __NAME__
channels:
  - conda-forge
dependencies:
  - python=3.10
  - cuda-version=12.8
"""


def die(message):
  # Print an error to stderr and exit nonzero.
  sys.stderr.write('Error: %s\n' % message)
  sys.exit(1)


def find_seed_environment_file(scriptDirectory, newDevName):
  # Return the environment file of the most recent existing -dev directory to
  # seed from (highest date first; the date sits at a fixed position in the name,
  # so a reverse sort works), or None.  Skips the new directory itself.
  devDirectoryNames = [entry for entry in os.listdir(scriptDirectory)
                       if entry.startswith('apt-') and entry.endswith('-dev')
                       and os.path.isdir(os.path.join(scriptDirectory, entry))]
  for devDirectoryName in sorted(devDirectoryNames, reverse=True):
    if devDirectoryName == newDevName:
      continue
    for baseName in ('environment.yaml', 'environment.yml'):
      candidatePath = os.path.join(scriptDirectory, devDirectoryName, baseName)
      if os.path.isfile(candidatePath):
        return candidatePath
  return None


def seed_text_from_template(templateFilePath, newDevName):
  # Return the template's text with the first top-level name: line rewritten to
  # the new dev name.
  outputLines = []
  didRewriteName = False
  with open(templateFilePath, 'r') as templateFile:
    for line in templateFile:
      if not didRewriteName and re.match(r'^name:\s', line):
        outputLines.append('name: %s\n' % newDevName)
        didRewriteName = True
      else:
        outputLines.append(line)
  if not didRewriteName:
    outputLines.insert(0, 'name: %s\n' % newDevName)
  return ''.join(outputLines)


def main():
  # Prompt for a tag, create the dev folder, and seed its environment.yaml.
  scriptDirectory = os.path.dirname(os.path.abspath(__file__))

  # Get the tag from the command line, or prompt for it.
  tag = sys.argv[1] if len(sys.argv) > 1 else ''
  if not tag:
    tag = input('Tag (e.g. tf215-pytorch21-hopper): ').strip()
  if not tag:
    die('No tag given; aborting.')

  # Synthesize the dev folder name from today's date.
  today = datetime.date.today().strftime('%Y%m%d')
  devName = 'apt-%s-%s-dev' % (today, tag)
  devDirectory = os.path.join(scriptDirectory, devName)

  if os.path.exists(devDirectory):
    die('Directory already exists: %s' % devDirectory)

  seedFilePath = find_seed_environment_file(scriptDirectory, devName)

  os.mkdir(devDirectory)
  targetFilePath = os.path.join(devDirectory, 'environment.yaml')

  if seedFilePath is not None:
    with open(targetFilePath, 'w') as targetFile:
      targetFile.write(seed_text_from_template(seedFilePath, devName))
    print('Seeded %s from %s'
          % (targetFilePath, os.path.relpath(seedFilePath, scriptDirectory)))
  else:
    with open(targetFilePath, 'w') as targetFile:
      targetFile.write(SKELETON_ENVIRONMENT_TEMPLATE.replace('__NAME__', devName))
    print('Wrote a minimal %s (no existing -dev environment to copy from)' % targetFilePath)

  print('')
  print('Created dev environment folder %s.' % devName)
  print('Edit %s as needed, then build the environment with:' % targetFilePath)
  print('  ./create-conda-env.py %s' % devName)


if __name__ == '__main__':
  main()
