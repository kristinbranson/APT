#! /bin/bash

# Create a new development environment folder for a conda/Docker/Apptainer
# complement.  Prompts for a tag (e.g. "tf215-pytorch21-hopper"), synthesizes the
# folder name from today's date as apt-<YYYYMMDD>-<tag>-dev, creates the folder,
# and seeds an environment.yaml in it (copied from the most recent existing -dev
# environment, with the name: rewritten to match the new folder).  Run this from
# deepnet/scripts.
#
# Usage:
#   ./create-dev-env-folder.bash [tag]
#
# If the tag is not given on the command line, you will be prompted for it.

set -e

scriptDirectory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$scriptDirectory"

# Get the tag from the command line, or prompt for it.
tag="$1"
if [ -z "$tag" ] ; then
  read -r -p 'Tag (e.g. tf215-pytorch21-hopper): ' tag
fi
if [ -z "$tag" ] ; then
  echo "No tag given; aborting." 1>&2
  exit 1
fi

# Synthesize the dev folder name from today's date.
today="$(date +%Y%m%d)"
devName="apt-${today}-${tag}-dev"
devDirectory="$scriptDirectory/$devName"

if [ -e "$devDirectory" ] ; then
  echo "Directory already exists: $devDirectory" 1>&2
  exit 1
fi

# Find the most recent existing -dev environment file to seed from (highest date
# first; the date sits at a fixed position in the name, so a reverse sort works).
templateFile=""
for candidate in $(ls -d apt-*-dev 2>/dev/null | sort -r) ; do
  if [ "$candidate" = "$devName" ] ; then
    continue
  fi
  for baseName in environment.yaml environment.yml ; do
    if [ -f "$candidate/$baseName" ] ; then
      templateFile="$candidate/$baseName"
      break
    fi
  done
  if [ -n "$templateFile" ] ; then
    break
  fi
done

mkdir "$devDirectory"
targetFile="$devDirectory/environment.yaml"

if [ -n "$templateFile" ] ; then
  # Copy the template, rewriting the first name: line to match the new folder.
  sed "0,/^name:.*/s//name: $devName/" "$templateFile" > "$targetFile"
  echo "Seeded $targetFile from $templateFile"
else
  # No existing -dev environment to copy; write a minimal skeleton.
  {
    echo "name: $devName"
    echo "channels:"
    echo "  - conda-forge"
    echo "dependencies:"
    echo "  - python=3.10"
    echo "  - cuda-version=12.8"
  } > "$targetFile"
  echo "Wrote a minimal $targetFile (no existing -dev environment to copy from)"
fi

echo ""
echo "Created dev environment folder $devName."
echo "Edit $targetFile as needed, then build the environment with:"
echo "  ./create-conda-env.bash $devName"
