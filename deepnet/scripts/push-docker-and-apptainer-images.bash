#! /bin/bash

# Share the production images with the world: push the Docker image to Docker
# Hub and copy the Apptainer .sif into the shared image directory.  Run this from
# the production directory (push-apptainer-image.bash copies the .sif from the
# current directory).

set -e

scriptDirectory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$scriptDirectory/push-docker-image.bash"
"$scriptDirectory/push-apptainer-image.bash"
