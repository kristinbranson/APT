FROM ubuntu:22.04

# Copy the conda env .yml file into the image
RUN mkdir -p /apt-20250505-tf215-pytorch21-hopper
COPY environment.yml /apt-20250505-tf215-pytorch21-hopper/environment.yml

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

# Create the conda env from the environment .yml file
RUN CONDA_OVERRIDE_CUDA="12.4" PIP_NO_DEPS=1 \
  conda env create --file /apt-20250505-tf215-pytorch21-hopper/environment.yml \
                   --prefix /apt-20250505-tf215-pytorch21-hopper/environment

# "Manually" activate the environment
# This seems like it might be fragile...
ENV PATH="/apt-20250505-tf215-pytorch21-hopper/environment/bin:${PATH}"
ENV GSETTINGS_SCHEMA_DIR=/apt-20250505-tf215-pytorch21-hopper/environment/share/glib-2.0/schemas

# Define default command.
CMD ["bash"]
