FROM ubuntu:22.04

# Copy the conda env .yaml file into the image
RUN mkdir -p /schtuff
COPY pinned_ampere_conda_environment.yaml /schtuff/pinned_ampere_conda_environment.yaml

# Want to use bash instead of sh for RUN commands
SHELL ["/bin/bash", "-c"]

# Update package lists
RUN apt -y update

# This will get us libGl.so
RUN apt -y install libgl1

# This will get us libjpeg.so, and a bunch of other stuff
RUN apt -y install ffmpeg

# Get micromamba
RUN apt -y install curl
RUN apt -y install bzip2
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

# Create the conda env from the environment .yaml file
RUN CONDA_OVERRIDE_CUDA="11.8" micromamba env create -f /schtuff/pinned_ampere_conda_environment.yaml --prefix /schtuff/pinned-ampere-env

# "Manually" activate the environment
# This seems like it might be fragile...
ENV PATH /schtuff/pinned-ampere-env/bin:$PATH
ENV GSETTINGS_SCHEMA_DIR /schtuff/pinned-ampere-env/share/glib-2.0/schemas

# Define default command.
CMD ["bash"]
