FROM ubuntu:22.04

# Copy the conda env .yaml file into the image
RUN mkdir -p /environments
COPY apt_20230427_tf211_pytorch113_ampere.env.yaml /environments/apt_20230427_tf211_pytorch113_ampere.env.yaml

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
RUN CONDA_OVERRIDE_CUDA="11.8" micromamba env create -f /environments/apt_20230427_tf211_pytorch113_ampere.env.yaml --prefix /environments/apt_20230427_tf211_pytorch113_ampere

# "Manually" activate the environment
# This seems like it might be fragile...
ENV PATH /environments/apt_20230427_tf211_pytorch113_ampere/bin:$PATH
ENV GSETTINGS_SCHEMA_DIR /environments/apt_20230427_tf211_pytorch113_ampere/share/glib-2.0/schemas

# Define default command.
CMD ["bash"]
