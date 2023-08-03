# first launch singularity env with the singularity/docker image
# sing image: /groups/branson/home/kabram/bransonlab/singularity/tf23_mmdetection.sif
# then activate the conda enviroment of the image
conda activate /opt/conda
# then list all the install packages from history
conda env export --from-history > conda_env1.yaml
# then find the exact version for important packages
conda env export > conda_env2.yaml
# Add the exact versions to conda_env1.yaml -- This can be automated but since there are only few important packages I'm doing in manually for: pytorch, python, numpy, torchvision
# Also add pip and its version explicityly 
# Then find pip packages that were installed
pip freeze > pip_list.yaml
# Then copy the pip packages into conda_env1.yaml
# remove conda stuff from pip list
# move mkl from pip to conda. change - to _ in mkl-fft and mkl-random
# change mmcv from 1.3.2 to 1.3.3 because 1.3.2 isn't available anymore.
# change pip to 22 because older version (20) doesn't work well with NFS
# give the git link for poseval git+https://github....git
# change version for tf-slim to 1.1.0
# install tensorflow using conda -- because xtcocotools and tf require different versions of numpy. There are newer versions of TF that are compatible with numpy requirements of xtcocotools but they are missing old style batchnormalization layers. So the way around this is to install TF using conda so that it gets installed first and then install xtcocotools. Ideally xtcocotools should have been installed first to make the order compatible with docker file, but can't install xtcocotools using conda.
# remove versions for almost all packages except h5py, keras, keras-applications, mmcv, scipy, tfslim, torch, torchvision


