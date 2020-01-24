#sudo -H pip2.7 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp27-none-linux_x86_64.whl

# initial AMI:  ami-060865e8b5914b4c4
sudo -H pip2.7 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.13.1-cp27-none-linux_x86_64.whl
sudo -H pip2.7 install imageio Keras==2.2.4 h5py enum EasyDict future scikit-image hdf5storage

# build opencv
# uninstall and reinstall cmake if it doesn't work.
wget https://github.com/opencv/opencv/archive/2.4.13.6.zip -O opencv3.zip 
unzip -q opencv3.zip && mv opencv-2.4.13.6 ~/opencv && rm opencv3.zip 
mkdir /opencv/build && cd /opencv/build \
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D BUILD_PYTHON_SUPPORT=ON \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D BUILD_EXAMPLES=OFF \
      -D WITH_IPP=OFF \
      -D WITH_FFMPEG=ON \
      -D WITH_V4L=ON \
	  -D WITH_CUDA=OFF ..

# install
cd /opencv/build && make -j$(nproc) 
sudo make install 
sudo ldconfig
#clean
sudo apt-get -y remove build-essential cmake git pkg-config libatlas-base-dev gfortran \
libjasper-dev libgtk2.0-dev libavcodec-dev libavformat-dev \
libswscale-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libv4l-dev \
apt-get clean 
cd ~
rm -rf ~/opencv ~/opencv_contrib
sudo pip install opencv-python


#install ffmpeg
apt-get install -y ffmpeg libav-tools x264 x265 
