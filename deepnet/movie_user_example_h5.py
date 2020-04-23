import os
import sys

import cv2
import numpy as np
import h5py #,cv2 #conda install h5py, cv2
from scipy import io as sio

'''
Example of user/custom movie-reader class for use with APT/movie.py. In this example,
the custom class reads image frames from an h5 dataset with key specified by an auxiliary 
(mat)-file. 
'''

def loadmat(filename):
    '''
    Load a matfile
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True, appendmat=False)
    return data

class MovieReader:
    def __init__(self, movname):
        movdir = os.path.dirname(movname)
        movds = os.path.join(movdir, 'dataset.mat')

        assert os.path.exists(movds), "movie_dataset file {} does not exist.".format(movds)
        ds = loadmat(movds)
        cam = ds['movie_dataset']
        if cam.endswith('mov1'):
            cam = 'mov1'
        elif cam.endswith('mov0'):
            cam = 'mov0'
        else:
            assert False, "Unrecognized camera spec in {}".format(movds)

        data = h5py.File(movname, 'r')
        movdata = data[cam]

        self.filename = movname
        self.h5file = data
        self.cam = cam
        self.movdata = movdata

    def get_frame(self, framenumber):
        """
        Return numpy array containing frame data.

        framenumber: index or slice etc
        """
        im = self.movdata[framenumber, ...]
        nd = im.ndim
        # 'transpose' last two axes to match with matlab
        # Note: assumes grayscale im
        im = np.swapaxes(im, nd-2, nd-1)
        return im, framenumber

    def get_n_frames(self):
        return self.movdata.shape[0]

    def get_width(self):
        # 'transpose' to match with matlab
        return self.movdata.shape[1]

    def get_height(self):
        # 'transpose' to match with matlab
        return self.movdata.shape[2]

    def get_all_timestamps(self):
        nf = self.get_n_frames()
        return np.arange(nf)

    def close(self):
        if hasattr(self, 'h5file'):
            self.h5file.close()

def main(argv):
    filename = argv[0]
    km = MovieReader(filename)
    print("mov shape is ({},{},{})".format(km.get_n_frames(),
        km.get_height(), km.get_width()))

    cv2.namedWindow('Playback')
    for frame in range(100):
        im, _ = km.get_frame(frame)
        cv2.imshow('Playback', im)
        cv2.waitKey(16)

    cv2.destroyAllWindows()
    print("Closing window")

if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) == 0:
        argv.append('/dat0/Dropbox/aptCondaKelly20200227/2187759634/movie.h5')
    main(argv)

