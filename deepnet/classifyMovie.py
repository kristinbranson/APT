import argparse
import tensorflow
import importlib
import hdf5storage
import PoseUNet
import cv2
from cvc import cvc
import numpy as np
import re

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("conffile",
                        help="config file",
                        required=True)
    parser.add_argument("confname",
                        help="conf name for importing",
                        required=True)
    parser.add_argument("net_name",
                        help="Name the network to classify with",
                        required=True)
    parser.add_argument("movie",
                        help="movie to classify",
                        required=True)
    parser.add_argument("out name",
                        help="file to save results to",
                        required=True)

    args = parser.parse_args()
    imp_mod = importlib.import_module(args.conffile)
    conf = imp_mod.__dict__[args.confname]
    oname = re.sub('!', '__', conf.getexpname(args.movie))

    self = PoseUNet.PoseUNet(conf, args.net_name)
    sess = self.init_net_meta(0,True)

    predList = self.classify_movie(args.movie, sess, flipud=True)

    cap = cv2.VideoCapture(args.movie)
    height = int(cap.get(cvc.FRAME_HEIGHT))
    width = int(cap.get(cvc.FRAME_WIDTH))
    rescale = conf.unet_rescale
    orig_crop_loc = conf.cropLoc[(height,width)]
    crop_loc = [int(x/rescale) for x in orig_crop_loc]
    end_pad = [int((height-conf.imsz[0])/rescale)-crop_loc[0],int((width-conf.imsz[1])/rescale)-crop_loc[1]]
    pp = [(0,0),(crop_loc[0],end_pad[0]),(crop_loc[1],end_pad[1]),(0,0)]
    predScores = np.pad(predList[1],pp,mode='constant',constant_values=-1.)

    predLocs = predList[0]
    predLocs[:,:,0] += orig_crop_loc[1]
    predLocs[:,:,1] += orig_crop_loc[0]

    hdf5storage.savemat(args.outname,{'locs':predLocs,'scores':predScores,'expname':args.movie},appendmat=False,truncate_existing=True)

    print('Done Detecting:%s'%oname)


if __name__ == "__main__":
   main(sys.argv[1:])
