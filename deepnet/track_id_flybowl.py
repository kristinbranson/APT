import sys
import APT_interface as apt
import TrkFile
import link_trajectories as lnk
from poseConfig import conf
import argparse
from pathlib import Path
import numpy as np
import tempfile

def parse_args(argv):

    parser = argparse.ArgumentParser(description='Apply ID tracking to fly tracker or CTrax tracked data')
    parser.add_argument("-mov", dest="mov",help="movie(s) to track. For multi-view projects, specify movies for all the views or specify the view for the single movie using -view", nargs='+')
    parser.add_argument("-trx", dest="trx",help='trx file for movie. Default is to replace movie with "registered_trx.mat"', default=None, nargs='*')
    parser.add_argument('-out', dest='out_files', help='file to save tracking results to', required=True, nargs='+')
    parser.add_argument('-crop_sz', help='crop sz to use for ID tracking',type=int,default=108)

    args = parser.parse_args(argv)
    return args


def track_id_flybowl(argv):
    args = parse_args(argv)
    if args.trx is None:
        args.trx = [m.replace(Path(m).name,'registered_trx.mat') for m in args.mov]

    msz = args.crop_sz
    conf.imsz = [int(msz), int(msz)]  # [128,128]
    conf.has_trx_file = False
    conf.use_bbox_trx = False
    conf.use_ht_trx = True
    conf.img_dim = 3
    conf.trx_align_theta = True
    conf.link_id = True

    tfiles = []
    for trx_file in args.trx:

        tmp_trx = tempfile.mkstemp()[1]
        tfiles.append(tmp_trx)
        trx = TrkFile.load_trx(trx_file)

        nfr = trx['endframes'].max().astype('int')+1
        ncount = np.zeros(nfr)
        for sf,ef in zip(trx['startframes'],trx['endframes']):
            ncount[sf:(ef+1)] += 1
        nt = ncount.max().astype('int')
        plocs = np.ones([nfr, nt, 2, 2]) * np.nan
        ndone = np.zeros(nfr).astype('int')

        all_locs = []
        for ndx in range(len(trx['x'])):
            theta = trx['theta'][ndx]
            locs_hx = trx['x'][ndx] + trx['a'][ndx] * 2 * np.cos(theta)
            locs_hy = trx['y'][ndx] + trx['a'][ndx] * 2 * np.sin(theta)
            locs_tx = trx['x'][ndx] - trx['a'][ndx] * 2 * np.cos(theta)
            locs_ty = trx['y'][ndx] - trx['a'][ndx] * 2 * np.sin(theta)
            locs_h = np.array([locs_hx, locs_hy])
            locs_t = np.array([locs_tx, locs_ty])
            locs = np.array([locs_h, locs_t])
            for zz in range(locs.shape[2]):
                curf = trx['startframes'][ndx] + zz
                plocs[curf, ndone[curf]] = locs[..., zz]
                ndone[curf] += 1

            all_locs.append(locs)

        locs_lnk = np.transpose(plocs, [2, 3, 0, 1])

        ts = np.ones_like(locs_lnk[:, 0, ...])
        tag = np.ones(ts.shape) * np.nan  # tag which is always false for now.
        locs_conf = None

        trk = TrkFile.Trk(p=locs_lnk, pTrkTS=ts, pTrkTag=tag, pTrkConf=locs_conf)
        trk.convert2sparse()
        trk = lnk.link_pure(trk, conf)
        trk.save(tmp_trx, saveformat='tracklet')

    lnk.link_trklets(tfiles,conf,args.mov,args.out_files)


if __name__ == "__main__":
    track_id_flybowl(sys.argv[1:])