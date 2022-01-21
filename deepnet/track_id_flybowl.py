import sys
import APT_interface as apt
import TrkFile
import link_trajectories as lnk
from poseConfig import conf
import argparse
from pathlib import Path
import numpy as np
import tempfile
from TrkFile import to_mat

def parse_args(argv):

    parser = argparse.ArgumentParser(description='Apply ID tracking to fly tracker or CTrax tracked data')
    parser.add_argument("-mov", dest="mov",help="movie(s) to track. For multi-view projects, specify movies for all the views or specify the view for the single movie using -view", nargs='+')
    parser.add_argument("-trx", dest="trx",help='trx file for movie. Default is to replace movie with "registered_trx.mat"', default=None, nargs='*')
    parser.add_argument('-out', dest='out_files', help='file to save tracking results to', required=True, nargs='+')
    parser.add_argument('-crop_sz', help='crop sz to use for ID tracking',type=int,default=108)

    args = parser.parse_args(argv)
    return args

def convert_trk2trx(trk):
    ntargets = trk.ntargets
    trx = {'x': [None, ] * ntargets, 'y': [None, ] * ntargets, \
           'theta': [None, ] * ntargets, 'a': [None, ] * ntargets, \
           'b': [None, ] * ntargets, 'startframes': np.zeros(ntargets, dtype=int), \
           'endframes': np.zeros(ntargets, dtype=int),
           'nframes': np.zeros(ntargets, dtype=int)}

    for itgt in range(ntargets):
        pts = trk.pTrk.data[itgt]
        trx['x'][itgt] = pts[0].mean(axis=1).flatten()
        trx['y'][itgt] = pts[1].mean(axis=1).flatten()
        trx['theta'][itgt] = np.arctan2(pts[1,0]-pts[1,1],pts[0,0]-pts[0,1])
        trx['a'][itgt] = np.linalg.norm(pts[0]-pts[1],axis=0)/4
        trx['b'][itgt] = td['trx']['b'][0, itgt].flatten()
        trx['startframes'][itgt] = trk.firstframes[itgt]
        trx['endframes'][itgt] = trk.endframes[itgt] + 1

    trx['x'] = to_mat(trx['x'])
    trx['y'] = to_mat(trx['y'])
    trx['startframes'] = to_mat(trx['startframes'])
    trx['endframes'] = to_mat(trx['endframes'])
    trx['nframes'] = trx['endframes'] - trx['startframes'] + 1

    return trx


def track_id_flybowl(argv):
    args = parse_args(argv)
    if args.trx is None:
        args.trx = [m.replace(Path(m).name,'registered_trx.mat') for m in args.mov]

    msz = args.crop_sz
    conf.multi_animal_crop_sz = int(msz)  # [128,128]
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
        tag = np.ones(ts.shape) <0  # tag which is always false for now.
        locs_conf = None

        trk = TrkFile.Trk(p=locs_lnk, pTrkTS=ts, pTrkTag=tag, pTrkConf=locs_conf)
        trk.convert2sparse()
        trk = lnk.link_pure(trk, conf)
        trk.save(tmp_trx, saveformat='tracklet')

    trks = lnk.link_trklets(tfiles,conf,args.mov,args.out_files)
    for trk, out_file in zip(trks,args.out_files):
        trk.save(out_file,saveformat='tracklet')


if __name__ == "__main__":
    track_id_flybowl(sys.argv[1:])