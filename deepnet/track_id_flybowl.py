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
import hdf5storage
from scipy import io as sio
from numpy.core.records import fromarrays


def parse_args(argv: list):

    parser = argparse.ArgumentParser(description='Apply ID tracking to fly tracker or CTrax tracked data')
    parser.add_argument("-mov", dest="mov",help="movie(s) to track. For multi-view projects, specify movies for all the views or specify the view for the single movie using -view", nargs='+')
    parser.add_argument("-trx", dest="trx",help='trx file for movie. Default is to replace movie with "registered_trx.mat"', default=None, nargs='*')
    parser.add_argument('-out', dest='out_files', help='file to save tracking results to', required=True, nargs='+')
    parser.add_argument('-crop_sz', help='crop sz to use for ID tracking',type=int,default=108)

    args = parser.parse_args(argv)
    return args

def convert_trk2trx(trk, orig_trx_file):
    """
    Convert the id linked trk into JAABA compatible trx file
    :param trk:
    :type trk:
    :param orig_trx_file:
    :type orig_trx_file:
    :return:
    :rtype:
    """
    ntargets = trk.ntargets
    trx = [{} for n in range(ntargets)]
    otrx_a = hdf5storage.loadmat(orig_trx_file)
    otrx = otrx_a['trx']

    o_keys = ['a','b','x_mm','y_mm','theta_mm','a_mm','b_mm',
              'wing_angler','wing_anglel','xwingl','xwingr','ywingl','ywingr',
              'sex','timestamps']

    # Convert head-tail points back to trx
    for itgt in range(ntargets):
        pts = trk.pTrk.data[itgt]
        x_mat = to_mat(pts[:,0].mean(axis=0).flatten())
        y_mat = to_mat(pts[:,1].mean(axis=0).flatten())
        theta = np.arctan2(pts[0,1]-pts[1,1],pts[0,0]-pts[1,0])
        cur_len = len(x_mat)
        trx[itgt]['x'] = x_mat
        trx[itgt]['y'] = y_mat
        trx[itgt]['theta'] = theta
        trx[itgt]['firstframe'] = float(trk.startframes[itgt]+1)
        trx[itgt]['endframe'] = float(trk.endframes[itgt] + 1)
        trx[itgt]['nframes'] = float(trx[itgt]['endframe'] - trx[itgt]['firstframe'] + 1)
        trx[itgt]['off'] = -float(trk.startframes[itgt])
        trx[itgt]['pxpermm'] = otrx[0,0]['pxpermm'][0,0]
        trx[itgt]['arena'] = {'x':otrx[0,0]['arena']['x'][0,0][0,0],
                              'y':otrx[0,0]['arena']['y'][0,0][0,0],
                              'r':otrx[0,0]['arena']['r'][0,0][0,0]}

        for fn in o_keys:
            if fn == 'sex':
                trx[itgt][fn] = [np.array(['m']) for i in range(cur_len)]
            else:
                trx[itgt][fn] = np.ones(cur_len)*np.nan

        # For all the other remaining fields find the match in trx and fill them in
        for ndx in range(len(x_mat)):
            if np.isnan(x_mat[ndx]): continue
            curf = trk.startframes[itgt] + ndx
            match = None
            all_d = []
            for tgt in range(otrx.shape[1]):
                if otrx['firstframe'][0,tgt][0,0] > (curf+1): continue
                if otrx['endframe'][0,tgt][0,0] < (curf+1): continue
                off = curf + otrx['off'][0,tgt][0,0]
                cur_tx = otrx['x'][0,tgt][0,off]
                cur_ty = otrx['y'][0,tgt][0,off]
                cur_theta = otrx['theta'][0,tgt][0,off]
                d = abs( cur_tx-x_mat[ndx]) + abs(cur_ty-y_mat[ndx]) + \
                    abs(cur_theta- theta[ndx])
                all_d.append(d)
                if d < 1e-4:
                    match = tgt
                    break

            assert match is not None, f'Could not find match for frame {curf} target {itgt}'

            off = curf + otrx['off'][0, match][0,0]
            for fn in o_keys:
                trx[itgt][fn][ndx] = otrx[fn][0,match][0,off]
        trx[itgt]['dt'] = np.diff(trx[itgt]['timestamps'])
        trx[itgt]['sex'][0].dtype = np.void

    # convert the data into records so that they get saved as struct array.
    k = list(trx[0].keys())
    v_trx = []
    for curk in k:
        curv = []
        for ndx in range(len(trx)):
            curv.append(trx[ndx][curk])
        v_trx.append(curv)

    out_trx = {'trx':fromarrays(v_trx, names=k)}
    for k in otrx_a.keys():
        if k.startswith('__') or k == 'trx': continue
        out_trx[k] = otrx_a[k]
    return out_trx


def track_id_flybowl(argv: list):
    """
    ID tracks fly trx data
    :param argv:
    :return:
    :rtype:
    """
    args = parse_args(argv)
    if args.trx is None:
        args.trx = [m.replace(Path(m).name,'registered_trx.mat') for m in args.mov]

    msz = args.crop_sz

    # conf parameters for fly id tracking
    conf.multi_animal_crop_sz = int(msz)  # [128,128]
    conf.has_trx_file = False
    conf.use_bbox_trx = False
    conf.use_ht_trx = True
    conf.img_dim = 3
    conf.trx_align_theta = True
    conf.link_id = True

    tfiles = []
    for trx_file in args.trx:
        # Convert ctrx or flytracker trx data into head-tail trk data and do pure linking.

        tmp_trx = tempfile.mkstemp()[1]
        tfiles.append(tmp_trx)
        trx = TrkFile.load_trx(trx_file)

        nfr = trx['endframes'].max().astype('int')+1
        ncount = np.zeros(nfr)
        for sf,ef in zip(trx['startframes'],trx['endframes']):
            ncount[sf:(ef+1)] += 1
        nt = ncount.max().astype('int')
        plocs = np.ones([nfr, nt, 2, 2]) * np.nan
        ts = np.ones([nfr,nt, 2]) *np.nan
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
                ts[curf,ndone[curf],:] = trx['b'][ndx][zz]
                ndone[curf] += 1

            all_locs.append(locs)

        locs_lnk = np.transpose(plocs, [2, 3, 0, 1])
        ts = np.transpose(ts, [2, 0, 1])
        # ts = np.ones_like(locs_lnk[:, 0, ...])
        tag = np.ones(ts.shape) <0  # tag which is always false for now.
        locs_conf = None

        trk = TrkFile.Trk(p=locs_lnk, pTrkTS=ts, pTrkTag=tag, pTrkConf=locs_conf)
        trk.convert2sparse()
        # Undo trx linking an do pure linking for id tracking
        trk = lnk.link_pure(trk, conf)
        trk.save(tmp_trx, saveformat='tracklet')

    # Do id tracking based linking
    trks = lnk.link_trklets(tfiles,conf,args.mov,args.out_files)

    # convert back into trx
    for ndx  in range(len(trks)):
        trk = trks[ndx]
        out_file = args.out_files[ndx]
        trx_file = args.trx[ndx]
        out_trx = convert_trk2trx(trk, trx_file)
        sio.savemat(out_file,out_trx,appendmat=False)


if __name__ == "__main__":
    track_id_flybowl(sys.argv[1:])