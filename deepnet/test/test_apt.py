import sys
if sys.version_info.major > 2:
    import urllib.request as url_lib
else:
    import urllib as url_lib
sys.path.append('..')
import tempfile
import os
import APT_interface as apt
import ast
import glob
import re
import numpy as np
import PoseTools
import apt_expts
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    # net_types = ['mdn','deeplabcut','openpose','leap','unet','resnet_unet']
    net_types = ['mdn']
    n_views = 1
    exp_name = 'alice_test'
    op_af_graph = '(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16)'
    bsz = 8
    dl_steps = 150 #15000 #
    ptiles = [50,75,90,95]
    tdir = tempfile.mkdtemp()

    lbl_file_url = 'https://www.dropbox.com/s/mo2jfdmzzsz6btc/alice_sz90_stripped.lbl?dl=1'
    lbl_file = os.path.join(tdir, 'alice_stripped.lbl')
    url_lib.urlretrieve(lbl_file_url, lbl_file)

    gt_file_url = 'https://www.dropbox.com/s/71glyy7bgkry7sm/gtdata_view0.tfrecords?dl=1'
    gt_file = os.path.join(tdir, 'gt_data.tfrecords')
    url_lib.urlretrieve(gt_file_url, gt_file)

    res_file_url = 'https://www.dropbox.com/s/cr702321rvv3htl/alice_view0_time.mat?dl=1'
    res_file = os.path.join(tdir,'alice_view0_time.mat')
    url_lib.urlretrieve(res_file_url,res_file)

    cmd = '-cache {} -name {} -conf_params batch_size {} dl_steps {} op_affinity_graph {} -type {{}} {} train -use_cache '.format(tdir, exp_name, bsz, dl_steps,op_af_graph, lbl_file)


    ##
    import h5py
    R = h5py.File(res_file,'r')

    for net in net_types:
        apt.main(cmd.format(net).split())

        conf = apt.create_conf(lbl_file, 0, exp_name, tdir, net)
        conf.batch_size = 1
        # if data_type == 'stephen' and train_type == 'mdn':
        #     conf.mdn_use_unet_loss = False
        if op_af_graph is not None:
            conf.op_affinity_graph = ast.literal_eval(op_af_graph.replace('\\', ''))
        files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format('deepnet'))
        files.sort(key=os.path.getmtime)
        files = [f for f in files if os.path.splitext(f)[1] in ['.index', '']]
        aa = [int(re.search('-(\d*)', f).groups(0)[0]) for f in files]
        aa = [b - a for a, b in zip(aa[:-1], aa[1:])]
        if any([a < 0 for a in aa]):
            bb = int(np.where(np.array(aa) < 0)[0]) + 1
            files = files[bb:]
        files = files[-1:]
        # n_max = 10
        # if len(files)> n_max:
        #     gg = len(files)
        #     sel = np.linspace(0,len(files)-1,n_max).astype('int')
        #     files = [files[s] for s in sel]


        afiles = [f.replace('.index', '') for f in files]
        mdn_out = apt_expts.classify_db_all(conf, gt_file, afiles, net, name='deepnet')
        preds = mdn_out[0][0]
        labels = mdn_out[0][1]

        res_iters = [R[e]['model_iter'].value[0,0] for e in R[net][:,0]]
        cndx = np.array([abs(r-dl_steps) for r in res_iters]).argmin()
        pred_saved = R[R[net][cndx,0]]['pred'].value.transpose([2,1,0])
        labels_saved = R[R[net][cndx,0]]['labels'].value.transpose([2,1,0])
        dd_saved = np.sqrt(np.sum((pred_saved-labels)**2,-1))
        pp_saved = np.percentile(dd_saved,ptiles,axis=0)

        dd_cur = np.sqrt(np.sum((preds-labels)**2,-1))
        pp_cur = np.percentile(dd_cur,ptiles,axis=0)
        assert(np.array_equal(labels,labels_saved))

        print(' ---- {} Results  ----- '.format(net))
        print(pp_cur-pp_saved)

if __name__ == "__main__":
    main()

