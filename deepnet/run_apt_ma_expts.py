import run_apt_expts_2 as rae
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from importlib import reload
import APT_interface as apt
reload(apt)
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.ion()
import PoseTools as pt
import numpy as np
import cv2
from matplotlib.pyplot import imshow
import scipy.optimize as opt
import pathlib
import pickle
import json
import glob
import re
import datetime
import multiResData
import h5py
from scipy import io as sio
import multiResData
from reuse import *
import hdf5storage
import pickle

out_dir = '/groups/branson/bransonlab/mayank/apt_results'
loc_file_str = 'loc.json'
loc_file_str_neg = 'loc_neg.json'  # loc file with neg ROIs
loc_file_str_inc = 'loc_neg_inc_{}.json'
alg_names = ['2stageHT', '2stageBBox', 'grone', 'openpose']
cache_dir = '/groups/branson/bransonlab/mayank/apt_cache_2'
run_dir = '/groups/branson/bransonlab/mayank/APT/deepnet'
n_round_inc = 8
n_min_inc = 10


class ma_expt(object):

    bdir = None
    dstr = None
    mask = None
    crops = None
    name = None
    imsz = None

    def __init__(self, data_type):
        self.name = data_type
        if data_type == 'roian':
            self.bdir = '/groups/branson/home/kabram/temp/ma_expts/roian'
            self.dstr = '23022022'# '08022022' #
            self.imsz = 2048
            self.train_dstr = '23022022' #'10022022' #
            self.crop_types = ('crop','nocrop')
            self.mask_types = ('mask','nomask')
            # self.params = {}
            self.params = {('grone','nocrop'):[{'rescale':2,'batch_size':4},{}],
                          ('2stageHT', 'nocrop'): [{'rescale': 2,'batch_size':4},{}],
#                            ('2stageBBox',):[{'batch_size':3},{}],
#                            ('grone',):[{'batch_size':4}],
                           ('openpose','crop'):[{'rescale':2},{}],
#                            ('openpose','crop'):[{'rescale':2}],
                           ('openpose','nocrop'):[{'rescale':4,'batch_size':4}]
                           }
            self.ex_db = '/nrs/branson/mayank/apt_cache_2/four_points_180806/mdn_joint_fpn/view_0/roian_split_ht_grone_single/val_TF.tfrecords'
            self.n_pts = 4
            self.dropoff = 0.4

        elif data_type == 'alice':
            self.bdir = '/groups/branson/home/kabram/temp/ma_expts/alice'
            self.dstr = '23022022'#'14022022'
            self.imsz = 1024
            self.train_dstr = '23022022'#'14022022'
            self.crop_types = ('crop', 'nocrop')
            self.mask_types = ('mask', )
            self.params = {('2stageHT', 'nocrop'): [{'rescale': 2}, {}],
            #                ('2stageBBox',): [{'batch_size': 3}, {}],
                           ('grone', 'nocrop'): [{'rescale': 2},{}],
                           ('openpose', 'nocrop'): [{'rescale': 2},{}],
            #                ('openpose', 'crop'): [{'rescale': 2}]
                           }
            self.ex_db = '/nrs/branson/mayank/apt_cache_2/alice_ma/mdn_joint_fpn/view_0/alice_split_ht_grone_single/val_TF.tfrecords'
            self.n_pts = 17
            self.dropoff = 0.7
        self.trnp_dir = os.path.join(self.bdir,f'trn_packdir_{self.dstr}')
        self.gt_dir = os.path.join(self.bdir,f'gt_packdir_{self.dstr}')
        self.log_dir = os.path.join(self.bdir,'log')
        self.im_dir = os.path.join(self.bdir,'sample_ims')
        self.trk_dir = os.path.join(self.bdir,'trks')
        os.makedirs(self.log_dir,exist_ok=True)
        os.makedirs(self.im_dir,exist_ok=True)
        os.makedirs(self.trk_dir,exist_ok=True)


    def get_neg_roi_alice(self, debug=False):
        im_sz = (1024, 1024)
        boxsz = 80
        num_negs = 200 if not debug else 10
        negbox_sz = 240

        in_loc = os.path.join(self.trnp_dir, 'grone', loc_file_str)
        T = pt.json_load(in_loc)

        # Add neg ROI boxes based on ctrax trx
        done_ix = []
        totcount = 0
        neg_info = []
        while totcount < num_negs:
            ix = np.random.choice(len(T['locdata']))
            done_ix.append(ix)
            curp = T['locdata'][ix]
            tt = os.path.split(T['movies'][curp['imov'] - 1])
            trx = sio.loadmat(os.path.join(tt[0], 'registered_trx.mat'))['trx'][0]
            ntrx = len(trx)
            fnum = curp['frm']-1
            all_mask = np.zeros(im_sz)
            boxes = []
            for tndx in range(ntrx):
                cur_trx = trx[tndx]
                if fnum > cur_trx['endframe'][0, 0] - 1:
                    continue
                if fnum < cur_trx['firstframe'][0, 0] - 1:
                    continue
                x, y, theta = multiResData.read_trx(cur_trx, fnum)
                x = int(round(x))
                y = int(round(y))
                x_min = max(0, x - boxsz)
                x_max = min(im_sz[1], x + boxsz)
                y_min = max(0, y - boxsz)
                y_max = min(im_sz[0], y + boxsz)
                all_mask[y_min:y_max, x_min:x_max] = 1
                boxes.append([x_min, x_max, y_min, y_max])

            done = False
            selb = []
            for count in range(20):
                negx_min = np.random.randint(im_sz[1] - negbox_sz)
                negy_min = np.random.randint(im_sz[0] - negbox_sz)
                if np.any(all_mask[negy_min:negy_min + negbox_sz, negx_min:negx_min + negbox_sz] > 0):
                    continue
                done = True
                selb = [negx_min, negx_min + negbox_sz, negy_min, negy_min + negbox_sz]
                break

            if debug and done:
                im = cv2.imread(os.path.join(self.trnp_dir, curp['img'][0]), cv2.IMREAD_UNCHANGED)
                f, ax = plt.subplots(1, 2)
                ax[0].cla()
                ax[0].imshow(im, 'gray')
                for b in boxes:
                    ax[0].plot([b[0], b[1], b[1], b[0], b[0]], [b[2], b[2], b[3], b[3], b[2]])
                ax[0].plot([selb[0], selb[1], selb[1], selb[0], selb[0]],
                           [selb[2], selb[2], selb[3], selb[3], selb[2]])
                cc = all_mask.copy()
                cc[selb[2]:selb[3], selb[0]:selb[1]] = -1
                ax[1].imshow(cc)

            if done:
                neg_info.append({'loc_ndx':ix,
                                 'mov_ndx':curp['imov']-1,
                                 'frm': curp['frm']-1,
                                'neg_box':selb
                })

                totcount = totcount + 1
                # print(f'Adding Roi for {ix}')

        return neg_info


    def add_neg_roi_alice(self, force=False, reload=False):

        assert self.name =='alice'

        save_file = os.path.join(self.trnp_dir,'neg_roi_data.pkl')
        if reload:
            neg_info = pt.pickle_load(save_file)
        else:
            neg_info = self.get_neg_roi_alice()
            with open(save_file,'wb') as f:
                pickle.dump(neg_info,f)


        for alg in alg_names:
            in_loc = os.path.join(self.trnp_dir, alg, loc_file_str)
            out_loc = os.path.join(self.trnp_dir, alg, loc_file_str_neg)
            T = pt.json_load(in_loc)
            for curn in neg_info:
                ix = curn['loc_ndx']
                curp = T['locdata'][ix]
                negx_min = curn['neg_box'][0]
                negy_min = curn['neg_box'][2]
                negbox_sz = curn['neg_box'][1] - curn['neg_box'][0]
                curp['extra_roi'] = [negx_min, negx_min,
                                     negx_min + negbox_sz,
                                     negx_min + negbox_sz,
                                     negy_min, negy_min + negbox_sz,
                                     negy_min + negbox_sz, negy_min]
                curp['nextraroi'] = 1

            if not os.path.exists(out_loc) or force:
                with open(out_loc,'w') as f:
                    json.dump(T,f)
            else:
                print(f'Neg loc file {out_loc} exists, so not overwriting')


    def add_neg_roi_roian(self, force=False, reload=False):
        assert self.name =='roian'
        n_add = 200

        in_loc = os.path.join(self.trnp_dir,'grone',loc_file_str)
        A = pt.json_load(in_loc)
        n_fr = len(A['locdata'])

        save_file = os.path.join(self.trnp_dir,'neg_roi_data.pkl')
        if reload:
            sel_fr, sel_starts = pt.pickle_load(save_file)
        else:
            sel_fr = np.random.choice(n_fr,n_add,replace=False)
            sel_starts = (np.random.rand(n_add*2).reshape([n_add,2])*self.imsz/2).round()
            with open(save_file,'wb') as f:
                pickle.dump([sel_fr,sel_starts],f)

        for alg in alg_names:
            in_loc = os.path.join(self.trnp_dir, alg, loc_file_str)
            out_loc = os.path.join(self.trnp_dir, alg, loc_file_str_neg)
            A = pt.json_load(in_loc)
            locs = A['locdata']
            for ndx in range(n_add):
                cur_l = locs[sel_fr[ndx]]
                cur_s = sel_starts[ndx]
                hsz = self.imsz/2
                cur_roi = [cur_s[0],cur_s[0], cur_s[0]+hsz, cur_s[0]+hsz, cur_s[1],cur_s[1]+hsz, cur_s[1]+hsz, cur_s[1]]
                cur_l['extra_roi'] = cur_roi
                cur_l['nextra_roi'] = 1

            if not os.path.exists(out_loc) or force:
                with open(out_loc,'w') as f:
                    json.dump(A,f)
            else:
                print(f'Neg loc file {out_loc} exists, so not overwriting')


    def get_settings(self, t_type):
        # Gets all the files and other stuff for a combination of alg, stage, crop-type etc

        alg = t_type[0]
        stg = t_type[-1]
        crop = t_type[1]
        params = {}
        if 'mask' in t_type:
            params['multi_loss_mask'] = True
        if 'nomask' in t_type:
            params['multi_loss_mask'] = False

        for curk in self.params.keys():
            # Add combination specific parameters. Usefull for setting batch sizes etc
            if not set(curk).issubset(set(t_type)): continue
            add_params = self.params[curk][1] if stg == 'second' else self.params[curk][0]
            params.update(add_params)

        loc_file = os.path.join(self.trnp_dir, alg, loc_file_str_neg)
        loc_file_inc = os.path.join(self.trnp_dir, alg, loc_file_str_inc)
        lbl_file = os.path.join(self.trnp_dir, alg,  f'conf_{crop}.json')
        conf = pt.json_load(lbl_file)
        if alg.startswith('2stage'):
            stg_str = f'-stage {stg}'
            train_name = '_'.join(t_type + (self.train_dstr,))
            cndx = 0 if stg == 'first' else 1
            net_type = conf['TrackerData'][cndx]['trnNetTypeString']

        else:
            stg_str = ''
            train_name = '_'.join(t_type[:-1] + (self.train_dstr,))
            net_type = conf['TrackerData']['trnNetTypeString']


        A = pt.json_load(lbl_file)
        proj_name = A['ProjName']
        train_cache_dir = os.path.join(cache_dir,proj_name,net_type,'view_{}',train_name)
        nviews = A['Config']['NumViews']
        settings = {'stg_str': stg_str,
                    'train_name': train_name,
                    'net_type': net_type,
                    'loc_file': loc_file,
                    'lbl_file': lbl_file,
                    'params': params,
                    'proj_name': proj_name,
                    'nviews': nviews,
                    'loc_file_inc':loc_file_inc,
                    'train_cache_dir': train_cache_dir}

        return settings

    def get_types(self,t_types):
        all_types = []
        for alg in alg_names:
            for stg in ['first', 'second']:
                if not alg.startswith('2stage') and stg == 'second':
                    continue
                if stg == 'second':
                    all_types.append((alg, 'nocrop', stg))
                    continue
                for c in self.crop_types:
                    for m in self.mask_types:
                        all_types.append((alg, c, m, stg))
        if t_types is None:
            t_types = all_types
        else:
            use_types = []
            for cur_in_type in t_types:
                for cur_type in all_types:
                    if set(cur_in_type).issubset(set(cur_type)):
                        use_types.append(cur_type)
            t_types = use_types

        new_t_types = []
        for cur_type in t_types:
            if set(('2stageBBox','crop')).issubset(set(cur_type)):
                continue
            else:
                new_t_types.append(cur_type)

        t_types = new_t_types

        return t_types

    def run_train(self, t_types=None, redo=False,queue='gpu_tesla',run_type='dry'):
        t_types = self.get_types(t_types)
        for t in t_types:
            settings = self.get_settings(t)
            lbl_file = settings['lbl_file']
            train_name = settings['train_name']
            loc_file = settings['loc_file']
            stg_str = settings['stg_str']
            net_type = settings['net_type']
            params= settings['params']
            alg = t[0]
            job_name = self.name + '_' + train_name

            conf_str = ' '.join([f'{k} {v}' for k, v in params.items()])
            err_file = os.path.join(self.trnp_dir,alg,train_name + '.err')

            cmd = f'APT_interface.py {lbl_file} -name {train_name} -json_trn_file {loc_file} -conf_params {conf_str} -cache {cache_dir} {stg_str} -type {net_type} train -use_cache'
            precmd = 'export CUDA_VISIBLE_DEVICES=0'

            if run_type == 'dry':
                print()
                print(f'{job_name}:')
                print(f'{cmd}')
            elif run_type == 'submit':
                rae.run_jobs(job_name,
                         cmd,
                         redo=redo,
                         run_dir=run_dir,
                         queue=queue,
                         precmd='',
                         logdir=self.log_dir, nslots=3)
        print(f'Total jobs {len(t_types)}')

    def show_samples(self,t_types=None,save=False):
        t_types = self.get_types(t_types)
        for t in t_types:
            settings = self.get_settings(t)
            train_name = settings['train_name']
            train_cache_dir = settings['train_cache_dir']
            nviews = settings['nviews']
            for v in range(nviews):
                sample_file = os.path.join(train_cache_dir.format(v),'deepnet_training_samples.mat')
                f = pt.show_sample_images(sample_file,extra_txt=train_name)
                if save:
                    f.savefig(os.path.join(self.im_dir,train_name + '.png'))


    def get_model_files(self, settings, view):
        train_cache_dir = settings['train_cache_dir'].format(view)
        train_name = settings['train_name']

        run_name = 'deepnet'

        train_dist = -1
        val_dist = -1

        files1 = glob.glob(os.path.join(train_cache_dir, "{}-[0-9]*").format(run_name))
        files2 = glob.glob(os.path.join(train_cache_dir, "{}_202[0-9][0-9][0-9][0-9][0-9]-[0-9]*").format(run_name))
        files = files1 + files2

        files.sort(key=os.path.getmtime)
        files = [f for f in files if os.path.splitext(f)[1] in ['.index', '']]
        return files

    def get_status(self, t_types=None):
        t_types = self.get_types(t_types)
        for t in t_types:
            settings = self.get_settings(t)
            lbl_file = settings['lbl_file']
            train_name = settings['train_name']
            loc_file = settings['loc_file']
            stg_str = settings['stg_str']
            net_type = settings['net_type']
            params= settings['params']
            proj_name = settings['proj_name']
            alg = t[0]
            run_name = 'deepnet'
            for v in range(settings['nviews']):
                files = self.get_model_files(settings,v)

                train_cache_dir = settings['train_cache_dir'].format(v)
                scriptfile = os.path.join(self.log_dir, train_name + '.bsub.sh')
                submit_time = os.path.getmtime(scriptfile)
                # latest model, time, train_dist etc
                if len(files) > 0:
                    latest = files[-1]
                    latest_model_iter = int(re.search('-(\d*)', latest).groups(0)[0])
                    latest_time = os.path.getmtime(latest)
                    if latest_time < submit_time:
                        latest_time = np.nan
                        latest_model_iter = -1
                    else:
                        # if submit_time is nan, assume the latest model is up to date
                        tfile = rae.get_traindata_file_flexible(train_cache_dir, run_name, proj_name)
                        A = pt.pickle_load(tfile)
                        if type(A) is list:
                            train_dist = A[0]['train_dist'][-1]
                            val_dist = A[0]['val_dist'][-1]
                else:
                    latest_model_iter = -1
                    latest_time = np.nan
                    train_dist = -1
                    val_dist = -1

            # trn time
            sec_per_iter = np.nan
            if len(files) > 0:
                first = files[0]
                first_model_iter = int(re.search('-(\d*)', first).groups(0)[0])
                first_time = os.path.getmtime(first)
                if latest_time > submit_time and first_time > submit_time:
                    diter = latest_model_iter - first_model_iter
                    dtime = latest_time - first_time
                    sec_per_iter = np.array(dtime) / np.array(diter)
                    # min_per_5kiter = sec_per_iter * 5000 / 60
            if np.isnan(sec_per_iter) or np.isinf(sec_per_iter):
                trntime5kiter = '---'
            else:
                trntime5kiter = str(datetime.timedelta(seconds=np.round(sec_per_iter * 5000)))

            print(
                'latest iter: {:06d} at {}, {:45s}. submit: {}. train:{:.2f} val:{:.2f}. trntime/5kiter:{}'.format(
                    latest_model_iter, rae.get_tstr(latest_time), train_name, rae.get_tstr(submit_time),
                    train_dist, val_dist, trntime5kiter))

    def setup_incremental(self, force=False, reload=False):

        n_rounds = n_round_inc
        n_min = n_min_inc

        in_loc = os.path.join(self.trnp_dir, 'grone', loc_file_str)
        ldata = pt.json_load(in_loc)
        ntgts = np.array([l['ntgt'] for l in ldata['locdata']])
        nlabels = ntgts.sum()
        n_samples = np.logspace(np.log10(n_min), np.log10(nlabels), n_rounds).round().astype('int')

        save_file = os.path.join(self.trnp_dir,'inc_data.pkl')
        if reload:
            inc_info = pt.pickle_load(save_file)
            assert np.array_equal(ntgts,inc_info['ntgts'])
            assert np.array_equal(n_samples,inc_info['n_samples'])
            sel = inc_info['sel']
            perm_lbls = inc_info['perm_lbls']
        else:

            perm_lbls = np.random.permutation(len(ntgts))
            lbl_count = np.cumsum(ntgts[perm_lbls])
            sel = []
            for ndx in range(n_rounds):
                thres_ndx = np.where(lbl_count >= n_samples[ndx])[0][0]
                cur_sel = perm_lbls[:thres_ndx + 1]
                sel.append(cur_sel)
            inc_info = {'sel':sel,'perm_lbls':perm_lbls,'ntgts':ntgts,'n_samples':n_samples}
            with open(save_file,'wb') as f:
                pickle.dump(inc_info,f)


        for ndx in range(n_rounds):
            for alg in alg_names:
                in_loc = os.path.join(self.trnp_dir, alg, loc_file_str_neg)
                out_loc = os.path.join(self.trnp_dir, alg, loc_file_str_inc.format(ndx))
                T = pt.json_load(in_loc)
                T['splitnames'] = ['trn','val']
                trn_count = 0
                val_count = 0
                for ix,t in enumerate(T['locdata']):
                    if ix in sel[ndx]:
                        if type(t['split']) in ['list','tuple']:
                            t['split'] = [1 for _ in t['split']]
                        else:
                            t['split'] = 1
                        trn_count +=1
                    else:
                        if type(t['split']) in ['list','tuple']:
                            t['split'] = [2 for _ in t['split']]
                        else:
                            t['split'] = 2
                        val_count +=1

                if not os.path.exists(out_loc) or force:
                    print(f'Round: {ndx} samples: {n_samples[ndx]} Trn: {trn_count} Val: {val_count} alg:{alg}')
                    with open(out_loc,'w') as f:
                        json.dump(T,f)
                else:
                    print(f'Inc loc file {out_loc} exists, so not overwriting')

    def run_incremental_train(self, t_types=None, redo=False,queue='gpu_rtx',run_type='dry'):
        t_types = self.get_types(t_types)
        j_count = 0
        for t in t_types:
            settings = self.get_settings(t)
            lbl_file = settings['lbl_file']
            train_name = settings['train_name']
            stg_str = settings['stg_str']
            net_type = settings['net_type']
            params= settings['params']
            alg = t[0]
            inc_loc_file = settings['loc_file_inc']

            conf_str = ' '.join([f'{k} {v}' for k, v in params.items()])

            for ndx in range(n_round_inc):
                job_name = self.name + '_' + train_name + f'_inc_{ndx}'
                inc_train_name = f'{train_name}_inc_{ndx}'
                err_file = os.path.join(self.trnp_dir,alg,train_name + f'_inc_{ndx}.err')
                loc_file = inc_loc_file.format(ndx)
                cmd = f'APT_interface.py {lbl_file} -name {inc_train_name} -json_trn_file {loc_file} -conf_params {conf_str} -cache {cache_dir} {stg_str} -type {net_type} train -use_cache'
                precmd = 'export CUDA_VISIBLE_DEVICES=0'

                j_count += 1
                if run_type == 'dry':
                    print()
                    print(f'{job_name}:')
                    print(f'{cmd}')
                elif run_type == 'submit':
                    rae.run_jobs(job_name,
                             cmd,
                             redo=redo,
                             run_dir=run_dir,
                             queue=queue,
                             precmd='',
                             logdir=self.log_dir, nslots=3)
        print(f'Total jobs {j_count}')


    def create_gt_db(self,run_type='dry'):
        loc_file = os.path.join(self.gt_dir,loc_file_str)
#        lbl_file = os.path.join(self.gt_dir,'gt_stripped.json')
        lbl_file = os.path.join(self.trnp_dir,'grone',  f'conf_nocrop.json')

        train_name = 'gt_db'
        net_type = 'multi_mdn_joint_torch'

        cmd = f'APT_interface.py {lbl_file} -name {train_name} -json_trn_file {loc_file} -cache {cache_dir} -type {net_type} train -use_cache -only_aug'

        if run_type == 'dry':
            print()
            print(f'{train_name}:')
            print(f'{cmd}')
        elif run_type == 'submit':
            rae.run_jobs(train_name+'_'+self.name,
                     cmd,
                     run_dir=run_dir,
                     queue='gpu_rtx',
                     precmd='',
                     logdir=self.log_dir, nslots=3)

    def get_results(self, t_types=None, only_last=True):
        if t_types is None:
            t_types = self.get_types((('first',),))
        else:
            t_types = self.get_types(t_types)

        dstr = pt.datestr()
        res = {}
        for curt in t_types:
            settings = self.get_settings(curt)
            net_type = settings['net_type']
            proj_name = settings['proj_name']
            alg = curt[0]
            run_name = 'deepnet'
            all_res_views = []
            curt_str = '_'.join(curt)
            cur_res_file = os.path.join(out_dir, f'{self.name}_ma_res_{dstr}_{curt_str}.pkl')
            for v in range(settings['nviews']):
                gt_db = os.path.join(cache_dir, proj_name, 'multi_mdn_joint_torch', 'view_{}', 'gt_db', 'train_TF.json').format(v)
                train_cache_dir = settings['train_cache_dir'].format(v)
                tfile = rae.get_traindata_file_flexible(train_cache_dir, run_name, proj_name)
                conf = pt.pickle_load(tfile)[1]
                conf.imsz = (self.imsz,self.imsz)
                conf.img_dim = 3
                conf.batch_size = 1
                conf.db_format = 'coco'
                conf.img_dim = 3
                models = self.get_model_files(settings, v)
                if alg.startswith('2stage'):
                    curt2 = (curt[0],'nocrop','second')
                    settings2 = self.get_settings(curt2)
                    train_cache_dir2 = settings2['train_cache_dir'].format(v)
                    tfile2 = rae.get_traindata_file_flexible(train_cache_dir2,run_name,proj_name)
                    conf2 = pt.pickle_load(tfile2)[1]
                    conf2.db_format = 'coco'
                    conf2.img_dim = 3
                    net_type2 = settings2['net_type']
                    models2 = self.get_model_files(settings2,v)
                else:
                    conf2 = None
                    net_type2 = None
                    models2 = None

                cur_res =apt.classify_db_all(net_type,conf,gt_db,conf2=conf2,model_type2=net_type2,img_dir='train',fullret=True)
                preds, locs, info, model_file = cur_res[:4]
                preds = apt.to_mat(preds['locs'])
                locs = apt.to_mat(locs)
                info = apt.to_mat(info)
                out_dict = {'pred_locs': preds, 'labeled_locs': locs, 'list': info, 'model_file': model_file,'all_res':cur_res}

                all_res_views.append([out_dict])

            res[curt_str] = all_res_views
            with open(cur_res_file,'wb') as f:
                pickle.dump(all_res_views,f)

        self.compile_results(dstr)

    def compile_results(self,dstr):
        res_file = os.path.join(out_dir, f'{self.name}_ma_res_{dstr}')
        t_types = self.get_types((('first',),))
        res = {}
        for curt in t_types:
            settings = self.get_settings(curt)
            net_type = settings['net_type']
            proj_name = settings['proj_name']
            alg = curt[0]
            run_name = 'deepnet'
            all_res_views = []
            curt_str = '_'.join(curt)
            cur_res_file = os.path.join(out_dir, f'{self.name}_ma_res_{dstr}_{curt_str}.pkl')
            all_res_views = pt.pickle_load(cur_res_file)
            res['res_'+curt_str] = all_res_views

        hdf5storage.savemat(res_file,{'results':res})

    def get_incremental_results(self, t_types=None, only_last=True):
        if t_types is None:
            t_types = self.get_types((('first',),))
        else:
            t_types = self.get_types(t_types)

        dstr = pt.datestr()
        res = {}
        res_file = os.path.join(out_dir, f'{self.name}_ma_res_inc_{dstr}')
        for curt in t_types:
            settings = self.get_settings(curt)
            net_type = settings['net_type']
            proj_name = settings['proj_name']
            alg = curt[0]
            run_name = 'deepnet'
            curt_str = '_'.join(curt)
            cur_res_file = os.path.join(out_dir, f'{self.name}_ma_res_{dstr}_{curt_str}.pkl')
            train_name = settings['train_name']
            all_res_inc = []
            for v in range(settings['nviews']):
                gt_db = os.path.join(cache_dir, proj_name, 'multi_mdn_joint_torch', 'view_{}', 'gt_db', 'train_TF.json').format(v)
                for ndx in range(n_round_inc):
                    job_name = self.name + '_' + train_name + f'_inc_{ndx}'
                    inc_train_name = f'{train_name}_inc_{ndx}'
                    train_cache_dir = settings['train_cache_dir'].format(v)
                    train_cache_dir = train_cache_dir.replace(train_name,inc_train_name)
                    tfile = rae.get_traindata_file_flexible(train_cache_dir, run_name, proj_name)
                    conf = pt.pickle_load(tfile)[1]
                    conf.imsz = (self.imsz,self.imsz)
                    conf.img_dim = 3
                    conf.batch_size = 1
                    conf.db_format = 'coco'
                    conf.img_dim = 3
                    models = self.get_model_files(settings, v)
                    if alg.startswith('2stage'):
                        curt2 = (curt[0],'nocrop','second')
                        settings2 = self.get_settings(curt2)
                        train_cache_dir2 = settings2['train_cache_dir'].format(v)
                        tfile2 = rae.get_traindata_file_flexible(train_cache_dir2,run_name,proj_name)
                        conf2 = pt.pickle_load(tfile2)[1]
                        conf2.db_format = 'coco'
                        conf2.img_dim = 3
                        net_type2 = settings2['net_type']
                        models2 = self.get_model_files(settings2,v)
                    else:
                        conf2 = None
                        net_type2 = None
                        models2 = None

                    cur_res =apt.classify_db_all(net_type,conf,gt_db,conf2=conf2,model_type2=net_type2,img_dir='train',fullret=True)
                    preds, locs, info, model_file = cur_res[:4]
                    preds = apt.to_mat(preds['locs'])
                    locs = apt.to_mat(locs)
                    info = apt.to_mat(info)
                    out_dict = {'pred_locs': preds, 'labeled_locs': locs, 'list': info, 'model_file': model_file,'all_res':cur_res}

                    all_res_inc.append([out_dict])

            res['res_'+curt_str] = all_res_inc
            with open(cur_res_file,'wb') as f:
                pickle.dump(all_res_inc,f)

        hdf5storage.savemat(res_file,{'results':res})



    def compare_conf(self, t_types=None):
        t_types = self.get_types(t_types)
        confs = []
        t_str = []
        for curt in t_types:
            settings = self.get_settings(curt)
            proj_name = settings['proj_name']
            run_name = 'deepnet'
            t_str.append('_'.join(curt))
            for v in range(settings['nviews']):
                gt_db = os.path.join(cache_dir, proj_name, 'multi_mdn_joint_torch', 'view_{}', 'gt_db', 'train_TF.json').format(v)
                train_cache_dir = settings['train_cache_dir'].format(v)
                tfile = rae.get_traindata_file_flexible(train_cache_dir, run_name, proj_name)
                conf = pt.pickle_load(tfile)[1]
                confs.append(conf)

        diff_vals = []
        ignore = ['labelfile', 'project_file', 'db_format', 'clip_gradients','selpts','stage','n_classes','multi_only_ht','max_n_animals','json_trn_file','is_multi','use_bbox_trx','use_ht_trx','unet_rescale','mdn_use_unet_loss','cachedir','multi_crop_ims','multi_frame_sz','set_exp_name','mmpose_net','print_dataaug_flds','sel_sz']
        for k in dir(confs[0]):
            if k.startswith('__') or k.startswith('get') or k.startswith('op_im'):
                continue
            if k in ignore: continue
            if not all([hasattr(c,k) for c in confs]):
                print(f'{k} not found in all')
                continue
            vals = [getattr(c,k) for c in confs]
            if vals.count(vals[0]) == len(vals):
                continue
            diff_vals.append([k,vals])
        for v in diff_vals:
            print(v[0])
            print(list(zip(t_str,v[1])))


    def show_results1(self, dstr, t_types=None):

        X = multiResData.read_and_decode_without_session(self.ex_db,self.n_pts)
        ex_im = X[0][0]
        ex_loc = X[1][0]

        if t_types is None:
            t_types = self.get_types((('first',),))
        else:
            t_types = self.get_types(t_types)

        n_types = len(t_types)
        nc = int(np.ceil(np.sqrt(n_types)))
        nr = int(np.ceil(n_types/float(nc)))
        prcs = [50,75,90,95,98,99]
        cmap = pt.get_cmap(len(prcs), 'cool')
        f, axx = plt.subplots(nr, nc, figsize=(12, 8), squeeze=False)
        axx = axx.flat
        dropoff = self.dropoff


        for idx,curt in enumerate(t_types):
            settings = self.get_settings(curt)
            net_type = settings['net_type']
            proj_name = settings['proj_name']
            alg = curt[0]
            run_name = 'deepnet'
            all_res_views = []
            curt_str = '_'.join(curt)
            cur_res_file = os.path.join(out_dir, f'{self.name}_ma_res_{dstr}_{curt_str}.pkl')
            all_res_views = pt.pickle_load(cur_res_file)[0][0]

            pp = all_res_views['pred_locs']
            ll = all_res_views['labeled_locs']
            ll[ll < -1000] = np.nan
            dd = np.linalg.norm(ll[:, None] - pp[:, :, None], axis=-1)
            dd1 = find_dist_match(dd)
            valid_l = np.any(~np.isnan(ll[:, :, :, 0]), axis=-1)

            cur_dist = dd1[valid_l]

            ax = axx[idx]
            if ex_im.ndim == 2:
                ax.imshow(ex_im, 'gray')
            elif ex_im.shape[2] == 1:
                ax.imshow(ex_im[:, :, 0], 'gray')
            else:
                ax.imshow(ex_im)

            vv = cur_dist.copy()
            vv[np.isnan(vv)] = 60.
            mm = np.nanpercentile(vv, prcs, axis=0, interpolation='nearest')
            for pts in range(ex_loc.shape[0]):
                for pp in range(mm.shape[0]):
                    c = plt.Circle(ex_loc[pts, :], mm[pp, pts], color=cmap[pp, :], fill=False,
                                   alpha=1 - ((pp + 1) / mm.shape[0]) * dropoff)
                    ax.add_patch(c)
            ttl = '{} '.format(curt_str)
            ax.set_title(ttl)
            ax.axis('off')

        for ndx in range(cmap.shape[0]):
            axx[nc - 1].plot(np.ones([1, 2]), np.ones([1, 2]), color=cmap[ndx, :],
                             alpha=1 - ((pp + 1) / mm.shape[0]) * dropoff)
        axx[nc - 1].legend([f'{ppt}' for ppt in prcs])
        for ax in axx:
            ax.set_xlim([0, ex_im.shape[1]])
            ax.set_ylim([ex_im.shape[0], 0])

        f.tight_layout()
        return f

    def show_results(self, t_types=None):
        res_file_ptn = os.path.join(out_dir,f'{self.name}_ma_res_[0-9]*.mat')

        files = glob.glob(res_file_ptn)
        files.sort(key=os.path.getmtime)
        # res = hdf5storage.loadmat(files[-1])
        K = h5py.File(files[-1],'r')
        X = multiResData.read_and_decode_without_session(self.ex_db,self.n_pts)
        ex_im = X[0][0]
        ex_loc = X[1][0]

        if t_types is None:
            t_types = self.get_types((('first',),))
        else:
            t_types = self.get_types(t_types)

        n_types = len(t_types)
        nc = int(np.ceil(np.sqrt(n_types)))
        nr = int(np.ceil(n_types/float(nc)))
        prcs = [50,75,90,95,98,99]
        cmap = pt.get_cmap(len(prcs), 'cool')
        f, axx = plt.subplots(nr, nc, figsize=(12, 8), squeeze=False)
        axx = axx.flat
        dropoff = self.dropoff

        all_dist = []
        for idx,curt in enumerate(t_types):
            curt_str = '_'.join(curt)

            # K = res[curt_str]
            # ll = K['labeled_locs']
            # pp = K['pred_locs']

            pp = K[K[K['results']['res_'+curt_str][0,0]][0,0]]['pred_locs'][()].T
            ll = K[K[K['results']['res_'+curt_str][0,0]][0,0]]['labeled_locs'][()].T
            ll[ll<-1000] = np.nan
            dd = np.linalg.norm(ll[:,None]-pp[:,:,None],axis=-1)
            dd1 = find_dist_match(dd)
            valid_l = np.any(~np.isnan(ll[:,:,:,0]),axis=-1)

            cur_dist = dd1[valid_l]
            all_dist.append(cur_dist)

            ax = axx[idx]
            if ex_im.ndim == 2:
                ax.imshow(ex_im, 'gray')
            elif ex_im.shape[2] == 1:
                ax.imshow(ex_im[:, :, 0], 'gray')
            else:
                ax.imshow(ex_im)

            vv = cur_dist.copy()
            vv[np.isnan(vv)] = 60.
            mm = np.nanpercentile(vv,prcs,axis=0,interpolation='nearest')
            for pts in range(ex_loc.shape[0]):
                for pp in range(mm.shape[0]):
                    c = plt.Circle(ex_loc[pts, :], mm[pp, pts], color=cmap[pp, :], fill=False,alpha=1-((pp+1)/mm.shape[0])*dropoff)
                    ax.add_patch(c)
            ttl = '{} '.format(curt_str)
            ax.set_title(ttl)
            ax.axis('off')

        for ndx in range(cmap.shape[0]):
            axx[nc-1].plot(np.ones([1, 2]), np.ones([1, 2]), color=cmap[ndx,:],alpha=1-((pp+1)/mm.shape[0])*dropoff)
        axx[nc-1].legend([f'{ppt}' for ppt in prcs])
        for ax in axx:
            ax.set_xlim([0,ex_im.shape[1]])
            ax.set_ylim([ex_im.shape[0],0])

        f.tight_layout()
        return f


    def show_incremental_results(self, t_type):
        X = multiResData.read_and_decode_without_session(self.ex_db,self.n_pts)
        ex_im = X[0][0]
        ex_loc = X[1][0]

        n_types = n_round_inc
        nc = int(np.ceil(np.sqrt(n_types)))
        nr = int(np.ceil(n_types/float(nc)))
        prcs = [50,75,90,95,98,99]
        cmap = pt.get_cmap(len(prcs), 'cool')
        f, axx = plt.subplots(nr, nc, figsize=(12, 8), squeeze=False)
        axx = axx.flat
        dropoff = self.dropoff

        curt = t_type
        curt_str = '_'.join(curt)

        curt_str = '_'.join(curt)
        res_file_ptn = os.path.join(out_dir, f'{self.name}_ma_res_[0-9]*_{curt_str}.pkl')
        files = glob.glob(res_file_ptn)
        files.sort(key=os.path.getmtime)
        # res = hdf5storage.loadmat(files[-1])
        res = pt.pickle_load(files[-1])
        n_rounds = n_round_inc
        n_min = n_min_inc

        in_loc = os.path.join(self.trnp_dir, 'grone', loc_file_str)
        ldata = pt.json_load(in_loc)
        ntgts = np.array([l['ntgt'] for l in ldata['locdata']])
        nlabels = ntgts.sum()
        n_samples = np.logspace(np.log10(n_min), np.log10(nlabels), n_rounds).round().astype('int')

        for ndx in range(n_round_inc):

            pp = res[ndx][0]['pred_locs']
            ll = res[ndx][0]['labeled_locs']
            ll[ll<-1000] = np.nan
            dd = np.linalg.norm(ll[:,None]-pp[:,:,None],axis=-1)
            dd1 = find_dist_match(dd)
            valid_l = np.any(~np.isnan(ll[:,:,:,0]),axis=-1)

            cur_dist = dd1[valid_l]

            ax = axx[ndx]
            if ex_im.ndim == 2:
                ax.imshow(ex_im, 'gray')
            elif ex_im.shape[2] == 1:
                ax.imshow(ex_im[:, :, 0], 'gray')
            else:
                ax.imshow(ex_im)

            vv = cur_dist.copy()
            vv[np.isnan(vv)] = 60.
            mm = np.nanpercentile(vv,prcs,axis=0,interpolation='nearest')
            for pts in range(ex_loc.shape[0]):
                for pp in range(mm.shape[0]):
                    c = plt.Circle(ex_loc[pts, :], mm[pp, pts], color=cmap[pp, :], fill=False,alpha=1-((pp+1)/mm.shape[0])*dropoff)
                    ax.add_patch(c)
            ttl = '{} '.format(n_samples[ndx])
            ax.set_title(ttl)
            ax.axis('off')

        for ndx in range(cmap.shape[0]):
            axx[nc-1].plot(np.ones([1, 2]), np.ones([1, 2]), color=cmap[ndx,:],alpha=1-((pp+1)/mm.shape[0])*dropoff)
        axx[nc-1].legend([f'{ppt}' for ppt in prcs])
        for ax in axx:
            ax.set_xlim([0,ex_im.shape[1]])
            ax.set_ylim([ex_im.shape[0],0])

        f.tight_layout()
        return f

    def track(self, mov_file, trk_file, t_types=None,run_type='dry',queue='gpu_rtx'):
        t_types = self.get_types(t_types)
        for t in t_types:
            settings = self.get_settings(t)
            lbl_file = settings['lbl_file']
            train_name = settings['train_name']
            loc_file = settings['loc_file']
            stg_str = settings['stg_str']
            net_type = settings['net_type']
            params= settings['params']
            alg = t[0]

            assert not 'second' in t, 'Use only first stage for tracking'
            if alg.startswith('2stage'):
                import poseConfig, tempfile
                import PoseCommon_pytorch
                t2 = [tt for tt in t if 'mask' not in tt]
                t2[-1] = 'second'
                t2 = tuple(t2)
                settings2 = self.get_settings(t2)
                cache_dir2 = settings2['train_cache_dir'].format(0)
                cc = poseConfig.conf
                cc.cachedir = cache_dir2
                net2 = settings2['net_type']
                sobj = PoseCommon_pytorch.PoseCommon_pytorch(cc)
                latest_model_file = sobj.get_latest_model_file()

                params2 = settings2['params']
                conf_str2 = ' '.join([f'{k} {v}' for k, v in params2.items()])
                t_file = trk_file.replace('.trk','_stg1.trk')
                #t_file = tempfile.mkstemp()[1]

                stg2_str = f'-stage multi -model_files2 {latest_model_file} -conf_params2 {conf_str2} -type2 {settings2["net_type"]} -name2 {settings2["train_name"]}'
                stg2_str_track = f'-trx {t_file}'
            else:
                stg2_str = ''
                stg2_str_track = ''

            trk_name = os.path.split(trk_file)[1]
            trk_name = os.path.splitext(trk_name)[0]
            job_name = self.name + '_' + train_name + '_' + trk_name

            conf_str = ' '.join([f'{k} {v}' for k, v in params.items()])
            conf_str += ' link_id True link_id_training_iters 100000'

            cmd = f'APT_interface.py {lbl_file} -name {train_name} -json_trn_file {loc_file} -conf_params {conf_str} -cache {cache_dir} {stg2_str} -type {net_type} track -mov {mov_file} -out {trk_file} {stg2_str_track}'
            precmd = 'export CUDA_VISIBLE_DEVICES=0'

            if run_type == 'dry':
                print()
                print(f'{job_name}:')
                print(f'{cmd}')
            elif run_type == 'submit':
                print(f'{cmd}')
                rae.run_jobs(job_name,
                         cmd,
                         run_dir=run_dir,
                         queue=queue,
                         precmd='',
                         logdir=self.log_dir, nslots=10,timeout=160*60)
        print(f'Total jobs {len(t_types)}')
