## Generic code for running types of shit.
import PoseTools as pt
import copy
import os

alice_dstr = '20210628'

sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'
# simg = '/groups/branson/bransonlab/mayank/singularity/tf23_mmdetection.sif'
simg = '/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif'
cache_dir = '/nrs/branson/mayank/apt_cache_2'
second_time = False

gt = False
if gt:
    alice_lbl = f'/nrs/branson/mayank/apt_cache_2/alice_ma/trnpack_{alice_dstr}/trn/alice_ma.lbl_multianimal.lbl'
    alice_split_json = None
    alice_json = f'/nrs/branson/mayank/apt_cache_2/alice_ma/trnpack_{alice_dstr}/trn/loc_neg.json'
    alice_full_json = alice_json
    alice_split_json = alice_json
    alice_tight_json = f'/nrs/branson/mayank/apt_cache_2/alice_ma/trnpack_{alice_dstr}/trn/loc_neg_tight.json'
    alice_gt_json = f'/nrs/branson/mayank/apt_cache_2/alice_ma/trnpack_{alice_dstr}/gt/loc.json'
    alice_gt_tight_json = f'/nrs/branson/mayank/apt_cache_2/alice_ma/trnpack_{alice_dstr}/gt/loc_tight.json'
    alice_val_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/gt_db_20210628/train_TF.json'
    alice_val_out_dir = '/nrs/branson/mayank/apt_cache_2/alice_ma/gt_results'
else:
    alice_lbl = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'
    alice_split_json = '/nrs/branson/mayank/apt_cache_2/alice_ma/loc_split_neg.json'
    alice_full_json = '/nrs/branson/mayank/apt_cache_2/alice_ma/loc.json'
    alice_json = alice_split_json
    alice_tight_json = alice_json
    alice_gt_json = alice_json
    alice_gt_tight_json = alice_json
    alice_val_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/alice_split_full_ims_grone_pose_multi/val_TF.json'
    alice_val_out_dir = '/nrs/branson/mayank/apt_cache_2/alice_ma/val_results'

    roian_full_tight_json = '/nrs/branson/mayank/apt_cache_2/four_points_180806/loc_neg_tight.json'
    roian_full_json = '/nrs/branson/mayank/apt_cache_2/four_points_180806/loc_neg.json'

def setup():
    opts = {}

    opts[('alice',)] = {
        'lbl_file':alice_lbl,
        'conf':{'max_n_animals':10},
        'mov':'/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00030_CsChr_RigC_20150826T144616/movie.ufmf',
        'out_dir': '/groups/branson/home/kabram/temp/alice_multi',
        'exp':'cx_GMR_SS00030_CsChr_RigC_20150826T144616',
        'trk_bs':2,
        'val_file':alice_val_file,
        'val_out_dir':alice_val_out_dir}
    if not gt:
        opts[('alice',)]['conf']['op_affinity_graph']='\\(\\(0,1\\),\\(0,5\\),\\(1,2\\),\\(3,4\\),\\(3,5\\),\\(5,6\\),\\(5,7\\),\\(5,9\\),\\(3,16\\),\\(9,10\\),\\(10,15\\),\\(9,14\\),\\(4,11\\),\\(7,8\\),\\(8,12\\),\\(7,13\\)\\)'

    opts[('alice','single')] = {'lbl_file':alice_lbl, 'conf':{'max_n_animals':10}}
    opts[('alice','split')] = {'train_json':alice_split_json}
    opts[('alice','full')] = {'train_json':alice_full_json}
    opts[('alice', 'bbox')] = {'lbl_file':alice_lbl,'train_json': alice_tight_json}

        # Full doesnt have neg bboxes for validation!!
    opts[('alice','ht')] = {'conf':{'ht_pts':'\\(0,6\\)'}}
    opts[('alice','multi','ht')] = {'conf':{'rescale':2}}
    opts[('alice','multi','openpose')] = {'queue':'gpu_tesla'}
    opts[('alice','full_ims','pose')] = {'queue':'gpu_tesla'}
    opts[('alice','multi','bbox')] = {'conf':{'rescale':2}}
    opts[('alice','single')] = {'conf':{'imsz':'\\(192,192\\)','scale_factor_range':1.1,'use_scale_factor_range':True}}
    opts[('alice', 'multi', 'openpose', 'full_ims', 'pose')] = {'conf': {'batch_size': 2, 'dl_steps': 400000}}
    opts[('alice', 'multi', 'grone', 'full_ims', 'pose')] = {'conf': {'batch_size': 4, 'dl_steps': 200000}}
#    opts[('alice','single')] = {'conf':{'rescale':0.5,'dl_steps':200000,'batch_size':4}}

    opts[('roian',)] = {
        'lbl_file':'/nrs/branson/mayank/apt_cache_2/four_points_180806/20210326T070533_20210326T070535.lbl',
        'conf':{'max_n_animals':2,'op_affinity_graph':'\\(\\(0,1\\),\\(0,2\\),\\(0,3\\)\\)'},
        'mov':'/groups/branson/home/kabram/temp/roian_multi/200918_m170234vocpb_m170234_odor_m170232_f0180322.mjpg',
        'out_dir':'/groups/branson/home/kabram/temp/roian_multi',
        'exp':'200918_m170234vocpb_m170234_odor_m170232_f0180322',
        'trk_bs':2,
        'val_file':'/nrs/branson/mayank/apt_cache_2/four_points_180806/multi_mdn_joint_torch/view_0/roian_split_full_ims_grone_pose_multi/val_TF.json',
        'val_out_dir':'/nrs/branson/mayank/apt_cache_2/four_points_180806/val_results'}
    opts[('roian','split')] = {'train_json':'/nrs/branson/mayank/apt_cache_2/four_points_180806/loc_split_neg.json'}
    opts[('roian','split','bbox')] = {'train_json':'/nrs/branson/mayank/apt_cache_2/four_points_180806/loc_split_neg_tight.json'}
    opts[('roian','full')] = {'train_json':'/nrs/branson/mayank/apt_cache_2/four_points_180806/loc_neg.json'}
    opts[('roian','full','bbox')] = {'train_json':'/nrs/branson/mayank/apt_cache_2/four_points_180806/loc_neg_tight.json'}
    opts[('roian','multi','full_ims')] = {'conf':{'rescale':'4','multi_use_mask':False,'multi_loss_mask':True,'op_hires_ndeconv':2}}
    opts[('roian','multi','full_maskless')] = {'conf':{'rescale':'4','multi_use_mask':False,'multi_loss_mask':False,'op_hires_ndeconv':2}}
    opts[('roian','multi','crop_ims')] = {'conf':{'multi_use_mask':False,'multi_loss_mask':True,'batch_size':4,'dl_steps':200000}}
    opts[('roian','multi','openpose')] = {'queue':'gpu_tesla'}
    opts[('roian','multi','bbox')] = {'conf':{'rescale':4}}
    opts[('roian','single')] = {'conf':{'imsz':'\\(352,352\\)'}}
    opts[('roian','ht')] = {'conf':{'ht_pts':'\\(0,1\\)'}}
    opts[('roian','multi','ht')] = {'conf':{'rescale':4}}


    # opts[('maskim',)] = {'conf':{'multi_use_mask':True,'multi_loss_mask':False},'train_dir':'maskim'}
    # opts[('maskloss',)] = {'conf':{'multi_use_mask':False,'multi_loss_mask':True},'train_dir':'nomask'}
    opts[('full_ims',)] = {'conf':{'multi_crop_ims':False,'multi_loss_mask':True},'train_dir':'full_ims'}
    opts[('crop_ims',)] = {'conf':{'multi_crop_ims':True,'multi_loss_mask':True},'train_dir':'crop_ims'}
    opts[('full_maskless',)] = {'conf':{'multi_crop_ims':False,'multi_loss_mask':False},'train_dir':'full_maskless'}
    opts[('split',)] = {'train_dir':'split'}
    opts[('full',)] = {'train_dir':'full'}
    opts[('multi','ht')] = {'train_dir':'ht','conf':{'multi_only_ht':True,'flipLandmarkMatches':'\\{\\}','op_affinity_graph':'\\(\\(0,1\\),\\)','multi_use_mask':False,'multi_loss_mask':True,'multi_crop_ims':False}}
    opts[('single','ht')] = {'conf':{'trange':5,'rrange':10,'use_ht_trx':True,'trx_align_theta':True}}
    opts[('multi','bbox')] = {'conf':{'multi_use_mask':False,'multi_loss_mask':True,'multi_crop_ims':False,'dl_steps':20000}}
    opts[('multi','bbox','frcnn')] = {'conf':{'mmdetect_net':'\\"frcnn\\"'}}
    opts[('multi','bbox','detr')] = {'conf':{'mmdetect_net':'\\"detr\\"','batch_size':3}}
    opts[('single','bbox')] = {'conf':{'trange':5,'rrange':180,'use_bbox_trx':True,'trx_align_theta':False}}
    opts[('single',)] = {'conf':{'is_multi':False}}

    opts[('multi',)] = {'conf':{'rrange':180,'trange':30,'is_multi':True}}
    opts[('multi','grone')] = {'type':'multi_mdn_joint_torch'}
    opts[('multi','mmpose')] = {'type':'multi_mmpose'}
    opts[('multi','openpose')] = {'type':'multi_openpose'}
    opts[('multi','mmdetect')] = {'type':'detect_mmdetect'}
    opts[('single','grone')] = {'type':'mdn_joint_fpn','conf':{'db_format':'\\"tfrecord\\"'}}
    opts[('single','mspn')] = {'type':'mmpose','conf':{'mmpose_net':'\\"mspn\\"'}}
    opts[('single','openpose')] = {'type':'openpose'}
    opts[('openpose',)] = {'conf':{'db_format':'\\"tfrecord\\"'}}
    # opts[('mspn',)] = {'conf':{'mmpose_net':'\\"mspn\\"'}}
    # opts[('alice','multi','grone','crop_ims')] = {'conf_opts':{'learning_rate_multiplier':((0.2,'lr_0p2'),)}}
    return opts

copts = {'conf':{'db_format':'\\"coco\\"','mmpose_net':'\\"higherhrnet\\"','dl_steps':100000,'save_step':10000,'batch_size':8},'train_dir':'apt','queue':'gpu_rtx','conf_opts':{}}

##

# Alice split
# train_set = {'name': ('alice',), 'mask': ('maskim', 'maskloss'), 'splits': ('split',), 'crops': ('full_ims', 'crop_ims'), 'nets': ('grone', 'mmpose', 'openpose'), 'type': ('multi',)}

# train_set = {'name': ('alice',), 'mask': ('maskloss',), 'splits': ('split',), 'crops': ('full_ims', 'crop_ims'), 'nets': ( 'openpose',), 'type': ('multi',)}

# Alice grone lr
# train_set = {'name': ('alice',), 'mask': ('maskim',), 'splits': ('split',), 'crops': ('crop_ims',), 'nets': ('grone', ), 'type': ('multi',)}


# alice neg
# train_set = {'name': ('alice_neg',), 'splits': ('split',), 'nets': ('grone', 'openpose','mmpose'), 'type': ('multi',)}

# Roian
# train_set = {'name': ('roian',), 'splits': ('split',),'crops': ('full_ims', 'crop_ims','full_maskless'), 'nets': ('grone', 'mmpose', 'openpose'), 'type': ('multi',)}

# alice ht track
# train_set = {'name': ('alice_ht',), 'mask': ('maskim', 'maskloss'), 'splits': ('full',), 'crops': ( 'crop_ims',), 'nets': ('grone', 'mmpose', 'openpose'), 'type': ('multi',)}

train_sets = []

# Order is important for the name. So name, splits, pts, nets, crops, type
# pts - whether pose (bottom-up) or ht/bbox.
# crops - whether to crop for training
#         full_ims is full images with mask
# type - whether multi or single (second stage).

# !!! CROSS VALIDATION !!!
# MA training
train_sets.append({'name': ('alice',), 'splits': ('split',), 'pts':('pose',), 'nets': ('grone', 'mmpose', 'openpose'),'crops': ('full_ims', 'crop_ims'),'type': ('multi',)})
train_sets.append({'name': ('roian',), 'splits': ('split',), 'pts':('pose',),'nets': ('grone', 'mmpose', 'openpose'), 'crops': ('full_ims', 'crop_ims','full_maskless'),'type': ('multi',)})

# HT 1stage training
train_sets.append({'name': ('alice',), 'splits': ('split',),'pts':('ht',), 'nets': ('grone', 'openpose'), 'crops':('full_ims',) ,'type': ('multi',)})
train_sets.append({'name': ('roian',), 'splits': ('split',),'pts':('ht',), 'nets': ('grone', 'openpose'), 'crops':('full_ims','full_maskless') ,'type': ('multi',)})

# bbox 1 stage training
train_sets.append( {'name': ('alice',), 'splits': ('split',),'pts':('bbox',), 'nets': ('mmdetect',), 'crops':('full_ims',),'type': ('multi',)})
train_sets.append( {'name': ('roian',), 'splits': ('split',),'pts':('bbox',), 'nets': ('mmdetect',), 'crops':('full_ims','full_maskless'), 'type': ('multi',), 'mmdetect_nets':('frcnn','detr')})

# HT 2 stage training
train_sets.append({'name': ('roian','alice'), 'splits': ('split',),'pts':('ht','bbox'), 'nets': ( 'grone',),'type': ('single',)})

# !!!!GROUND TRUTHING
## ALICE
train_sets = []
train_sets.append({'name': ('alice',), 'splits': ('full',), 'pts':('pose',), 'nets': ('grone', 'mmpose', 'openpose'),'crops': ('full_ims', 'crop_ims'),'type': ('multi',)})

# HT 1stage training
train_sets.append({'name': ('alice',), 'splits': ('full',),'pts':('ht',), 'nets': ('grone', 'openpose'), 'crops':('full_ims','crop_ims') ,'type': ('multi',)})

# bbox 1 stage training
train_sets.append( {'name': ('alice',), 'splits': ('full',),'pts':('bbox',), 'nets': ('mmdetect',), 'crops':('full_ims',),'type': ('multi',),'mmdetect_nets':('frcnn','detr')})

# HT 2 stage training
train_sets.append({'name': ('alice',), 'splits': ('full',),'pts':('ht','bbox'), 'nets': ( 'grone','openpose','mspn'),'type': ('single',)})

## ROIAN
train_sets = []
train_sets.append({'name': ('roian',), 'splits': ('split',), 'pts':('pose',),'nets': ('grone', 'mmpose', 'openpose'), 'crops': ('full_ims', 'crop_ims','full_maskless'),'type': ('multi',)})
train_sets.append({'name': ('roian',), 'splits': ('split',),'pts':('ht',), 'nets': ('grone', 'openpose'), 'crops':('full_ims','full_maskless') ,'type': ('multi',)})
train_sets.append( {'name': ('roian',), 'splits': ('split',),'pts':('bbox',), 'nets': ('mmdetect',), 'crops':('full_ims','full_maskless'), 'type': ('multi',),'mmdetect_nets':('frcnn','detr')})
train_sets.append({'name': ('roian',), 'splits': ('split',),'pts':('ht','bbox'), 'nets': ( 'grone',),'type': ('single',)})

##
import h5py
import cv2
import numpy as np
from scipy import io as sio
import multiResData
import json
from matplotlib import pyplot as plt
def update_lbls_alice(debug=False):
    trndir = os.path.join(cache_dir,'alice_ma',f'trnpack_{alice_dstr}')
    tpack = os.path.join(trndir,'trn','loc.json')
    lbl_file = os.path.join(trndir,'trn','alice_ma.lbl_multianimal.lbl')
    jj = os.path.splitext(tpack)
    neg_file = jj[0] + '_neg' + jj[1]
    bbox_file = jj[0] + '_neg_tight' + jj[1]

    # Add neg ROI box
    im_sz = (1024, 1024)
    boxsz = 80
    negbox_sz = 240
    num_negs = 150
    bdir, _ = os.path.split(tpack)
    T = pt.json_load(tpack)
    lbl = h5py.File(lbl_file, 'r')
    done_ix = []
    totcount = 0
    while totcount < num_negs:
        ix = np.random.choice(len(T['locdata']))
        done_ix.append(ix)
        curp = T['locdata'][ix]
        tt = os.path.split(T['movies'][curp['imov'] - 1])
        trx = sio.loadmat(os.path.join(tt[0], 'registered_trx.mat'))['trx'][0]
        ntrx = len(trx)
        fnum = curp['frm']
        all_mask = np.zeros(im_sz)
        boxes = []
        for tndx in range(ntrx):
            cur_trx = trx[tndx]
            if fnum > cur_trx['endframe'][0, 0] - 1:
                continue
            if fnum < cur_trx['firstframe'][0, 0] - 1:
                continue
            x, y, theta = multiResData.read_trx(cur_trx, fnum)
            x = int(round(x));
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
            im = cv2.imread(os.path.join(bdir, curp['img'][0]), cv2.IMREAD_UNCHANGED)
            f,ax = plt.subplots(1,2)
            ax[0].cla()
            ax[0].imshow(im, 'gray')
            for b in boxes:
                ax[0].plot([b[0], b[1], b[1], b[0], b[0]], [b[2], b[2], b[3], b[3], b[2]])
            ax[0].plot([selb[0], selb[1], selb[1], selb[0], selb[0]], [selb[2], selb[2], selb[3], selb[3], selb[2]])
            cc = all_mask.copy()
            cc[selb[2]:selb[3], selb[0]:selb[1]] = -1
            ax[1].imshow(cc)

        if done:
            totcount = totcount+1
            print(f'Adding Roi for {ix}')
            curp['extra_roi'] = [negx_min, negx_min, negx_min + negbox_sz, negx_min + negbox_sz, negy_min, negy_min + negbox_sz,
                                 negy_min + negbox_sz, negy_min]
            curp['nextraroi'] = 1

    with open(neg_file, 'w') as f:
        json.dump(T, f)


    # tighten bboxes for object detection
    H = pt.json_load(neg_file)
    for hh in H['locdata']:
        pts = np.array(hh['pabs']).reshape([2,17,-1])
        mn = pts.min(axis=1)
        mx = pts.max(axis=1)
        roi = np.array([mn[0] ,mn[0], mx[0], mx[0],mn[1],mx[1],mx[1],mn[1]])
        hh['roi'] = roi.tolist()

    with open(bbox_file, 'w') as f:
        json.dump(H, f)

    # tight boxes for GT just in case
    tpack = os.path.join(trndir,'gt','loc.json')
    lbl_file = os.path.join(trndir,'gt','alice_ma.lbl_multianimal.lbl')
    jj = os.path.splitext(tpack)
    bbox_file = jj[0] + '_tight' + jj[1]
    H = pt.json_load(tpack)
    for hh in H['locdata']:
        pts = np.array(hh['pabs']).reshape([2, 17, -1])
        mn = pts.min(axis=1)
        mx = pts.max(axis=1)
        roi = np.array([mn[0], mn[0], mx[0], mx[0], mn[1], mx[1], mx[1], mn[1]])
        hh['roi'] = roi.tolist()

    with open(bbox_file, 'w') as f:
        json.dump(H, f)


##
def _get_sets(ts,kk):
    sets = []
    if len(kk) > 1:
        for c in ts[kk[0]]:
            for i in _get_sets(ts,kk[1:]):
                i.append(c)
                sets.append(i)
    else:
        sets = [[t] for t in ts[kk[0]]]
    return sets

def get_sets(train_sets):
    cur_sets = []
    for train_set in train_sets:
        cur_sets.extend(_get_sets(train_set,list(train_set.keys())))
    okeys = list(setup().keys())
    okeys.sort(key = lambda x:len(x))
    for c in cur_sets:
        print(c)
    return cur_sets, okeys

## Training

def train(train_sets,submit=False):

    opts = setup()
    cur_sets,okeys = get_sets(train_sets)
    for c in cur_sets:
        cur_dict = copy.deepcopy(copts)
        for curk in okeys:
            if not set(curk).issubset(set(c)):
                continue
            cur_opt_dict = opts[curk]
            for k in cur_opt_dict.keys():
                if k == 'conf':
                    cur_dict['conf'].update(cur_opt_dict[k])
                elif k =='conf_opts':
                    cur_dict['conf_opts'].update(cur_opt_dict[k])
                elif k == 'train_dir':
                    cur_dict['train_dir'] = '_'.join([cur_opt_dict[k], cur_dict['train_dir']])
                else:
                    cur_dict[k] = cur_opt_dict[k]

        opt_str = ' '.join([f'{k} {cur_dict["conf"][k]}' for k in cur_dict['conf'].keys()])
        n = '_'.join(c[::-1])
        job_name = n
        if len(cur_dict['conf_opts'].keys())>0:
            train_name = None
            for ok in cur_dict['conf_opts'].keys():
                for curv,curn in cur_dict['conf_opts'][ok]:
                    if curv is None:
                        continue
                    opt_str += f' {ok} {curv}'
                    train_name = f'{train_name}_{curn}' if train_name is not None else curn
                    job_name += f'_{curn}'
                    cmd = f'APT_interface.py {cur_dict["lbl_file"]} -conf_params {opt_str} -json_trn_file {cur_dict["train_json"]} -type {cur_dict["type"]} -name {n} -train_name {train_name} -cache {cache_dir} train -use_cache -skip_db'

                    print(cur_dict['queue'], cmd)
                    if submit:
                        pt.submit_job(job_name, cmd, sdir, cur_dict['queue'], sing_image=simg, numcores=4)

        else:
            job_name = n
            cmd = f'APT_interface.py {cur_dict["lbl_file"]} -conf_params {opt_str} -json_trn_file {cur_dict["train_json"]} -type {cur_dict["type"]} -name {n} -cache {cache_dir} train -use_cache'

            print(cur_dict['queue'],cmd)
            print()
            if submit:
                pt.submit_job(job_name,cmd,sdir,cur_dict['queue'],sing_image=simg,numcores=4)

## tracking

def track(train_sets,submit=False):
    cur_sets,okeys = get_sets(train_sets)

    opts = setup()
    for c in cur_sets:
        cur_dict = copy.deepcopy(copts)
        for curk in okeys:
            if not set(curk).issubset(set(c)):
                continue
            cur_opt_dict = opts[curk]
            for k in cur_opt_dict.keys():
                if k == 'conf':
                    cur_dict['conf'].update(cur_opt_dict[k])
                elif k == 'train_dir':
                    cur_dict['train_dir'] = '_'.join([cur_opt_dict[k], cur_dict['train_dir']])
                else:
                    cur_dict[k] = cur_opt_dict[k]
        cur_dict['conf']['batch_size'] = cur_dict['trk_bs']
        opt_str = ' '.join([f'{k} {cur_dict["conf"][k]}' for k in cur_dict['conf'].keys()])
        n = '_'.join(c[::-1])
        job_name = n
        cmd = f'APT_interface.py {cur_dict["lbl_file"]} -name {n} -type {cur_dict["type"]} -conf_params {opt_str} -cache {cache_dir} track -mov {cur_dict["mov"]} -out {cur_dict["out_dir"]}/{cur_dict["exp"]}_{n}.trk -end_frame 5000'
        job_name += '_trk'

        print(cur_dict['queue'],cmd)
        if submit:
            pt.submit_job(job_name,cmd,sdir,cur_dict['queue'],sing_image=simg,numcores=4)


## Classify db

def classify(train_sets,submit=False):
    cur_sets,okeys = get_sets(train_sets)
    opts = setup()
    for c in cur_sets:
        cur_dict = copy.deepcopy(copts)
        for curk in okeys:
            if not set(curk).issubset(set(c)):
                continue
            cur_opt_dict = opts[curk]
            for k in cur_opt_dict.keys():
                if k == 'conf':
                    cur_dict['conf'].update(cur_opt_dict[k])
                elif k == 'train_dir':
                    cur_dict['train_dir'] = '_'.join([cur_opt_dict[k], cur_dict['train_dir']])
                else:
                    cur_dict[k] = cur_opt_dict[k]
        cur_dict['conf']['batch_size'] = cur_dict['trk_bs']
        cur_dict['conf']['db_format'] = '\\"coco\\"'
        opt_str = ' '.join([f'{k} {cur_dict["conf"][k]}' for k in cur_dict['conf'].keys()])
        n = '_'.join(c[::-1])
        job_name = n
        cmd = f'APT_interface.py {cur_dict["lbl_file"]} -name {n} -type {cur_dict["type"]} -conf_params {opt_str} -cache {cache_dir} classify -db_file {cur_dict["val_file"]} -out_file {cur_dict["val_out_dir"]}/{job_name}'
        job_name += '_classify'

        print(cur_dict['queue'],cmd)
        if submit:
            pt.submit_job(job_name,cmd,sdir,cur_dict['queue'],sing_image=simg,numcores=4)


## Classify db 2 stage

def classify_2(train_sets,second_net='grone'):
    import APT_interface as apt
    import os
    import hdf5storage
    import multiprocessing
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opts = setup()
    cur_sets,okeys = get_sets(train_sets)

    def run_classify(arg_dict,out_file):

        res = apt.classify_db_all(**arg_dict)
        preds, locs, info,model_file = res[:4]
        preds = apt.to_mat(preds)
        locs = apt.to_mat(locs)
        info = apt.to_mat(info)
        out_dict = {'pred_locs': preds, 'labeled_locs': locs, 'list': info, 'model_file': model_file}
        hdf5storage.savemat(out_file,out_dict,appendmat=False,truncate_existing=True)

    for c in cur_sets:
        cur_dict = copy.deepcopy(copts)
        for curk in okeys:
            if not set(curk).issubset(set(c)):
                continue
            cur_opt_dict = opts[curk]
            for k in cur_opt_dict.keys():
                if k == 'conf':
                    cur_dict['conf'].update(cur_opt_dict[k])
                elif k == 'train_dir':
                    cur_dict['train_dir'] = '_'.join([cur_opt_dict[k], cur_dict['train_dir']])
                else:
                    cur_dict[k] = cur_opt_dict[k]
        cur_dict['conf']['batch_size'] = cur_dict['trk_bs']
        cur_dict['conf']['db_format'] = '\"coco\"'

        cur_dict2 = copy.deepcopy(copts)
        c2 = copy.copy(c)
        if 'full_ims' in c2:
            _ = c2.pop(c2.index('full_ims'))
        elif 'full_maskless' in c2:
            _ = c2.pop(c2.index('full_maskless'))
        if 'mmdetect' in c2:
            _ = c2.pop(c2.index('mmdetect'))
        c2[c2.index('multi')] = 'single'
        c2[0] = second_net
        for curk in okeys:
            if not set(curk).issubset(set(c2)):
                continue
            cur_opt_dict = opts[curk]
            for k in cur_opt_dict.keys():
                if k == 'conf':
                    cur_dict2['conf'].update(cur_opt_dict[k])
                elif k == 'train_dir':
                    cur_dict2['train_dir'] = '_'.join([cur_opt_dict[k], cur_dict2['train_dir']])
                else:
                    cur_dict2[k] = cur_opt_dict[k]


        n = '_'.join(c[::-1])
        n2 = '_'.join(c2[::-1])

        opt_str = ' '.join([f'{k} {cur_dict["conf"][k]}' for k in cur_dict['conf'].keys()])
        opt_str = opt_str.replace('\\','')
        opt_str2 = ' '.join([f'{k} {cur_dict2["conf"][k]}' for k in cur_dict2['conf'].keys()])
        opt_str2 = opt_str2.replace('\\','')
        lbl_file = cur_dict['lbl_file']
        conf = apt.create_conf(lbl_file,0,n,cache_dir,net_type=cur_dict['type'],conf_params=opt_str.split())
        conf.img_dim = 3
        conf.batch_size = 1
        # conf.n_classes = 2
        conf2 = apt.create_conf(lbl_file,0,n2,cache_dir,net_type=cur_dict2['type'],conf_params=opt_str2.split())

        arg_dict = {'model_type':cur_dict['type'], 'model_type2': cur_dict2['type'], 'conf': conf, 'conf2' : conf2, 'db_file':cur_dict[        'val_file']}
        out_file = cur_dict['val_out_dir'] + "/" + n + f'_{second_net}_0.mat'
        run_classify(arg_dict,out_file)
        # p = multiprocessing.Process(target=run_classify,args=[arg_dict,out_file])
        # p.start()
        # p.join()


## status

def status(train_sets,tt='train'):

    cur_sets,okeys = get_sets(train_sets)
    import subprocess
    from termcolor import colored
    opts = setup()
    for c in cur_sets:
        cur_dict = copy.deepcopy(copts)
        for curk in okeys:
            if not set(curk).issubset(set(c)):
                continue
            cur_opt_dict = opts[curk]
            for k in cur_opt_dict.keys():
                if k == 'conf':
                    cur_dict['conf'].update(cur_opt_dict[k])
                elif k =='conf_opts':
                    cur_dict['conf_opts'].update(cur_opt_dict[k])
                elif k == 'train_dir':
                    cur_dict['train_dir'] = '_'.join([cur_opt_dict[k], cur_dict['train_dir']])
                else:
                    cur_dict[k] = cur_opt_dict[k]

        n = '_'.join(c[::-1])
        job_name = n
        if len(cur_dict['conf_opts'].keys())>0:
            train_name = None
            for ok in cur_dict['conf_opts'].keys():
                for curv,curn in cur_dict['conf_opts'][ok]:
                    if curv is None:
                        continue
                    opt_str += f' {ok} {curv}'
                    train_name = f'{train_name}_{curn}' if train_name is not None else curn
                    job_name += f'_{curn}'

        else:
            job_name = n

        if tt == 'classify':
            job_name += '_classify'

        err_file = os.path.join(sdir,job_name + '.err')
        ff = open(err_file,'r')
        lines = ff.readlines()

        cmd = ['ssh','login1', f'bjobs -w | grep {job_name}']
        try:
            bout = subprocess.check_output(cmd)
        except subprocess.CalledProcessError:
            bout = 'Job not found'

        print(colored(job_name,'cyan'))
        print(colored(bout,'red'))
        print(lines[-5:])
        print()

if False:

## Validation results.
    import multiResData
    import PoseTools
    import h5py
    from reuse import *
    res = {}

    dtype = 'roian'
    if dtype == 'roian':
        dropoff = 0.4
    else:
        dropoff = 0.7
    opts = setup()
    cur_sets, okeys = get_sets(train_sets)

    for c in cur_sets:
        if dtype not in c: continue
        cur_dict = copy.deepcopy(copts)
        for curk in okeys:
            if not set(curk).issubset(set(c)):
                continue
            cur_opt_dict = opts[curk]
            for k in cur_opt_dict.keys():
                if k == 'conf':
                    cur_dict['conf'].update(cur_opt_dict[k])
                elif k == 'train_dir':
                    cur_dict['train_dir'] = '_'.join([cur_opt_dict[k], cur_dict['train_dir']])
                else:
                    cur_dict[k] = cur_opt_dict[k]
        cur_dict['conf']['batch_size'] = cur_dict['trk_bs']
        cur_dict['conf']['db_format'] = '\\"coco\\"'
        opt_str = ' '.join([f'{k} {cur_dict["conf"][k]}' for k in cur_dict['conf'].keys()])
        n = '_'.join(c[::-1])
        job_name = n
        res_file = f'{cur_dict["val_out_dir"]}/{job_name}_0.mat'

        cur_dict = copy.deepcopy(copts)
        if not os.path.exists(res_file): continue

        K = h5py.File(res_file,'r')
        ll = K['labeled_locs'][()].T
        pp = K['pred_locs']['locs'][()].T
        ll[ll<-1000] = np.nan
        dd = np.linalg.norm(ll[:,None]-pp[:,:,None],axis=-1)
        dd1 = find_dist_match(dd)
        K.close()
        valid_l = np.any(~np.isnan(ll[:,:,:,0]),axis=-1)

        res['_'.join(c)] = dd1[valid_l]

    if 'alice' == dtype:
        ex_db = '/nrs/branson/mayank/apt_cache_2/alice_ma/mdn_joint_fpn/view_0/alice_split_ht_grone_single/val_TF.tfrecords'
        X = multiResData.read_and_decode_without_session(ex_db,17)
    else:
        ex_db = '/nrs/branson/mayank/apt_cache_2/four_points_180806/mdn_joint_fpn/view_0/roian_split_ht_grone_single/val_TF.tfrecords'
        X = multiResData.read_and_decode_without_session(ex_db,4)

    ex_im = X[0][0]
    ex_loc = X[1][0]

    n_types = len(res)
    nc = 3 #n_types  # int(np.ceil(np.sqrt(n_types)))
    nr = 2  # int(np.ceil(n_types/float(nc)))
    prcs = [50,75,90,95,98]
    cmap = PoseTools.get_cmap(len(prcs), 'cool')
    f, axx = plt.subplots(nr, nc, figsize=(12, 8), squeeze=False)
    axx = axx.flat
    for idx,k  in enumerate(res.keys()):
        ax = axx[idx]
        if ex_im.ndim == 2:
            ax.imshow(ex_im, 'gray')
        elif ex_im.shape[2] == 1:
            ax.imshow(ex_im[:, :, 0], 'gray')
        else:
            ax.imshow(ex_im)

        vv = res[k].copy()
        vv[np.isnan(vv)] = 60.
        mm = np.nanpercentile(vv,prcs,axis=0)
        for pt in range(ex_loc.shape[0]):
            for pp in range(mm.shape[0]):
                c = plt.Circle(ex_loc[pt, :], mm[pp, pt], color=cmap[pp, :], fill=False,alpha=1-((pp+1)/mm.shape[0])*dropoff)
                ax.add_patch(c)
        ttl = '{} '.format(k)
        ax.set_title(ttl)
        ax.axis('off')

    for ndx in range(cmap.shape[0]):
        axx[nc-1].plot(np.ones([1, 2]), np.ones([1, 2]), color=cmap[ndx,:],alpha=1-((pp+1)/mm.shape[0])*dropoff)
    axx[nc-1].legend([f'{ppt}' for ppt in prcs])
    for ax in axx:
        ax.set_xlim([0,ex_im.shape[1]])
        ax.set_ylim([ex_im.shape[0],0])
    f.tight_layout()



## Results
    import multiResData
    import PoseTools
    import h5py
    from reuse import *
    res = {}

    dtype = 'roian'
    if dtype == 'roian':
        dropoff = 0.4
    else:
        dropoff = 0.7

    exclude = ['crop']
    exclude = None
    include = None
    out_dirs = {'alice':alice_val_out_dir,
                'roian':'/nrs/branson/mayank/apt_cache_2/four_points_180806/val_results'}
    all_res_files = os.listdir(out_dirs[dtype])
    all_res_files = [a for a in all_res_files if a.startswith(dtype)]
    if exclude is not None:
        for estr in exclude:
            all_res_files = [a for a in all_res_files if not (a.count(estr)>0)]

    if include is not None:
        for estr in include:
            all_res_files = [a for a in all_res_files if (a.count(estr)>0)]

    str_rem  = ['split','full_ims','multi','0.mat','_pose_']

    for c in all_res_files:
        res_file = os.path.join(out_dirs[dtype],c)

        cur_name = c
        for st in str_rem:
            cur_name = cur_name.replace(st,'')
        cur_name = cur_name.replace('_',' ')
        cur_name = ' '.join(cur_name.split())

        K = h5py.File(res_file,'r')
        ll = K['labeled_locs'][()].T
        pp = K['pred_locs']['locs'][()].T
        ll[ll<-1000] = np.nan
        dd = np.linalg.norm(ll[:,None]-pp[:,:,None],axis=-1)
        dd1 = find_dist_match(dd)
        valid_l = np.any(~np.isnan(ll[:,:,:,0]),axis=-1)

        res[cur_name] = dd1[valid_l]
        K.close()

    if 'alice' == dtype:
        ex_db = '/nrs/branson/mayank/apt_cache_2/alice_ma/mdn_joint_fpn/view_0/alice_split_ht_grone_single/val_TF.tfrecords'
        X = multiResData.read_and_decode_without_session(ex_db,17)
    else:
        ex_db = '/nrs/branson/mayank/apt_cache_2/four_points_180806/mdn_joint_fpn/view_0/roian_split_ht_grone_single/val_TF.tfrecords'
        X = multiResData.read_and_decode_without_session(ex_db,4)

    ex_im = X[0][0]
    ex_loc = X[1][0]

    n_types = len(res)
    nc = 5 #n_types  # int(np.ceil(np.sqrt(n_types)))
    nr = 4  # int(np.ceil(n_types/float(nc)))
    prcs = [50,75,90,95,98,99]
    cmap = PoseTools.get_cmap(len(prcs), 'cool')
    f, axx = plt.subplots(nr, nc, figsize=(12, 8), squeeze=False)
    axx = axx.flat
    for idx,k  in enumerate(res.keys()):
        ax = axx[idx]
        if ex_im.ndim == 2:
            ax.imshow(ex_im, 'gray')
        elif ex_im.shape[2] == 1:
            ax.imshow(ex_im[:, :, 0], 'gray')
        else:
            ax.imshow(ex_im)

        vv = res[k].copy()
        vv[np.isnan(vv)] = 60.
        mm = np.nanpercentile(vv,prcs,axis=0,interpolation='nearest')
        for pt in range(ex_loc.shape[0]):
            for pp in range(mm.shape[0]):
                c = plt.Circle(ex_loc[pt, :], mm[pp, pt], color=cmap[pp, :], fill=False,alpha=1-((pp+1)/mm.shape[0])*dropoff)
                ax.add_patch(c)
        ttl = '{} '.format(k)
        ax.set_title(ttl)
        ax.axis('off')

    for ndx in range(cmap.shape[0]):
        axx[nc-1].plot(np.ones([1, 2]), np.ones([1, 2]), color=cmap[ndx,:],alpha=1-((pp+1)/mm.shape[0])*dropoff)
    axx[nc-1].legend([f'{ppt}' for ppt in prcs])
    for ax in axx:
        ax.set_xlim([0,ex_im.shape[1]])
        ax.set_ylim([ex_im.shape[0],0])

    f.tight_layout()

## Create gt db alice
    import os
    gt_dstr = '20210628'
    lbl_file = f'/nrs/branson/mayank/apt_cache_2/alice_ma/trnpack_{gt_dstr}/gt/alice_ma.lbl_multianimal.lbl'
    json_trn = '/nrs/branson/mayank/apt_cache_2/alice_ma/trnpack_20210628/gt/loc.json'
    import APT_interface as apt
    conf = apt.create_conf(lbl_file,0,'gt_db',cache_dir,net_type='gt_db',json_trn_file=json_trn)
    conf.cachedir = os.path.join(cache_dir,conf.expname,f'gt_db_{gt_dstr}')
    apt.create_coco_db(conf,False)
##

