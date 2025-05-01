from reuse import *
json_file = '/groups/branson/bransonlab/mayank/apt_cache_2/multitarget_bubble/mmpose/view_0/alice_randsplit_round_7/train_TF.json'
im_dir = ''
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/alice_inc'

##
import json
val_frac = 0.2
J = pt.json_load(json_file)
n_full = len(J['annotations'])
randp = np.random.permutation(n_full)
bpt = int(val_frac*n_full)
rtr = randp[bpt:]
rval = randp[:bpt]
J['annotations'] = [jj for ndx,jj in enumerate(J['annotations']) if ndx in rtr]
uid = np.unique([a['image_id'] for a in J['annotations']])
ims = J['images']
sel_ims = [ims[ndx] for ndx,i in enumerate(ims) if i['id'] in uid]
J['images'] = sel_ims
with open(bdir + '/train_TF.json','w') as f:
    json.dump(J,f)
J = pt.json_load(json_file)
J['annotations'] = [jj for ndx,jj in enumerate(J['annotations']) if ndx in rval]
uid = np.unique([a['image_id'] for a in J['annotations']])
ims = J['images']
sel_ims = [ims[ndx] for ndx,i in enumerate(ims) if i['id'] in uid]
J['images'] = sel_ims
with open(bdir + '/val_TF.json','w') as f:
    json.dump(J,f)


##

from reuse import *
from copy import deepcopy
import json

full_json = os.path.join(bdir,'train_TF.json')
J = pt.json_load(full_json)
jq = np.array([jj['keypoints'] for jj in J['annotations']]).reshape([-1,17,3])
invalid = np.all(jq[:,:,2]==0,axis=1)
n_train = np.count_nonzero(~invalid)
jj = np.random.permutation(n_train)
inc_size = 200
n_grs = int(n_train/inc_size)
sel_ann = [jj for ndx,jj in enumerate(J['annotations']) if not invalid[ndx]]
J['annotations'] = sel_ann

## create the dirs and training json files
for ndx in range(n_grs):
    cur_sz = range((ndx+1)*inc_size)
    cur_sel = jj[cur_sz].tolist()
    J1 = deepcopy(J)
    J1['annotations'] = [J1['annotations'][ix] for ix in cur_sel]
    uid = np.unique([a['image_id'] for a in J1['annotations']])
    ims = J1['images']
    sel_ims = [ims[ndx] for ndx, i in enumerate(ims) if i['id'] in uid]
    J1['images'] = sel_ims
    cdir = bdir + f'/train_{ndx}'
    os.makedirs(cdir,exist_ok=True)
    with open(cdir + f'/train_TF.json','w') as f:
        json.dump(J1,f)

## submit training jobs
import ap36_train as a36
conf = pt.pickle_load('/groups/branson/bransonlab/mayank/apt_cache_2/multitarget_bubble/mdn_joint_fpn/view_0/apt_expt/multitarget_bubble_deepnet_20200706_traindata')[1]
conf.dl_steps = 20000
conf.batch_size = 16
conf.db_format = 'coco'
conf.rot_prob = 0.7
conf.is_multi = False
conf.multi_match_dist_factor = 0.2
for ndx in range(n_grs):
    cdir = bdir + f'/train_{ndx}'
    conf.cachedir = cdir
    a36.train_bsub(conf,'mdn_joint_fpn','deepnet',f'alice_inc_{ndx}')

## check accuracy
import ap36_train as a36
val_file = '/groups/branson/bransonlab/mayank/apt_cache_2/alice_inc/val_TF.json'
ppd = []
dds = []
for ndx in range(n_grs):
    cdir = bdir + f'/train_{ndx}'
    C = pt.pickle_load(cdir + '/traindata')
    conf = C[1]
    conf.coco_im_dir = im_dir

    # imdir= '/groups/branson/bransonlab/mayank/apt_cache_2/ap36k_topdown_cow/val'
    # ename = f'train_{ndx}'
    # conf = a36.create_conf_sa({},cdir,imdir,ename)
    # sigmas = [0.01,]*7 + [0.04,]*4 + [0.09,]*6
    sigmas = 0.01*np.array([1.02824052, 1.197518  , 1.20009346, 1.09318899, 1.07948815,
        1.51404181, 1.51965481, 1.67886249, 2.05974873, 1.69516461,
        2.00565759, 4.70209639, 2.97893987, 4.97722848, 4.94152492,
        2.94189999, 4.48909379])
    # sigmas = 0.01*np.array([1.28618643, 1.40018312, 1.5528589 , 1.29503387, 1.34988427,
    #    2.05064234, 2.02291948, 2.20446902, 3.04643631, 2.27786311,
    #    2.70105468, 6.46569436, 5.8863953 , 7.79436939, 8.23480698,
    #    5.67569008, 7.14263513])



    ap = a36.compute_perf_sa(conf,'mdn_joint_fpn',val_file,'deepnet','',sigmas=sigmas)
    ppd.append(ap[0])
    dds.append(ap[1].mean(axis=0))

ppd = np.array(ppd)
dds = np.array(dds)

## do incremental training

# start with 120 labels which is ndx=2
import time
import json
from reuse import *
import ap36_train as a36

net_type = 'mdn_joint_fpn'
gap = 2
nadd = 200

bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/alice_inc'
full_json = os.path.join(bdir,'train_TF.json')

rand_sample = True
# if rand_sample:
#     os.environ['CUDA_VISIBLE_DEVICES'] = ''

if rand_sample:
    r_start = 9
else:
    r_start = 1
val_file = '/groups/branson/bransonlab/mayank/apt_cache_2/alice_inc/val_TF.json'

for round in range(r_start,20):
    if round==1:
        prev_dir = bdir + '/train_2'
    else:
        prev_dir = bdir + f'/round_front_{round-1}'
        if rand_sample:
            prev_dir += '_rand'

    while True and round>r_start:
        time.sleep(300)
        prev_status = a36.get_status(f'alice_fb_front_{round-1}{"_rand" if rand_sample else ""}',prev_dir + '/deepnet')
        if prev_status[0]:
            curt = time.strftime('%m%d%H%M')
            print(f'{curt}: {prev_status[2]} - {prev_status[1]}')
        else:
            break



    C = pt.pickle_load(prev_dir + '/traindata')
    conf = C[1]
    conf.coco_im_dir = im_dir

    preds,labels,_,_ = apt.classify_db_all(net_type,conf,full_json,name='deepnet',fullret=True)

    #
    ll = labels
    pp = preds['locs']
    ll[ll < -1000] = np.nan
    dd = np.linalg.norm(ll - pp, axis=-1)
    dd = np.nansum(dd[:,[11,16]],axis=-1)

    ord = np.argsort(dd)[::-1]

    j_prev = pt.json_load(prev_dir+ '/train_TF.json')
    j_full = pt.json_load(full_json)
    j_full_ndx = np.array([jx['image_id'] for jx in j_full['annotations']])
    prev_ids = [jx['image_id'] for jx in j_prev['annotations']]

    to_add = []
    curndx = 0
    while len(to_add)<nadd*3:
        if j_full_ndx[ord[curndx]] not in prev_ids:
            to_add.append(ord[curndx])
        curndx+=1

    if rand_sample:
        to_add = np.array(to_add)[np.random.permutation(nadd*3)]
    to_add = to_add[:nadd]

    for ix in to_add:
        full_ix = ix
        j_prev['annotations'].append(j_full['annotations'][full_ix])

    uid = np.unique([a['image_id'] for a in j_prev['annotations']])
    ims = j_full['images']
    sel_ims = [ims[ndx] for ndx, i in enumerate(ims) if i['id'] in uid]
    j_prev['images'] = sel_ims

    cdir = bdir + f'/round_front_{round}{"_rand" if rand_sample else ""}'
    os.makedirs(cdir,exist_ok=True)
    with open(cdir + '/train_TF.json','w') as f:
        json.dump(j_prev,f)
    if not os.path.exists(cdir + '/val_TF.json'):
        os.symlink(val_file,cdir+'/val_TF.json')

    #
    conf = pt.pickle_load(
        '/groups/branson/bransonlab/mayank/apt_cache_2/multitarget_bubble/mdn_joint_fpn/view_0/apt_expt/multitarget_bubble_deepnet_20200706_traindata')[
        1]
    conf.dl_steps = 20000
    conf.batch_size = 16
    conf.db_format = 'coco'
    conf.rot_prob = 0.7
    conf.is_multi = False
    conf.multi_match_dist_factor = 0.2
    conf.cachedir = cdir
    a36.train_bsub(conf,'mdn_joint_fpn','deepnet',f'alice_fb_front_{round}{"_rand" if rand_sample else ""}')




##

from reuse import *
import ap36_train as a36
val_file = '/groups/branson/bransonlab/mayank/apt_cache_2/alice_inc/val_TF.json'
pp_fb = []
dds_fb = []
round = 1
rand_sample = True

while True:

    cdir = bdir + f'/round_front_{round}{"_rand" if rand_sample else ""}'
    if not os.path.exists(cdir+'/traindata'):
        break
    C = pt.pickle_load(cdir + '/traindata')
    conf = C[1]
    conf.coco_im_dir = im_dir
    # sigmas = [0.01,]*7 + [0.04,]*4 + [0.09,]*6
    sigmas = 0.01*np.array([1.02824052, 1.197518  , 1.20009346, 1.09318899, 1.07948815,
        1.51404181, 1.51965481, 1.67886249, 2.05974873, 1.69516461,
        2.00565759, 4.70209639, 2.97893987, 4.97722848, 4.94152492,
        2.94189999, 4.48909379])
    # sigmas = 0.01*np.array([1.28618643, 1.40018312, 1.5528589 , 1.29503387, 1.34988427,
    #    2.05064234, 2.02291948, 2.20446902, 3.04643631, 2.27786311,
    #    2.70105468, 6.46569436, 5.8863953 , 7.79436939, 8.23480698,
    #    5.67569008, 7.14263513])

    ap = a36.compute_perf_sa(conf,'mdn_joint_fpn',val_file,'deepnet','',sigmas=sigmas)
    pp_fb.append(ap[0])
    dds_fb.append(ap[1].mean(axis=0))
    round +=1

dds_fb = np.array(dds_fb)

##
plt.figure()
plt.plot(ppd)
plt.plot(np.concatenate([ppd[:3],pp_fb],axis=0))
plt.figure()
plt.plot(dds.mean(axis=1))
plt.plot(np.concatenate([dds[:,3],dds_fb],axis=0).mean(axis=1))


## Roian's data


##
json_file = '/groups/branson/bransonlab/mayank/apt_cache_2/four_points_180806/multi_mdn_joint_torch/view_0/grone_crop_mask_23022022/train_TF.json'

im_dir = ''
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/roian_inc'

##
import json
val_frac = 0.2
J = pt.json_load(json_file)
n_full = len(J['images'])
randp = np.random.permutation(n_full)
bpt = int(val_frac*n_full)
rtr = randp[bpt:]
rval = randp[:bpt]
J['images'] = [jj for ndx,jj in enumerate(J['images']) if ndx in rtr]
uid = np.unique([a['id'] for a in J['images']])
anns = J['annotations']
sel_anns = []
for ndx,i in enumerate(anns):
    if np.all(np.array(i['keypoints'])==0.0): continue
    if i['image_id'] in uid:
        sel_anns.append(i)
J['annotations'] = sel_anns
with open(bdir + '/ma_train_TF.json','w') as f:
    json.dump(J,f)
J = pt.json_load(json_file)
J['images'] = [jj for ndx,jj in enumerate(J['images']) if ndx in rval]
uid = np.unique([a['id'] for a in J['images']])
anns = J['annotations']
sel_anns = []
for ndx,i in enumerate(anns):
    if np.all(np.array(i['keypoints'])==0.0): continue
    if i['image_id'] in uid:
        sel_anns.append(i)
J['annotations'] = sel_anns
with open(bdir + '/ma_val_TF.json','w') as f:
    json.dump(J,f)

##
import ap36_train as a36
import copy
from reuse import *
conf = pt.pickle_load('/groups/branson/bransonlab/mayank/apt_cache_2/four_points_180806/mdn_joint_fpn/view_0/2stageHT_nocrop_second_23022022/traindata')[1]
conf.multi_scale_by_bbox = False
conf.trx_align_theta = True
conf.cachedir = bdir

a36.write_coco_single_animal(conf,'',bdir+'/ma_train_TF.json',bdir+'/train_TF','train',pad=0)
a36.write_coco_single_animal(conf,'',bdir+'/ma_val_TF.json',bdir+'/val_TF','val',pad=0)

##

from reuse import *
from copy import deepcopy
import json

full_json = os.path.join(bdir,'train_TF.json')
J = pt.json_load(full_json)
jq = np.array([jj['keypoints'] for jj in J['annotations']]).reshape([-1,4,3])
invalid = np.all(jq[:,:,2]==0,axis=1)
n_train = np.count_nonzero(~invalid)
jj = np.random.permutation(n_train)
inc_size = 100
n_grs = int(n_train/inc_size)
sel_ann = [jj for ndx,jj in enumerate(J['annotations']) if not invalid[ndx]]
J['annotations'] = sel_ann

## create the dirs and training json files
for ndx in range(n_grs):
    cur_sz = range((ndx+1)*inc_size)
    cur_sel = jj[cur_sz].tolist()
    J1 = deepcopy(J)
    J1['annotations'] = [J1['annotations'][ix] for ix in cur_sel]
    uid = np.unique([a['image_id'] for a in J1['annotations']])
    ims = J1['images']
    sel_ims = [ims[ndx] for ndx, i in enumerate(ims) if i['id'] in uid]
    J1['images'] = sel_ims
    cdir = bdir + f'/train_{ndx}'
    os.makedirs(cdir,exist_ok=True)
    with open(cdir + f'/train_TF.json','w') as f:
        json.dump(J1,f)

## submit training jobs
import ap36_train as a36
conf = pt.pickle_load('/groups/branson/bransonlab/mayank/apt_cache_2/four_points_180806/mdn_joint_fpn/view_0/2stageHT_nocrop_second_23022022/traindata')[1]

conf.dl_steps = 20000
conf.batch_size = 16
conf.db_format = 'coco'
conf.rot_prob = 0.7
conf.is_multi = False
conf.multi_match_dist_factor = 0.2
conf.rrange = 20
for ndx in range(n_grs):
    cdir = bdir + f'/train_{ndx}'
    conf.cachedir = cdir
    a36.train_bsub(conf,'mdn_joint_fpn','deepnet',f'roian_inc_{ndx}')

## check accuracy
import ap36_train as a36
from reuse import *
val_file = '/groups/branson/bransonlab/mayank/apt_cache_2/roian_inc/val_TF.json'
ppd = []
dds = []
n_grs = 25
for ndx in range(n_grs):
    cdir = bdir + f'/train_{ndx}'
    C = pt.pickle_load(cdir + '/traindata')
    conf = C[1]
    conf.coco_im_dir = im_dir

    # imdir= '/groups/branson/bransonlab/mayank/apt_cache_2/ap36k_topdown_cow/val'
    # ename = f'train_{ndx}'
    # conf = a36.create_conf_sa({},cdir,imdir,ename)
    # sigmas = [0.01,]*7 + [0.04,]*4 + [0.09,]*6
    sigmas = 0.0012*np.array([6.61506409, 6.83956899, 7.38154773, 7.90749107])
    # sigmas = 0.01*np.array([1.28618643, 1.40018312, 1.5528589 , 1.29503387, 1.34988427,
    #    2.05064234, 2.02291948, 2.20446902, 3.04643631, 2.27786311,
    #    2.70105468, 6.46569436, 5.8863953 , 7.79436939, 8.23480698,
    #    5.67569008, 7.14263513])



    ap = a36.compute_perf_sa(conf,'mdn_joint_fpn',val_file,'deepnet','',sigmas=sigmas,n_classes=conf.n_classes)
    ppd.append(ap[0])
    dds.append(ap[1])

ppd = np.array(ppd)
dds = np.array(dds)

## do incremental training

# start with 120 labels which is ndx=2
import time
import json
from reuse import *
import ap36_train as a36

net_type = 'mdn_joint_fpn'
nadd = 100

bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/roian_inc'
full_json = os.path.join(bdir,'train_TF.json')

rand_sample = False
# if rand_sample:
#     os.environ['CUDA_VISIBLE_DEVICES'] = ''

if rand_sample:
    r_start = 12
else:
    r_start = 1
val_file = '/groups/branson/bransonlab/mayank/apt_cache_2/roian_inc/val_TF.json'

for round in range(r_start,20):
    if round==1:
        prev_dir = bdir + '/train_1'
    else:
        prev_dir = bdir + f'/round_{round-1}'
        if rand_sample:
            prev_dir += '_rand'

    while True and round>r_start:
        time.sleep(300)
        prev_status = a36.get_status(f'roian_fb_{round-1}{"_rand" if rand_sample else ""}',prev_dir + '/deepnet')
        if prev_status[0]:
            curt = time.strftime('%m%d%H%M')
            print(f'{curt}: {prev_status[2]} - {prev_status[1]}')
        else:
            break


    C = pt.pickle_load(prev_dir + '/traindata')
    conf = C[1]
    conf.coco_im_dir = im_dir

    preds,labels,_,_ = apt.classify_db_all(net_type,conf,full_json,name='deepnet',fullret=True)

    #
    ll = labels
    pp = preds['locs']
    ll[ll < -1000] = np.nan
    dd = np.linalg.norm(ll - pp, axis=-1)
    dd = np.nansum(dd[:,:],axis=-1)

    ord = np.argsort(dd)[::-1]

    j_prev = pt.json_load(prev_dir+ '/train_TF.json')
    j_full = pt.json_load(full_json)
    j_full_ndx = np.array([jx['image_id'] for jx in j_full['annotations']])
    prev_ids = [jx['image_id'] for jx in j_prev['annotations']]

    to_add = []
    curndx = 0
    while len(to_add)<nadd*3:
        if j_full_ndx[ord[curndx]] not in prev_ids:
            to_add.append(ord[curndx])
        curndx+=1

    if rand_sample:
        to_add = np.array(to_add)[np.random.permutation(nadd*3)]
    to_add = to_add[:nadd]

    for ix in to_add:
        full_ix = ix
        j_prev['annotations'].append(j_full['annotations'][full_ix])

    uid = np.unique([a['image_id'] for a in j_prev['annotations']])
    ims = j_full['images']
    sel_ims = [ims[ndx] for ndx, i in enumerate(ims) if i['id'] in uid]
    j_prev['images'] = sel_ims

    cdir = bdir + f'/round_{round}{"_rand" if rand_sample else ""}'
    os.makedirs(cdir,exist_ok=True)
    with open(cdir + '/train_TF.json','w') as f:
        json.dump(j_prev,f)
    if not os.path.exists(cdir + '/val_TF.json'):
        os.symlink(val_file,cdir+'/val_TF.json')

    #
    conf = pt.pickle_load(
        '/groups/branson/bransonlab/mayank/apt_cache_2/four_points_180806/mdn_joint_fpn/view_0/2stageHT_nocrop_second_23022022/traindata')[
        1]

    conf.dl_steps = 20000
    conf.batch_size = 16
    conf.db_format = 'coco'
    conf.rot_prob = 0.7
    conf.is_multi = False
    conf.multi_match_dist_factor = 0.2
    conf.rrange = 20
    conf.cachedir = cdir
    a36.train_bsub(conf,'mdn_joint_fpn','deepnet',f'roian_fb_{round}{"_rand" if rand_sample else ""}')




##

from reuse import *
import ap36_train as a36
val_file = '/groups/branson/bransonlab/mayank/apt_cache_2/roian_inc/val_TF.json'
pp_fb = []
dds_fb = []
round = 1
rand_sample = True

while True:

    cdir = bdir + f'/round_{round}{"_rand" if rand_sample else ""}'
    if not os.path.exists(cdir+'/traindata'):
        break
    C = pt.pickle_load(cdir + '/traindata')
    conf = C[1]
    conf.coco_im_dir = im_dir
    # sigmas = [0.01,]*7 + [0.04,]*4 + [0.09,]*6
    # sigmas = 0.01*np.array([1.02824052, 1.197518  , 1.20009346, 1.09318899, 1.07948815,
    #     1.51404181, 1.51965481, 1.67886249, 2.05974873, 1.69516461,
    #     2.00565759, 4.70209639, 2.97893987, 4.97722848, 4.94152492,
    #     2.94189999, 4.48909379])
    sigmas = 0.0012*np.array([6.61506409, 6.83956899, 7.38154773, 7.90749107])

    # sigmas = 0.01*np.array([1.28618643, 1.40018312, 1.5528589 , 1.29503387, 1.34988427,
    #    2.05064234, 2.02291948, 2.20446902, 3.04643631, 2.27786311,
    #    2.70105468, 6.46569436, 5.8863953 , 7.79436939, 8.23480698,
    #    5.67569008, 7.14263513])

    ap = a36.compute_perf_sa(conf,'mdn_joint_fpn',val_file,'deepnet','',sigmas=sigmas,n_classes=conf.n_classes)
    pp_fb.append(ap[0])
    dds_fb.append(ap[1])
    round +=1

dds_fb = np.array(dds_fb)

##
plt.figure()
cs  = ['r','g','b','c']
for ndx in range(ppd.shape[1]):
    plt.plot(np.arange(1,26)/25,ppd[:,ndx],cs[ndx])
for ndx in range(ppd.shape[1]):
    plt.plot(np.arange(1,22)/25,np.concatenate([ppd[:2],pp_fb],axis=0)[:,ndx],cs[ndx]+':')
plt.legend(['mAP','AP.5','AP.75'])
plt.ylabel('accuracy')
plt.xlabel('Fraction of training data')
plt.figure()
gg = np.concatenate([dds[:2], dds_fb], axis=0)
for ndx in range(dds.shape[2]):
    plt.plot(np.arange(1,26)/25,np.percentile(dds[:,:,ndx],[90,95],axis=1).T,cs[ndx])
for ndx in range(dds.shape[2]):
    plt.plot(np.arange(1,22)/25,np.percentile(gg[:,:,ndx],[90,95],axis=1).T,cs[ndx]+':')
plt.grid()
plt.legend(['nose','tail','left ear','right ear'])
plt.ylabel('avg. error of a landmark')
plt.xlabel('Fraction of training data')


## UNMARKED MICE!! My dataset!!!

## Training on increasing interactive labels

## create the datasets
import tempfile
import tarfile
from reuse import *
from scipy import io as sio
import copy
import json
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc'
idir = os.path.join(bdir,'interactive')
all_json = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/loc.json'
A = pt.json_load(all_json)
for ndx in range(1,8):
    # lbl_file = f'/groups/branson/home/kabram/APT_projects/unmarkedMice_round{ndx}_trained.lbl'
    lbl_file = f'/groups/branson/home/kabram/APT_projects/unmarkedMice_round{ndx}_trained_new.lbl'
    ltar = tarfile.TarFile(lbl_file)
    tempdir = tempfile.mkdtemp()
    ltar.extractall(tempdir)
    K = sio.loadmat(os.path.join(tempdir,'label_file.lbl'))
    B = copy.deepcopy(A)
    new_l = []
    for lndx in range(2):
        all_frs = [[ix,bb['frm']] for ix,bb in enumerate(B['locdata']) if bb['imov']==(lndx+1)]
        cur_fr = K['labels'][lndx,0]['frm'][0,0][:,0]
        frs,cc = np.unique(cur_fr,return_counts=True)
        for ix,cf in all_frs:
            if cf not in frs: continue
            fix = np.where(frs==cf)[0][0]
            assert B['locdata'][ix]['ntgt']==cc[fix]
            new_l.append(B['locdata'][ix])
    B['locdata'] = new_l
    with open(os.path.join(idir,f'loc_{ndx}.json'),'w') as f:
        json.dump(B,f)

## Run training
from reuse import *
from scipy import io as sio
import copy
import json
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc'
idir = os.path.join(bdir,'interactive')
# cfile = f'{idir}/20230823T041646_20230823T041650.json'
cfile = f'{idir}/20241231T012358_20241231T012358.json'

for ndx in range(1,8):
    cmd = f'APT_interface.py {cfile} -name round{ndx} -json_trn_file {idir}/loc_{ndx}.json -type multi_mdn_joint_torch -cache {idir} train -use_cache'
    pt.submit_job(f'unmarked_mice_inc_round{ndx}',cmd,f'{idir}/run_info',queue='gpu_a100',sing_image='/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif')


## Run training on the random label dataset
# these doesn't get used. The models in the lbl files are used

import tempfile
import tarfile
from reuse import *
from scipy import io as sio
import copy
import json
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc'
idir = os.path.join(bdir,'random')
all_json = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/loc.json'
A = pt.json_load(all_json)
for ndx in range(2):
    estr = '_more' if ndx==1 else ''
    lbl_file = f'/groups/branson/home/kabram/APT_projects/unmarkedMice_rand_labels{estr}.lbl'
    ltar = tarfile.TarFile(lbl_file)
    tempdir = tempfile.mkdtemp()
    ltar.extractall(tempdir)
    K = sio.loadmat(os.path.join(tempdir,'label_file.lbl'))
    B = copy.deepcopy(A)
    new_l = []
    for lndx in range(2):
        all_frs = [[ix,bb['frm']] for ix,bb in enumerate(B['locdata']) if bb['imov']==(lndx+1)]
        cur_fr = K['labels'][lndx,0]['frm'][0,0][:,0]
        frs,cc = np.unique(cur_fr,return_counts=True)
        for ix,cf in all_frs:
            if cf not in frs: continue
            fix = np.where(frs==cf)[0][0]
            assert B['locdata'][ix]['ntgt']==cc[fix]
            new_l.append(B['locdata'][ix])
    B['locdata'] = new_l
    with open(os.path.join(idir,f'loc_{ndx}.json'),'w') as f:
        json.dump(B,f)

## Run training
from reuse import *
from scipy import io as sio
import copy
import json
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc'
idir = os.path.join(bdir,'random')

for ndx in range(2):
    cmd = f'APT_interface.py {bdir}/interactive/20230823T041646_20230823T041650.json -name round{ndx} -json_trn_file {idir}/loc_{ndx}.json -type multi_mdn_joint_torch -cache {idir} train -use_cache'
    pt.submit_job(f'unmarked_mice_inc_round{ndx}',cmd,f'{idir}/run_info',queue='gpu_a100',sing_image='/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif')


## the GT datasets are created by running dummy "training" on the corresponding gt label files
# the label files are /groups/branson/home/kabram/APT_projects/unmarkedMice_comparison_labels.lbl and /groups/branson/home/kabram/APT_projects/unmarkedMice_comparison_labels_missing.lbl

# missing is where one of the trackers misses the mouse

# fix the file names
A = pt.json_load('/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/gt/comparison/train_TF.json')
# replace the file locations A['images] to /groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/gt/comparison/
for aa in A['images']:
    aa['file_name'] =aa['file_name'].replace('/groups/branson/home/kabram/.apt/tp31bdab16_eb6f_4ff5_83fa_41375b757b7c/unmarkedMice/multi_mdn_joint_torch/view_0/20230831T022309','/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/gt/comparison')
import json
with open('/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/gt/comparison/train_TF_fixed.json','w') as f:
    json.dump(A,f)

# do the above for missing
A = pt.json_load('/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/gt/missing/train_TF.json')
# replace the file locations A['images] to /groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/gt/missing/
for aa in A['images']:
    aa['file_name'] = aa['file_name'].replace('/groups/branson/home/kabram/.apt/tpf2edc516_39e4_4e29_a826_bec2fad733f6/unmarkedMice/multi_mdn_joint_torch/view_0/20230831T022524','/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/gt/missing')

import json
with open('/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/gt/missing/train_TF_fixed.json','w') as f:
    json.dump(A,f)

## classify the dbs
from reuse import *
conf = pt.pickle_load('/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/unmarkedMice/multi_mdn_joint_torch/view_0/round1/traindata')[1]

db_file = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/gt/comparison/train_TF_fixed.json'
db_file_missing = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/gt/missing/train_TF_fixed.json'
all_preds = []
dii = []
ipp = []
for round in range(1,8):
    conf.imsz = [2048,2048]
    conf.cachedir = f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/unmarkedMice/multi_mdn_joint_torch/view_0/round{round}'
    preds_dict,locs,info = apt.classify_db_all('multi_mdn_joint_torch',conf,db_file)
    preds = preds_dict['locs']
    preds_dict,locsm,info = apt.classify_db_all('multi_mdn_joint_torch',conf,db_file_missing)
    predsm = preds_dict['locs']
    preds = np.concatenate((preds,predsm),axis=0)
    ipp.append(preds)
    locs = np.concatenate((locs,locsm),axis=0)
    dd = np.linalg.norm(preds[:,None]-locs[:,:,None],axis=-1)
    ddm = find_dist_match(dd)
    dii.append(ddm)
    ddv = np.reshape(ddm,(-1,conf.n_classes))
    vv = ~np.isnan(ddv[:,0])
    ddv = ddv[vv,:]
    all_preds.append(ddv)

all_preds = np.array(all_preds)
dii = np.array(dii)
ipp = np.array(ipp)

## random labeling

from reuse import *
conf = pt.pickle_load('/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/unmarkedMice/multi_mdn_joint_torch/view_0/round1/traindata')[1]

db_file = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/gt/comparison/train_TF_fixed.json'
db_file_missing = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/gt/missing/train_TF_fixed.json'
rand_preds = []
drr = []
rpp = []
for round in range(2):
    conf.imsz = [2048,2048]
    conf.cachedir = f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/unmarkedMice/multi_mdn_joint_torch/view_0/round{round}'
    preds_dict,locs,info = apt.classify_db_all('multi_mdn_joint_torch',conf,db_file)
    preds = preds_dict['locs']
    preds_dict,locsm,info = apt.classify_db_all('multi_mdn_joint_torch',conf,db_file_missing)
    predsm = preds_dict['locs']
    preds = np.concatenate((preds,predsm),axis=0)
    rpp.append(preds)
    locs = np.concatenate((locs,locsm),axis=0)
    dd = np.linalg.norm(preds[:,None]-locs[:,:,None],axis=-1)
    ddm = find_dist_match(dd)
    drr.append(ddm)
    ddv = np.reshape(ddm,(-1,conf.n_classes))
    vv = ~np.isnan(ddv[:,0])
    ddv = ddv[vv,:]
    rand_preds.append(ddv)

rand_preds = np.array(rand_preds)
rpp = np.array(rpp)
drr = np.array(drr)
##
pp = np.percentile(all_preds, [50, 75, 90], axis=1)
ss = np.percentile(rand_preds, [50, 75, 90], axis=1)
f, ax = plt.subplots(2, 2)
ax = ax.flatten()
for ndx in range(4):
    ax[ndx].plot(pp[:, :, ndx].T)
    ax[ndx].set_ylim([0, 200])
    ax[ndx].plot([0, 7], [ss[:,0, ndx], ss[:,0, ndx]],'r')
    ax[ndx].plot([0, 7], [ss[:,1, ndx], ss[:,1, ndx]],'b')

##
J1 = pt.json_load(db_file)
J2 = pt.json_load(db_file_missing)

xx = np.where(np.abs(drr[-1]-(dii[-1])>20))[0]
xx = np.unique(xx)
##
x = np.random.choice(xx)
if x >= len(J1['images']):
    im_f = J2['images'][x-len(J1['images'])]['file_name']
else:
    im_f = J1['images'][x]['file_name']
im = cv2.imread(im_f)
plt.figure()
plt.imshow(im)
skl = [[0,1],[0,2],[0,3]]
mdskl(ipp[-1][x],skl,cc='r')
mdskl(rpp[-1][x],skl,cc='b')
plt.title(f'{x}')

## comparing my interactive labels to random labels
# the gt movie was tracked within GUI

# the trained model are in the lbl files
# /groups/branson/home/kabram/APT_projects/unmarkedMice_rand_labels.lbl
# /groups/branson/home/kabram/APT_projects/unmarkedMice_round7_trained_from_scratch.lbl

import TrkFile
from reuse import *

# S = TrkFile.Trk('/groups/branson/home/kabram/temp/roian_unmarked_mice_rand_track.trk') # random
# S = TrkFile.Trk('/groups/branson/home/kabram/temp/roian_unmarked_mice_rand_more_track_linked_tracklet.trk') # random 2x
# S = TrkFile.Trk('/groups/branson/home/kabram/temp/roian_unmarked_mice_rand_4x_movie3_track_tracklet.trk') # random 4x
S = TrkFile.Trk('/groups/branson/home/kabram/temp/roian_unmarked_mice_rand_more_roi_movie3_track_tracklet.trk') # random 2x with more roi
# S = TrkFile.Trk('/groups/branson/home/kabram/temp/roian_unmarked_mice_rand_4x_roi_movie3_track_tracklet.trk') # random 4x with more roi

Q = TrkFile.Trk('/groups/branson/home/kabram/temp/roian_unmarkedmice_interactive_labels.trk')
Q.convert2dense()
S.convert2dense()

##
dds = np.ones([108000, 5, 4])*np.nan

ci = np.zeros([108000])
cr = ci.copy()

for ndx in range(108000):
    aa = Q.pTrk[:, :, ndx, :]
    aa = aa[:, :, ~np.all(np.isnan(aa[:, 0, :]), axis=0)]
    ci[ndx] = aa.shape[2]
    bb = S.pTrk[:, :, ndx]
    bb = bb[:, :, ~np.all(np.isnan(bb[:, 0, :]), axis=0)]
    cr[ndx] = bb.shape[2]

    dd = np.linalg.norm(aa[..., None] - bb[:, :, None], axis=1)
    dd_m = find_dist_match(np.transpose(dd[None], [0, 2, 3, 1]))

    dds[ndx, :dd_m.shape[1]] = dd_m[0, :]
    if bb.shape[2]<aa.shape[2]:
        extra_det = np.argmax(dd.min(axis=2).sum(axis=0))
        dds[ndx,dd_m.shape[1],:] = dd.min(axis=2)[:,extra_det]

xx = np.nanmax(np.max(dds, axis=-1), axis=1)
fpr = np.count_nonzero(cr==5)
fpi = np.count_nonzero(ci==5)
fnr = np.count_nonzero(cr==3)
fni = np.count_nonzero(ci==3)

##
n_tot = 240
# thresh = 30 # for same number rand
thresh = 50 #for 2x rand and 4x rand
sel =np.where(xx>thresh)[0]
sugg = []
done = []
np.random.seed(3354)
for ndx in range(n_tot):
    cc = np.random.choice(sel)
    while cc in done:
        cc = np.random.choice(sel)
    aa = Q.pTrk[:, :, cc, :]
    aa = aa[:, :, ~np.all(np.isnan(aa[:, 0, :]), axis=0)]
    bb = S.pTrk[:, :, cc]
    bb = bb[:, :, ~np.all(np.isnan(bb[:, 0, :]), axis=0)]

    dd = np.linalg.norm(aa[..., None] - bb[:, :, None], axis=1)
    tt = 0
    if (bb.shape[2] == 5) and (aa.shape[2]==5):
        continue
    elif bb.shape[2] ==5:
        # false positives
        dd_m = find_dist_match(np.transpose(dd[None], [0, 3, 2, 1]))[0]
        if np.all(dd_m.flat<thresh):
            continue
        ix,iy = np.where(dd_m>thresh)
        ax = ix[0]; iy =iy[0]

        minn = aa[:,:,ax].min(axis=0)
        maxx = aa[:,:,ax].max(axis=0)
        tt=3
    elif aa.shape[2]<bb.shape[2]:
        bx = np.where(np.any(np.nanmin(dd,axis=1)>thresh,axis=0))[0][0]
        minn = bb[:, :, bx].min(axis=0)
        maxx = bb[:, :, bx].max(axis=0)
        tt= 1
    elif bb.shape[2]<aa.shape[2]:
        ax = np.where(np.any(np.nanmin(dd,axis=2)>thresh,axis=0))[0][0]
        minn = aa[:, :, ax].min(axis=0)
        maxx = aa[:, :, ax].max(axis=0)
        tt=2
    else:
        # dd_m = find_dist_match(np.transpose(dd[None], [0, 2, 3, 1]))[0]
        ix,iy = np.where(dds[cc]>thresh)
        bx = ix[0]; iy =iy[0]
        ax = np.where(dd[iy,:,bx]==dds[cc,bx,iy])[0][0]

        minn = aa[:,:,ax].min(axis=0)
        maxx = aa[:,:,ax].max(axis=0)
        tt=0

    sugg.append([cc,minn,maxx,tt])
    done.append(cc)


## save the suggestions as trk

sf = []
data = []
for ss in sugg:
    # if ss[3]==0:
    #     continue
    # if ss[3]!=0:
    #     continue
    sf.append(ss[0])
    data.append(np.array(ss[1:3])[:,:,None])
sf = np.array(sf)
pTrk = TrkFile.Tracklet(size_rest=[2,2],ntargets=len(data),defaultval=np.nan)
pTrk.setdata_tracklet(data,sf,sf,np.nan)

pTrkTS = TrkFile.Tracklet(size_rest=[2,],ntargets=len(data),defaultval=-np.inf)
tdat = [np.zeros_like(dt[:,0]) for dt in data]
pTrkTS.setdata_tracklet(tdat,sf,sf,-np.inf)

pTrkTag = TrkFile.Tracklet(size_rest=[2,],ntargets=len(data),defaultval=False)
tdat = [np.zeros_like(dt[:,0])>1 for dt in data]
pTrkTag.setdata_tracklet(tdat,sf,sf,False)

pTrkConf = TrkFile.Tracklet(size_rest=[2,],ntargets=len(data),defaultval=np.nan)
tdat = [np.ones_like(dt[:,0]) for dt in data]
pTrkConf.setdata_tracklet(tdat,sf,sf,np.nan)

J = TrkFile.Trk(p=pTrk,pTrkTS=pTrkTS,pTrkTag=pTrkTag,pTrkConf=pTrkConf)

# J.save('/groups/branson/home/kabram/temp/roian_inc_suggestions_missing.trk')
# J.save('/groups/branson/home/kabram/temp/roian_inc_suggestions.trk') # for same number rand labels. thresh 30
J.save('/groups/branson/home/kabram/temp/roian_inc_suggestions_4x.trk') # for 2x rand labels. thresh 50

# After this do the labeling. And then create the gt dataset by running training. Rememeber to deselect the cropping.

##
from scipy import io as sio
import TrkFile
from reuse import *
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
import torch
import movies

use_roi = False

# A copy of files should exist in the temp directory where they were originally created
# 20210924_four_female_mice_again_labels.trk has missing GT labels
# 20210924_four_female_mice_again_comparison_labels.trk has comparison GT labels

gt_lbls = [['/groups/branson/home/kabram/APT_projects/20210924_four_female_mice_again_comparison_labels.trk','//groups/branson/home/kabram/APT_projects/20210924_four_female_mice_again_labels.trk'],'/groups/branson/home/kabram/APT_projects/20210924_four_female_mice_again_comparison_labels_2x.trk','/groups/branson/home/kabram/APT_projects/20210924_four_female_mice_again_comparison_labels_4x.trk']

rand_preds = ['/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/trks/roian_unmarked_mice_rand_track.trk','/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/trks/roian_unmarked_mice_rand_more_roi_movie3_track_tracklet.trk','/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/trks/roian_unmarked_mice_rand_4x_roi_movie3_track_tracklet.trk']

rand_preds_noroi = ['/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/trks/roian_unmarked_mice_rand_track.trk','/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/trks/roian_unmarked_mice_rand_more_track_linked_tracklet.trk','/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/trks/roian_unmarked_mice_rand_4x_movie3_track_tracklet.trk']

if not use_roi:
    rand_preds = rand_preds_noroi

# G = sio.loadmat('/groups/branson/home/kabram/temp/20210924_four_female_mice_again_comparison_labels.trk')
# G = sio.loadmat('/groups/branson/home/kabram/temp/20210924_four_female_mice_again_comparison_labels_2x.trk')
# G = sio.loadmat('/groups/branson/home/kabram/temp/20210924_four_female_mice_again_comparison_labels_4x.trk')
#Gm = sio.loadmat('/groups/branson/home/kabram/temp/20210924_four_female_mice_again_labels.trk')

# S = TrkFile.Trk('/groups/branson/home/kabram/temp/roian_unmarked_mice_rand_track.trk')
# S = TrkFile.Trk('/groups/branson/home/kabram/temp/roian_unmarked_mice_rand_more_track_linked_tracklet.trk')
# S = TrkFile.Trk('/groups/branson/home/kabram/temp/roian_unmarked_mice_rand_4x_movie3_track_tracklet.trk')
# S = TrkFile.Trk('/groups/branson/home/kabram/temp/roian_unmarked_mice_rand_more_roi_movie3_track_tracklet.trk')
# S = TrkFile.Trk('/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/trks/roian_unmarked_mice_rand_4x_roi_movie3_track_tracklet.trk')
Q = TrkFile.Trk('/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/trks/roian_unmarkedmice_interactive_labels.trk')
mov_file = Q.trkData['trkInfo']['mov_file']
mcap = movies.Movie(mov_file)

S_all = [TrkFile.Trk(rand_preds[pndx]) for pndx in range(len(rand_preds))]


## Show the histogram and the order plots

f,ax = plt.subplots(len(gt_lbls),len(rand_preds)+1,figsize=(12.56,9.52))
f1,ax1 = plt.subplots(1,len(gt_lbls),figsize=(15.77,6))
#
from run_apt_ma_expts import ma_expt
# mae = ma_expt('roian')
# import multiResData
# X = multiResData.read_and_decode_without_session(mae.ex_db, mae.n_pts)
# ex_im = X[0][0]
# ex_loc = X[1][0]

fr_sel = 35000
psz = 350

a_sel = 2
im = mcap.get_frame(fr_sel)[0]
ex_pts = Q.getframe(fr_sel)[:,:,0]
vi = ~np.all(np.isnan(ex_pts[:, 0]), axis=0)  # all valid interactive label predictions
ex_pts = ex_pts[:, :, vi][:,:,a_sel]
centeroid = np.nanmean(ex_pts, axis=0)
ex_pts = ex_pts - centeroid[None]
ex_pts = ex_pts + np.array([psz / 2, psz / 2])[None]
ex_loc = ex_pts
ex_im = im[int(centeroid[1] - psz / 2):int(centeroid[1] + psz / 2), int(centeroid[0] - psz / 2):int(centeroid[0] + psz / 2)]

for lndx in range(len(gt_lbls)):
    cur_gt_lbls = gt_lbls[lndx]
    if isinstance(cur_gt_lbls,list):
        G = sio.loadmat(cur_gt_lbls[0])
        Gm = sio.loadmat(cur_gt_lbls[1])
        frs = np.concatenate([G['frm'],Gm['frm']])
        pts = np.concatenate([G['p'],Gm['p']],axis=1)
    else:
        G = sio.loadmat(cur_gt_lbls)
        frs = G['frm']
        pts = G['p']
    frs = frs-1
    pts = pts-1


    for pndx in range(len(rand_preds)):
        S = S_all[pndx]
        gg = []
        dd_ii = []
        dd_rr = []
        ov_i = []
        ov_r = []
        m_frs = []
        for ndx,fr in enumerate(frs):
           ii = Q.getframe(fr)[:,:,0]
           vi = ~np.all(np.isnan(ii[:,0]),axis=0) # all valid interactive label predictions
           ii = ii[:,:,vi]
           rr = S.getframe(fr)[:,:,0]
           vi = ~np.all(np.isnan(rr[:,0]),axis=0) # all valid interactive random predictions
           rr = rr[:,:,vi]
           if rr.size==0:
               continue
           m_frs.append(fr)
           g_cur = pts[:,ndx].reshape([2,4]).T
           dd_i = np.linalg.norm(g_cur[:,:,None]-ii,axis=1)
           mm_i = np.argmin(np.nanmean(dd_i,axis=0))
           dd_ii.append(dd_i[:,mm_i])
           dd_r = np.linalg.norm(g_cur[:,:,None]-rr,axis=1)
           mm_r = np.argmin(np.nanmean(dd_r,axis=0))
           dd_rr.append(dd_r[:,mm_r])
           az = np.array([g_cur.min(axis=0),g_cur.max(axis=0)]).reshape([-1,4])
           bz = np.array([rr.min(axis=0),rr.max(axis=0)]).reshape([4,-1]).T
           orr = tn(bbox_overlaps(torch.tensor(az),torch.tensor(bz)))[0,mm_r]
           ov_r.append(orr)
           cz = np.array([ii.min(axis=0),ii.max(axis=0)]).reshape([4,-1]).T
           oii = tn(bbox_overlaps(torch.tensor(az),torch.tensor(cz)))[0,mm_i]
           ov_i.append(oii)

        dd_ii = np.array(dd_ii)
        dd_rr = np.array(dd_rr)
        ov_i = np.array(ov_i)
        ov_r = np.array(ov_r)

        miss_thresh = 0.1
        missed_r = ov_r<miss_thresh
        missed_i = ov_i<miss_thresh

        missed = missed_r | missed_i


        prcs = [50,75,90,95]
        # f,ax = plt.subplots(1,2,figsize=(10,5))
        pp_i = np.percentile(dd_ii[~missed],prcs,axis=0)
        pp_r = np.percentile(dd_rr[~missed],prcs,axis=0)
        if pndx==0:
            pt.show_result_hist(ex_im,ex_loc,pp_i,ax=ax[lndx,0],dropoff=0.4)
        pt.show_result_hist(ex_im,ex_loc,pp_r,ax=ax[lndx,pndx+1],dropoff=0.4)
        # ax[lndx,0].set_title(f'Interactive, missed {missed_i.sum()/missed_i.size:.2f}, total {missed_i.size}')
        # ax[lndx,pndx+1].set_title(f'Random, missed {missed_r.sum()/missed_r.size:.2f}, total {missed_r.size}')
        ax[lndx,0].axis('off')
        ax[lndx,pndx+1].axis('off')

        #
        msz = 5
        malpha = 0.5
        oo_ii = np.argsort(dd_ii.mean(axis=1))
        oo_rr = np.argsort(dd_rr.mean(axis=1))
        if pndx==0:
            # ax1[lndx].scatter(np.tile(range(dd_ii.shape[0]),[4,1]).T,dd_ii[oo_ii,:],alpha=malpha,s=msz,c='b')
            ax1[lndx].scatter(range(dd_ii.shape[0]),dd_ii[oo_ii].mean(axis=1),marker='^',color='b')
        cc = plt.colormaps['Reds'](pndx/2+(2-pndx)/5)
        # ax1[lndx].scatter(np.tile(range(dd_ii.shape[0]),[4,1]).T,dd_rr[oo_rr],alpha=malpha,s=msz,c=cc)
        ax1[lndx].scatter(range(dd_ii.shape[0]),dd_rr[oo_rr].mean(axis=1),marker='^',color=cc)
        ax1[lndx].set_title(f'GT Set {lndx+1}')
        ax1[lndx].set_ylim([0,200])

    ax[0,0].set_title('Interactive')
    ax[0,1].set_title('Random 1x')
    ax[0,2].set_title('Random 2x')
    ax[0,3].set_title('Random 4x')
    for ndx in range(3):
        ax[ndx,0].axis('on')
        ax[ndx, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[ndx, 0].tick_params(axis='y', which='both', left=False, right=False, labelleft=True)
        ax[ndx, 0].set_yticks([])
    ax[0,0].set_ylabel('Set 1')
    ax[1,0].set_ylabel('Set 2')
    ax[2,0].set_ylabel('Set 3')
    plt.figure(f.number)
    plt.tight_layout()

    ax1[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax1[2].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax1[0].set_ylabel('Mean Error (pixels)')
    ax1[2].legend(['Interactive','1x','2x','4x'],loc='upper right')
    plt.figure(f1.number)
    plt.tight_layout()

## Show the match between interactive and random

ddqs = []
for pndx in range(3):
    dds = np.ones([108000, 5, 4])*np.nan
    S = S_all[pndx]
    ci = np.zeros([108000])
    cr = ci.copy()

    for ndx in range(108000):
        aa = Q.getframe(ndx)[:, :, 0, :]
        aa = aa[:, :, ~np.all(np.isnan(aa[:, 0, :]), axis=0)]
        ci[ndx] = aa.shape[2]
        bb = S.getframe(ndx)[:, :, 0]
        bb = bb[:, :, ~np.all(np.isnan(bb[:, 0, :]), axis=0)]
        cr[ndx] = bb.shape[2]

        dd = np.linalg.norm(aa[..., None] - bb[:, :, None], axis=1)
        dd_m = find_dist_match(np.transpose(dd[None], [0, 2, 3, 1]))

        dds[ndx, :dd_m.shape[1]] = dd_m[0, :]
        if bb.shape[2]<aa.shape[2]:
            extra_det = np.argmax(dd.min(axis=2).sum(axis=0))
            dds[ndx,dd_m.shape[1],:] = dd.min(axis=2)[:,extra_det]

    ddq = dds[:,:4].mean(axis=-1).flatten()
    ddqs.append(ddq)
##
f = plt.figure()
for pndx in range(3):
    ddqs[pndx].sort()
    cc = plt.colormaps['Reds'](pndx / 2 + (2 - pndx) / 5)
    plt.scatter(np.arange(len(ddqs[pndx]))/len(ddqs[0]),ddqs[pndx],color=cc)
ax = plt.gca()
ax.set_xlim([0.97,1.])
ax.set_ylim([0,100])
ax.set_xlabel('Frames (fraction) sorted by difference')
ax.set_ylabel('Mean difference (pixels)')
ax.legend(['1x','2x','4x'],loc='upper right')
ax.set_title('Agreement between interactive and random labels')

###### plotting examples

## Show examples for each round of interactive labeling

lbls = []
isz = 320
prev_ims = None
for round in range(1,8):
    cdir = f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/unmarkedMice/multi_mdn_joint_torch/view_0/round{round}'
    F = pt.json_load(f'{cdir}/train_TF.json')
    anns = F['annotations']
    ims = F['images']
    cur_ims = [[x['movid'], x['frm']] for x in ims]
    if prev_ims is not None:
        cur_ims = [x for x in cur_ims if x not in prev_ims]
    prev_ims = [[x['movid'], x['frm']] for x in ims]
    cur_anns = [x for x in anns if [ims[x['image_id']]['movid'], ims[x['image_id']]['frm']] in cur_ims]
    cur_lbls = []
    for cc in cur_anns:
        im = cv2.imread(ims[cc['image_id']]['file_name'],cv2.IMREAD_UNCHANGED)
        bctr = np.array(cc['bbox'][:2]) + np.array(cc['bbox'][2:]) / 2
        bctr = np.round(bctr).astype(np.int32)
        # grab a patch of size iszxisz centered on the bbox. Pad if necessary

        padamt = []
        sz = im.shape
        idx = []
        for i in range(2):
            b1 = bctr[1-i] - isz // 2
            b2 = bctr[1-i] + isz // 2
            p1 = -min(b1, 0)
            p2 = max(b2+ sz[i] + 1, 0)
            padamt.append([p1, p2])
            idx.append([b1 + p1, b2 + p1])
        imp = np.pad(im, padamt, 'constant', constant_values=0)
        patch = imp[idx[0][0]:idx[0][1], idx[1][0]:idx[1][1]]

        cur_a = np.array(cc['keypoints']).reshape([-1, 3])
        if np.all(cur_a[:,2]==0):
            continue
        cur_a[:,0] -= bctr[0] - isz//2
        cur_a[:,1] -= bctr[1] - isz//2
        cur_lbls.append([patch,cur_a])
    lbls.append(cur_lbls)


## Show examples for random labeling

lbls = []
isz = 320
prev_ims = None
for round in range(1,8):
    cdir = f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/unmarkedMice/multi_mdn_joint_torch/view_0/round1'
    F = pt.json_load(f'{cdir}/train_TF.json')
    anns = F['annotations']
    ims = F['images']
    cur_ims = [[x['movid'], x['frm']] for x in ims]
    cur_anns = [x for x in anns if [ims[x['image_id']]['movid'], ims[x['image_id']]['frm']] in cur_ims]
    cur_lbls = []
    for cc in cur_anns:
        im = cv2.imread(ims[cc['image_id']]['file_name'],cv2.IMREAD_UNCHANGED)
        bctr = np.array(cc['bbox'][:2]) + np.array(cc['bbox'][2:]) / 2
        bctr = np.round(bctr).astype(np.int32)
        # grab a patch of size iszxisz centered on the bbox. Pad if necessary

        padamt = []
        sz = im.shape
        idx = []
        for i in range(2):
            b1 = bctr[1-i] - isz // 2
            b2 = bctr[1-i] + isz // 2
            p1 = -min(b1, 0)
            p2 = max(b2+ sz[i] + 1, 0)
            padamt.append([p1, p2])
            idx.append([b1 + p1, b2 + p1])
        imp = np.pad(im, padamt, 'constant', constant_values=0)
        patch = imp[idx[0][0]:idx[0][1], idx[1][0]:idx[1][1]]

        cur_a = np.array(cc['keypoints']).reshape([-1, 3])
        if np.all(cur_a[:,2]==0):
            continue
        cur_a[:,0] -= bctr[0] - isz//2
        cur_a[:,1] -= bctr[1] - isz//2
        cur_lbls.append([patch,cur_a])
    lbls.append(cur_lbls)

## Show examples for GT labeling

from scipy import io as sio
import movies

gt_lbls = [['/groups/branson/home/kabram/APT_projects/20210924_four_female_mice_again_comparison_labels.trk','//groups/branson/home/kabram/APT_projects/20210924_four_female_mice_again_labels.trk'],'/groups/branson/home/kabram/APT_projects/20210924_four_female_mice_again_comparison_labels_2x.trk','/groups/branson/home/kabram/APT_projects/20210924_four_female_mice_again_comparison_labels_4x.trk']


lbls = []
isz = 320
mov_file = '/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice_again/20210924_four_female_mice_again.mjpg'
mcap = movies.Movie(mov_file)

for lndx in range(len(gt_lbls)):
    cur_gt_lbls = gt_lbls[lndx]
    if isinstance(cur_gt_lbls, list):
        G = sio.loadmat(cur_gt_lbls[0])
        Gm = sio.loadmat(cur_gt_lbls[1])
        frs = np.concatenate([G['frm'], Gm['frm']])
        pts = np.concatenate([G['p'], Gm['p']], axis=1)
    else:
        G = sio.loadmat(cur_gt_lbls)
        frs = G['frm']
        pts = G['p']
    frs = frs - 1
    pts = pts - 1
    pts = pts.reshape([2,4,pts.shape[1]])

    cur_lbls = []
    for ix,cc in enumerate(frs[:,0]):

        im = mcap.get_frame(cc)[0]
        bctr = np.mean(pts[:,:,ix],axis=1).astype(np.int32)

        # grab a patch of size iszxisz centered on the bbox. Pad if necessary
        padamt = []
        sz = im.shape
        idx = []
        for i in range(2):
            b1 = bctr[1-i] - isz // 2
            b2 = bctr[1-i] + isz // 2
            p1 = -min(b1, 0)
            p2 = max(b2+ sz[i] + 1, 0)
            padamt.append([p1, p2])
            idx.append([b1 + p1, b2 + p1])
        imp = np.pad(im, padamt, 'constant', constant_values=0)
        patch = imp[idx[0][0]:idx[0][1], idx[1][0]:idx[1][1]]

        cur_a = pts[:,:,ix].T
        cur_a[:,0] -= bctr[0] - isz//2
        cur_a[:,1] -= bctr[1] - isz//2
        cur_lbls.append([patch,cur_a])
    lbls.append(cur_lbls)

## plot the examples

n_ex = 6
f,ax = plt.subplots(len(lbls),n_ex,figsize=(n_ex*2,len(lbls)*2))
for rnd in range(len(lbls)):
    done = []
    for ndx in range(n_ex):
        while True:
            ix = np.random.randint(0, len(lbls[rnd]))
            if ix not in done:
                done.append(ix)
                break
        ax[rnd,ndx].imshow(lbls[rnd][ix][0],'gray')
        ax[rnd,ndx].scatter(lbls[rnd][ix][1][:,0],lbls[rnd][ix][1][:,1],color='r')
        ax[rnd,ndx].axis('off')
        # ax[rnd,ndx].set_title(f'Round {rnd+1}')
    ax[rnd,0].set_ylabel(f'Round {rnd+1}')
f.tight_layout()
for ndx in range(7):
    ax[ndx, 0].axis('on')
    ax[ndx, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[ndx, 0].tick_params(axis='y', which='both', left=False, right=False, labelleft=True)
    ax[ndx, 0].set_yticks([])
    # ax[ndx, 0].set_ylabel(f'Round {ndx+1}')
    ax[ndx, 0].set_ylabel(f'Set {ndx+1}')


# f.savefig('/groups/branson/home/kabram/temp/unmarked_mice_interactive_labels_rounds.svg',format='svg')
# f.savefig('/groups/branson/home/kabram/temp/unmarked_mice_interactive_labels_rounds.png',format='png')
# f.savefig('/groups/branson/home/kabram/temp/unmarked_mice_random_labels_rounds.svg',format='svg')
# f.savefig('/groups/branson/home/kabram/temp/unmarked_mice_random_labels_rounds.png',format='png')
f.savefig('/groups/branson/home/kabram/temp/unmarked_mice_gt_labels_rounds.svg',format='svg')
f.savefig('/groups/branson/home/kabram/temp/unmarked_mice_gt_labels_rounds.png',format='png')


## Results on 4x labels using 2x and interactive

## classify the dbs
from reuse import *
conf = pt.pickle_load('/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/unmarkedMice/multi_mdn_joint_torch/view_0/round1/traindata')[1]

db_file = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/unmarkedMice/multi_mdn_joint_torch/view_0/round3/train_TF.json'
db_file_2x = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/unmarkedMice/multi_mdn_joint_torch/view_0/round1/train_TF.json'

j1 = pt.json_load(db_file)
j2 = pt.json_load(db_file_2x)
i1 = [[jj['movid'],jj['frm']] for jj in j1['images']]
i2 = [[jj['movid'],jj['frm']] for jj in j2['images']]

sel = [ix for ix in range(len(i1)) if i1[ix] not in i2]

conf.imsz = [608,608]

conf.cachedir = f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/unmarkedMice/multi_mdn_joint_torch/view_0/round7'
preds_dict,locs,info = apt.classify_db_all('multi_mdn_joint_torch',conf,db_file)
preds = preds_dict['locs']
dd = np.linalg.norm(preds[:,None]-locs[:,:,None],axis=-1)
ddm = find_dist_match(dd)
ddm = ddm[sel]
ddv = np.reshape(ddm,(-1,conf.n_classes))
vv = ~np.isnan(ddv[:,0])
ddv = ddv[vv,:]
dii = ddv

conf.cachedir = f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/unmarkedMice/multi_mdn_joint_torch/view_0/round1'
preds_dict,locs,info = apt.classify_db_all('multi_mdn_joint_torch',conf,db_file)
preds = preds_dict['locs']
dd = np.linalg.norm(preds[:,None]-locs[:,:,None],axis=-1)
ddm = find_dist_match(dd)
ddm = ddm[sel]
ddv = np.reshape(ddm,(-1,conf.n_classes))
vv = ~np.isnan(ddv[:,0])
ddv = ddv[vv,:]
drr = ddv

##
import movies
import TrkFile

Q = TrkFile.Trk('/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/trks/roian_unmarkedmice_interactive_labels.trk')
mov_file = Q.trkData['trkInfo']['mov_file']
mcap = movies.Movie(mov_file)
psz = 350

fr_sel = 35000
a_sel = 2
im = mcap.get_frame(fr_sel)[0]
ex_pts = Q.getframe(fr_sel)[:,:,0]
vi = ~np.all(np.isnan(ex_pts[:, 0]), axis=0)  # all valid interactive label predictions
ex_pts = ex_pts[:, :, vi][:,:,a_sel]
centeroid = np.nanmean(ex_pts, axis=0)
ex_pts = ex_pts - centeroid[None]
ex_pts = ex_pts + np.array([psz / 2, psz / 2])[None]
ex_loc = ex_pts
ex_im = im[int(centeroid[1] - psz / 2):int(centeroid[1] + psz / 2), int(centeroid[0] - psz / 2):int(centeroid[0] + psz / 2)]

prcs = [50,75,90,95,98]
f,ax = plt.subplots(1,2,figsize=(10,5))
pp_i = np.percentile(dii, prcs, axis=0)
pp_r = np.percentile(drr, prcs, axis=0)
pt.show_result_hist(ex_im, ex_loc, pp_i, ax=ax[0], dropoff=0.4)
pt.show_result_hist(ex_im, ex_loc, pp_r, ax=ax[1], dropoff=0.4)
ax[0].axis('off')
ax[1].axis('off')
ax[0].set_title('Interactive')
ax[1].set_title('Random')
f.tight_layout()
f.savefig('/groups/branson/home/kabram/temp/interactive_vs_2x_on_4x.png',format='png')
f.savefig('/groups/branson/home/kabram/temp/interactive_vs_2x_on_4x.svg',format='svg')


################################################
################################################
# normal mice labels on unmarked mice vids

S = TrkFile.Trk('/groups/branson/home/kabram/temp/ma_expts/roian/trks/20210924_four_female_mice_0_grone_crop_mask_tracklet.trk')
Q = TrkFile.Trk('/groups/branson/home/kabram/temp/ma_expts/roian/trks/20210924_four_female_mice_0_2stageBBox_hrformer_nomask_tracklet.trk')
Q.convert2dense()
S.convert2dense()

dds = np.ones([108000, 5, 4])*np.nan

ci = np.zeros([108000])
cr = ci.copy()

for ndx in range(108000):
    aa = Q.pTrk[:, :, ndx, :]
    aa = aa[:, :, ~np.all(np.isnan(aa[:, 0, :]), axis=0)]
    ci[ndx] = aa.shape[2]
    bb = S.pTrk[:, :, ndx]
    bb = bb[:, :, ~np.all(np.isnan(bb[:, 0, :]), axis=0)]
    cr[ndx] = bb.shape[2]

    dd = np.linalg.norm(aa[..., None] - bb[:, :, None], axis=1)
    dd_m = find_dist_match(np.transpose(dd[None], [0, 2, 3, 1]))

    dds[ndx, :dd_m.shape[1]] = dd_m[0, :]
    if bb.shape[2]<aa.shape[2]:
        extra_det = np.argmax(dd.min(axis=2).sum(axis=0))
        dds[ndx,dd_m.shape[1],:] = dd.min(axis=2)[:,extra_det]

xx = np.nanmax(np.max(dds, axis=-1), axis=1)
fpr = np.count_nonzero(cr==5)
fpi = np.count_nonzero(ci==5)
fnr = np.count_nonzero(cr==3)
fni = np.count_nonzero(ci==3)

n_tot = 240
# thresh = 30 # for same number rand
thresh = 50 #for 2x rand and 4x rand
sel = np.where( (xx>thresh)&(cr==4)&(ci==4))[0]
sugg = []
done = []
np.random.seed(3354)
for ndx in range(n_tot):
    cc = np.random.choice(sel)
    while cc in done:
        cc = np.random.choice(sel)
    aa = Q.pTrk[:, :, cc, :]
    aa = aa[:, :, ~np.all(np.isnan(aa[:, 0, :]), axis=0)]
    bb = S.pTrk[:, :, cc]
    bb = bb[:, :, ~np.all(np.isnan(bb[:, 0, :]), axis=0)]

    dd = np.linalg.norm(aa[..., None] - bb[:, :, None], axis=1)
    tt = 0
    if (bb.shape[2] != 4) or (aa.shape[2]!=4):
        continue
    else:
        # dd_m = find_dist_match(np.transpose(dd[None], [0, 2, 3, 1]))[0]
        ix,iy = np.where(dds[cc]>thresh)
        bx = ix[0]; iy =iy[0]
        ax = np.where(dd[iy,:,bx]==dds[cc,bx,iy])[0][0]

        minn = aa[:,:,ax].min(axis=0)
        maxx = aa[:,:,ax].max(axis=0)
        tt=0

    sugg.append([cc,minn,maxx,tt])
    done.append(cc)

sf = []
data = []
for ss in sugg:
    # if ss[3]==0:
    #     continue
    # if ss[3]!=0:
    #     continue
    sf.append(ss[0])
    data.append(np.array(ss[1:3])[:,:,None])
sf = np.array(sf)
pTrk = TrkFile.Tracklet(size_rest=[2,2],ntargets=len(data),defaultval=np.nan)
pTrk.setdata_tracklet(data,sf,sf,np.nan)

pTrkTS = TrkFile.Tracklet(size_rest=[2,],ntargets=len(data),defaultval=-np.inf)
tdat = [np.zeros_like(dt[:,0]) for dt in data]
pTrkTS.setdata_tracklet(tdat,sf,sf,-np.inf)

pTrkTag = TrkFile.Tracklet(size_rest=[2,],ntargets=len(data),defaultval=False)
tdat = [np.zeros_like(dt[:,0])>1 for dt in data]
pTrkTag.setdata_tracklet(tdat,sf,sf,False)

pTrkConf = TrkFile.Tracklet(size_rest=[2,],ntargets=len(data),defaultval=np.nan)
tdat = [np.ones_like(dt[:,0]) for dt in data]
pTrkConf.setdata_tracklet(tdat,sf,sf,np.nan)

J = TrkFile.Trk(p=pTrk,pTrkTS=pTrkTS,pTrkTag=pTrkTag,pTrkConf=pTrkConf)

J.save('/groups/branson/home/kabram/temp/ma_expts/20210924_four_female_mice_0_labeling_suggestion.trk')


###################
###################

## Show heavy tail using just the labels

## Distance of random labeling examples to itself

from reuse import *
data = 'alice'
if data=='roian':
    from reuse import *
    J = pt.json_load('/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/unmarkedMice/multi_mdn_joint_torch/view_0/rand_2/train_TF.json')

    kk = np.array([jj['keypoints'] for jj in J['annotations'] if jj['iscrowd']==0]).reshape([-1,4,3])[:,:,:2]
    ht = [0,1]
    npp = 4
    spts = range(4)
    skel = [[0,1],[0,2],[0,3]]
else:
    from scipy import io as sio
    G = sio.loadmat('/groups/branson/home/kabram/temp/alice_close.mat')
    kk = G['p'][()].astype('float32').transpose([2,0,1])
    # kk = kk.astype(np.float32)
    ht = [0,6]
    npp = 17
    spts = range(11,17)
    skel = [ [1,2],[1,6],[1,3],[6,15],[6,7],[6,12],[6,17],[6,14], [6,13], [6,16],]
    skel = np.array(skel)-1
    triads = [[12,6,7],[13,6,7],[14,6,7],[15,6,7],[16,6,7],[17,6,7]]#,[17,6,7],[16,6,7],[15,6,7],[16,6,1],[13,6,1],[12,6,1],[17,6,1],[16,6,1],[15,6,1]]#,]
    triads = np.array(triads)-1


def smallest_angle_range(angles):
    # Convert angles to range [0, 360)
    angles = np.mod(angles, 2*np.pi)

    # Sort the list of angles
    angles.sort()

    # Compute differences between successive elements
    diffs = [np.mod(angles[i+1] - angles[i], 2*np.pi) for i in range(len(angles) - 1)]

    # Add the circular difference between the first and last element
    diffs.append(np.mod(angles[0] - angles[-1] + 2*np.pi, 2*np.pi))

    # The smallest range is 360 minus the maximum difference
    return 2*np.pi - max(diffs),angles[np.mod(np.argmax(diffs)+1,len(angles))]

skel = np.array(skel)
kk1 = kk-np.mean(kk[:,ht],axis=1,keepdims=True)
th = -np.arctan2(kk1[:,ht[0],1],kk1[:,ht[0],0])
kr_x = kk1[:,:,0]*np.cos(th)[:,None] - kk1[:,:,1]*np.sin(th)[:,None]
kr_y = kk1[:,:,0]*np.sin(th)[:,None] + kk1[:,:,1]*np.cos(th)[:,None]
kr = np.array([kr_x,kr_y]).transpose([1,2,0])

to_use = 'rotated'
if to_use == 'rotated':
    k_use = kr.copy()
else:
    k_use = kk.copy()
    k_use = k_use - np.mean(k_use[:,:2],axis=1,keepdims=True)
k_use = k_use.reshape([-1,npp*2])
kkm = k_use.mean(axis=0)
gg = np.cov((k_use-kkm).T)
gg = gg
# gg = np.diag(np.diag(gg))
# gg[[0,4,6,2]][:,[0,4,6,2]] = np.mean(gg[[0,4,6,2]][:,[0,4,6,2]])
# gg[[1,3,5,7]][:,[1,3,5,7]] = np.mean(gg[[1,3,5,7]][:,[1,3,5,7]])
kk = kk.reshape([-1,npp,2])

# traverse through the skeleton and find the angles between the edges
# first convert the skeleton to a graph
import networkx as nx
G = nx.Graph()
for ndx in range(npp):
    G.add_node(ndx)
G.add_edges_from(skel)
kk_a = []
kk_b = []
kk_a_list = []

for ndx in range(len(triads)):
    a = triads[ndx][0]
    b = triads[ndx][2]
    c = triads[ndx][1]
    cur_a = np.arctan2(kk[:,a,1]-kk[:,b,1],kk[:,a,0]-kk[:,b,0]) - np.arctan2(kk[:,c,1]-kk[:,b,1],kk[:,c,0]-kk[:,b,0])
    cur_a = np.mod(cur_a,2*np.pi)
    cur_ar,cur_an = smallest_angle_range(cur_a)
    cur_a = np.mod(cur_a-cur_an,2*np.pi)
    cur_a = np.mod(cur_a-np.mean(cur_a)+np.pi, 2 * np.pi)
    kk_b.append(cur_a)
    kk_a.append(np.cos(cur_a))
    kk_a.append(np.sin(cur_a))
    kk_a_list.append([a,c,b])

# for a in range(npp):
#     for b in range(a+1,npp):
#         if G.has_edge(a,b):
#             continue
#         if not nx.has_path(G,a,b):
#             continue
#         # find all the paths between a and b
#         paths = nx.shortest_path(G,a,b)
#         if len(paths)>3:
#             continue
#
#         for c in paths[1:-1]:
#             cur_a = np.arctan2(kk[:,a,1]-kk[:,c,1],kk[:,a,0]-kk[:,c,0]) - np.arctan2(kk[:,b,1]-kk[:,c,1],kk[:,b,0]-kk[:,c,0])
#             cur_a = np.mod(cur_a-np.median(cur_a)+np.pi, 2 * np.pi)
#             # cur_a = np.abs(cur_a-np.median(cur_a))
#             # cur_a = np.mod(cur_a, 2 * np.pi)
#             kk_b.append(cur_a)
#             kk_a.append(np.cos(cur_a))
#             kk_a.append(np.sin(cur_a))
#             kk_a_list.append([a,c,b])

kk_a = np.array(kk_a).T
kk_a = kk_a.reshape([-1,len(kk_a_list),2])
kk_b = np.array(kk_b).T

kkn1 = np.random.multivariate_normal(kkm,gg,size=kk.shape[0]*2).reshape([-1,npp,2])
kkn = np.random.randn(*kk.shape)
kku = np.random.uniform(-1,1,kk.shape)
kkp = np.random.pareto(4,kk.shape)
kkp1 = np.random.pareto(2,kk.shape)

kk_bm = np.array(kk_b).mean(axis=0)
gg_b = np.cov((kk_b-kk_bm).T)
gg_b = np.diag(np.diag(gg_b))
if gg_b.size==1:
    gg_b = np.array([[gg_b]])
# kk_bn = np.random.multivariate_normal(kk_bm,gg_b,size=kk.shape[0])
kk_bn = np.random.multivariate_normal(kk_bm,gg_b,size=kk.shape[0])
kk_an = np.concatenate([np.cos(kk_bn)[...,None],np.sin(kk_bn)[...,None]],axis=2)

kk_b = np.abs(np.pi-kk_b)
kk_bn = np.abs(np.pi-kk_bn)

dd = np.ones([kk.shape[0],kk.shape[0],len(spts)])*np.nan
dd_a = np.ones([kk.shape[0],kk.shape[0],len(kk_a_list)])*np.nan
dd_an = np.ones([kk.shape[0],kk.shape[0],len(kk_a_list)])*np.nan
# ddn = np.ones([kk.shape[0],kk.shape[0],npp])*np.nan
# ddu = np.ones([kk.shape[0],kk.shape[0],npp])*np.nan
# ddp = np.ones([kk.shape[0],kk.shape[0],npp])*np.nan
# ddp1 = np.ones([kk.shape[0],kk.shape[0],npp])*np.nan
ddn1 = np.ones([kk.shape[0],kk.shape[0],len(spts)])*np.nan
# for xx in range(len(kk)):
#     for yy in range(len(kk)):
        # dsum,d,r,t = pt.align_points(kk[xx,spts],kk[yy,spts])
        # dd[xx,yy,:] = d
        # dsum,d,r,t = pt.align_points(kkn[xx],kkn[yy])
        # ddn[xx,yy,:] = d
        # dsum,d,r,t = pt.align_points(kku[xx],kku[yy])
        # ddu[xx,yy,:] = d
        # dsum,d,r,t = pt.align_points(kkp[xx],kkp[yy])
        # ddp[xx,yy,:] = d
        # dsum,d,r,t = pt.align_points(kkp1[xx],kkp1[yy])
        # ddp1[xx,yy,:] = d
        # dsum,d,r,t = pt.align_points(kkn1[xx,spts],kkn1[yy,spts])
        # ddn1[xx,yy,:] = d
        # dd_a[xx,yy] = np.minimum( np.minimum(np.abs(np.mod(kk_b[xx]-kk_b[yy],2*np.pi)),
        #                 np.abs(np.mod(kk_b[yy]-kk_b[xx],2*np.pi))),
        #                 np.minimum(np.abs(np.mod(kk_b[xx] - kk_b[yy]+np.pi, 2 * np.pi)),
        #                   np.abs(np.mod(kk_b[yy] - kk_b[xx] + np.pi, 2 * np.pi))))
        # dd_an[xx,yy] =  np.minimum( np.minimum(np.abs(np.mod(kk_bn[xx]-kk_bn[yy],2*np.pi)),
        #                 np.abs(np.mod(kk_bn[yy]-kk_bn[xx],2*np.pi))),
        #                 np.minimum(np.abs(np.mod(kk_bn[xx] - kk_bn[yy]+np.pi, 2 * np.pi)),
        #                   np.abs(np.mod(kk_bn[yy] - kk_bn[xx] + np.pi, 2 * np.pi))))

# dd = np.linalg.norm(kr[None]-kr[:,None],axis=-1)

# ff();
# for pj in [2,5,10,15,20]:
#     px = np.nanpercentile(dd.sum(axis=-1),pj,axis=1)
#     cx = np.histogram(px,np.arange(0,200,10))[0]
#     plt.plot(np.arange(0,190,10),cx)
# plt.legend([2,5,10,15,20])
# plt.ylabel('Number of Examples')
# plt.xlabel('Distance')
# plt.yscale('log')

# hill estimate of density
import scipy

f,ax = plt.subplots()
prc = 5
smooth = False
moment = True
dt = ['data_angle','normal_angle']#'normal_data','data','normal','uniform', 'data','pareto','pareto1']
ss = ['-','--']
cc = pt.get_cmap(len(kk_a_list))
for tt in dt:

    if tt=='normal':
        inv_d = np.percentile(ddn.sum(axis=-1),prc,axis=0)
    if tt == 'normal_data':
        inv_d = np.percentile(ddn1.sum(axis=-1), prc, axis=0)
    elif tt == 'uniform':
        inv_d = np.percentile(ddu.sum(axis=-1), prc, axis=0)
    elif tt == 'pareto':
        inv_d = np.percentile(ddp.sum(axis=-1), prc, axis=0)
    elif tt == 'pareto1':
        inv_d = np.percentile(ddp1.sum(axis=-1), prc, axis=0)
    elif tt == 'data':
        inv_d = np.percentile(dd.sum(axis=-1),prc,axis=0)
    elif tt == 'data_angle':
        # inv_d = np.percentile(dd_a,prc,axis=0)
        inv_d = kk_b*180/np.pi+1
    elif tt == 'normal_angle':
        # inv_d = np.percentile(dd_an,prc,axis=0)
        inv_d = kk_bn*180/np.pi+1


    # zz = np.sort(inv_d)[::-1]
    # h_e = []
    # for k in range(1,len(zz)-1):
    #     hh = (np.sum(np.log(zz[:k+1]/zz[k])))/(k+1)
    #     if moment:
    #         hh = (np.sum(np.log(zz[:k + 1] / zz[k+1]))) / (k+1)
    #         hh2 = np.sum(np.log(zz[:k+1]/zz[k+1])**2)/(k+1)
    #         hh = hh + 1 - 1/(1-hh**2/hh2)*0.5
    #         h_e.append(hh)
    #     else:
    #         h_e.append(1 / hh)
    #
    # h_e = np.array(h_e)
    # if smooth:
    #     # smooth with u=2
    #     h_s = []
    #     for j in range(len(h_e)//2):
    #         hh = h_e[j+1:2*j+1].mean()
    #         h_s.append(hh)
    #     h_e = np.array(h_s)
    #
    # ax.plot(h_e)


    for r in range(len(kk_a_list)):

        zz = np.sort(inv_d[:,r])[::-1]+0.00001
        h_e = []
        for k in range(1,len(zz)-1):
            hh = (np.sum(np.log(zz[:k+1]/(zz[k]+0.00001))))/(k+1)
            if moment:
                hh = (np.sum(np.log(zz[:k + 1] / (zz[k+1] + 0.00001)))) / (k+1)
                hh2 = np.sum(np.log(zz[:k+1]/ (zz[k+1]+0.00001))**2)/(k+1)
                hh = hh + 1 - 1/(1-hh**2/hh2)*0.5
                h_e.append(hh)
            else:
                h_e.append(1 / hh)

        h_e = np.array(h_e)
        if smooth:
            # smooth with u=2
            h_s = []
            for j in range(len(h_e)//2):
                hh = h_e[j+1:2*j+1].mean()
                h_s.append(hh)
            h_e = np.array(h_s)

        ax.plot(h_e,color=cc[r],linestyle=ss[tt=='normal_angle'])

# ax.legend(dt)
ax.legend(np.array(kk_a_list)+1)

if not moment:
    ax.set_ylim([0,15])
    ax.plot([0,len(h_e)],[3,3],'r--')
    plt.title('Hill estimator')

else:

    ax.plot([0,len(h_e)],[1/3,1/3],'r--')
    plt.ylim([-2,2])
    # ax.set_ylim([-1,1])
    ax.plot([0, len(h_e)], [0, 0], 'k')

    plt.title('Hill Moment estimator')
# if tt == 'data':
#     ax.set_ylim([0,5])
# else:
#     ax.set_ylim([0,10])

## multivariate estimator

from qpsolvers import solve_qp
inv_d = np.percentile(dd_a, prc, axis=0)
alpha = 0.75

si = []
for i in range(inv_d.shape[1]):
    si.append(np.sort(inv_d[:,i])[::-1])
lambdas = np.zeros([inv_d.shape[0],inv_d.shape[1]])
h_k = []

C = np.diag(np.ones(inv_d.shape[1]))
C = np.concatenate([np.ones([1, inv_d.shape[1]]) * -1, C* -1], axis=0)
d = np.zeros(inv_d.shape[1] + 1)
d[0] = -1
for k in range(1,inv_d.shape[0]-1):
    tau = np.zeros([inv_d.shape[1], inv_d.shape[1]])
    for i in range(inv_d.shape[1]):
        for j in range(inv_d.shape[1]):
            tau[i,j] = np.count_nonzero( (inv_d[:,i]>si[i][k]) & (inv_d[:,j]>si[j][k]) )/k

    tau = tau/(alpha**2)
    # solve quadratic equation min(l.T * tau * l) s.t. C.l <= d

    h_e = []
    for i in range(inv_d.shape[1]):
        zz = np.sort(inv_d[:,i])[::-1] + 0.00001
        hh = (np.sum(np.log(zz[:k + 1] / zz[k]))) / (k + 1)
        hh = (np.sum(np.log(zz[:k + 1] / zz[k + 1]))) / (k + 1)
        hh2 = np.sum(np.log(zz[:k + 1] / zz[k + 1]) ** 2) / (k + 1)
        hh = hh + 1 - 1 / (1 - hh ** 2 / hh2) * 0.5
        h_e.append(hh)

    h_e = np.array(h_e)
    x = solve_qp(tau, np.zeros(inv_d.shape[1]), C, d, solver='cvxopt')
    # lambdas[k] = x[1:]
    h_k.append( (h_e*x).sum())

ff(); plt.plot(h_k)
plt.ylim([-2,2])

## Pickands estimator

f,ax = plt.subplots()
dt = ['normal','uniform', 'data','pareto','normal_data']

for tt in dt:

    if tt=='normal':
        inv_d = np.percentile(ddn.sum(axis=-1),prc,axis=0)
    elif tt == 'uniform':
        inv_d = np.percentile(ddu.sum(axis=-1), prc, axis=0)
    elif tt == 'pareto':
        inv_d = np.percentile(ddp.sum(axis=-1), prc, axis=0)
    elif tt == 'normal_data':
        inv_d = np.percentile(ddn1.sum(axis=-1), prc, axis=0)
    elif tt == 'data':
        inv_d = np.percentile(dd.sum(axis=-1),prc,axis=0)

    zz = np.sort(inv_d)[::-1]
    h_e = []
    for k in range(1,len(zz)//4-1):
        hh = np.log( (zz[k]-zz[2*k])/(zz[2*k]-zz[4*k]) )/np.log(2)
        h_e.append(hh)

    ax.plot(h_e)

ax.legend(dt)
ax.plot([0, len(h_e)], [1 / 3, 1 / 3], 'r--')
ax.plot([0,len(h_e)],[0,0],'k')
plt.title('Pickands estimator')

## Peaks over threshold estimator (install evt package)
from evt.dataset import Dataset
from evt.estimators.hill import Hill
from evt.methods.peaks_over_threshold import PeaksOverThreshold as POT
import pandas as pd

f, ax = plt.subplots()
prc = 20

dt = ['normal','uniform', 'data','pareto','normal_data']
for tt in dt:
    if tt=='normal':
        inv_d = np.percentile(ddn.sum(axis=-1),prc,axis=0)
    elif tt == 'uniform':
        inv_d = np.percentile(ddu.sum(axis=-1), prc, axis=0)
    elif tt == 'pareto':
        inv_d = np.percentile(ddp.sum(axis=-1), prc, axis=0)
    elif tt == 'normal_data':
        inv_d = np.percentile(ddn1.sum(axis=-1), prc, axis=0)
    elif tt == 'data':
        inv_d = np.percentile(dd.sum(axis=-1), prc, axis=0)
    ss = pd.Series(inv_d)
    ss.index.name = 'Index'
    dts = Dataset(ss)
    peaks_over_threshold = POT(dts,np.percentile(inv_d,prc))
    hill_e = Hill(peaks_over_threshold)

    self = hill_e
    max_number_of_order_statistics = len(self.order_statistics) - 1
    x_axis = hill_e.order_statistics.index.values[1:max_number_of_order_statistics]
    estimates, ci_lowers, ci_uppers = map(np.array, zip(*[
        self.estimate(number_of_order_statistics)[0]
        for number_of_order_statistics in x_axis
    ]))

    ax.plot(x_axis, estimates)

ax.legend(dt)
ax.plot([0, max_number_of_order_statistics], [1 / 3, 1 / 3], 'r--')
plt.title('Peaks over threshold estimator')

## log log plot after clustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

aa = linkage(kr.reshape([-1,8]),'single','euclidean')
prcs = np.arange(40,70,10)
trs = np.percentile(aa[:,2],prcs)
f,ax= plt.subplots(1,len(trs))
ax = ax.flatten()
for ndx,tr in enumerate(trs):
    Z = fcluster(aa,tr,'distance')
    bb = np.histogram(Z,np.arange(max(Z)+1)+0.5)
    cc = np.sort(bb[0])[::-1]
    ax[ndx].plot(bb[1][1:]-0.5,cc)
    ax[ndx].grid()
    ax[ndx].set_xscale('log'); ax[ndx].set_yscale('log')
    ax[ndx].set_title(f'Thresh {tr:.2f}, Prc {prcs[ndx]}')

## plot the distances

f,ax = plt.subplots()
dt = ['normal','uniform', 'data','pareto']
cc = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf']

inv_d = np.percentile(dd.sum(axis=-1), 5, axis=0)
p50 = np.percentile(inv_d, 50, axis=0)
for prc in range(5,25,5):
    for ndx,tt in enumerate(dt):
        if tt=='normal':
            inv_d = np.percentile(ddn.sum(axis=-1),prc,axis=0)
        elif tt == 'normal_data':
            inv_d = np.percentile(ddn1.sum(axis=-1), prc, axis=0)
        elif tt == 'uniform':
            inv_d = np.percentile(ddu.sum(axis=-1), prc, axis=0)
        elif tt == 'pareto':
            inv_d = np.percentile(ddp.sum(axis=-1), prc, axis=0)
        elif tt == 'data':
            inv_d = np.percentile(dd.sum(axis=-1),prc,axis=0)

        zz = np.sort(inv_d)
        q50 = np.percentile(zz,50)
        plt.plot(zz*p50/q50,color=cc[ndx])

ax.legend(dt)
ax.set_ylabel('Distance')
ax.set_xlabel('Sorted examples')
ax.set_title('Order plot of distances')



##################
##################


## Distance of 4x labels not in 2x to 2x labels
from reuse import *
conf = pt.pickle_load('/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/unmarkedMice/multi_mdn_joint_torch/view_0/round1/traindata')[1]

db_file = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/unmarkedMice/multi_mdn_joint_torch/view_0/round3/train_TF.json'
db_file_2x = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/unmarkedMice/multi_mdn_joint_torch/view_0/round1/train_TF.json'

j1 = pt.json_load(db_file)
j2 = pt.json_load(db_file_2x)
i1 = [[jj['movid'],jj['frm']] for jj in j1['images']]
i2 = [[jj['movid'],jj['frm']] for jj in j2['images']]

sel = [ix for ix in range(len(i1)) if i1[ix] not in i2]

conf.imsz = [608,608]

conf.cachedir = f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/unmarkedMice/multi_mdn_joint_torch/view_0/round1'
preds_dict,locs,info = apt.classify_db_all('multi_mdn_joint_torch',conf,db_file,full_ret=True)
preds = preds_dict['locs']
dd = np.linalg.norm(preds[:,None]-locs[:,:,None],axis=-1)
ddm = find_dist_match(dd)
ddm = ddm[sel]
ddv = np.reshape(ddm,(-1,conf.n_classes))
vv = ~np.isnan(ddv[:,0])
ddv = ddv[vv,:]
drr = ddv

##
gt = locs[sel].copy()
gt = gt.reshape([-1,4,2])
gt = gt[vv]
gt = gt-np.mean(gt,axis=1,keepdims=True)
th_g = -np.arctan2(gt[:,0,1],gt[:,0,0])
gt_x = gt[:,:,0]*np.cos(th_g)[:,None] - gt[:,:,1]*np.sin(th_g)[:,None]
gt_y = gt[:,:,0]*np.sin(th_g)[:,None] + gt[:,:,1]*np.cos(th_g)[:,None]
gt_r = np.array([gt_x,gt_y]).transpose([1,2,0])
dg_rr = np.linalg.norm(gt_r[:,None]-kr[None],axis=-1).sum(axis=-1)
px_rr = np.nanpercentile(dg_rr,10,axis=1)
occ = np.zeros(locs.shape[0:3])
occ_c = np.zeros(locs.shape[0]).astype('int')
ii = np.array(info)
J = pt.json_load(db_file)
j_a_img = np.array([jj['image_id'] for jj in J['annotations']])
j_img = np.array([jj['id'] for jj in J['images']])
j_fr = np.array([jj['frm'] for jj in J['images']])

for jj in J['annotations']:
    cur_j_ix = np.where(j_img==jj['image_id'])[0]
    cur_fr = j_fr[cur_j_ix]
    cur_o_ix = np.where(ii[:,1]==cur_fr)[0]
    occ[cur_o_ix,occ_c[cur_o_ix],:] = np.array(jj['keypoints']).reshape(4,3)[:,2]

occ = occ[sel]
occ = occ.reshape([-1,4])
occ = occ[vv]

occ_s = occ.sum(axis=1)
ff()
plt.scatter(px_rr,drr.sum(axis=1),c=occ_s/4)

##############################################

## Do the comparison with detr bbox detection
from scipy import io as sio
import TrkFile
from reuse import *
# 20210924_four_female_mice_again_labels.trk has missing GT labels
# 20210924_four_female_mice_again_comparison_labels.trk has comparison GT labels
G = sio.loadmat('/groups/branson/home/kabram/temp/20210924_four_female_mice_again_comparison_labels.trk')
# G = sio.loadmat('/groups/branson/home/kabram/temp/20210924_four_female_mice_again_comparison_labels_2x.trk')
#Gm = sio.loadmat('/groups/branson/home/kabram/temp/20210924_four_female_mice_again_labels.trk')

# S = TrkFile.Trk('/groups/branson/home/kabram/temp/roian_unmarked_mice_rand_track.trk')
S = TrkFile.Trk('/groups/branson/home/kabram/temp/roian_unmarked_mice_2stage_linked_stg1.trk')

Q = TrkFile.Trk('/groups/branson/home/kabram/temp/roian_unmarkedmice_interactive_labels.trk')

##
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
import torch
gg = []
dd_ii = []
dd_rr = []
frs = G['frm']-1
pts = G['p']-1
# frs = np.concatenate([G['frm'],Gm['frm']])
# pts = np.concatenate([G['p'],Gm['p']],axis=1)
ov_i = []
ov_r = []
for ndx,fr in enumerate(frs):
   ii = Q.getframe(fr)[:,:,0]
   vi = ~np.all(np.isnan(ii[:,0]),axis=0)
   ii = ii[:,:,vi]
   iib = np.array([ii.min(axis=0),ii.max(axis=0)]).reshape([4,-1]).T
   rr = S.getframe(fr)[:,:,0]
   vi = ~np.all(np.isnan(rr[:,0]),axis=0)
   rr = rr[:,:,vi].reshape([4,-1]).T
   g_cur = pts[:,ndx].reshape([2,4]).T
   g_curb = np.array([g_cur.min(axis=0),g_cur.max(axis=0)]).reshape([4,-1]).T
   dd_i = tn(bbox_overlaps(torch.tensor(g_curb),torch.tensor(iib))).max()
   dd_ii.append(dd_i)
   dd_r = tn(bbox_overlaps(torch.tensor(g_curb),torch.tensor(rr))).max()
   dd_rr.append(dd_r)

dd_ii = np.array(dd_ii)
dd_rr = np.array(dd_rr)


## compare randoms to interactive's rounds

rand_lbls = ['/groups/branson/home/kabram/APT_projects/unmarkedMice_rand_labels.lbl','/groups/branson/home/kabram/APT_projects/unmarkedMice_rand_labels_more.lbl','/groups/branson/home/kabram/APT_projects/unmarkedMice_rand_labels_4x.lbl']

int_lbls = [f'/groups/branson/home/kabram/APT_projects/unmarkedMice_round{ix}_trained.lbl' for ix in range(1,8)]
int_lbls.append('/groups/branson/home/kabram/APT_projects/unmarkedMice_round7_trained_from_scratch.lbl')

gts = ['/groups/branson/home/kabram/temp/20210924_four_female_mice_again_comparison_labels.trk','/groups/branson/home/kabram/temp/20210924_four_female_mice_again_comparison_labels_2x.trk','/groups/branson/home/kabram/temp/20210924_four_female_mice_again_comparison_labels_4x.trk']

from scipy import io as sio
import tempfile
import tarfile
import glob
import os
import json

##
for ndx,g in enumerate(gts):
    list_json = os.path.join(f'/groups/branson/home/kabram/temp/unmarked_gt_res/gt_list_{ndx}.json')
    G = sio.loadmat(g)
    frs = G['frm'][:,0]
    toTrack = [[1,0,int(fr)] for fr in frs]
    out_dict = {'movieFiles':['/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice_again/20210924_four_female_mice_again.mjpg'],'toTrack':toTrack,'trxFiles':[],'cropLocs':[]}
    with open(list_json,'w') as fid:
        json.dump(out_dict,fid)

##
# do gt using random labels
import APT_interface as apt
import h5py
rand_res = []
for rndx,lbl_file in enumerate(rand_lbls):
    # untar rr into a temp directory
    ltar = tarfile.TarFile(lbl_file)
    tempdir = tempfile.mkdtemp()
    ltar.extractall(tempdir)
    # find the json file in the extracted directory
    lbl_json = glob.glob(os.path.join(tempdir,'unmarkedMice','*.json'))[0]
    # find the directory that contains the model
    cache_dir = os.path.join(os.path.dirname(lbl_json),'multi_mdn_joint_torch','view_0')
    cache_dir = glob.glob(os.path.join(cache_dir,'*'))[0]
    train_name = os.path.basename(cache_dir)
    cur_res = []
    for ndx in range(len(gts)):
        list_json = os.path.join(f'/groups/branson/home/kabram/temp/unmarked_gt_res/gt_list_{ndx}.json')
        cmd = f"{lbl_json} -name {train_name} -type multi_mdn_joint_torch -cache {tempdir} track -list_file {list_json} -out //groups/branson/home/kabram/temp/unmarked_gt_res/rand_{rndx}_gt_{ndx}.mat"
        apt.main(cmd.split())
        res = h5py.File(f'/groups/branson/home/kabram/temp/unmarked_gt_res/rand_{rndx}_gt_{ndx}.mat','r')
        ll = res['labeled_locs'][()]
        pp = res['pred_locs']['locs'][()]
        cur_res.append([ll,pp])

    rand_res.append(cur_res)

# do gt using interactive labels
import APT_interface as apt
import h5py

int_res = []
for rndx, lbl_file in enumerate(int_lbls):
    # untar rr into a temp directory
    ltar = tarfile.TarFile(lbl_file)
    tempdir = tempfile.mkdtemp()
    ltar.extractall(tempdir)
    # find the json file in the extracted directory
    lbl_json = glob.glob(os.path.join(tempdir, 'unmarkedMice', '*.json'))[0]
    # find the directory that contains the model
    cache_dir = os.path.join(os.path.dirname(lbl_json), 'multi_mdn_joint_torch', 'view_0')
    cache_dir = glob.glob(os.path.join(cache_dir, '*'))[0]
    train_name = os.path.basename(cache_dir)
    cur_res = []
    for ndx in range(len(gts)):
        list_json = os.path.join(f'/groups/branson/home/kabram/temp/unmarked_gt_res/gt_list_{ndx}.json')
        cmd = f"{lbl_json} -name {train_name} -type multi_mdn_joint_torch -cache {tempdir} track -list_file {list_json} -out //groups/branson/home/kabram/temp/unmarked_gt_res/int_{rndx}_gt_{ndx}.mat"
        apt.main(cmd.split())
        res = h5py.File(f'/groups/branson/home/kabram/temp/unmarked_gt_res/int_{rndx}_gt_{ndx}.mat', 'r')
        ll = res['labeled_locs'][()]
        pp = res['pred_locs']['locs'][()]
        cur_res.append([ll, pp])

    int_res.append(cur_res)


##
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
import torch
from run_apt_ma_expts import ma_expt

mae = ma_expt('roian')
import multiResData

X = multiResData.read_and_decode_without_session(mae.ex_db, mae.n_pts)
ex_im = X[0][0]
ex_loc = X[1][0]

all_rand_res = []
all_int_res = []
for gndx in range(len(gts)):
    G = sio.loadmat(gts[gndx])
    gg = []
    frs = G['frm']-1
    pts = G['p']-1
    cur_res = []
    for rndx in range(len(rand_res)):
        dd_rr = []
        ov_r = []
        for ndx in range(len(rand_res[0][gndx][0][0,0,0])):
           rr =rand_res[rndx][gndx][1][:,:,:,ndx]
           vi = ~np.all(np.isnan(rr[:,0]),axis=0)
           rr = rr[:,:,vi]
           g_cur = pts[:, ndx].reshape([2, 4])
           dd_i = np.linalg.norm(g_cur[:,:,None]-rr,axis=0)
           mm_r = np.argmin(np.nanmean(dd_i,axis=0))
           dd_rr.append(dd_i[:,mm_r])
           az = np.array([g_cur.min(axis=1),g_cur.max(axis=1)]).reshape([-1,4])
           bz = np.array([rr.min(axis=1),rr.max(axis=1)]).reshape([4,-1]).T
           orr = tn(bbox_overlaps(torch.tensor(az),torch.tensor(bz)))[0,mm_r]
           ov_r.append(orr)

        dd_rr = np.array(dd_rr)
        ov_r = np.array(ov_r)
        cur_res.append([dd_rr,ov_r])
    all_rand_res.append(cur_res)

    cur_res = []
    for rndx in range(len(int_res)):
        dd_rr = []
        ov_r = []
        for ndx in range(len(int_res[0][gndx][0][0,0,0])):
           rr = int_res[rndx][gndx][1][:,:,:,ndx]
           vi = ~np.all(np.isnan(rr[:,0]),axis=0)
           rr = rr[:,:,vi]
           g_cur = pts[:, ndx].reshape([2, 4])
           dd_i = np.linalg.norm(g_cur[:,:,None]-rr,axis=0)
           mm_r = np.argmin(np.nanmean(dd_i,axis=0))
           dd_rr.append(dd_i[:,mm_r])
           az = np.array([g_cur.min(axis=1),g_cur.max(axis=1)]).reshape([-1,4])
           bz = np.array([rr.min(axis=1),rr.max(axis=1)]).reshape([4,-1]).T
           orr = tn(bbox_overlaps(torch.tensor(az),torch.tensor(bz)))[0,mm_r]
           ov_r.append(orr)

        dd_rr = np.array(dd_rr)
        ov_r = np.array(ov_r)
        cur_res.append([dd_rr,ov_r])
    all_int_res.append(cur_res)


## comaprison table
C_res = np.ones([len(rand_res),len(int_res),len(gts),2])*np.nan
for rndx in range(len(rand_res)):
    for indx in range(len(int_res)):
        for gndx in range(len(gts)):
            a1 = np.count_nonzero(all_rand_res[gndx][rndx][0]>all_int_res[gndx][indx][0]+15)
            a2 = np.count_nonzero(all_int_res[gndx][indx][0]>all_rand_res[gndx][rndx][0]+15)
            frac = (a1-a2)/(a1+a2)
            C_res[rndx,indx,gndx,0] = frac

#
gndx = 2
plt.figure(); plt.imshow(C_res[:,:,gndx,0])
plt.colorbar()
plt.ylabel('Random Labels')
ax = plt.gca()
ax.set_yticks(range(len(rand_res)))
ax.set_yticklabels(['x','2x','4x'])
ax.set_xticklabels([f'{x}' for x in range(len(int_res)+1)])
plt.xlabel('Interactive Label Rounds')

## do ID tracking using various models


from reuse import *

movs = ['/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice/20210924_four_female_mice_0.mjpg','/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice/20210924_four_female_mice_1.mjpg']

rand_lbls = ['/groups/branson/home/kabram/APT_projects/unmarkedMice_rand_labels.lbl','/groups/branson/home/kabram/APT_projects/unmarkedMice_rand_labels_more_roi.lbl','/groups/branson/home/kabram/APT_projects/unmarkedMice_rand_labels_4x_roi.lbl']

int_lbls = [f'/groups/branson/home/kabram/APT_projects/unmarkedMice_round{ix}_trained.lbl' for ix in range(1,8)]

all_lbls = rand_lbls[-1:]+int_lbls[-1:]
out_dir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/trks'
submit = False
for mov in movs:
    for lbl in all_lbls:
        mn = os.path.basename(mov).split('.')[0]
        lname = os.path.basename(lbl).split('.')[0]
        name = f'{mn}_{lname}'
        cmd = f'APT_track.py -mov {mov} -lbl_file {lbl} -conf_params link_id True link_id_training_iters 100000 -out {out_dir}/{name}.trk'
        # cmd = f'APT_track.py -mov {mov} -lbl_file {lbl} -conf_params link_id True link_id_training_iters 100000 link_id_mining_steps 1 -out {out_dir}/{name}_nohardmine.trk'
        if submit:
            pt.submit_job(name,cmd,f'{out_dir}/run_info',queue='gpu_a100',sing_image='/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif')
        else:
            print(cmd)


##
from reuse import *
import TrkFile as trkf

movs = ['/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice/20210924_four_female_mice_0.mjpg','/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice/20210924_four_female_mice_1.mjpg']


rand_lbls = ['/groups/branson/home/kabram/APT_projects/unmarkedMice_rand_labels.lbl','/groups/branson/home/kabram/APT_projects/unmarkedMice_rand_labels_more_roi.lbl','/groups/branson/home/kabram/APT_projects/unmarkedMice_rand_labels_4x_roi.lbl']

int_lbls = [f'/groups/branson/home/kabram/APT_projects/unmarkedMice_round{ix}_trained.lbl' for ix in range(1,8)]

all_lbls = rand_lbls[-1:]+int_lbls[-1:]
out_dir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/trks'

missing = []
acc = []
for cur_mov in movs:
    ac = []
    exp_name = os.path.basename(cur_mov).split('.')[0]
    for lbl in all_lbls:
        lname = os.path.basename(lbl).split('.')[0]
        name = f'{exp_name}_{lname}'
        out_trk = f'{out_dir}/{name}.trk'
        if not os.path.exists(out_trk):
            missing.append([cur_mov,curt])
            continue
        trk = trkf.Trk(out_trk)
        counts = []
        for xx in range(trk.ntargets):
            jj = trk.gettarget(xx)[0, 0, :, 0]
            clen = np.count_nonzero(~np.isnan(jj))
            counts.append(clen)

        counts= np.array(counts)
        nfr = max(trk.nframes)
        ac.append([lname,np.count_nonzero(counts>0.9*nfr),np.sum(counts)/10])
    acc.append([exp_name,ac])

print(acc)



##############################################
##############################################

## SHOW HEAVY TAIL USING INCREMENTAL TRAINING


## for this train using exponential subsets of random labels
from reuse import *

idir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/'
rand_json = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/loc_0.json'
n_rounds = 4
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/'

## create subsets

import json
S = pt.json_load(rand_json)

def do_rectangles_intersect(rect1, rect2):
  """
  This function checks if two rectangles defined by their bottom-left and top-right corners intersect.

  Args:
      rect1: A tuple (x1, y1, x2, y2) representing the bottom-left and top-right corners of the first rectangle.
      rect2: A tuple (x1, y1, x2, y2) representing the bottom-left and top-right corners of the second rectangle.

  Returns:
      True if the rectangles intersect, False otherwise.
  """

  # Check if one rectangle is completely to the left or right of the other
  if rect1[0] >= rect2[2] or rect2[0] >= rect1[2]:
    return False

  # Check if one rectangle is completely above or below the other
  if rect1[1] >= rect2[3] or rect2[1] >= rect1[3]:
    return False

  # Otherwise, there is an intersection
  return True


intersecting = []
for xx,ll in enumerate(S['locdata']):
    if ll['nextra_roi']==0:
        continue
    nroi = np.array(ll['extra_roi']).reshape([8,-1])
    for ndx in range(ll['nextra_roi']):
        roi = np.array(ll['roi']).reshape([8,-1])
        for ix in range(roi.shape[1]):
            if do_rectangles_intersect(roi[[0,4,2,5],ix],nroi[[0,4,2,5],ndx]):
                intersecting.append(xx)
                print(f'Intersecting {xx}')
                break


ntgts = np.array([ll['ntgt'] for ll in S['locdata']])
intersecting_n = ntgts[intersecting].sum()

tot_n = ntgts.sum()

np.random.seed(3354)
rand_ord = np.random.permutation(len(ntgts))
for rndx in range(n_rounds):
    cur_tot = tot_n//(2**(rndx+1))
    cur_sel = intersecting.copy()
    for cc in rand_ord:
        if cc in cur_sel:
            continue
        cur_sel.append(cc)
        if ntgts[cur_sel].sum()>cur_tot:
            cur_sel = cur_sel[:-1]
            break

    ldata = [ss for ndx,ss in enumerate(S['locdata']) if ndx in cur_sel]
    out_json = f'{idir}/loc_1_{rndx}.json'
    with open(out_json,'w') as fid:
        json.dump({'movies':S['movies'],'splitnames':S['splitnames'],'locdata':ldata},fid)


## train on these subsets. 1_x indicates 1/2^(x+1) of the total data

from reuse import *
import tarfile
import APT_interface as apt
import tempfile
import glob
import shutil

# lbl_file = '/groups/branson/home/kabram/APT_projects/unmarkedMice_rand_labels.lbl'

# ltar = tarfile.TarFile(lbl_file)
# tempdir = tempfile.mkdtemp()
# ltar.extractall(tempdir)
# # find the json file in the extracted directory
# lbl_json = glob.glob(os.path.join(tempdir, 'unmarkedMice', '*.json'))[0]
# out_lbl = '/groups/branson/home/kabram/temp/unmarkedMice_rand_labels.json'
# shutil.copy(lbl_json,out_lbl)
# out_lbl = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/20230823T041646_20230823T041650.json'
out_lbl = f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/20241231T012358_20241231T012358.json'

# These are subsets of the random labels
for rndx in range(n_rounds):
    train_name = f'rand_1_{rndx}'
    cmd = f"APT_interface.py {out_lbl} -name {train_name} -conf_params mdn_pred_dist True batch_size 4 dl_steps 100000 multi_crop_ims False -type multi_mdn_joint_torch -cache {idir} -json_trn_file {idir}/loc_1_{rndx}.json train -use_cache"
    pt.submit_job(f'unmarked_mice_inc_rand_1_round{rndx}', cmd, f'{idir}/run_info', queue='gpu_a100', sing_image='/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif')

# train 1x,2x and 4x
for rndx in range(3):
    train_name = f'rand_{rndx}'
    if rndx>0:
        estr = '_roi'
    else:
        estr = ''
    cmd = f"APT_interface.py {out_lbl} -name {train_name} -conf_params mdn_pred_dist True batch_size 4 dl_steps 100000 multi_crop_ims False -type multi_mdn_joint_torch -cache {idir} -json_trn_file {idir}/loc_{rndx}{estr}.json train -use_cache"
    pt.submit_job(f'unmarked_mice_inc_rand_round{rndx}', cmd, f'{idir}/run_info', queue='gpu_a100', sing_image='/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif')

## train using interactive labels

from reuse import *
# out_lbl = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/20230823T041646_20230823T041650.json'
out_lbl = f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/20241231T012358_20241231T012358.json'
cdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/'
for rndx in range(1,8):
    train_name = f'int_{rndx}'
    cmd = f"APT_interface.py {out_lbl} -name {train_name} -conf_params mdn_pred_dist True batch_size 4 dl_steps 100000 multi_crop_ims False -type multi_mdn_joint_torch -cache {cdir} -json_trn_file {cdir}/loc_{rndx}.json train -use_cache"
    pt.submit_job(f'unmarked_mice_inc_int_round{rndx}', cmd, f'{cdir}/run_info', queue='gpu_a100', sing_image='/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif')


## combine 4x_roi and interactive label to create a joint project
import torch
import mmpose
from packaging import version
if version.parse(mmpose.__version__).major>0:
    from mmdet.structures.bbox import bbox_overlaps
else:
    from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps

int_json = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/loc_7.json'
rand_json = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/loc_2_roi.json'

out_json = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/joint/loc.json'
l1 = pt.json_load(int_json)
l2 = pt.json_load(rand_json)

a1 = [(ll['imov'],ll['frm']) for ll in l1['locdata']]
a2 = [(ll['imov'],ll['frm']) for ll in l2['locdata']]

for ndx,ll in enumerate(l2['locdata']):
    if (ll['imov'],ll['frm']) in a1:
        print(f'Intersecting {ndx}')
        ndx1 = a1.index((ll['imov'],ll['frm']))
        p1 = np.array(l1['locdata'][ndx1]['pabs'])
        p2 = np.array(ll['pabs'])
        if p2.size>0:
            r1 = np.array(l1['locdata'][ndx1]['roi'])
            r2 = np.array(ll['roi'])

            rr1 = r1.reshape([2,4,-1])[:,[0,2],:].T.reshape([-1,4])
            rr2 = r2.reshape([2,4,-1])[:,[0,2],:].T.reshape([-1,4])
            dd = tn(bbox_overlaps(torch.tensor(rr1), torch.tensor(rr2)))
            if p1.ndim==1:
                p1 = p1[:,None]
            if p2.ndim==1:
                p2 = p2[:,None]
            for ix in range(dd.shape[1]):
                if np.all(dd[:,ix]<0.7):
                    print(f'No match {ix}')
                    p1 = np.concatenate([p1,p2[:,ix][:,None]],axis=1)
                    r1 = np.concatenate([r1,r2[:,ix][:,None]],axis=1)
            l1['locdata'][ndx1]['pabs'] = p1.tolist()
            l1['locdata'][ndx1]['roi'] = r1.tolist()
            l1['locdata'][ndx1]['ntgt'] = p1.shape[1]
        l1['locdata'][ndx1]['nextra_roi'] = l1['locdata'][ndx1]['nextra_roi']+ll['nextra_roi']
        l1['locdata'][ndx1]['extra_roi'] = l1['locdata'][ndx1]['extra_roi']+ll['extra_roi']
    else:
        l1['locdata'].append(ll)

import json
with open(out_json,'w') as fid:
    json.dump(l1,fid)

## train on the joint dataset

from reuse import *
# out_lbl = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/20230823T041646_20230823T041650.json'
out_lbl = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/20241231T012358_20241231T012358.json'
cdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/joint/'
out_json = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/joint/loc.json'

train_name = f'joint_re'
cmd = f"APT_interface.py {out_lbl} -name {train_name} -conf_params mdn_pred_dist True batch_size 4 dl_steps 100000 multi_crop_ims False -type multi_mdn_joint_torch -cache {cdir} -json_trn_file {cdir}/loc.json train -use_cache"
pt.submit_job(f'unmarked_mice_inc_joint', cmd, f'{cdir}/run_info', queue='gpu_a100', sing_image='/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif')

## track the gt movie using the trained models for the random labels
from reuse import *

gt_movie = '/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice_again/20210924_four_female_mice_again.mjpg'

out_lbl = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/20230823T041646_20230823T041650.json'
idir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/'
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random'
n_rounds = 4
for rndx in range(n_rounds):
    train_name = f'rand_1_{rndx}'
    cmd = f"APT_interface.py {out_lbl} -name {train_name} -type multi_mdn_joint_torch -conf_params mdn_pred_dist True link_id True link_id_training_iters 100000 -cache {idir} -json_trn_file {idir}/loc_1_{rndx}.json track -mov {gt_movie} -out {bdir}/trks/20210924_four_female_mice_again_rand_1_{rndx}.trk"
    pt.submit_job(f'unmarked_mice_inc_rand_round_1_{rndx}_track', cmd, f'{idir}/run_info', queue='gpu_a100', sing_image='/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif')

for rndx in range(3):
    train_name = f'rand_{rndx}'
    cmd = f"APT_interface.py {out_lbl} -name {train_name} -type multi_mdn_joint_torch -conf_params mdn_pred_dist True link_id True link_id_training_iters 100000 -cache {idir} -json_trn_file {idir}/loc_1_{rndx}.json track -mov {gt_movie} -out {bdir}/trks/20210924_four_female_mice_again_rand_{rndx}.trk"
    pt.submit_job(f'unmarked_mice_inc_rand_round_{rndx}_track', cmd, f'{idir}/run_info', queue='gpu_a100', sing_image='/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif')

## track the gt movie using interactive labels

gt_movie = '/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice_again/20210924_four_female_mice_again.mjpg'
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive'
#out_lbl = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/20230823T041646_20230823T041650.json'
out_lbl = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/20241231T012358_20241231T012358.json'

out_dir = f'{bdir}/trks'
for rndx in range(1,8):
    train_name = f'int_{rndx}'
    cmd = f"APT_interface.py {out_lbl} -name {train_name} -type multi_mdn_joint_torch -conf_params mdn_pred_dist True link_id True link_id_training_iters 100000 -cache {bdir} -json_trn_file {bdir}/loc_{rndx}.json track -mov {gt_movie} -out {bdir}/trks/20210924_four_female_mice_again_rand_{rndx}.trk"
    pt.submit_job(f'unmarked_mice_inc_rand_round_int_{rndx}_track', cmd, f'{bdir}/run_info', queue='gpu_a100', sing_image='/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif')

## track the gt movie using the joint labels

gt_movie = '/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice_again/20210924_four_female_mice_again.mjpg'
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/joint'
# out_lbl = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/20230823T041646_20230823T041650.json'
out_lbl = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/20241231T012358_20241231T012358.json'

estr = '_re'
out_dir = f'{bdir}/trks'
train_name = f'joint{estr}'
cmd = f"APT_interface.py {out_lbl} -name {train_name} -type multi_mdn_joint_torch -conf_params mdn_pred_dist True link_id True link_id_training_iters 100000 -cache {bdir} -json_trn_file {bdir}/loc.json track -mov {gt_movie} -out {bdir}/trks/20210924_four_female_mice_again_joint{estr}.trk"
pt.submit_job(f'unmarked_mice_inc_rand_round_joint_track{estr}', cmd, f'{bdir}/run_info', queue='gpu_a100', sing_image='/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif')


## Compare tracking
import TrkFile
import mmpose
from packaging import version
if version.parse(mmpose.__version__).major>0:
    from mmdet.structures.bbox import bbox_overlaps
else:
    from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
import torch
import tqdm
from reuse import *

n_rounds = 4
idir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random'
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc'

trks = [f'{idir}/trks/20210924_four_female_mice_again_rand_1_{rndx}_tracklet.trk' for rndx in range(n_rounds-1,-1,-1)]
trks1 = [f'{idir}/trks/20210924_four_female_mice_again_rand_{rndx}_tracklet.trk' for rndx in range(3)]

trks = trks + trks1

trks_j = f'{bdir}/joint/trks/20210924_four_female_mice_again_joint_tracklet.trk'
# S_all = [TrkFile.Trk(trks[pndx]) for pndx in range(len(trks))]
S_j = TrkFile.Trk(trks_j)
# S_4x = TrkFile.Trk(trks_i[-1])
S_j.convert2dense()

ddqs = []
for pndx in range(len(trks)):
    dds = np.ones([108000, 4, 4])*np.nan
    fp = np.zeros([108000])
    fn = np.zeros([108000])
    ee = np.zeros([108000])
    oo = np.zeros([108000,4])
    S = TrkFile.Trk(trks[pndx])
    S.convert2dense()
    ci = np.zeros([108000])
    cr = ci.copy()

    for ndx in tqdm.tqdm(range(108000)):
        aa = S_j.pTrk[:,:,ndx]
        # aa = S_4x.getframe(ndx)[:, :, 0, :]
        aa = aa[:, :, ~np.all(np.isnan(aa[:, 0, :]), axis=0)]
        ci[ndx] = aa.shape[2]
        # bb = S.getframe(ndx)[:, :, 0]
        bb = S.pTrk[:,:,ndx]
        bb = bb[:, :, ~np.all(np.isnan(bb[:, 0, :]), axis=0)]
        cr[ndx] = bb.shape[2]

        dd = np.linalg.norm(aa[..., None] - bb[:, :, None], axis=1)
        # dd_m = find_dist_match(np.transpose(dd[None], [0, 2, 3, 1]))

        az = np.array([aa.min(axis=0), aa.max(axis=0)]).reshape([4,-1]).T
        bz = np.array([bb.min(axis=0), bb.max(axis=0)]).reshape([4, -1]).T
        orr = tn(bbox_overlaps(torch.tensor(az), torch.tensor(bz)))

        matched1 = np.zeros(orr.shape[0])
        matched2 = np.zeros(orr.shape[1])
        count = 0
        for ix in range(orr.shape[0]):
            jx = np.argmax(orr[ix])
            if count<4:
                oo[ndx, count] = orr[ix, jx]
            if orr[ix].max()>0.1:
                if count==4:
                    ee[ndx] = 1
                    break
                jx = np.argmax(orr[ix])
                dds[ndx,count,:] = dd[:,ix,jx]
                matched1[ix] = 1
                matched2[jx] = 1
                oo[ndx,count] = orr[ix,jx]
                count = count+1
        fn[ndx] = 4-np.count_nonzero(matched1==1)
        fp[ndx] = np.count_nonzero(matched2==0)


    ddq = dds[:,:4]
    ddqs.append([ddq,fn,fp,ee,oo])

ff(); [plt.plot(np.sort(dd1[0].flatten())) for dd1 in ddqs]
plt.legend([f'{x+1}' for x in range(len(trks))])
plt.title('Random')
plt.xlim([1.4e6,1.75e6])
plt.ylim([-5,50])
plt.ylabel('Distance (px) between predictions')
plt.xlabel('Predictions Order')
plt.title('Prediction agreement of randomly labeled models')
# plt.savefig('/groups/branson/home/kabram/temp/random_matches.svg')
# plt.savefig('/groups/branson/home/kabram/temp/random_matches.png')


## Distance of interactive labels to final interactive

import TrkFile
import mmpose
if version.parse(mmpose.__version__).major>0:
    from mmdet.structures.bbox import bbox_overlaps
else:
    from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
import torch
import tqdm

out_dir = f'{bdir}/interactive/trks'
trks_i = [f'{out_dir}/20210924_four_female_mice_again_rand_{ndx}_tracklet.trk' for ndx in range(1,8)]

trks_j = f'{bdir}/joint/trks/20210924_four_female_mice_again_joint_tracklet.trk'
# S_all = [TrkFile.Trk(trks[pndx]) for pndx in range(len(trks))]
S_j = TrkFile.Trk(trks_j)
# S_4x = TrkFile.Trk(trks_i[-1])
S_j.convert2dense()

ddqi = []
for pndx in range(len(trks_i)):
    dds = np.ones([108000, 4, 4])*np.nan
    fp = np.zeros([108000])
    fn = np.zeros([108000])
    ee = np.zeros([108000])
    oo = np.ones([108000,4])*np.nan
    S = TrkFile.Trk(trks_i[pndx])
    S.convert2dense()
    ci = np.zeros([108000])
    cr = ci.copy()

    for ndx in tqdm.tqdm(range(108000)):
        aa = S_j.pTrk[:,:,ndx]
        aa = aa[:, :, ~np.all(np.isnan(aa[:, 0, :]), axis=0)]
        ci[ndx] = aa.shape[2]
        bb = S.pTrk[:,:,ndx]
        bb = bb[:, :, ~np.all(np.isnan(bb[:, 0, :]), axis=0)]
        cr[ndx] = bb.shape[2]

        dd = np.linalg.norm(aa[..., None] - bb[:, :, None], axis=1)
        # dd_m = find_dist_match(np.transpose(dd[None], [0, 2, 3, 1]))

        az = np.array([aa.min(axis=0), aa.max(axis=0)]).reshape([4,-1]).T
        bz = np.array([bb.min(axis=0), bb.max(axis=0)]).reshape([4, -1]).T
        orr = tn(bbox_overlaps(torch.tensor(az), torch.tensor(bz)))

        matched1 = np.zeros(orr.shape[0])
        matched2 = np.zeros(orr.shape[1])
        count = 0
        for ix in range(orr.shape[0]):
            jx = np.argmax(orr[ix])
            if count<4:
                oo[ndx, count] = orr[ix, jx]
            if orr[ix].max()>0.1:
                if count==4:
                    ee[ndx] = 1
                    break
                dds[ndx,count,:] = dd[:,ix,jx]
                matched1[ix] = 1
                matched2[jx] = 1
                count = count+1
        fn[ndx] = 4-np.count_nonzero(matched1==1)
        fp[ndx] = np.count_nonzero(matched2==0)

    ddq = dds[:,:4]
    ddqi.append([ddq,fn,fp,ee,oo])


ff(); [plt.plot(np.sort(dd1[0].flatten())) for dd1 in ddqi]
plt.legend([f'{x+1}' for x in range(len(trks_i))])
plt.title('Interactive')
plt.xlim([1.4e6,1.75e6])
plt.ylim([-5,50])
plt.ylabel('Distance (px) between predictions')
plt.xlabel('Predictions Order')
plt.title('Prediction agreement of interactively labeled models')
# plt.savefig('/groups/branson/home/kabram/temp/interactive_matches.svg')
# plt.savefig('/groups/branson/home/kabram/temp/interactive_matches.png')

##

sel = np.zeros([108000])
for ndx in tqdm.tqdm(range(108000)):
    aa = S_j.pTrk[:, :, ndx]
    aa = aa[:, :, ~np.all(np.isnan(aa[:, 0, :]), axis=0)]
    az = np.array([aa.min(axis=0)-20, aa.max(axis=0)+20]).reshape([4, -1]).T

    orr = tn(bbox_overlaps(torch.tensor(az), torch.tensor(az)))
    orr = orr.flatten()
    orr[::(az.shape[0] + 1)] = 0.
    if np.any(orr>0.1):
        sel[ndx] = 1

sel = sel>0.5
sel = np.ones([108000])>0.5
##

tfiles= [f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/loc_{ix}.json' for ix in range(1,8)]
n_train = []
for tt in tfiles:
    J = pt.json_load(tt)
    nn = sum([ll['ntgt'] for ll in J['locdata']])
    n_train.append(nn)

# avg animal size is 75px
tr = 75/2/2/2
ff()

jj = []
for dd1 in ddqi:
    cj = np.count_nonzero(sel)*4-np.count_nonzero(dd1[0][sel].mean(axis=-1)<tr)
    cj = cj+dd1[1][sel].sum() + dd1[2][sel].sum()
    jj.append(cj)
plt.plot(n_train,jj,marker='o')
plt.title('Interactive')

tfiles = [f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/loc_1_{ix}.json' for ix in range(n_rounds-1,-1,-1)]
tfiles.extend([f'{bdir}/random/loc_0.json', f'{bdir}/random/loc_1_roi.json',f'{bdir}/random/loc_2_roi.json'])
n_train = []
for tt in tfiles:
    J = pt.json_load(tt)
    nn = sum([ll['ntgt'] for ll in J['locdata']])
    n_train.append(nn)

jj = []
for dd1 in ddqs:
    cj = np.count_nonzero(sel)*4-np.count_nonzero(dd1[0][sel].mean(axis=-1)<tr)
    cj = cj+dd1[1][sel].sum() + dd1[2][sel].sum()
    jj.append(cj)

plt.plot(n_train,jj,marker='o')
plt.legend(['Interactive','Random'])
plt.xlabel('Number of training examples')
plt.ylabel(f'Number of predictions with error {tr}px')
plt.title('Prediction agreement of interactively and randomly labeled models')
plt.yscale('log')
plt.xscale('log')
# plt.savefig('/groups/branson/home/kabram/temp/incremental_training_vs_random_training.svg')


##

dd1 = ddqi[-1][0]
dd2 = ddqs[-1][0]


pp = np.ones([108000,4,4,2])*np.nan
for ndx in range(108000):
    aa = S_j.pTrk[:, :, ndx]
    aa = aa[:, :, ~np.all(np.isnan(aa[:, 0, :]), axis=0)]
    aa = aa[:,:,:np.clip(aa.shape[2],0,4)]
    pp[ndx,:aa.shape[2]] = np.transpose(aa,[2,0,1])


xa = np.where( (dd1>30)& (dd2<20))
sel_p = pp[xa[0],xa[1]]
sz = np.sqrt(np.prod(sel_p.max(axis=1)-sel_p.min(axis=1),axis=-1))
all_sz = np.sqrt(np.prod(pp.max(axis=2)-pp.min(axis=2),axis=-1)).flatten()

# xa = np.where( (dd2>30)& (dd1<20))
# sel_p = pp[xa[0],xa[1]]
# sz1 = np.sqrt(np.prod(sel_p.max(axis=1)-sel_p.min(axis=1),axis=-1))

ff(); plt.hist([all_sz,sz],density=True)
plt.legend(['All','Interactive worse than random'])
plt.xlabel('Size of bounding box')
plt.ylabel('Density')
plt.title('Size of bounding box for predictions')

##############################################################
## PLOT TRACKING EXAMPLES
gt_movie = '/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice_again/20210924_four_female_mice_again.mjpg'
trk_file = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/trks/20210924_four_female_mice_again_rand_7.trk'
import TrkFile
tt = TrkFile.Trk(trk_file)
import movies
import PoseTools as pt
cap = movies.Movie(gt_movie)

##
cur_info = {
'sel_int': [23300,23360],
'n_fr' : 5,
'x_lim' : [150,450 ],
'y_lim' : [1170,1470],
'skel': [[1,[2,3]],[0,2],[0,3],[3,2]],
}
info = []
info.append(cur_info)
cur_info = {
'sel_int': [24020,24060],
'n_fr' : 5,
'x_lim' : [350,650 ],
'y_lim' : [1510,1810],
'skel': [[1,[2,3]],[0,2],[0,3],[3,2]],
}
info.append(cur_info)

cur_info = { # 3 mice, some incorrect localizations
'sel_int': [59680,59740],
'n_fr' : 5,
'x_lim' : [1600,1900 ],
'y_lim' : [150,450],
'skel': [[1,[2,3]],[0,2],[0,3],[3,2]],
}
info.append(cur_info)

cur_info = { # 3 mice
'sel_int': [2850,3000],
'n_fr' : 5,
'x_lim' : [875,1275],
'y_lim' : [1300,1700 ],
'skel': [[1,[2,3]],[0,2],[0,3],[3,2]],
}
info.append(cur_info)

cur_info = { # 3 mice
'sel_int': [72170,72270],
'n_fr' : 5,
'x_lim' : [300,600],
'y_lim' : [1500,1800 ],
'skel': [[1,[2,3]],[0,2],[0,3],[3,2]],
}
info.append(cur_info)

sel_id = 4
sel_int = info[sel_id]['sel_int']
n_fr = info[sel_id]['n_fr']
x_lim = info[sel_id]['x_lim']
y_lim = info[sel_id]['y_lim']
skel = info[sel_id]['skel']

f,ax = plt.subplots(1,n_fr,figsize=(n_fr*2,2))
frs = np.linspace(sel_int[0],sel_int[1],n_fr).astype(int)
for i,ff in enumerate(frs):
    curf = cap.get_frame(ff)[0]
    curt,extra = tt.getframe(ff,extra=True)
    curt = curt[:,:,0]
    occ = extra['pTrkTag'][:,0]
    vv = np.where(~np.all(np.isnan(curt[:,0]),axis=0))[0]
    curt = curt[:,:,vv]
    occ = occ[:,vv]

    ax[i].imshow(curf,cmap='gray')
    cmap = plt.get_cmap('tab10', curt.shape[0])
    for j in range(curt.shape[0]):
        for x in range(curt.shape[2]):
            if np.all(np.isnan(curt[j,:,x])):
                continue
            if occ[j,x]>0:
                ax[i].scatter(curt[j,0,x],curt[j,1,x],edgecolor=cmap(j),s=15,marker='o',facecolor='none')
            else:
                ax[i].scatter(curt[j,0,x],curt[j,1,x],color=cmap(j),s=15,marker='x')
    for ee in skel:
        if np.all(np.isnan(curt[ee[0],0])):
            continue
        if type(ee[1])==list:
            xx1 = curt[ee[1],0,:].mean(axis=0)
            yy1 = curt[ee[1],1,:].mean(axis=0)
        else:
            xx1 = curt[ee[1],0,:]
            yy1 = curt[ee[1],1,:]
        if type(ee[0])==list:
            xx2 = curt[ee[0],0,:].mean(axis=0)
            yy2 = curt[ee[0],1,:].mean(axis=0)
        else:
            xx2 = curt[ee[0],0,:]
            yy2 = curt[ee[0],1,:]

        ax[i].plot([xx1,xx2],[yy1,yy2],color='red',linewidth=1,alpha=0.25)

    ax[i].axis('off')
    ax[i].axis('image')
    ax[i].set_xlim(x_lim)
    ax[i].set_ylim(y_lim[::-1])
    ax[i].text((x_lim[0]+x_lim[1])/2,y_lim[0]+20,f'Frame {ff}',ha='center',va='center',fontsize=10,color='white')

f.tight_layout()

plt.savefig(f'/groups/branson/home/kabram/temp/tracking_example_{sel_id}.svg')
plt.savefig(f'/groups/branson/home/kabram/temp/tracking_example_{sel_id}.png')
#
import cv2
from reuse import *
out_file = f'/groups/branson/home/kabram/temp/tracking_example_{sel_id}_mov.mp4'
fps = 5
fourcc = cv2.VideoWriter_fourcc(*'X264')
x = x_lim
y = y_lim

f = plt.figure(figsize=[4,4*(y[1]-y[0])/(x[1]-x[0])])
f.set_dpi(400)
ax = f.add_axes([0, 0, 1, 1])
trk_fr,extra = tt.getframe(np.arange(sel_int[0],sel_int[1]+1),extra=True)
occ = extra['pTrkTag']

offset = np.array([x[0], y[0]])
cc = cmap

fr_s = sel_int[0]
fr_e = sel_int[1]+1
for fr in range(fr_s, fr_e):
    ax.clear()
    im = cap.get_frame(fr)[0]
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    ax.imshow(im[y[0]:y[1], x[0]:x[1]])
    ax.axis('off')

    for j in range(trk_fr.shape[0]):
        for xn in range(trk_fr.shape[3]):
            if np.all(np.isnan(trk_fr[j,:,fr-fr_s,xn])):
                continue
            if occ[j,fr-fr_s,xn]>0:
                ax.scatter(trk_fr[j,0,fr-fr_s,xn]-x[0],trk_fr[j,1,fr-fr_s,xn]-y[0],edgecolor=cmap(j),s=55,marker='o',facecolor='none')
            else:
                ax.scatter(trk_fr[j,0,fr-fr_s,xn]-x[0],trk_fr[j,1,fr-fr_s,xn]-y[0],color=cmap(j),s=55,marker='x')


        # ax.scatter(trk_fr[j,0,fr-fr_s,:]-x[0],trk_fr[j,1,fr-fr_s,:]-y[0],color=cmap(j),s=55,marker='+')

    # for ix in range(trk_fr.shape[3]):
    #     dskl(trk_fr[..., fr - fr_s, ix] - offset[None], skel, cc='red',alpha=0.5, ax=ax)

    curt = trk_fr[:, :, fr - fr_s, :] - offset[None,:,None]
    vv = np.where(~np.all(np.isnan(curt[:,0]),axis=0))[0]
    curt = curt[:,:,vv]

    for ee in skel:
        if np.all(np.isnan(curt[ee[0], 0])):
            continue
        if type(ee[1]) == list:
            xx1 = curt[ee[1], 0, :].mean(axis=0)
            yy1 = curt[ee[1], 1, :].mean(axis=0)
        else:
            xx1 = curt[ee[1], 0, :]
            yy1 = curt[ee[1], 1, :]
        if type(ee[0]) == list:
            xx2 = curt[ee[0], 0, :].mean(axis=0)
            yy2 = curt[ee[0], 1, :].mean(axis=0)
        else:
            xx2 = curt[ee[0], 0, :]
            yy2 = curt[ee[0], 1, :]

        ax.plot([xx1, xx2], [yy1, yy2], color='red', linewidth=1, alpha=0.25)

    ax.set_xlim([0, x[1] - x[0]])
    ax.set_ylim([ y[1] - y[0],0])
    f.canvas.draw()
    img = np.frombuffer(f.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(f.canvas.get_width_height()[::-1] + (3,))
    if fr == fr_s:
        fr_sz = img.shape[:2]
        out = cv2.VideoWriter(out_file, fourcc, fps, (fr_sz[1], fr_sz[0]))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out.write(img)

out.release()

##############################################
##############################################
## Using Confidence for labeling

from reuse import *

idir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/confidence/'
import os
os.makedirs(idir,exist_ok=True)
rand_json = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/loc_0.json'
full_rand_json = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/confidence/full_train.json'
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/'

rfiles = ['rand_1_3','rand_1_2','rand_1_1','rand_1_0','rand_0','rand_1','rand_2']

import APT_interface as apt
import time

## classify the db

start_round = 0
estr = '_rep_2'
for round in range(start_round,8):

    # check jobs status
    # time.sleep(100)
    # while pt.get_job_status(f'inc_confidence_grone_round_{round}') in ['PENDING','RUN']:
    #     time.sleep(100)

    tfile = f'{idir}/unmarkedMice/multi_mdn_joint_torch/view_0/round_{round}{estr}/traindata'
    A = pt.pickle_load(tfile)
    conf = A[1]
    res = apt.classify_db_all('multi_mdn_joint_torch',conf,full_rand_json)

    cc = res[0]['conf']
    locs = res[1]
    pp = res[0]['locs']
    dd = np.linalg.norm(pp[:, :,None] - locs[:, None], axis=-1)
    dd_m = find_dist_match(dd)
    n_pred = np.count_nonzero(~np.all(np.isnan(pp),axis=(-1,-2)),axis=-1)
    n_lbl = np.count_nonzero(~np.all(np.isnan(locs),axis=(-1,-2)),axis=-1)
    no_detect = np.where( np.any((np.mean(dd_m,axis=-1)>75/2),axis=1)|(n_pred<n_lbl) )[0]
    vv = ~np.all(np.isnan(pp),axis=(-1,-2))

    # ff(); plt.scatter(cc[vv],dd_m[vv])
    ##
    R = pt.json_load(f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/unmarkedMice/multi_mdn_joint_torch/view_0/{rfiles[round+1]}/train_TF.json')
    C = pt.json_load(f'{idir}/unmarkedMice/multi_mdn_joint_torch/view_0/round_{round}{estr}/train_TF.json')
    n_extra = len(R['annotations'])-len(C['annotations'])
    aa_r = np.array([jj['iscrowd'] for jj in R['annotations']])
    aa_c = np.array([jj['iscrowd'] for jj in C['annotations']])

    n_roi_r = np.count_nonzero(aa_r==1)-np.count_nonzero(aa_r==0)
    n_roi_c = np.count_nonzero(aa_c==1)-np.count_nonzero(aa_c==0)
    n_roi_extra = n_roi_r-n_roi_c

    R_f = pt.json_load(full_rand_json)
    done = np.array([ (xx['movid'],xx['frm']) for xx in C['images']])
    info = np.array(res[2])[:,:2]

    matched_conf = np.ones([len(R_f['images']),4])*np.nan
    for ndx in range(len(R_f['images'])):
        for ix in range(4):
            if np.all(np.isnan(locs[ndx,ix,:,0])):
                continue
            dd = np.linalg.norm(locs[ndx,ix,None]-pp[ndx],axis=-1).mean(axis=-1)
            mid = np.nanargmin(dd)
            matched_conf[ndx,ix] = cc[ndx,mid].mean(axis=-1)

    # ord = np.argsort(np.nanmin(cc.mean(axis=-1),axis=-1))
    ord = np.argsort(np.nanmin(matched_conf,axis=-1))

    im_idx = np.array([xx['image_id'] for xx in R_f['annotations']])
    aa_f = np.array([jj['iscrowd'] for jj in R_f['annotations']])
    bb_f = np.array([jj['bbox'] for jj in R_f['annotations']])
    bb_f_lbl = bb_f[aa_f==0]
    #R_f has bbox repeated for masking for dekr
    for ndx in range(len(aa_f)):
        if aa_f[ndx]==0:
            continue
        if np.any( (bb_f_lbl==bb_f[ndx]).all(axis=-1)):
            aa_f[ndx] = 0

    sel = []
    count = 0
    n_ann = 0
    n_roi = 0

    for dx in range(done.shape[0]):
        sel.append(np.where((info == done[dx]).all(axis=-1))[0][0])

    roi_sel = []
    while n_roi<=n_roi_extra and count < len(ord):
        cur_s = np.random.choice(im_idx[aa_f==1])

        jx = np.where((info[cur_s] == done).all(axis=-1))
        if len(jx[0])==0:
            sel.append(cur_s)
            n_ann += np.count_nonzero(im_idx==R_f['images'][cur_s]['id'])
            n_roi += np.count_nonzero(aa_f[im_idx==R_f['images'][cur_s]['id']]==1)
            roi_sel.append(cur_s)
        count = count+1

    count = 0
    while (n_ann<n_extra) and count < len(no_detect):

        if np.all(np.isnan(locs[no_detect[count]])):
            # if a patch has only ROI
            if no_detect[count] not in sel:
                sel.append(no_detect[count])
                count = count+1
                continue
        if no_detect[count] in sel:
            count = count+1
            continue

        sel.append(no_detect[count])
        n_ann += np.count_nonzero(im_idx==R_f['images'][no_detect[count]]['id'])
        count = count+1

    count = 0
    while (n_ann<n_extra) and count < len(ord):

        if np.all(np.isnan(locs[ord[count]])):
            # if a patch has only ROI
            if ord[count] not in sel:
                sel.append(ord[count])
                count = count+1
                continue
        if ord[count] in sel:
            count = count+1
            continue

        jx = np.where((info[ord[count]] == done).all(axis=-1))
        if len(jx[0])==0:
            sel.append(ord[count])
            n_ann += np.count_nonzero(im_idx==R_f['images'][ord[count]]['id'])
        count = count+1

    C['images'] = []
    C['annotations'] = []
    for curs in sel:
        C['images'].append(R_f['images'][curs])
        idx = R_f['images'][curs]['id']
        asel = np.where(im_idx==idx)[0]
        C['annotations'].extend([R_f['annotations'][xx] for xx in asel])


    new_dir = f'{idir}/unmarkedMice/multi_mdn_joint_torch/view_0/round_{round+1}{estr}'
    os.makedirs(new_dir,exist_ok=True)
    import json
    with open(f'{new_dir}/train_TF.json','w') as fid:
        json.dump(C,fid)

    import ap36_train as ap36
    conf.cachedir = new_dir
    conf.rescale = 1.
    apt.gen_train_samples1(conf,'multi_mdn_joint_torch',out_file=f'{conf.cachedir}/train_samples',nsamples=25)
    ap36.train(conf,'multi_mdn_joint_torch','deepnet')
    # ap36.train_bsub(conf,'multi_mdn_joint_torch','deepnet',name=f'inc_confidence_grone_round_{round+1}')


    # time.sleep(100)
    # while pt.get_job_status(f'inc_confidence_grone_round_{round+1}') in ['PENDING','RUN']:
    #     time.sleep(100)



## train on the random subsets. Not required. They should be already trained earlier!!

from reuse import *
import tarfile
import APT_interface as apt
import tempfile
import glob
import shutil

# lbl_file = '/groups/branson/home/kabram/APT_projects/unmarkedMice_rand_labels.lbl'

# ltar = tarfile.TarFile(lbl_file)
# tempdir = tempfile.mkdtemp()
# ltar.extractall(tempdir)
# # find the json file in the extracted directory
# lbl_json = glob.glob(os.path.join(tempdir, 'unmarkedMice', '*.json'))[0]
# out_lbl = '/groups/branson/home/kabram/temp/unmarkedMice_rand_labels.json'
# shutil.copy(lbl_json,out_lbl)
out_lbl = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/20230823T041646_20230823T041650.json'

# These are subsets of the random labels
for rndx in range(n_rounds):
    train_name = f'rand_1_{rndx}'
    cmd = f"APT_interface.py {out_lbl} -name {train_name} -conf_params mdn_pred_dist True batch_size 4 dl_steps 100000 multi_crop_ims False -type multi_mdn_joint_torch -cache {idir} -json_trn_file {idir}/loc_1_{rndx}.json train -use_cache"
    pt.submit_job(f'unmarked_mice_inc_rand_1_round{rndx}', cmd, f'{idir}/run_info', queue='gpu_a100', sing_image='/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif')

# train 1x,2x and 4x
for rndx in range(3):
    train_name = f'rand_{rndx}'
    if rndx>0:
        estr = '_roi'
    else:
        estr = ''
    cmd = f"APT_interface.py {out_lbl} -name {train_name} -conf_params mdn_pred_dist True batch_size 4 dl_steps 100000 multi_crop_ims False -type multi_mdn_joint_torch -cache {idir} -json_trn_file {idir}/loc_{rndx}{estr}.json train -use_cache"
    pt.submit_job(f'unmarked_mice_inc_rand_round{rndx}', cmd, f'{idir}/run_info', queue='gpu_a100', sing_image='/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif')

## track gt movie using confidence labels

gt_movie = '/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice_again/20210924_four_female_mice_again.mjpg'
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc'
idir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/confidence/'


out_dir = f'{idir}/trks'
import os
os.makedirs(out_dir,exist_ok=True)
os.makedirs(f'{bdir}/run_info',exist_ok=True)
import ap36_train as ap36
from reuse import *
nets = [['multi_mdn_joint_torch',{}],]

for rndx in range(7):
    train_name = f'round_{rndx}'
    for net in nets:
        tfile = f'{idir}/unmarkedMice/{net[0]}/view_0/round_{rndx}/traindata'
        A = pt.pickle_load(tfile)
        conf = A[1]
        if net[0] == 'multi_mdn_joint_torch':
            sing_img = '/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif'
            nname = 'grone'
        else:
            sing_img = '/groups/branson/home/kabram/bransonlab/singularity/mmpose_1x_pycharm.sif'
            nname = 'dekr'
        ap36.track_bsub(conf,net[0],'deepnet',gt_movie,f'{out_dir}/round_{rndx}_{nname}.trk',sing_img=sing_img,name=f'confidence_{nname}_round_{rndx}')


## Compare tracking. .Random results
import TrkFile
from packaging import version
import mmpose
if version.parse(mmpose.__version__).major>0:
    from mmdet.structures.bbox import bbox_overlaps
else:
    from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
import torch
import tqdm
from reuse import *

n_rounds = 4
idir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random'
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc'

trks = [f'{idir}/trks/20210924_four_female_mice_again_rand_1_{rndx}_tracklet.trk' for rndx in range(n_rounds-1,-1,-1)]
trks1 = [f'{idir}/trks/20210924_four_female_mice_again_rand_{rndx}_tracklet.trk' for rndx in range(3)]


trks_j = f'{bdir}/joint/trks/20210924_four_female_mice_again_joint_tracklet.trk'
trks_j_re = f'{bdir}/joint/trks/20210924_four_female_mice_again_joint_re_tracklet.trk'

trks = trks + trks1 + [trks_j_re,]

# S_all = [TrkFile.Trk(trks[pndx]) for pndx in range(len(trks))]
S_j = TrkFile.Trk(trks_j)
# S_4x = TrkFile.Trk(trks_i[-1])
S_j.convert2dense()

ddqs = []
for pndx in range(len(trks)):
    dds = np.ones([108000, 4, 4])*np.nan
    fp = np.zeros([108000])
    fn = np.zeros([108000])
    ee = np.zeros([108000])
    oo = np.zeros([108000,4])
    S = TrkFile.Trk(trks[pndx])
    S.convert2dense()
    ci = np.zeros([108000])
    cr = ci.copy()

    for ndx in tqdm.tqdm(range(108000)):
        aa = S_j.pTrk[:,:,ndx]
        # aa = S_4x.getframe(ndx)[:, :, 0, :]
        aa = aa[:, :, ~np.all(np.isnan(aa[:, 0, :]), axis=0)]
        ci[ndx] = aa.shape[2]
        # bb = S.getframe(ndx)[:, :, 0]
        bb = S.pTrk[:,:,ndx]
        bb = bb[:, :, ~np.all(np.isnan(bb[:, 0, :]), axis=0)]
        cr[ndx] = bb.shape[2]

        dd = np.linalg.norm(aa[..., None] - bb[:, :, None], axis=1)
        # dd_m = find_dist_match(np.transpose(dd[None], [0, 2, 3, 1]))

        az = np.array([aa.min(axis=0), aa.max(axis=0)]).reshape([4,-1]).T
        bz = np.array([bb.min(axis=0), bb.max(axis=0)]).reshape([4, -1]).T
        orr = tn(bbox_overlaps(torch.tensor(az), torch.tensor(bz)))

        matched1 = np.zeros(orr.shape[0])
        matched2 = np.zeros(orr.shape[1])
        count = 0
        for ix in range(orr.shape[0]):
            jx = np.argmax(orr[ix])
            if count<4:
                oo[ndx, count] = orr[ix, jx]
            if orr[ix].max()>0.1:
                if count==4:
                    ee[ndx] = 1
                    break
                jx = np.argmax(orr[ix])
                dds[ndx,count,:] = dd[:,ix,jx]
                matched1[ix] = 1
                matched2[jx] = 1
                oo[ndx,count] = orr[ix,jx]
                count = count+1
        fn[ndx] = 4-np.count_nonzero(matched1==1)
        fp[ndx] = np.count_nonzero(matched2==0)


    ddq = dds[:,:4]
    ddqs.append([ddq,fn,fp,ee,oo])

ddqj_re = ddqs[-1]
ddqs = ddqs[:-1]

ff(); [plt.plot(np.sort(dd1[0].flatten())) for dd1 in ddqs]
plt.legend([f'{x+1}' for x in range(len(trks))])
plt.plot(np.sort(ddqj_re[0].flatten()),'--')
plt.title('Random')
plt.xlim([1.4e6,1.75e6])
plt.ylim([-5,50])
plt.ylabel('Distance (px) between predictions')
plt.xlabel('Predictions Order')
plt.title('Prediction agreement of randomly labeled models')
# plt.savefig('/groups/branson/home/kabram/temp/random_matches_new.svg')
# plt.savefig('/groups/branson/home/kabram/temp/random_matches_new.png')


## Distance of confidence labels to final interactive

import TrkFile
import mmpose
from packaging import version
if version.parse(mmpose.__version__).major>0:
    from mmdet.structures.bbox import bbox_overlaps
else:
    from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
import torch
import tqdm

out_dir = f'{bdir}/confidence/trks'
trks_i = [f'{out_dir}/round_{ndx}_grone_tracklet.trk' for ndx in range(0,7)]

trks_j = f'{bdir}/joint/trks/20210924_four_female_mice_again_joint_tracklet.trk'
# S_all = [TrkFile.Trk(trks[pndx]) for pndx in range(len(trks))]
S_j = TrkFile.Trk(trks_j)
# S_4x = TrkFile.Trk(trks_i[-1])
S_j.convert2dense()

ddqc = []
for pndx in range(len(trks_i)):
    dds = np.ones([108000, 4, 4])*np.nan
    fp = np.zeros([108000])
    fn = np.zeros([108000])
    ee = np.zeros([108000])
    oo = np.ones([108000,4])*np.nan
    S = TrkFile.Trk(trks_i[pndx])
    S.convert2dense()
    ci = np.zeros([108000])
    cr = ci.copy()

    for ndx in tqdm.tqdm(range(108000)):
        aa = S_j.pTrk[:,:,ndx]
        aa = aa[:, :, ~np.all(np.isnan(aa[:, 0, :]), axis=0)]
        ci[ndx] = aa.shape[2]
        bb = S.pTrk[:,:,ndx]
        bb = bb[:, :, ~np.all(np.isnan(bb[:, 0, :]), axis=0)]
        cr[ndx] = bb.shape[2]

        dd = np.linalg.norm(aa[..., None] - bb[:, :, None], axis=1)
        # dd_m = find_dist_match(np.transpose(dd[None], [0, 2, 3, 1]))

        az = np.array([aa.min(axis=0), aa.max(axis=0)]).reshape([4,-1]).T
        bz = np.array([bb.min(axis=0), bb.max(axis=0)]).reshape([4, -1]).T
        orr = tn(bbox_overlaps(torch.tensor(az), torch.tensor(bz)))

        matched1 = np.zeros(orr.shape[0])
        matched2 = np.zeros(orr.shape[1])
        count = 0
        for ix in range(orr.shape[0]):
            jx = np.argmax(orr[ix])
            if count<4:
                oo[ndx, count] = orr[ix, jx]
            if orr[ix].max()>0.1:
                if count==4:
                    ee[ndx] = 1
                    break
                dds[ndx,count,:] = dd[:,ix,jx]
                matched1[ix] = 1
                matched2[jx] = 1
                count = count+1
        fn[ndx] = 4-np.count_nonzero(matched1==1)
        fp[ndx] = np.count_nonzero(matched2==0)

    ddq = dds[:,:4]
    ddqc.append([ddq,fn,fp,ee,oo])


ff(); [plt.plot(np.sort(dd1[0].flatten())) for dd1 in ddqc]
plt.legend([f'{x+1}' for x in range(len(trks_i))])
plt.title('Tracker difference')
plt.xlim([1.4e6,1.75e6])
plt.ylim([-5,50])
plt.ylabel('Distance (px) between predictions')
plt.xlabel('Predictions Order')
plt.title('Prediction agreement of confidence labeled models')
plt.savefig('/groups/branson/home/kabram/temp/confidence_matches.svg')
plt.savefig('/groups/branson/home/kabram/temp/confidence_matches.png')


##

sel = np.ones([108000])>0.5

# avg animal size is 75px
tr = 75/4
ff()

n_rounds = 4
tfiles = [f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/loc_1_{ix}.json' for ix in range(n_rounds-1,-1,-1)]
tfiles.extend([f'{bdir}/random/loc_0.json', f'{bdir}/random/loc_1_roi.json',f'{bdir}/random/loc_2_roi.json'])
n_train = []
for tt in tfiles:
    J = pt.json_load(tt)
    nn = sum([ll['ntgt'] for ll in J['locdata']])
    n_train.append(nn)

jj = []
for dd1 in ddqs:
    cj = np.count_nonzero(sel)*16-np.count_nonzero(dd1[0][sel]<tr)
    cj = cj+dd1[1][sel].sum() + dd1[2][sel].sum()
    jj.append(cj)

plt.plot(n_train,jj,marker='o')

jj = []
for dd1 in ddqc:
    cj = np.count_nonzero(sel)*16-np.count_nonzero(dd1[0][sel]<tr)
    cj = cj+dd1[1][sel].sum() + dd1[2][sel].sum()
    jj.append(cj)
plt.plot(n_train[:len(jj)],jj,marker='o')
plt.title('Confidence')

plt.legend(['Random','Confidence'])
plt.xlabel('Number of training examples')
plt.ylabel(f'Number of predictions with error {tr}px')
plt.title('Prediction agreement of Confidence and randomly labeled models')
plt.yscale('log')
plt.xscale('log')
# plt.savefig('/groups/branson/home/kabram/temp/incremental_training_vs_random_training.svg')





############################################## TRACKER DIFFERENCE
##############################################

## train an initial dekr tracker
out_lbl = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/20230823T041646_20230823T041650.json'
from reuse import *

idir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/'
rand_json = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/loc_0.json'
n_rounds = 4
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/'

rndx = 3
train_name = f'rand_1_{rndx}'

tfile = f'{idir}/unmarkedMice/multi_mdn_joint_torch/view_0/rand_1_3/traindata'
A = pt.pickle_load(tfile)
conf = A[1]
conf.mmpose_net = 'dekr'
conf.cachedir = f'{idir}/unmarkedMice/multi_mmpose/view_0/rand_1_3'

# copy files from multi_mdn_joint_torch/view_0/rand_1_3/train_TF.json to multi_mmpose/view_0/rand_1_3/
import ap36_train as ap36
import time

# ap36.train(conf,'multi_mmpose','deepnet')
ap36.train_bsub(conf,'multi_mmpose','deepnet',name='inc_dekr_rand_1_3',sing_img='/groups/branson/bransonlab/mayank/singularity/mmpose_1x_pycharm.sif')

# cmd = f"APT_interface.py {out_lbl} -name {train_name} -conf_params mdn_pred_dist True batch_size 4 dl_steps 100000 multi_crop_ims False mmpose_net dekr -type multi_mmpose -cache {idir} -json_trn_file {idir}/loc_1_{rndx}.json train -use_cache"
# pt.submit_job(f'unmarked_mice_inc_rand_1_round{rndx}_dekr', cmd, f'{idir}/run_info', queue='gpu_a100',
#               sing_image='/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif')

## Using two trackers for labeling

from reuse import *

idir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/tracker_diff/'
import os
os.makedirs(idir,exist_ok=True)
rand_json = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/loc_0.json'
full_rand_json = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/confidence/full_train.json'
n_rounds = 4
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/'

rfiles = ['rand_1_3','rand_1_2','rand_1_1','rand_1_0','rand_0','rand_1','rand_2']

import time
import APT_interface as apt

nets = [['multi_mdn_joint_torch',{}],['multi_mmpose',{'mmpose_net':'dekr'}]]
## classify the db

start_round = 0
estr = '_rep_2'
for round in range(start_round,8):
    # time.sleep(100)
    # while (pt.get_job_status(f'inc_diff_dekr_round_{round}{estr}') in ['PENDING','RUN']) or  (pt.get_job_status(f'inc_diff_grone_round_{round}{estr}') in ['PENDING','RUN']):
    #     time.sleep(100)

    a_res = []
    for net in nets:
        tfile = f'{idir}/unmarkedMice/{net[0]}/view_0/round_{round}{estr}/traindata'
        A = pt.pickle_load(tfile)
        conf = A[1]
        res = apt.classify_db_all(net[0],conf,full_rand_json)
        a_res.append(res)

    pp1 = a_res[0][0]['locs']
    pp2 = a_res[1][0]['locs']
    locs = a_res[0][1]


    # for each label in locs, find the closest prediction in pp1 and pp2
    distances_pp1 = np.linalg.norm(pp1[:,:,None] - locs[:,None], axis=-1).mean(axis=-1)
    distances_pp2 = np.linalg.norm(pp2[:,:,None] - locs[:,None], axis=-1).mean(axis=-1)

    closest_pp1 = np.ones(locs.shape)*np.nan
    closest_pp2 = np.ones(locs.shape)*np.nan
    for a1 in range(closest_pp1.shape[0]):
        for a2 in range(closest_pp1.shape[1]):
            if np.all(np.isnan(locs[a1,a2])):
                continue
            if np.all(np.isnan(distances_pp1[a1,:,a2])):
                closest_pp1[a1,a2] = 0.
            else:
                closest_pp1[a1,a2] = pp1[a1,np.nanargmin(distances_pp1[a1,:,a2])]
            if np.all(np.isnan(distances_pp2[a1,:,a2])):
                closest_pp2[a1,a2] = 512.
            else:
                closest_pp2[a1,a2] = pp2[a1,np.nanargmin(distances_pp2[a1,:,a2])]
    # find the distance between the two closest predictions
    distances_closest = np.linalg.norm(closest_pp1-closest_pp2,axis=-1).mean(axis=-1)


    # ff(); plt.scatter(cc[vv],dd_m[vv])
    #
    if round == 6:
        C = pt.json_load(
            f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/unmarkedMice/multi_mdn_joint_torch/view_0/{rfiles[-1]}{estr}/train_TF.json')
    else:

        R = pt.json_load(f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/unmarkedMice/multi_mdn_joint_torch/view_0/{rfiles[round+1]}/train_TF.json')
        C = pt.json_load(f'{idir}/unmarkedMice/multi_mdn_joint_torch/view_0/round_{round}{estr}/train_TF.json')
        n_extra = len(R['annotations'])-len(C['annotations'])
        aa_r = np.array([jj['iscrowd'] for jj in R['annotations']])
        aa_c = np.array([jj['iscrowd'] for jj in C['annotations']])

        n_roi_r = np.count_nonzero(aa_r==1)-np.count_nonzero(aa_r==0)
        n_roi_c = np.count_nonzero(aa_c==1)-np.count_nonzero(aa_c==0)
        n_roi_extra = n_roi_r-n_roi_c

        R_f = pt.json_load(full_rand_json)
        done = np.array([ (xx['movid'],xx['frm']) for xx in C['images']])
        info = np.array(res[2])[:,:2]
        ord = np.argsort(np.nanmax(distances_closest,axis=-1))[::-1]

        im_idx = np.array([xx['image_id'] for xx in R_f['annotations']])
        aa_f = np.array([jj['iscrowd'] for jj in R_f['annotations']])
        bb_f = np.array([jj['bbox'] for jj in R_f['annotations']])
        bb_f_lbl = bb_f[aa_f==0]
        #R_f has bbox repeated for masking for dekr
        for ndx in range(len(aa_f)):
            if aa_f[ndx]==0:
                continue
            if np.any( (bb_f_lbl==bb_f[ndx]).all(axis=-1)):
                aa_f[ndx] = 0

        sel = []
        count = 0
        n_ann = 0
        n_roi = 0

        for dx in range(done.shape[0]):
          sel.append(np.where((info == done[dx]).all(axis=-1))[0][0])

        roi_sel = []
        while n_roi<=n_roi_extra and count < len(ord):
            cur_s = np.random.choice(im_idx[aa_f==1])

            jx = np.where((info[cur_s] == done).all(axis=-1))
            if len(jx[0])==0:
                sel.append(cur_s)
                n_ann += np.count_nonzero(im_idx==R_f['images'][cur_s]['id'])
                n_roi += np.count_nonzero(aa_f[im_idx==R_f['images'][cur_s]['id']]==1)
                roi_sel.append(cur_s)
            count = count+1

        count = 0
        while (n_ann<n_extra) and count < len(ord):
            if np.all(np.isnan(locs[ord[count]])):
                jx = np.where((info[ord[count]] == done).all(axis=-1))
                if len(jx[0])==0:
                    sel.append(ord[count])
                    count = count+1
                    continue
            if ord[count] in roi_sel:
                count = count+1
                continue

            jx = np.where((info[ord[count]] == done).all(axis=-1))
            if len(jx[0])==0:
                sel.append(ord[count])
                n_ann += np.count_nonzero(im_idx==R_f['images'][ord[count]]['id'])
            count = count+1

        C['images'] = []
        C['annotations'] = []
        for curs in sel:
            C['images'].append(R_f['images'][curs])
            idx = R_f['images'][curs]['id']
            asel = np.where(im_idx==idx)[0]
            C['annotations'].extend([R_f['annotations'][xx] for xx in asel])


    new_dir = f'{idir}/unmarkedMice/multi_mdn_joint_torch/view_0/round_{round+1}{estr}'
    os.makedirs(new_dir,exist_ok=True)
    import json
    with open(f'{new_dir}/train_TF.json','w') as fid:
        json.dump(C,fid)

    new_dir_dekr = f'{idir}/unmarkedMice/multi_mmpose/view_0/round_{round+1}{estr}'
    os.makedirs(new_dir_dekr,exist_ok=True)
    # remove existing soft link
    if not os.path.exists(f'{new_dir_dekr}/train_TF.json'):
        # create a soft link to the json file
        os.symlink(f'{new_dir}/train_TF.json',f'{new_dir_dekr}/train_TF.json')

    import ap36_train as ap36
    conf.cachedir = new_dir
    conf.rescale = 1
    conf.nan_as_occluded = False
    ap36.train_bsub(conf,'multi_mdn_joint_torch','deepnet',name=f'inc_diff_grone_round_{round+1}{estr}')

    conf.cachedir = new_dir_dekr
    conf.mmpose_net = 'dekr'
    ap36.train_bsub(conf,'multi_mmpose','deepnet',name=f'inc_diff_dekr_round_{round+1}{estr}',sing_img='/groups/branson/bransonlab/mayank/singularity/mmpose_1x_pycharm.sif')

    time.sleep(100)
    while (pt.get_job_status(f'inc_diff_dekr_round_{round+1}{estr}') in ['PENDING','RUN']) or  (pt.get_job_status(f'inc_diff_grone_round_{round+1}{estr}') in ['PENDING','RUN']):
        time.sleep(100)


## track gt movie using difference labels

gt_movie = '/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice_again/20210924_four_female_mice_again.mjpg'
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc'
idir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/tracker_diff/'


out_dir = f'{idir}/trks'
import os
os.makedirs(out_dir,exist_ok=True)
os.makedirs(f'{bdir}/run_info',exist_ok=True)
import ap36_train as ap36
from reuse import *
nets = [['multi_mdn_joint_torch',{}],['multi_mmpose',{'mmpose_net':'dekr'}]]

for rndx in range(7):
    train_name = f'round_{rndx}'
    for net in nets:
        tfile = f'{idir}/unmarkedMice/{net[0]}/view_0/round_{rndx}/traindata'
        A = pt.pickle_load(tfile)
        conf = A[1]
        if net[0] == 'multi_mdn_joint_torch':
            sing_img = '/groups/branson/home/kabram/bransonlab/singularity/ampere_pycharm_vscode.sif'
            nname = 'grone'
        else:
            sing_img = '/groups/branson/home/kabram/bransonlab/singularity/mmpose_1x_pycharm.sif'
            nname = 'dekr'
        ap36.track_bsub(conf,net[0],'deepnet',gt_movie,f'{out_dir}/round_{rndx}_{nname}.trk',sing_img=sing_img,name=f'inc_diff_{nname}_round_{rndx}')


## Compare tracking
import TrkFile
from packaging import version
import mmpose
if version.parse(mmpose.__version__).major>0:
    from mmdet.structures.bbox import bbox_overlaps
else:
    from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
import torch
import tqdm
from reuse import *

n_rounds = 4
idir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random'
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc'

trks = [f'{idir}/trks/20210924_four_female_mice_again_rand_1_{rndx}_tracklet.trk' for rndx in range(n_rounds-1,-1,-1)]
trks1 = [f'{idir}/trks/20210924_four_female_mice_again_rand_{rndx}_tracklet.trk' for rndx in range(3)]

trks = trks + trks1

trks_j = f'{bdir}/joint/trks/20210924_four_female_mice_again_joint_tracklet.trk'
# S_all = [TrkFile.Trk(trks[pndx]) for pndx in range(len(trks))]
S_j = TrkFile.Trk(trks_j)
# S_4x = TrkFile.Trk(trks_i[-1])
S_j.convert2dense()

ddqs = []
for pndx in range(len(trks)):
    dds = np.ones([108000, 4, 4])*np.nan
    fp = np.zeros([108000])
    fn = np.zeros([108000])
    ee = np.zeros([108000])
    oo = np.zeros([108000,4])
    S = TrkFile.Trk(trks[pndx])
    S.convert2dense()
    ci = np.zeros([108000])
    cr = ci.copy()

    for ndx in tqdm.tqdm(range(108000)):
        aa = S_j.pTrk[:,:,ndx]
        # aa = S_4x.getframe(ndx)[:, :, 0, :]
        aa = aa[:, :, ~np.all(np.isnan(aa[:, 0, :]), axis=0)]
        ci[ndx] = aa.shape[2]
        # bb = S.getframe(ndx)[:, :, 0]
        bb = S.pTrk[:,:,ndx]
        bb = bb[:, :, ~np.all(np.isnan(bb[:, 0, :]), axis=0)]
        cr[ndx] = bb.shape[2]

        dd = np.linalg.norm(aa[..., None] - bb[:, :, None], axis=1)
        # dd_m = find_dist_match(np.transpose(dd[None], [0, 2, 3, 1]))

        az = np.array([aa.min(axis=0), aa.max(axis=0)]).reshape([4,-1]).T
        bz = np.array([bb.min(axis=0), bb.max(axis=0)]).reshape([4, -1]).T
        orr = tn(bbox_overlaps(torch.tensor(az), torch.tensor(bz)))

        matched1 = np.zeros(orr.shape[0])
        matched2 = np.zeros(orr.shape[1])
        count = 0
        for ix in range(orr.shape[0]):
            jx = np.argmax(orr[ix])
            if count<4:
                oo[ndx, count] = orr[ix, jx]
            if orr[ix].max()>0.1:
                if count==4:
                    ee[ndx] = 1
                    break
                jx = np.argmax(orr[ix])
                dds[ndx,count,:] = dd[:,ix,jx]
                matched1[ix] = 1
                matched2[jx] = 1
                oo[ndx,count] = orr[ix,jx]
                count = count+1
        fn[ndx] = 4-np.count_nonzero(matched1==1)
        fp[ndx] = np.count_nonzero(matched2==0)


    ddq = dds[:,:4]
    ddqs.append([ddq,fn,fp,ee,oo])

ff(); [plt.plot(np.sort(dd1[0].flatten())) for dd1 in ddqs]
plt.legend([f'{x+1}' for x in range(len(trks))])
plt.title('Random')
plt.xlim([1.4e6,1.75e6])
plt.ylim([-5,50])
plt.ylabel('Distance (px) between predictions')
plt.xlabel('Predictions Order')
plt.title('Prediction agreement of randomly labeled models')
# plt.savefig('/groups/branson/home/kabram/temp/random_matches_new.svg')
# plt.savefig('/groups/branson/home/kabram/temp/random_matches_new.png')


## Distance of diff labels to final interactive

import TrkFile
import mmpose
if version.parse(mmpose.__version__).major>0:
    from mmdet.structures.bbox import bbox_overlaps
else:
    from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
import torch
import tqdm

out_dir = f'{bdir}/tracker_diff/trks'
trks_i = [f'{out_dir}/round_{ndx}_grone_tracklet.trk' for ndx in range(0,7)]

trks_j = f'{bdir}/joint/trks/20210924_four_female_mice_again_joint_tracklet.trk'
# S_all = [TrkFile.Trk(trks[pndx]) for pndx in range(len(trks))]
S_j = TrkFile.Trk(trks_j)
# S_4x = TrkFile.Trk(trks_i[-1])
S_j.convert2dense()

ddqd = []
for pndx in range(len(trks_i)):
    dds = np.ones([108000, 4, 4])*np.nan
    fp = np.zeros([108000])
    fn = np.zeros([108000])
    ee = np.zeros([108000])
    oo = np.ones([108000,4])*np.nan
    S = TrkFile.Trk(trks_i[pndx])
    S.convert2dense()
    ci = np.zeros([108000])
    cr = ci.copy()

    for ndx in tqdm.tqdm(range(108000)):
        aa = S_j.pTrk[:,:,ndx]
        aa = aa[:, :, ~np.all(np.isnan(aa[:, 0, :]), axis=0)]
        ci[ndx] = aa.shape[2]
        bb = S.pTrk[:,:,ndx]
        bb = bb[:, :, ~np.all(np.isnan(bb[:, 0, :]), axis=0)]
        cr[ndx] = bb.shape[2]

        dd = np.linalg.norm(aa[..., None] - bb[:, :, None], axis=1)
        # dd_m = find_dist_match(np.transpose(dd[None], [0, 2, 3, 1]))

        az = np.array([aa.min(axis=0), aa.max(axis=0)]).reshape([4,-1]).T
        bz = np.array([bb.min(axis=0), bb.max(axis=0)]).reshape([4, -1]).T
        orr = tn(bbox_overlaps(torch.tensor(az), torch.tensor(bz)))

        matched1 = np.zeros(orr.shape[0])
        matched2 = np.zeros(orr.shape[1])
        count = 0
        for ix in range(orr.shape[0]):
            jx = np.argmax(orr[ix])
            if count<4:
                oo[ndx, count] = orr[ix, jx]
            if orr[ix].max()>0.1:
                if count==4:
                    ee[ndx] = 1
                    break
                dds[ndx,count,:] = dd[:,ix,jx]
                matched1[ix] = 1
                matched2[jx] = 1
                count = count+1
        fn[ndx] = 4-np.count_nonzero(matched1==1)
        fp[ndx] = np.count_nonzero(matched2==0)

    ddq = dds[:,:4]
    ddqd.append([ddq,fn,fp,ee,oo])


ff(); [plt.plot(np.sort(dd1[0].flatten())) for dd1 in ddqd]
plt.legend([f'{x+1}' for x in range(len(trks_i))])
plt.title('Tracker difference')
plt.xlim([1.4e6,1.75e6])
plt.ylim([-5,50])
plt.ylabel('Distance (px) between predictions')
plt.xlabel('Predictions Order')
plt.title('Prediction agreement of difference labeled models')
# plt.savefig('/groups/branson/home/kabram/temp/tracker_diff_matches.svg')
# plt.savefig('/groups/branson/home/kabram/temp/tracker_diff_matches.png')


##

sel = np.ones([108000])>0.5

# avg animal size is 75px
tr = 75/4
ff()


tfiles = [f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/loc_1_{ix}.json' for ix in range(n_rounds-1,-1,-1)]
tfiles.extend([f'{bdir}/random/loc_0.json', f'{bdir}/random/loc_1_roi.json',f'{bdir}/random/loc_2_roi.json'])
n_train = []
for tt in tfiles:
    J = pt.json_load(tt)
    nn = sum([ll['ntgt'] for ll in J['locdata']])
    n_train.append(nn)

jj = []
for dd1 in ddqs:
    cj = np.count_nonzero(sel)*4*4-np.count_nonzero(dd1[0][sel]<tr)
    cj = cj+dd1[1][sel].sum() + dd1[2][sel].sum()
    jj.append(cj)

plt.plot(n_train,jj,marker='o')

jj = []
for dd1 in ddqd:
    cj = np.count_nonzero(sel)*16-np.count_nonzero(dd1[0][sel]<tr)
    cj = cj+dd1[1][sel].sum() + dd1[2][sel].sum()
    jj.append(cj)
plt.plot(n_train[:len(jj)],jj,marker='o')
plt.title('Tracker difference')

plt.legend(['Random','Difference'])
plt.xlabel('Number of training examples')
plt.ylabel(f'Number of predictions with error {tr}px')
plt.title('Prediction agreement of interactively and randomly labeled models')
plt.yscale('log')
plt.xscale('log')
# plt.savefig('/groups/branson/home/kabram/temp/incremental_training_vs_random_training.svg')



#################################################
#################################################

## PLot ALL -- interactive, confidence and difference


sel = np.ones([108000])>0.5

# avg animal size is 75px
tr = 75/4
ff()


tfiles = [f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/random/loc_1_{ix}.json' for ix in range(n_rounds-1,-1,-1)]
tfiles.extend([f'{bdir}/random/loc_0.json', f'{bdir}/random/loc_1_roi.json',f'{bdir}/random/loc_2_roi.json'])
n_train = []
for tt in tfiles:
    J = pt.json_load(tt)
    nn = sum([ll['ntgt'] for ll in J['locdata']])
    n_train.append(nn)

jj = []
for dd1 in ddqs:
    cj = np.count_nonzero(sel)*4*4-np.count_nonzero(dd1[0][sel]<tr)
    cj = cj+dd1[1][sel].sum() + dd1[2][sel].sum()
    jj.append(cj)

plt.plot(n_train,jj,marker='o')

jj = []
for dd1 in ddqc:
    cj = np.count_nonzero(sel)*16-np.count_nonzero(dd1[0][sel]<tr)
    cj = cj+dd1[1][sel].sum() + dd1[2][sel].sum()
    jj.append(cj)
plt.plot(n_train[:len(jj)],jj,marker='o')

jj = []
for dd1 in ddqd:
    cj = np.count_nonzero(sel)*16-np.count_nonzero(dd1[0][sel]<tr)
    cj = cj+dd1[1][sel].sum() + dd1[2][sel].sum()
    jj.append(cj)
plt.plot(n_train[:len(jj)],jj,marker='o')
plt.title('Labeling Strategies')

tfiles= [f'/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/loc_{ix}.json' for ix in range(1,8)]
n_train_i = []
for tt in tfiles:
    J = pt.json_load(tt)
    nn = sum([ll['ntgt'] for ll in J['locdata']])
    n_train_i.append(nn)

jj = []
for dd1 in ddqi:
    cj = np.count_nonzero(sel)*16-np.count_nonzero(dd1[0][sel]<tr)
    cj = cj+dd1[1][sel].sum() + dd1[2][sel].sum()
    jj.append(cj)
plt.plot(n_train_i[:len(jj)],jj,marker='o')

plt.legend(['Random','Confidence','Difference','Interactive'])
plt.xlabel('Number of training examples')
plt.ylabel(f'Number of predictions with error {tr}px')
plt.title('Prediction agreement of interactively and randomly labeled models')
plt.yscale('log')
plt.xscale('log')



#################################################
#################################################

## tracking examples
mov_file = '/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice/20210924_four_female_mice_0.mjpg'
trk_file = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/trks/20210924_four_female_mice_0_unmarkedMice_round7_trained.trk'
out_file = '/groups/branson/home/kabram/temp/mice_close_tracking.avi'
skel = [[0,1],[0,2],[0,3]]
x = [100,700]
y = [1250,1850]
fig_size = [10,10]
from reuse import *
cmap = 'hsv'
st = 82600
en = 82800
out_file = '/groups/branson/home/kabram/temp/mice_close_tracking.avi'
pt.make_vid(mov_file,trk_file,out_file,skel,st,en,x,y,cmap=cmap,fig_size=fig_size)
st = 105800
en = 106000
out_file = '/groups/branson/home/kabram/temp/mice_close_tracking_1.avi'
pt.make_vid(mov_file,trk_file,out_file,skel,st,en,x,y,cmap=cmap,fig_size=fig_size)
en = 89670
st = 89500
out_file = '/groups/branson/home/kabram/temp/mice_close_tracking_2.avi'
pt.make_vid(mov_file,trk_file,out_file,skel,st,en,x,y,cmap=cmap,fig_size=fig_size)
st = 71965
en = 72450
out_file = '/groups/branson/home/kabram/temp/mice_close_tracking_3.avi'
x = [1400,2000]
y = [200,800]
pt.make_vid(mov_file,trk_file,out_file,skel,st,en,x,y,cmap=cmap,fig_size=fig_size)

# using random labels
trk_file = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/trks/20210924_four_female_mice_0_unmarkedMice_rand_labels_4x_roi.trk'
out_file = '/groups/branson/home/kabram/temp/mice_close_tracking_random.avi'
skel = [[0,1],[0,2],[0,3]]
x = [100,700]
y = [1250,1850]
fig_size = [10,10]
from reuse import *
cmap = 'hsv'
st = 82600
en = 82800
out_file = '/groups/branson/home/kabram/temp/mice_close_tracking_random.avi'
pt.make_vid(mov_file,trk_file,out_file,skel,st,en,x,y,cmap=cmap,fig_size=fig_size)
st = 105800
en = 106000
out_file = '/groups/branson/home/kabram/temp/mice_close_tracking_1_random.avi'
pt.make_vid(mov_file,trk_file,out_file,skel,st,en,x,y,cmap=cmap,fig_size=fig_size)
en = 89670
st = 89500
out_file = '/groups/branson/home/kabram/temp/mice_close_tracking_2_random.avi'
pt.make_vid(mov_file,trk_file,out_file,skel,st,en,x,y,cmap=cmap,fig_size=fig_size)
st = 71965
en = 72450
out_file = '/groups/branson/home/kabram/temp/mice_close_tracking_3_random.avi'
x = [1400,2000]
y = [200,800]
pt.make_vid(mov_file,trk_file,out_file,skel,st,en,x,y,cmap=cmap,fig_size=fig_size)


############################################################
#############################################################
## HEAVY TAIL OF CLOSE INTERACTIONS

import movies
gt_movie = '/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice_again/20210924_four_female_mice_again.mjpg'
bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/joint'
out_lbl = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc/interactive/20241231T012358_20241231T012358.json'

trk_file = f'{bdir}/trks/20210924_four_female_mice_again_joint.trk'

import TrkFile
tt = TrkFile.Trk(trk_file)

cap = movies.Movie(gt_movie)
tr = 300 # roughly 2x body len

close_pts = []
mice_pts = []
frs = []
for ndx in range(108000):
    curp = tt.getframe(ndx)[:,:,0]
    vv = ~np.all(np.isnan(curp[:,0]),axis=0)
    curp = curp[:,:,vv]

    curm =  curp[[0,1]].mean(axis=0)
    d = np.linalg.norm(curm[..., None] - curp[0,:,  None], axis=0) # d[i,j] has distance from the center of mouse i to the nose of mouse j
    d_all = np.linalg.norm(curp[...,None] - curp[:,:,None], axis=1)
    for ix in range(d.shape[0]):
        for jx in range(d.shape[1]):
            if ix==jx:
                continue
            if d[ix,jx]>d[jx,ix]:
                continue
            if d[ix,jx]<tr:
                ht1 = curp[[0,1],:,ix]
                ht2 = curp[[0,1],:,jx]

                # rotate and center ht2 into ht1 coordinates
                ht1 = ht1 - curm[:,ix]
                ht2 = ht2 - curm[:,ix]
                theta = -np.arctan2(ht1[0,1], ht1[0,0])

                # rotate ht1 by theta
                ht2_rotated = np.zeros(ht2.shape)
                ht2_rotated[:,0] = ht2[:,0]*np.cos(theta) - ht2[:,1]*np.sin(theta)
                ht2_rotated[:,1] = ht2[:,0]*np.sin(theta) + ht2[:,1]*np.cos(theta)
                close_pts.append(ht2_rotated)
                frs.append(ndx)

                ht1_rotated = np.zeros(ht1.shape)
                ht1_rotated[:,0] = ht1[:,0]*np.cos(theta) - ht1[:,1]*np.sin(theta)
                ht1_rotated[:,1] = ht1[:,0]*np.sin(theta) + ht1[:,1]*np.cos(theta)
                mice_pts.append(ht1_rotated)

close_pts = np.array(close_pts)
frs = np.array(frs)
mice_pts = np.array(mice_pts)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_polar_histogram(x, y, n_bins=36, figsize=(10, 10), color='skyblue',
                         edgecolor='darkblue', alpha=0.7, normalize=False):
    """
    Create a polar histogram from x,y coordinates.

    Parameters:
    -----------
    x : array-like
        X-coordinates of points
    y : array-like
        Y-coordinates of points (same length as x)
    n_bins : int, default=36
        Number of angular bins (e.g., 36 for 10-degree bins)
    figsize : tuple, default=(10, 10)
        Figure size
    color : str, default='skyblue'
        Color of the histogram bars
    edgecolor : str, default='darkblue'
        Color of the histogram bar edges
    alpha : float, default=0.7
        Transparency of the histogram bars
    normalize : bool, default=False
        Whether to normalize the histogram (sum to 1)

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Convert Cartesian (x,y) to polar coordinates (theta, r)
    theta = np.arctan2(y, x)

    # Create figure and axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')

    # Create the histogram
    counts, bin_edges = np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi))

    if normalize:
        counts = counts / np.sum(counts)

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot the histogram
    bars = ax.bar(bin_centers, counts, width=(2 * np.pi) / n_bins,
                  bottom=0.0, color=color, edgecolor=edgecolor, alpha=alpha)

    # Configure the plot
    ax.set_theta_zero_location("N")  # Set 0 degrees at the top
    ax.set_theta_direction(-1)  # Go clockwise

    # Add labels and title
    ax.set_title('Polar Histogram of Points', y=1.08)
    ax.set_xlabel('Angle (radians)')
    ax.set_ylabel('Count')

    plt.tight_layout()
    return fig, ax

##
def plot_polar_histogram_with_distance(x, y, n_angular_bins=36, n_radial_bins=5,
                                       figsize=(12, 10), cmap='viridis',scale='normal',ax=None, x_max =None):
    """
    Create a 2D polar histogram showing both angle and distance distribution.

    Parameters:
    -----------
    x : array-like
        X-coordinates of points
    y : array-like
        Y-coordinates of points (same length as x)
    n_angular_bins : int, default=36
        Number of angular bins
    n_radial_bins : int, default=5
        Number of radial (distance) bins
    figsize : tuple, default=(12, 10)
        Figure size
    cmap : str, default='viridis'
        Colormap for the 2D histogram

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Convert Cartesian (x,y) to polar coordinates (theta, r)
    theta = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)

    if ax is None:
        # Create figure and axis
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
    else:
        fig = ax.figure

    # Set the maximum radius if not provided
    if x_max is None:
        x_max = np.max(r)
    # Create the 2D histogram
    hist, theta_edges, r_edges = np.histogram2d(
        theta, r,
        bins=[n_angular_bins, n_radial_bins],
        range=[[-np.pi, np.pi], [0, x_max]]
    )

    if scale == 'log':
        # hist = np.log(hist + 1)
        hist = hist + 1
    # Convert to mesh for plotting
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2

    # Create meshgrid for pcolormesh
    THETA, R = np.meshgrid(theta_edges, r_edges)

    # Plot as a pcolormesh
    if scale == 'log':
        cax = ax.pcolormesh(THETA, R, hist.T, cmap=cmap,norm=LogNorm(vmin=1,vmax=np.max(hist)))
    else:
        cax = ax.pcolormesh(THETA, R, hist.T, cmap=cmap)

    # Add a colorbar
    cbar = fig.colorbar(cax, ax=ax, pad=0.1)
    if scale == 'log':
        cbar.set_label('Count (Log Scale)')
    else:
        cbar.set_label('Count')

    # Configure the plot
    ax.set_theta_zero_location("N")  # Set 0 degrees at the top
    ax.set_theta_direction(-1)  # Go clockwise

    # Set the ylim (radial distance)
    ax.set_ylim(0, x_max)

    # Add title
    ax.set_title('2D Polar Histogram (Angle and Distance)', y=1.08)

    return hist

f = plt.figure(figsize=(14, 10))
ax = [f.add_subplot(2, 2,gg , projection='polar') for gg in range(1,5)]
h_all = []
curh = plot_polar_histogram_with_distance(close_pts[:,0,0],close_pts[:,0,1],n_angular_bins=36,n_radial_bins=10,scale='normal',ax=ax[0],x_max=150)
h_all.append(curh)
ax[0].set_title('nose relative to center (normal scale)')
plot_polar_histogram_with_distance(close_pts[:,0,0],close_pts[:,0,1],n_angular_bins=36,n_radial_bins=10,scale='log',ax=ax[2],x_max=150)

ax[2].set_title('nose relative to center (log scale)')
close_pts_nose = close_pts - mice_pts[:,0:1,:]
curh = plot_polar_histogram_with_distance(close_pts_nose[:,0,0],close_pts_nose[:,0,1],n_angular_bins=36,n_radial_bins=10,scale='normal',ax=ax[1],x_max=150)
h_all.append(curh)
ax[1].set_title('nose relative to nose (normal scale)')
plot_polar_histogram_with_distance(close_pts_nose[:,0,0],close_pts_nose[:,0,1],n_angular_bins=36,n_radial_bins=10,scale='log',ax=ax[3],x_max=150)
ax[3].set_title('nose relative to nose (log scale)')
plt.tight_layout()


if False:
    plt.savefig('/groups/branson/home/kabram/temp/close_interaction_polar_histogram.svg')
    plt.savefig('/groups/branson/home/kabram/temp/close_interaction_polar_histogram.png')

##
f,ax = plt.subplots(1, 2, figsize=(10, 5))
for ndx,curh in enumerate(h_all):
    cc = curh.flatten()
    cc = cc[cc > 0]
    ax[ndx].plot(np.sort(cc)[::-1])
    ss = curh.std()
    gg = np.random.normal(size=1000)*ss
    gg= gg[gg>0]
    ax[ndx].plot(np.sort(gg)[::-1])
    ax[ndx].set_yscale('log')
    ax[ndx].set_xscale('log')

ax[0].set_title('nose relative to center')
ax[0].set_xlabel('Cell location')
ax[0].set_ylabel('Count')

ax[1].set_title('nose relative to nose')
ax[1].set_xlabel('Cell location')
ax[1].set_ylabel('Count')

if False:
    plt.savefig('/groups/branson/home/kabram/temp/close_interaction_log_log.svg')
    plt.savefig('/groups/branson/home/kabram/temp/close_interaction_log_log.png')
