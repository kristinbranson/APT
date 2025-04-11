import PoseTools as pt
import numpy as np
import TrkFile

# 10 zebrafish
id_file = '//groups/branson/home/kabram/Downloads/idstuff/10_fish_group6/first/iddat.pkl'
mov_file = '/groups/branson/home/kabram/Downloads/idstuff/10_zebrafish/group_3/TU20171214_N10_Group6_first_01-18-18_17-35-02.avi'
# mov_file = '/home/kabram/temp/TU20171214_N10_Group6_first_01-18-18_17-35-02.avi'
# mov_file = '/groups/branson/home/kabram/Downloads/idstuff/10_zebrafish/group_3/out_dir/out_00000001.png'

# 60 flies
# id_file = '/groups/branson/home/kabram/temp/iddat_flies.pkl'
# mov_file = '/groups/branson/home/kabram/Downloads/idstuff/60_flies/Canton_N59_12-15-17_16-32-02/Canton_N59_12-15-17_16-32-02.avi'
# mov_file = '/home/kabram/temp/Canton_N59_12-15-17_16-32-02.avi'

id_dat = pt.pickle_load(id_file)
bbox = id_dat['bbox']
gt = id_dat['gt_id']
theta = id_dat['theta']
ctrs = id_dat['centers']
blen = id_dat['blen']
n_fr = bbox.shape[0]
n_id = bbox.shape[1]
msz = np.round(np.nanmedian(blen))

#
from tqdm import tqdm
# locs = np.zeros([2,2,n_fr,n_id])

# locs = np.transpose(bbox,[2,3,0,1])
locs1 = ctrs.copy()
locs2 = ctrs.copy()

#theta = theta + np.pi/4
locs1[...,0] += msz/2*np.cos(theta)
locs1[...,1] += msz/2*np.sin(theta)
locs2[...,0] -= msz/2*np.cos(theta)
locs2[...,1] -= msz/2*np.sin(theta)
locs = np.stack([locs1,locs2],axis=2)
locs = np.transpose(locs,[2,3,0,1])

ts = np.ones_like(locs[:,0])
tag = np.zeros_like(ts).astype('bool')

trk = TrkFile.Trk(p=locs,pTrkTS=ts,pTrkTag=tag)

ids = TrkFile.Tracklet(defaultval=-1, size=(1, trk.ntargets, n_fr))
[sf, ef] = trk.get_startendframes()
ids.allocate((1,), sf, ef)
orig_ids = []

maxid = 0
not_used = 0
max_prev = maxid
for aid in range(n_id):
    frs = np.where(np.isnan(bbox[:,aid, 0, 0]))[0]
    if frs.size == 0:
        ids.settargetframe(np.ones([n_fr])*maxid,aid,range(n_fr))
        maxid += 1
    else:
        if frs[0] > 0:
            ids.settargetframe(np.ones([frs[0] ]) * maxid, aid, range(frs[0]))
            maxid +=1
        for br_id in range(len(frs)):
            if br_id<(len(frs)-1) and ((frs[br_id+1]-frs[br_id]) == 1):
                not_used += 1
                continue

            if frs[br_id] == frs[-1]:
                if frs[br_id] == (n_fr-1):
                    not_used +=1
                    continue
                ids.settargetframe(np.ones([n_fr-frs[br_id]-1])*maxid,aid,range(frs[br_id]+1,n_fr))
                maxid+=1
            else:
                ids.settargetframe(np.ones([frs[br_id+1]-frs[br_id]-1])*maxid,aid,range(frs[br_id]+1,frs[br_id+1]))
                maxid+=1

    orig_ids.append([max_prev,maxid])
    max_prev = maxid
_,ids = ids.unique()
trk.apply_ids(ids)
orig_ids = np.array(orig_ids)

# do sanity checks
for fr in range(n_fr):
    aa = ids.getframe(fr)
    if not (np.count_nonzero(aa==-1) == np.count_nonzero(np.isnan(bbox[fr,:,0,0]))):
        print(f'{fr}')
        break

##
tmp_trx = '/groups/branson/home/kabram/temp/tt_fish.trk'
trk.save(tmp_trx,saveformat='tracklet')
##

import link_trajectories as lnk
import tempfile
from poseConfig import conf
conf.imsz = [int(msz),int(msz)] # [128,128]
conf.has_trx_file = False
conf.use_bbox_trx = False
conf.use_ht_trx = True
conf.img_dim = 3
conf.trx_align_theta = True

rescale = 0.5

# Run script_idtracker_create_dset.py to create and save the triplet dataset
tmp_trx = '/groups/branson/home/kabram/temp/tt.trk'
# tmp_trx = tempfile.mkstemp()[1]
# trk.save(tmp_trx, saveformat='tracklet')
id_classifier, loss_history = lnk.train_id_classifier('/groups/branson/home/kabram/temp/iddat_flies_25k.pkl', conf,save=True,save_file='/groups/branson/home/kabram/temp/id_fis',n_iters=60000, rescale=rescale, flip90=True,eval=False)



## link
def_params = lnk.get_default_params(conf)
trk_out, matched = lnk.link_trklet_id(trk.copy(),id_classifier,mov_file,conf,tmp_trx,min_len=def_params['maxframes_delete'],n_per_trk=25,rescale=rescale,min_len_select=20)

mids = np.where(matched>0)[0]
oids = [np.where( (orig_ids[:,0]<=r) & (orig_ids[:,1]>r))[0][0] for r in mids]
nids = [np.where( (orig_ids[:,0]<=r) & (orig_ids[:,1]>r))[0][0] for r in matched[mids]]
np.count_nonzero(np.array(oids)!=np.array(nids))
## load wts from saved mode

wt_file = '/groups/branson/home/kabram/temp/id_fly_net_wts_tracklet_wt_sampling_1k-15000.p'
from torchvision import models
import torch

net = models.resnet.resnet18(pretrained=True)
net.fc = torch.nn.Linear(in_features=512, out_features=32, bias=True)

checkpoint = torch.load(wt_file)
net.load_state_dict(checkpoint['model_state_params'])
id_classifier = net.cuda()


## quick test id network
import movies
n_per_trk = 25
import APT_interface as apt
import PoseTools
import torch
cap = movies.Movie(mov_file)
ss, ee = trk.get_startendframes()
trx_dict = apt.get_trx_info(tmp_trx, conf, cap.get_n_frames())
trx = trx_dict['trx']
dummy_locs = np.ones([n_per_trk, 2, 2]) * conf.imsz[0] / 2

# For each tracklet chose n_per_trk random examples and the find their embedding.

net = id_classifier
all_ims = []
net.eval()

preds = []
ix_range = np.where(~np.isnan(trk.getframe(2000)[0,0,:]))[0].tolist()
ix_range += np.where(~np.isnan(trk.getframe(18000)[0,0,:]))[0].tolist()

for ix in tqdm(ix_range):
    to_do_list = []
    for cc in range(n_per_trk):
        ndx = np.random.choice(np.arange(ss[ix], ee[ix] + 1))
        to_do_list.append([ndx, ix], )

    curims = apt.create_batch_ims(to_do_list, conf, cap, False, trx, None, use_bsize=False)
    all_ims.append(curims)
    if curims.shape[3] == 1:
        curims = np.tile(curims, [1, 1, 1, 3])
    zz, _ = PoseTools.preprocess_ims(curims, dummy_locs, conf, False, rescale)
    zz = zz.transpose([0, 3, 1, 2])
    zz = torch.tensor(zz, dtype=torch.float).cuda()
    zz = zz / 255.
    im_mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).to('cuda')
    im_std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).to('cuda')
    zz = zz- im_mean
    zz = zz/ im_std
    with torch.no_grad():
        oo = net(zz).cpu().numpy()
    preds.append(oo)

cap.close()

rr = np.array(preds)
kk = np.linalg.norm(rr[None,:,None,...]-rr[:,None,:,None],axis=-1)
ff = np.median(kk,axis=(-1,-2))
ff2 = ff[60:,:60].copy()
gg = ff.diagonal()
gg2 = ff2.diagonal().copy()
ff2[range(ff2.shape[0]),range(ff2.shape[0])] = 100.
hh = ff2.min(axis=1)
plt.figure(); plt.scatter(gg2,hh)
plt.figure(); plt.imshow(ff)
mm = np.median(kk,axis=(-1,-2))
gg = np.clip(gg,0.05,1000)
nn = (mm/np.sqrt(gg[None,:]))/np.sqrt(gg[:,None])
plt.figure(); plt.imshow(np.clip(nn,0,10))

## check data loaders
to_do_list = pt.pickle_load('/groups/branson/home/kabram/temp/tdlist.pkl')
len(to_do_list)
import link_trajectories as lnk
from poseConfig import conf

conf.imsz = [128,128]
conf.has_trx_file = False
conf.use_bbox_trx = True
conf.img_dim = 3
conf.trx_align_theta = False

trk_in = trk
import tempfile
tmp_trx = tempfile.mkstemp()[1]
trk_in.save(tmp_trx,saveformat='tracklet')

import APT_interface as apt
import movies
import torch
cap = movies.Movie(mov_file)
trx_dict = apt.get_trx_info(tmp_trx, conf, cap.get_n_frames())
trx = trx_dict['trx']

##
# kk = lnk.id_loader(mov_file,0,trx,conf,to_do_list)
# aa = torch.utils.data.DataLoader(kk,batch_size=5,num_workers=0,worker_init_fn=lambda id: np.random.seed(id))
# bb = iter(aa)
# cc = next(bb)
# cc.shape
if conf.trx_align_theta:
    conf.rrange = 15.
else:
    conf.rrange = 180.
conf.trange = min(conf.imsz) / 10
conf.horzFlip = False
conf.vertFlip = False
conf.scale_factor_range = 1
conf.brange = [-0.05, 0.05]
conf.crange = [0.95, 1.05]

kk = lnk.id_dset(mov_file,5,trx,conf,to_do_list)
aa = torch.utils.data.DataLoader(kk,batch_size=5,num_workers=5,worker_init_fn=lambda id: np.random.seed(id))
bb = iter(aa)
cc = next(bb)

qq = cc[0][...,0].reshape([32*3,128,128,1]).cpu().numpy()
pt.show_stack(qq[:4*3],4,3)

## tracklet based triplet generation/mining

## Setup trx and other stuff

import cv2
import multiprocessing as mp
import APT_interface as apt
import movies
import link_trajectories as lnk
import tempfile
from poseConfig import conf
from tqdm import tqdm

conf.imsz = [int(msz),int(msz)] # [128,128]
conf.has_trx_file = False
conf.use_bbox_trx = False
conf.use_ht_trx = True
conf.img_dim = 3
conf.trx_align_theta = True

# tmp_trx = tempfile.mkstemp()[1]
# trk.save(tmp_trx, saveformat='tracklet')
# tmp_trx = '/groups/branson/home/kabram/temp/tt.trk'

min_trx_len = 3
n_threads = min(12,mp.cpu_count())

ss, ee = trk.get_startendframes()
# Save the current trk to be used as trx. Could be avoided but the whole image patch extracting pipeline exists with saved trx file, so not rewriting it.

if np.count_nonzero((ee - ss) > min_trx_len) < conf.max_n_animals:
    min_trx_len = np.percentile((ee - ss), 20) - 1

sel_trk = np.where( (ee-ss)>min_trx_len)[0]
sel_trk_info = list(zip(sel_trk,ss[sel_trk],ee[sel_trk]))

cap = movies.Movie(mov_file)
trx_dict = apt.get_trx_info(tmp_trx, conf, cap.get_n_frames())
trx = trx_dict['trx']
cap.close()

## read the image in parallel

from importlib import reload
reload(lnk)
import time
train_data_file = '/groups/branson/home/kabram/temp/fish_id_tracklet_data.pkl'

a = time.time()
data = lnk.read_ims_par(trx,sel_trk_info,mov_file,conf)
b =time.time()
print(f'Time to read {np.round((b-a)/60)} min')
import pickle
with open(train_data_file, 'wb') as f:
    pickle.dump(data,f)


##
import PoseTools as pt
import link_trajectories as lnk
from poseConfig import conf

conf.imsz = [int(msz),int(msz)] # [128,128]
conf.has_trx_file = False
conf.use_bbox_trx = False
conf.use_ht_trx = True
conf.img_dim = 3
conf.trx_align_theta = True
rescale = 1

# tdata = pt.pickle_load(train_data_file)
train_data_file = '/groups/branson/home/kabram/temp/id_fish_tracking_data.pkl'
AA = pt.pickle_load(train_data_file)
tdata = AA['data']
sel_tgt = AA['sel_tgt']
id_classifier, loss_history = lnk.train_id_classifier(tdata,conf,trk,save=True,save_file='/groups/branson/home/kabram/temp/id_fish_net_wts_tracklet_ind_sampling',rescale=rescale,n_iters=60000)

##
import copy
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import PoseTools
import torch
from tqdm import tqdm

rescale = 0.5
flip90 = False
n_iters = 15000
bsz = 16
save_file = '/groups/branson/home/kabram/temp/id_fly_net_wts_tracklet_wt_sampling_1k'
save= True
use_sampling = True

## create data for tracking

tracking_data_file = '/groups/branson/home/kabram/temp/id_fish_tracking_data.pkl'
ss, ee = trk.get_startendframes()
trx_dict = apt.get_trx_info(tmp_trx, conf, cap.get_n_frames())
trx = trx_dict['trx']
min_len_select = 3
# For each tracklet chose n_per_trk random examples and the find their embedding.
sel_tgt = np.where((ee - ss + 1) >= min_len_select)[0]
sel_ss = ss[sel_tgt];
sel_ee = ee[sel_tgt]
trk_info = list(zip(sel_tgt, sel_ss, sel_ee))
start_t = time.time()
data = lnk.read_ims_par(trx, trk_info, mov_file, conf, n_ex=25)
end_t = time.time()

with open(tracking_data_file, 'wb') as f:
    pickle.dump({'sel_tgt': sel_tgt, 'sel_ss': sel_ss, 'sel_ee': sel_ee, 'data': data}, f)

## testing co clustering


from sklearn.datasets import make_biclusters
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score
from sklearn.cluster import spectral_clustering

AA = pt.pickle_load(tracking_data_file)
data = AA['data']
sel_tgt = AA['sel_tgt']
net = id_classifier
tgt_id = np.array([r[1] for r in data])
import torch
net.eval()
preds = []
for ix in tqdm(sel_tgt):
    curndx = np.where(tgt_id==ix)[0][0]
    curims = data[curndx][0]
    zz = lnk.process_id_ims(curims, conf, False, rescale)
    zz = torch.tensor(zz).float().cuda()
    with torch.no_grad():
        oo = net(zz).cpu().numpy()
    preds.append(oo)

rr = np.array(preds)
n_tr = rr.shape[0]
ddr = np.ones([n_tr,n_tr,25,25]) * np.nan
from tqdm import tqdm

for xx in tqdm(range(n_tr)):
    ddr[xx, :] = np.linalg.norm(rr[xx, None, :, None] - rr[:, None, :], axis=-1)
ddm = np.median(ddr,axis=(2,3))
gg = ddm[range(len(ddm)),range(len(ddm))]
ddz = (ddm/np.sqrt(gg[:,None]))/np.sqrt(gg[None,:])
##
yy = ddm
plt.figure(); plt.imshow(yy)

prev_id = 0
b_start = -0.5
id_aa = np.array(orig_ids)
for ndx, curid in enumerate(sel_tgt):
    cc = np.where((id_aa[:,0]<=curid)&(id_aa[:,1]>curid))[0][0]
    if cc != prev_id:
        prev_id = cc
        plt.plot([b_start,ndx-0.5,ndx-0.5,b_start,b_start],[b_start,b_start,ndx-0.5,ndx-0.5,b_start],c='b')
        b_start = ndx-0.5

orig_map = []
for curo in orig_ids:
    oo = np.where((sel_tgt>=curo[0]) & (sel_tgt<curo[1]))[0]
    orig_map.append(oo)

cluster_means = []
for mm in orig_map:
    cluster_means.append(np.mean(yy[np.ix_(mm,mm)]))
##

done = np.zeros(n_tr)
groupsid = np.ones(n_tr)*-1
groups = []
tline = []
tlen = []

# diagdd = ddm.diagonal()
# ddn = ddm/np.sqrt(diagdd[None])/np.sqrt(diagdd[:,None])
ddm_tr = ddm < 0.5
n_fr = max(sel_ee)
while not np.all(done):
    sel = np.random.choice(np.where(done==0)[0])
    x = np.zeros(n_tr)
    x[sel] = 1.
    y1 = np.matmul(ddm_tr,x)
    if y1.sum()==0:
        done[sel] = 1
        groups.append([sel])
        groupsid[sel] = len(groups)-1
        continue
    y2 = np.matmul(ddm_tr,y1)/y1.sum()
    cur_gr = np.where(y2>0.45)[0]
    assert(~np.any(done[cur_gr]))
    done[cur_gr] = 1
    groups.append(cur_gr)
    groupsid[cur_gr] = len(groups) - 1
    ctline = np.zeros(n_fr)
    ctlen = 0
    for cc in cur_gr:
        ctline[sel_ss[cc]:sel_ee[cc]+1] += 1
        ctlen += sel_ee[cc]-sel_ss[cc]+1
    tline.append(ctline)
    tlen.append(ctlen)



## On alices data

###
# trx_file = '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/nochr_TrpA65F12_Unknown_RigB_20201212T163629/registered_trx.mat'
# mov_file = '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/nochr_TrpA65F12_Unknown_RigB_20201212T163629/movie.ufmf'

trx_file = '/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/witinAssayData/cx_GMR_SS00006_CsChr_RigC_20151014T093157/registered_trx.mat'
mov_file = trx_file.replace('registered_trx.mat','movie.ufmf')
tmp_trx = '/groups/branson/home/kabram/temp/alice_fly_trk1.trk'
import APT_interface as apt
import TrkFile
import link_trajectories as lnk
import multiprocessing as mp
import APT_interface as apt
import movies
import link_trajectories as lnk
from poseConfig import conf

n_ex = 50
msz = 108
conf.imsz = [int(msz),int(msz)] # [128,128]
conf.has_trx_file = False
conf.use_bbox_trx = False
conf.use_ht_trx = True
conf.img_dim = 3
conf.trx_align_theta = True

params = {}
params['maxframes_delete'] = 3
params['maxcost_heuristic'] = 'secondorder'

min_len_sel = 11

nt = 11
nfr = 50662
plocs = np.ones([nfr,nt,2,2])*np.nan
ndone = np.zeros(nfr).astype('int')

trx = TrkFile.load_trx(trx_file)

all_locs = []
for ndx in range(len(trx['x'])):
    theta = trx['theta'][ndx]
    locs_hx = trx['x'][ndx] + trx['a'][ndx]*2*np.cos(theta)
    locs_hy = trx['y'][ndx] + trx['a'][ndx]*2*np.sin(theta)
    locs_tx = trx['x'][ndx] - trx['a'][ndx]*2*np.cos(theta)
    locs_ty = trx['y'][ndx] - trx['a'][ndx]*2*np.sin(theta)
    locs_h = np.array([locs_hx,locs_hy])
    locs_t = np.array([locs_tx,locs_ty])
    locs = np.array([locs_h,locs_t])
    for zz in range(locs.shape[2]):
        curf = trx['startframes'][ndx] + zz
        plocs[curf,ndone[curf]] = locs[...,zz]
        ndone[curf] +=1

    all_locs.append(locs)

locs_lnk = np.transpose(plocs, [2, 3, 0, 1])

ts = np.ones_like(locs_lnk[:, 0, ...])
tag = np.ones(ts.shape)*np.nan  # tag which is always false for now.
locs_conf = None

trk = TrkFile.Trk(p=locs_lnk, pTrkTS=ts, pTrkTag=tag, pTrkConf=locs_conf)
trk.convert2sparse()
trk = lnk.link(trk, conf, params_in=params, do_merge_close=False, do_stitch=False)
trk.save(tmp_trx,saveformat='tracklet')

# tlocs = TrkFile.Tracklet()
# tlocs.setdata(all_locs,np.nan,trx['startframes'],trx['endframes'])
# tts = TrkFile.Tracklet()
# all_ts = [np.ones_like(ll[0]) for ll in all_locs]
# tts.setdata(all_ts,-np.inf,trx['startframes'],trx['endframes'])
# ttg = TrkFile.Tracklet()
# all_tg = [np.ones_like(ll[0])<0 for ll in all_locs]
# ttg.setdata(all_tg,False,trx['startframes'],trx['endframes'])

# trk = TrkFile.Trk(p=tlocs,pTrkTS=tts,pTrkTag=ttg)
# trk.save(tmp_trx,saveformat='tracklet')

##

ss, ee = trk.get_startendframes()

# if np.count_nonzero((ee - ss) > min_trx_len) < conf.max_n_animals:
#     min_trx_len = np.percentile((ee - ss), 20) - 1

sel_trk = np.where((ee-ss)>min_len_sel)[0] #np.arange(trk.ntargets)
sel_trk_info = list(zip(sel_trk,ss[sel_trk],ee[sel_trk]))

cap = movies.Movie(mov_file)
trx_dict = apt.get_trx_info(tmp_trx, conf, cap.get_n_frames())
trx = trx_dict['trx']
cap.close()

from importlib import reload
reload(lnk)
import time
train_data_file = '/groups/branson/home/kabram/temp/alice_fly_tracklet_data.pkl'

a = time.time()
data = lnk.read_ims_par(trx,sel_trk_info,mov_file,conf,n_ex=n_ex)
b =time.time()
print(f'Time to read {np.round((b-a)/60)} min')
import pickle
with open(train_data_file, 'wb') as f:
    pickle.dump(data,f)


## train

if False:
    AA = pt.pickle_load(train_data_file)
    data = AA
rescale = 1
id_classifier, loss_history = lnk.train_id_classifier(data,conf,trk,save=True,save_file='/groups/branson/home/kabram/temp/alice_fly_net_wts_tracklet_ind_sampling',rescale=rescale,n_iters=40000)

## check results

from tqdm import tqdm
net = id_classifier

cap = movies.Movie(mov_file)
ss, ee = trk.get_startendframes()
trx_dict = apt.get_trx_info(tmp_trx, conf, cap.get_n_frames())
trx = trx_dict['trx']

# For each tracklet chose n_per_trk random examples and the find their embedding.
sel_tgt = np.where((ee - ss + 1) >= min_len_sel)[0]
sel_ss = ss[sel_tgt]
sel_ee = ee[sel_tgt]
trk_info = list(zip(sel_tgt, sel_ss, sel_ee))
tdata = lnk.read_ims_par(trx, trk_info, mov_file, conf, n_ex=n_ex)
adata = tdata
tgt_id = [r[1] for r in adata]
tgt_id = np.array(tgt_id)
# tgt_id1 = [r[1]+len(data) for r in tdata]
# tgt_id = np.array(tgt_id+tgt_id1)
import torch
net.eval()
preds = []
for ix in tqdm(sel_tgt):
    curndx = np.where(tgt_id==ix)[0][0]
    curims = adata[curndx][0]
    zz = lnk.process_id_ims(curims, conf, False, rescale)
    zz = torch.tensor(zz).float().cuda()
    with torch.no_grad():
        oo = net(zz).cpu().numpy()
    preds.append(oo)

rr = np.array(preds)
n_tr = rr.shape[0]
ddr = np.ones([n_tr,n_tr,n_ex,n_ex]) * np.nan

for xx in tqdm(range(n_tr)):
    ddr[xx, :] = np.linalg.norm(rr[xx, None, :, None] - rr[:, None, :], axis=-1)
ddm = np.median(ddr,axis=(2,3))
# plt.figure(); plt.imshow(ddm)


##

def_params = lnk.get_default_params(conf)
trk_out, matched = lnk.link_trklet_id(trk.copy(),id_classifier,mov_file,conf,tmp_trx,min_len=def_params['maxframes_delete'],n_per_trk=25,rescale=rescale,min_len_select=11)
trk_out.save('/groups/branson/home/kabram/temp/alice_fly_trk_out.trk',saveformat='tracklet')

##

tinfo = []
for xx in range(trk_out[0].ntargets):
    jj = trk_out[0].gettarget(xx)[0, 0, :, 0]
    clen = np.count_nonzero(~np.isnan(jj))
    qq = np.ones(jj.shape[0]+2)>0
    qq[1:-1] = np.isnan(jj)

    qq1 = np.where(   qq[:-2]   & (~qq[1:-1]) )[0]
    qq2 = np.where( (~qq[1:-1]) &   qq[2:] )[0]
    aa1 = np.array([np.where(sel_ss==qq)[0][0] for qq in qq1])
    aa2 = np.array([np.where(sel_ee==qq)[0][0] for qq in qq2])
    aa1 = np.array([sel_tgt[aa] for aa in aa1])+1
    aa2 = np.array([sel_tgt[aa] for aa in aa2])+1
    tinfo.append([clen,list(zip(qq1+1,qq2+1,aa1,aa2))])

## Test on a different movie

trx_file = '/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/withGenotypeData/cx_GMR_SS00238_CsChr_RigD_20150826T143306/registered_trx.mat'
mov_file = trx_file.replace('registered_trx.mat','movie.ufmf')

tmp_trx = '/groups/branson/home/kabram/temp/alice_fly_trk2.trk'
import APT_interface as apt
import TrkFile
import link_trajectories as lnk
import multiprocessing as mp
import APT_interface as apt
import movies
import link_trajectories as lnk
from poseConfig import conf

cap = movies.Movie(mov_file)
n_ex = 50
msz = 108
conf.imsz = [int(msz),int(msz)] # [128,128]
conf.has_trx_file = False
conf.use_bbox_trx = False
conf.use_ht_trx = True
conf.img_dim = 3
conf.trx_align_theta = True

params = {}
params['maxframes_delete'] = 3
params['maxcost_heuristic'] = 'secondorder'

min_len_sel = 11

nt = 11
nfr = cap.get_n_frames()
plocs = np.ones([nfr,nt,2,2])*np.nan
ndone = np.zeros(nfr).astype('int')

trx = TrkFile.load_trx(trx_file)

all_locs = []
for ndx in range(len(trx['x'])):
    theta = trx['theta'][ndx]
    locs_hx = trx['x'][ndx] + trx['a'][ndx]*2*np.cos(theta)
    locs_hy = trx['y'][ndx] + trx['a'][ndx]*2*np.sin(theta)
    locs_tx = trx['x'][ndx] - trx['a'][ndx]*2*np.cos(theta)
    locs_ty = trx['y'][ndx] - trx['a'][ndx]*2*np.sin(theta)
    locs_h = np.array([locs_hx,locs_hy])
    locs_t = np.array([locs_tx,locs_ty])
    locs = np.array([locs_h,locs_t])
    for zz in range(locs.shape[2]):
        curf = trx['startframes'][ndx] + zz
        plocs[curf,ndone[curf]] = locs[...,zz]
        ndone[curf] +=1

    all_locs.append(locs)


trk = lnk.link(plocs, conf, params_in=params, do_merge_close=False, do_stitch=False)
trk.save(tmp_trx,saveformat='tracklet')

import torch
from torchvision import models
net = models.resnet.resnet18(pretrained=True)
net.fc = torch.nn.Linear(in_features=512, out_features=32, bias=True)

cpt = torch.load('/groups/branson/home/kabram/temp/alice_fly_net_wts_tracklet_ind_sampling-40000.p')
net.load_state_dict(cpt['model_state_params'])
id_classifier = net.cuda()

rescale = 1
def_params = lnk.get_default_params(conf)
trk_out, matched = lnk.link_trklet_id(trk.copy(),id_classifier,mov_file,conf,tmp_trx,min_len=def_params['maxframes_delete'],n_per_trk=n_ex,rescale=rescale,min_len_select=11)

##

ss, ee = trk.get_startendframes()
sel_tgt = np.where((ee - ss + 1) >= 11)[0]
sel_ss = ss[sel_tgt]
sel_ee = ee[sel_tgt]

tinfo = []
for xx in range(trk_out[0].ntargets):
    jj = trk_out[0].gettarget(xx)[0, 0, :, 0]
    clen = np.count_nonzero(~np.isnan(jj))
    qq = np.ones(jj.shape[0]+2)>0
    qq[1:-1] = np.isnan(jj)

    qq1 = np.where(   qq[:-2]   & (~qq[1:-1]) )[0]
    qq2 = np.where( (~qq[1:-1]) &   qq[2:] )[0]
    # aa1 = np.array([np.where(sel_ss==qq)[0][0] for qq in qq1])
    # aa2 = np.array([np.where(sel_ee==qq)[0][0] for qq in qq2])
    # aa1 = np.array([sel_tgt[aa] for aa in aa1])+1
    # aa2 = np.array([sel_tgt[aa] for aa in aa2])+1
    tinfo.append([clen,list(zip(qq1+1,qq2+1))])#,aa1,aa2))])


## Roians data


## setup

import TrkFile
import multiprocessing as mp
import APT_interface as apt
import movies
import link_trajectories as lnk
from poseConfig import conf
import PoseTools as pt
trk_file = '/groups/branson/home/kabram/temp/roian_unmarked_1_raw.trk'
mov_file = '/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice/20210924_four_female_mice_1.mjpg'
trk_file_out = '/groups/branson/home/kabram/temp/roian_unmarked_1_raw_idlinked.trk'
train_data_file = '/groups/branson/home/kabram/temp/roian_mouse_train_data.pkl'
tracking_data_file = '/groups/branson/home/kabram/temp/roian_unmarkedmice_tracking_data.pkl'

tmp_trx = trk_file_out
n_ex = 50
msz = 250
conf.imsz = [int(msz),int(msz)] # [128,128]
conf.has_trx_file = False
conf.use_bbox_trx = False
conf.use_ht_trx = True
conf.img_dim = 3
conf.trx_align_theta = True
rescale = 1
min_len_sel = 11

#

trk_in = TrkFile.Trk(trkfile=trk_file)
params = {}
params['maxframes_delete'] = 3
params['maxcost_heuristic'] = 'secondorder'
do_delete_short = False

trk = lnk.link(trk_in, conf, params_in=params, do_merge_close=False, do_stitch=False,do_delete_short=do_delete_short)
trk.save(trk_file_out,saveformat='tracklet')

ss, ee = trk.get_startendframes()

# if np.count_nonzero((ee - ss) > min_trx_len) < conf.max_n_animals:
#     min_trx_len = np.percentile((ee - ss), 20) - 1

# sel_tgt = np.arange(trk.ntargets)
sel_tgt = np.where((ee - ss + 1) >= min_len_sel)[0]
sel_trk_info = list(zip(sel_tgt,ss[sel_tgt],ee[sel_tgt]))

cap = movies.Movie(mov_file)
trx_dict = apt.get_trx_info(tmp_trx, conf, cap.get_n_frames())
trx = trx_dict['trx']
cap.close()

##
trk = TrkFile.Trk(trkfile=trk_file_out)

##

from importlib import reload
reload(lnk)
import time
a = time.time()
data = lnk.read_ims_par(trx,sel_trk_info,mov_file,conf,n_ex=n_ex)
b =time.time()
print(f'Time to read {np.round((b-a)/60)} min')
import pickle
with open(train_data_file, 'wb') as f:
    pickle.dump(data,f)

##

conf.link_id_debug=False
if False:
    AA = pt.pickle_load(train_data_file)
    data = AA
id_classifier, loss_history = lnk.train_id_classifier([data],conf,[trk],save=True,save_file='/groups/branson/home/kabram/temp/roian_unmarked_mice_net_wts')

##
import torch
from torchvision import models
net = models.resnet.resnet18(pretrained=True)
net.fc = torch.nn.Linear(in_features=512, out_features=32, bias=True)

cpt = torch.load('/groups/branson/home/kabram/temp/roian_unmarked_mice_net_wts-40000.p')
net.load_state_dict(cpt['model_state_params'])
id_classifier = net.cuda()

##

trk_out = lnk.link_trklet_id([trk.copy()],id_classifier,[mov_file],conf,[trx],n_per_trk=50,rescale=rescale,min_len_select=5)
trk_out[0].save('/groups/branson/home/kabram/temp/roian_unmarked_trk_out1.trk',saveformat='tracklet')


##
from tqdm import tqdm
import pickle

cap = movies.Movie(mov_file)
ss, ee = trk.get_startendframes()
trx_dict = apt.get_trx_info(tmp_trx, conf, cap.get_n_frames())
trx = trx_dict['trx']

# For each tracklet chose n_per_trk random examples and the find their embedding.
sel_tgt = np.where((ee - ss + 1) >= min_len_sel)[0]
# sel_tgt = np.arange(trk.ntargets)
sel_ss = ss[sel_tgt]
sel_ee = ee[sel_tgt]
trk_info = list(zip(sel_tgt, sel_ss, sel_ee))
tdata = lnk.read_ims_par(trx, trk_info, mov_file, conf, n_ex=n_ex)

#
# with open(tracking_data_file, 'wb') as f:
#     pickle.dump({'sel_tgt': sel_tgt, 'sel_ss': sel_ss, 'sel_ee': sel_ee, 'tdata': tdata}, f)

##
net = id_classifier
adata = tdata
tgt_id = np.array([r[1] for r in tdata])
# tgt_id1 = [r[1]+len(data) for r in tdata]
# tgt_id = np.array(tgt_id+tgt_id1)
import torch
net.eval()
preds = []
for ix in tqdm(sel_tgt):
    curndx = np.where(tgt_id==ix)[0][0]
    curims = adata[curndx][0]
    zz = lnk.process_id_ims(curims, conf, False, rescale)
    zz = torch.tensor(zz).float().cuda()
    with torch.no_grad():
        oo = net(zz).cpu().numpy()
    preds.append(oo)

rr = np.array(preds)
n_tr = rr.shape[0]
ddr = np.ones([n_tr,n_tr,n_ex,n_ex]) * np.nan

for xx in tqdm(range(n_tr)):
    ddr[xx, :] = np.linalg.norm(rr[xx, None, :, None] - rr[:, None, :], axis=-1)
ddm = np.median(ddr,axis=(2,3))
# plt.figure(); plt.imshow(ddm)

##
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

ddn = ddm.copy()
ddn[range(n_tr),range(n_tr)] = 0.
distArray = ssd.squareform(ddn)  # distArray[{n choose 2}-{n-i choose 2} + (j-i-1)] is the distance between points i and j
Z = linkage(distArray,'average')
plt.figure()
dn = dendrogram(Z)

##
thres = 1.
F = fcluster(Z,thres,criterion='distance')

groups = []
tline = []
tlen = []
tmax = []
n_fr = max(sel_ee)
for ndx in range(max(F)):
    cur_gr = np.where(np.array(F)==(ndx+1))[0]
    groups.append(cur_gr)
    ctline = np.zeros(n_fr)
    ctlen = 0
    for cc in cur_gr:
        ctline[sel_ss[cc]:sel_ee[cc]] += 1
        ctlen += sel_ee[cc]- sel_ss[cc]
    tmax.append(max(ctline))
    tline.append(ctline)
    tlen.append(ctlen)

##

f1 = plt.figure()

for ndx, gr in enumerate(groups):
    if len(gr)<2: continue
    if tlen[ndx] < 1000: continue
    ims = []
    for gg in gr:
        ix = np.where(tgt_id==sel_tgt[gg])[0][0]
        ims.extend(tdata[ix][0])
    ims = np.array(ims)
    si = np.random.randint(0,len(ims),25)
    im_s = ims[si,...,0]
    isz1 = im_s.shape[1]
    isz2 = im_s.shape[2]
    im_s = im_s.reshape([5,5,isz1, isz2])
    im_s = im_s.transpose([0, 2, 1, 3])
    im_s = im_s.reshape([5 * isz1, 5 * isz2])

    plt.figure(f1)
    plt.imshow(im_s,'gray')
    plt.axis('off')
    plt.title(f'{ndx}, {len(gr)}, {tlen[ndx]}')
    plt.waitforbuttonpress()

##
gndx = 3
ss_t = sel_ss
ee_t = sel_ee
sg = np.where(np.array(tlen)>50000)[0]
for xx in groups[gndx]:
    overlap_tgts = lnk.get_overlap(ss_t,ee_t,ss_t[xx],ee_t[xx],xx)
    overlap_tgts = np.array(list(set(overlap_tgts) - set([xx])))
    ogr = np.unique(np.array(F)[overlap_tgts]-1)
    ogr1 = [oo for oo in ogr if oo in sg]
    print(xx, ogr1, ogr, sel_tgt[overlap_tgts]+1)


##
ss, ee = trk.get_startendframes()

tinfo = []
tlen = []
for xx in range(trk.ntargets):
    jj = trk.gettarget(xx)[0, 0, :, 0]
    clen = np.count_nonzero(~np.isnan(jj))
    qq = np.ones(jj.shape[0]+2)>0
    qq[1:-1] = np.isnan(jj)

    qq1 = np.where(   qq[:-2]   & (~qq[1:-1]) )[0]
    qq2 = np.where( (~qq[1:-1]) &   qq[2:] )[0]
    # aa1 = np.array([np.where(sel_ss==qq)[0][0] for qq in qq1])
    # aa2 = np.array([np.where(sel_ee==qq)[0][0] for qq in qq2])
    # aa1 = np.array([sel_tgt[aa] for aa in aa1])+1
    # aa2 = np.array([sel_tgt[aa] for aa in aa2])+1
    tlen.append(clen)
#    tinfo.append([clen,list(zip(qq1+1,qq2+1))])#,aa1,aa2))])

## id over multiple movies

import link_trajectories as lnk
import TrkFile
import movies
import APT_interface as apt

from poseConfig import conf
import PoseTools as pt
n_ex = 50
msz = 250
conf.imsz = [int(msz),int(msz)] # [128,128]
conf.has_trx_file = False
conf.use_bbox_trx = False
conf.use_ht_trx = True
conf.img_dim = 3
conf.trx_align_theta = True
rescale = 1
min_len_sel = 11

trk_files = ['/groups/branson/home/kabram/temp/roian_unmarked_0_pure_linked.trk','/groups/branson/home/kabram/temp/roian_unmarked_1_pure_linked.trk']
trk0 = TrkFile.Trk(trk_files[0])
trk1 = TrkFile.Trk(trk_files[1])

trks = [trk0,trk1]
mov_files = ['/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice/20210924_four_female_mice_0.mjpg', '/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice/20210924_four_female_mice_1.mjpg']
import torch
from torchvision import models
net = models.resnet.resnet18(pretrained=True)
net.fc = torch.nn.Linear(in_features=512, out_features=32, bias=True)

cpt = torch.load('/groups/branson/home/kabram/temp/out0_id_idwts.p-40000.p')
net.load_state_dict(cpt['model_state_params'])
id_classifier = net.cuda()

all_trx = []

for trk_file, mov_file in zip(trk_files, mov_files):

    cap = movies.Movie(mov_file)
    trx_dict = apt.get_trx_info(trk_file, conf, cap.get_n_frames())
    trx = trx_dict['trx']
    all_trx.append(trx)
    cap.close()

#
trk_out = lnk.link_trklet_id(trks,id_classifier,mov_files,conf,all_trx,n_per_trk=50,rescale=rescale,min_len_select=25)


## ID track an existing fly apt trk

import TrkFile
import link_trajectories as lnk
from reuse import *
trk_file = '/groups/branson/home/robiea/Projects_data/Labeler_APT/socialCsChr_JRC_SS56987_CsChrimson_RigB_20190910T163328/movie_multitarget_bubble_20200811_MDN20201112_40K_mdn.trk'
mov_file = '/groups/branson/home/robiea/Projects_data/Labeler_APT/socialCsChr_JRC_SS56987_CsChrimson_RigB_20190910T163328/movie.ufmf'

pure_file = '/groups/branson/home/kabram/temp/alice_fly_pure_1.trk'
out_file =   '/groups/branson/home/kabram/temp/alice_fly_id_out_1.trk'
trk = TrkFile.Trk(trk_file)

from poseConfig import conf

for k in trk.trkData['trkInfo']['params']:
    conf.__dict__[k] = trk.trkData['trkInfo']['params'][k]

n_ex = 50
msz = 108

conf.multi_animal_crop_sz = int(msz)  # [128,128]
conf.has_trx_file = False
conf.use_bbox_trx = False
conf.use_ht_trx = True
conf.img_dim = 3
conf.trx_align_theta = True
conf.link_id = True
conf.ht_pts = (0,6)

trk_p = lnk.link_pure(trk,conf)
trk_p.save(pure_file,saveformat='tracklet')

##
trk_i = lnk.link_trklets([pure_file], conf, [mov_file], [out_file])
trk_i[0].save(out_file,saveformat='tracklet')


## Tomato id tracking


## pure tracking
cmd = '-name 20220203T072424 -view 1 -cache /groups/branson/home/kabram/.apt/tp3a572117_4d4c_405e_a784_48174bd63da7  -stage multi -conf_params -type detect_mmdetect -conf_params2 use_bbox_trx True -type2 mdn_joint_fpn /groups/branson/home/kabram/.apt/tpfeebd0f0_4fed_4d14_8f68_957228574fcd/t/20220203T072424_20220203T072425.lbl track -out /groups/branson/home/kabram/temp/tomato.trk -mov /groups/branson/home/kabram/bransonlab/data/tomato/IMG_3009.avi  -trx /groups/branson/home/kabram/.apt/tpfeebd0f0_4fed_4d14_8f68_957228574fcd/t/detect_mmdetect/view_0/20220202T044923/trk/IMG_3010_trn20220202T044923_iter20000_20220202T080224_mov1_vwj1_rr.trk -track_type only_predict -end_frame 1235'

cmd1 = '-name 20220203T072424 -view 1 -cache /groups/branson/home/kabram/.apt/tp3a572117_4d4c_405e_a784_48174bd63da7  -stage multi -conf_params -type detect_mmdetect -conf_params2 use_bbox_trx True -type2 mdn_joint_fpn /groups/branson/home/kabram/.apt/tpfeebd0f0_4fed_4d14_8f68_957228574fcd/t/20220203T072424_20220203T072425.lbl track -out /groups/branson/home/kabram/temp/tomato_10.trk -mov /groups/branson/home/kabram/bransonlab/data/tomato/IMG_3010.avi  -trx /groups/branson/home/kabram/.apt/tpfeebd0f0_4fed_4d14_8f68_957228574fcd/t/detect_mmdetect/view_0/20220202T044923/trk/IMG_3010_trn20220202T044923_iter20000_20220202T080224_mov1_vwj1_rr.trk -track_type only_predict -end_frame 780' #1235'

from reuse import *
apt.main(cmd.split())
apt.main(cmd1.split())

## load the pure trk

import TrkFile
import link_trajectories as lnk

mov_files = ['/groups/branson/home/kabram/bransonlab/data/tomato/IMG_3009.avi','/groups/branson/home/kabram/bransonlab/data/tomato/IMG_3010.avi']
trk_pure_files = ['/groups/branson/home/kabram/temp/tomato_pure.trk', '/groups/branson/home/kabram/temp/tomato_10_pure.trk']
out_files = ['/groups/branson/home/kabram/temp/tomato_id_1.trk', '/groups/branson/home/kabram/temp/tomato_10_id_1.trk']

trk_p = [TrkFile.Trk(tt) for tt in trk_pure_files]

# do id tracking
from poseConfig import conf

for k in trk_p[0].trkData['trkInfo']['params']:
    conf.__dict__[k] = trk_p[0].trkData['trkInfo']['params'][k]

conf.has_trx_file = False
conf.use_bbox_trx = False
conf.use_ht_trx = True
conf.img_dim = 3
conf.trx_align_theta = False
conf.link_id = True
conf.ht_pts = (0,1)

trk_i = lnk.link_trklets(trk_pure_files, conf,mov_files, out_files)
trk_i[0].save(out_file[0],saveformat='tracklet')
trk_i[1].save(out_file[1],saveformat='tracklet')

## rerun linking on roians data -- adding missing links

# Roians --marked mice not much useful
# mov_files = ['/groups/branson/bransonlab/roian/apt_testing/multianimal/pb_assay_videos_for_testing/190701_m165837silpb_no_odor_m165836_f0164992.mjpg']
# trk_pure_files = ['/groups/branson/bransonlab/roian/apt_testing/multianimal/pb_assay_videos_for_testing/190701_m165837silpb_no_odor_m165836_f0164992_linked_pure.trk']
# out_files = ['/groups/branson/home/kabram/temp/190701_m165837silpb_no_odor_m165836_f0164992_linked_gap_filled.trk']
# prev_linked = ['/groups/branson/bransonlab/roian/apt_testing/multianimal/pb_assay_videos_for_testing/190701_m165837silpb_no_odor_m165836_f0164992_linked_unchecked_traj_id.trk']
# id_wts = '/groups/branson/bransonlab/roian/apt_testing/multianimal/pb_assay_videos_for_testing/190701_m165837silpb_no_odor_m165836_f0164992_linked_idwts.p'
# id_wts = None

# Roian's unmarked mice
trk_pure_files = ['/groups/branson/home/kabram/temp/20210924_four_female_mice_0_unlabeled_mice_grone_occluded_MA_Bottom_Up_tracklet.trk']
mov_files = ['/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/four_and_five_mice_recordings_210924/20210924_four_female_mice/20210924_four_female_mice_0.mjpg']
out_files = ['/groups/branson/home/kabram/temp/20210924_four_female_mice_0_unlabeled_mice_grone_occluded_MA_Bottom_Up_motion_link.trk']
gt_trk = '/groups/branson/home/kabram/temp/20210924_four_female_mice_0_gt.trk'
id_wts = '/groups/branson/home/kabram/temp/20210924_four_female_mice_0_unlabeled_mice_grone_occluded_MA_Bottom_Up_motion_link_idwts.p'

# alice
# GT flies

# trk_pure_files  =['/groups/branson/home/kabram/temp/ma_expts/alice/trks/cx_GMR_SS00030_CsChr_RigC_20150826T144616_bbox_tracklet.trk']
# id_wts = '/groups/branson/home/kabram/temp/ma_expts/alice/trks/cx_GMR_SS00030_CsChr_RigC_20150826T144616_bbox_idwts.p'
# out_files = ['/groups/branson/home/kabram/temp/a.trk']
# mov_files = ['/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00030_CsChr_RigC_20150826T144616/movie.ufmf']
# prev_linked = ['/groups/branson/home/kabram/temp/ma_expts/alice/trks/cx_GMR_SS00030_CsChr_RigC_20150826T144616_bbox.trk']

# alice social flies

# trk_pure_files  =['/groups/branson/home/kabram/temp/ma_expts/alice/trks/nochr_TrpA65F12_Unknown_RigA_20201212T163531_grone_tracklet.trk']
# trk_pure_files = ['/groups/branson/home/kabram/temp/ar_flytracker_purelinked.trk']
# trk_pure_files = ['/groups/branson/home/kabram/temp/ar_flytracker.trk']
# id_wts = '/groups/branson/home/kabram/temp/ma_expts/alice/trks/nochr_TrpA65F12_Unknown_RigA_20201212T163531_grone_idwts.p'
# id_wts = None
# id_wts = '/groups/branson/home/kabram/temp/ar_social_id_grone_idwts.p'
# id_wts = '/groups/branson/home/kabram/temp/ar_social_id_grone_70imsz_idwts.p'
# id_wts = None
# out_files = ['/groups/branson/home/kabram/temp/ar_social_id_grone_grone.trk']
# mov_files = ['/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigA_20201212T163531//movie.ufmf']
# prev_linked = ['/groups/branson/home/kabram/temp/ma_expts/alice/trks/cx_GMR_SS00030_CsChr_RigC_20150826T144616_bbox.trk']


# social movie. ntraj=15 with flytracker
# trk_pure_files = ['/groups/branson/home/kabram/temp/ar_flytracker2_purelinked.trk']
# # trk_pure_files = ['/groups/branson/home/kabram/temp/ar_flytracker2.trk']
# # id_wts = '/groups/branson/home/kabram/temp/ma_expts/alice/trks/nochr_TrpA65F12_Unknown_RigA_20201212T163531_grone_idwts.p'
# id_wts = '/groups/branson/home/kabram/temp/ar_social_id_idwts.p'
# # id_wts = None
# out_files = ['/groups/branson/home/kabram/temp/ar_social_id.trk']
# mov_files = ['/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigB_20201212T163629//movie.ufmf']
#
#
# # another alice social movie. ntraj=20
# mov_files = ['/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigC_20201216T155818/movie.ufmf']
# trk_pure_files = ['/groups/branson/home/kabram/temp/ar_flytracker3_purelinked.trk']
# id_wts = '/groups/branson/home/kabram/temp/ar_social_id_idwts.p'
# # id_wts = None
#
# # all social flies together
#
# trk_pure_files  =['/groups/branson/home/kabram/temp/ar_flytracker_purelinked.trk','/groups/branson/home/kabram/temp/ar_flytracker2_purelinked.trk','/groups/branson/home/kabram/temp/ar_flytracker3_purelinked.trk']
# mov_files = ['/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigA_20201212T163531//movie.ufmf','/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigB_20201212T163629//movie.ufmf','/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigC_20201216T155818/movie.ufmf']
# # id_wts = None
# id_wts = '/groups/branson/home/kabram/temp/ar_social_id_3_movs_idwts.p'
# out_files = ['/groups/branson/home/kabram/temp/ar_social_id_3_movs.trk','/groups/branson/home/kabram/temp/ar_social_id_3_movs_2.trk','/groups/branson/home/kabram/temp/ar_social_id_3_movs_3.trk']
#
# nails
# trk_pure_files  =['/groups/branson/bransonlab/mayank/data/nails/IMG_3123_nail_head_MA_Bottom_Up_tracklet.trk']
# id_wts = '/groups/branson/bransonlab/mayank/data/nails/IMG_3123_nail_head_MA_Bottom_Up_rect_idwts.p'
# out_files = ['/groups/branson/bransonlab/mayank/data/nails/IMG_3123_nail_head_MA_Bottom_Up_rect.trk']
# mov_files = ['/groups/branson/bransonlab/mayank/data/nails/IMG_3123.mp4']
# prev_linked = ['/groups/branson/home/kabram/temp/ma_expts/alice/trks/cx_GMR_SS00030_CsChr_RigC_20150826T144616_grone.trk']


##

fly_movs = ['/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/nochr_TrpA65F12_Unknown_RigC_20201216T164812',
            '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/socialCsChr_JHS_BDPAD_BDPDBD_CsChrimson_RigA_20190910T152328',
            '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/socialCsChr_JHS_BDPAD_BDPDBD_CsChrimson_RigC_20190910T152823',
            '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/nochr_TrpA65F12_Unknown_RigD_20201216T155952',
            '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/nochr_TrpA65F12_Unknown_RigD_20201216T175902',
            '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/nochr_TrpA71G01_Unknown_RigC_20201216T153727']

cur_mov = fly_movs[0]
mov_files = [cur_mov + '/movie.ufmf']
out_root = cur_mov.replace('/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/','/groups/branson/home/kabram/temp/ma_expts/alice/trks/')
out_root = out_root.replace('/movie.ufmf','_grone')
trk_pure_files = [out_root + '_tracklet.trk']
id_wts = out_root + '_idwts.p'
out_files = [out_root + '.trk']

trk_pure_files = ['/groups/branson/home/kabram/temp/ma_expts/alice/trks/nochr_TrpA65F12_Unknown_RigC_20201216T164812_grone_tracklet.trk']
mov_files = ['/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/nochr_TrpA65F12_Unknown_RigC_20201216T164812/movie.ufmf']
out_files = ['/groups/branson/home/kabram/temp/ma_expts/alice/trks/nochr_TrpA65F12_Unknown_RigC_20201216T164812_grone2.trk']
id_wts = '/groups/branson/home/kabram/temp/ma_expts/alice/trks/nochr_TrpA65F12_Unknown_RigC_20201216T164812_grone_idwts.p'
id_wts = None

from reuse import *
import TrkFile
import link_trajectories as lnk

trk_p = [TrkFile.Trk(tt) for tt in trk_pure_files]

# do id tracking
from poseConfig import conf

for k in trk_p[0].trkData['trkInfo']['params']:
    conf.__dict__[k] = trk_p[0].trkData['trkInfo']['params'][k]

conf.has_trx_file = False
conf.use_bbox_trx = False
conf.use_ht_trx = True
conf.img_dim = 3
conf.trx_align_theta = True
conf.link_id = True
# conf.ht_pts = (0,1)
conf.ht_pts = (0,6)
ww = int(conf.multi_animal_crop_sz/2*1.2)
conf.imsz = [ww,ww]
# conf.imsz = [conf.multi_animal_crop_sz,conf.multi_animal_crop_sz]
# conf.imsz = [70,70]
conf.horz_flip = False
conf.vert_flip = False
conf.link_id_training_iters = 200000
conf.link_id_min_tracklet_len = 6
# trk_p = [lnk.link_pure(tt,conf) for tt in trk_p]
# assert trk_pure_files[0] == '/groups/branson/home/kabram/temp/ma_expts/alice/trks/cx_GMR_SS00030_CsChr_RigC_20150826T144616_grone_pure.trk'
# conf.ht_pts = (0,2)

# conf.imsz = [120,200]
# conf.link_id_training_iters = 200000
conf.link_id_mining_steps = 20

trk_i = lnk.link_id(trk_p, trk_pure_files, mov_files, conf, out_files, id_wts,link_method='motion',save_debug_data=True)

#trk_i[0].save(out_files[0],saveformat='tracklet')
## gt info

gt_trk = '/groups/branson/home/kabram/temp/ar_social_fly_65F12_gt.trk'
gt = TrkFile.Trk(gt_trk)


tt = linked_trks[0]
npure = tt.ntargets
gt_i = np.ones(npure)*np.nan
gt_d = np.ones([npure,2])*np.nan
ss,ee = tt.get_startendframes()
for ndx in range(npure):
    sc = ss[ndx]; ec = ee[ndx]
    p = tt.gettargetframe(ndx,range(sc,ec+1))
    g = gt.getframe(range(sc,ec+1))
    dd = np.sum(np.abs(p[[0,6]]-g[[1,6]]),axis=(0,1,2))
    if dd.min()<(ec-sc+1)*15:
        gt_i[ndx] = dd.argmin()

    ff = dd.copy()/(ec-sc+1)
    ff.sort()
    gt_d[ndx,:] = ff[:2]


gt_s = gt_i[sel_tgt]

## quick stats

trk = trk_i[0]
counts = []
for xx in range(trk.ntargets):
    jj = trk.gettarget(xx)[0, 0, :, 0]
    clen = np.count_nonzero(~np.isnan(jj))
    counts.append(clen)

counts= np.array(counts)

counts

## Run tracking on social fly movies

fly_movs = ['/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/nochr_TrpA65F12_Unknown_RigC_20201216T164812',
            '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/socialCsChr_JHS_BDPAD_BDPDBD_CsChrimson_RigA_20190910T152328',
            '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/socialCsChr_JHS_BDPAD_BDPDBD_CsChrimson_RigC_20190910T152823',
            '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/nochr_TrpA65F12_Unknown_RigD_20201216T155952',
            '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/nochr_TrpA65F12_Unknown_RigD_20201216T175902',
            '/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/nochr_TrpA71G01_Unknown_RigC_20201216T153727']

fly_movs = [f+'/movie.ufmf' for f in fly_movs]

import os
import run_apt_ma_expts as rae_ma
import PoseTools as pt
robj = rae_ma.ma_expt('alice')

loc_file = os.path.join(robj.gt_dir, rae_ma.loc_file_str)
A = pt.json_load(loc_file)

run_type = 'dry'
for cur_mov in fly_movs:
    exp_name = os.path.split(os.path.split(cur_mov)[0])[1]
    out_trk = os.path.join(robj.trk_dir,exp_name + '_grone.trk')
    robj.track(cur_mov,out_trk,t_types=[('grone','crop')],run_type=run_type)
##

all_counts =[]
for cur_mov in fly_movs:
    trk_file = cur_mov
    trk_file = trk_file.replace('/groups/branson/bransonlab/from_tier2/fly_bubble/bubble_data/','/groups/branson/home/kabram/temp/ma_expts/alice/trks/')
    trk_file = trk_file.replace('/movie.ufmf','_grone.trk')
    trk =TrkFile.Trk(trk_file)
    counts = []
    for xx in range(trk.ntargets):
        jj = trk.gettarget(xx)[0, 0, :, 0]
        clen = np.count_nonzero(~np.isnan(jj))
        counts.append(clen)

    counts= np.array(counts)
    all_counts.append(np.sort(counts)[::-1])


## get size for id tracking
trk = '/groups/branson/home/kabram/temp/ar_flytracker_purelinked.trk'
ht_pts = [0,6]
trk = '/groups/branson/home/kabram/temp/20210924_four_female_mice_0_unlabeled_mice_grone_occluded_MA_Bottom_Up_tracklet.trk'
ht_pts = [0,1]
import TrkFile
T = TrkFile.Trk(trk)
import cv2
all_rects = []
all_sz = []
for tt in range(T.T):
    pts = T.getframe(tt)
    pid = T.real_idx(pts)
    pts = pts[...,pid]
    ht = pts[ht_pts]
    ctr = np.mean(ht,axis=0)
    theta = np.arctan2(ht[0, 1] - ht[1, 1], ht[0, 0]-ht[1,0])*180/np.pi
    # plt.figure();
    rects = []
    for idx in range(pts.shape[-1]):
        # plt.scatter(pts[:,0,idx],pts[:,1,idx])
        curp = pts[...,idx]-ctr[None,...,idx]
        rot_mat = cv2.getRotationMatrix2D((ctr[0,idx], ctr[1,idx]), theta[idx]+90, 1.)
        tp = np.matmul(pts[...,idx],rot_mat[:,:2].T)
        tp[...,0] += rot_mat[0,2]
        tp[...,1] += rot_mat[1,2]
        # plt.plot(tp[:,0],tp[:,1])
        minx = min(tp[:,0])-10; maxx = max(tp[:,0])+10
        miny = min(tp[:,1]); maxy = max(tp[:,1])
        all_sz.append([maxx-minx,maxy-miny])
        rect_pts = np.array( [[minx,miny],[minx,maxy],[maxx,maxy],[maxx,miny]])
        rot_mati = cv2.getRotationMatrix2D((ctr[0,idx], ctr[1,idx]), -theta[idx]-90, 1.)
        orig_rect = np.matmul(rect_pts,rot_mati[:,:2].T)
        orig_rect[...,0] += rot_mati[0,2]
        orig_rect[...,1] += rot_mati[1,2]
        rects.append(orig_rect)
        # plt.plot(orig_rect[:,0],orig_rect[:,1])
    all_rects.append(np.array(orig_rect))
    # plt.axis('equal')


## debug
i1 = 1187; i2 = 1189
t1 = np.where(sel_tgt==(i1-1))[0][0]
t2 = np.where(sel_tgt==(i2-1))[0][0]
ts = [t1 , t2]
print(f'End:{sel_ee[t1]}, Start:{sel_ss[t2]}')
sel_d = dist_mat[ts][:,ts]
ims1 = cur_data[t1][0][...,0]
ims2 = cur_data[t2][0][...,0]
c1 = np.where(dist_mat[t1]<close_thresh)[0]
c2 = np.where(dist_mat[t2]<close_thresh)[0]
##
sr1 = np.argsort(sel_ss[c1])
w1 = sel_tgt[c1[sr1]]+1
sr2 = np.argsort(sel_ss[c2])
w2 = sel_tgt[c2[sr2]]+1

i1 = np.array([cur_data[ww][0][...,0] for ww in c1[sr1]])
i2 = np.array([cur_data[ww][0][...,0] for ww in c2[sr2]])


##

selc = c2
id1 = 2

hh = sel_ee[selc]- sel_ss[selc]

selpred = preds[selc]
dcmat = np.linalg.norm(selpred[None,:,None]-selpred[:,None,:,None],axis=-1)
c1_id1 = np.where( (gt_s[selc]==id1) & (hh>50))[0]
c1_nid1 = np.where((gt_s[selc]!=id1)&~np.isnan(gt_s[selc])&(hh>50))[0]
d1 = dcmat[c1_id1][:,c1_nid1]
[a1,a2,b1,b2] = np.where(d1<0.1)

##
ni = len(a1)

rt = np.random.randint(ni)

x1 = c1_id1[a1[rt]]
x2 = c1_nid1[a2[rt]]
i1 = cur_data[selc[x1]][0][...,0]
i2 = cur_data[selc[x2]][0][...,0]

print(hh[x1],hh[x2])
# ipair = np.concatenate([i1[b1[rt]][None],i2[b2[rt]][None]],axis=0)
# pt.show_stack(ipair,1,2)

#
selid = np.where(gt_s==id1)[0]
y1 = selc[x1]
z1 = b1[rt]
mndx = np.where(selid==y1)[0]
selid = np.delete(selid,mndx)

dsmat = np.linalg.norm(preds[None,y1,None,z1]-preds[selid],axis=-1)
[v1,u1] = np.where(dsmat>1.6)
ns = len(v1)
ut = np.random.randint(ns)

g1 = selid[v1[ut]]
s1 = i1[z1]
s2 = cur_data[g1][0][u1[ut],...,0]
itrip = np.concatenate([i1[b1[rt]][None],s2[None],i2[b2[rt]][None]],axis=0)
pt.show_stack(itrip,1,3)

##
import movies

id_wts = '/groups/branson/home/kabram/temp/ar_social_id_3_movs_idwts.p'
net = lnk.load_id_wts(id_wts)
cap = movies.Movie(mov_files[0])
gt_i = []
gt_trx = apt.get_trx_info(gt_trk,conf,max(gt.endframes),use_ht_pts=True)
gt_trx = gt_trx['trx']
for ix in range(gt.ntargets):
    jj = [[i, ix] for i in range(0, 45011, 300)]
    jx = apt.create_batch_ims(jj, conf, cap, False, gt_trx, None, use_bsize=False)
    gt_i.append(jx)

gt_p = []
for ix in range(gt.ntargets):
    px = lnk.tracklet_pred([gt_i[ix]], net, conf, rescale=1)
    gt_p.append(px[0])

## motion linking stats

tndx = 0
st, en = linked_trks[tndx].get_startendframes()

tt = linked_trks[tndx]
maxn = max(en)
aq = []
info = []
for ndx in range(tt.ntargets):
    if st[ndx]==0 or en[ndx]==maxn: continue
    if st[ndx]==en[ndx]: continue
    if np.isnan(gt_i[ndx]): continue
    pt = np.where(en==st[ndx]-1)[0]
    if pt.size==0:
        aq.append(ndx)
        continue
    p1 = tt.gettargetframe(ndx,st[ndx])[...,0,0]
    p2 = tt.gettargetframe(ndx,st[ndx]+1)[...,0,0]
    vmag = np.linalg.norm(p1-p2,axis=-1).mean()
    p3 = tt.getframe(st[ndx]-1)[...,0,pt]
    mpred = p1-(p2-p1)
    dpred = np.linalg.norm(p3-mpred[...,None],axis=1).mean(axis=0)
    match_ndx = np.nanargmin(dpred)
    match_ndx_id = pt[match_ndx]
    if np.isnan(gt_i[pt[match_ndx]]): continue
    match_dist = dpred[match_ndx]
    d2 = dpred.copy()
    d2[match_ndx] = np.nan
    if np.all(np.isnan(d2)):
        match_ndx2_id = np.nan
        m2 = np.nan
    else:
        match_ndx2 = np.nanargmin(d2)
        m2 = d2[match_ndx2]
        match_ndx2_id = pt[match_ndx2]
    info.append([ndx,match_ndx_id,match_dist,vmag,gt_i[ndx],gt_i[pt[match_ndx]],match_ndx2_id,m2])

info = np.array(info)

## pure linking with motion

# after assign_ids in link_pure in link_trajectories

mpred_stats = []
for ndx in range(2000):
    ix = np.random.randint(int(T)-3)
    pp = trk.getframe(np.arange(ix,ix+3))
    ii = ids.getframe(np.arange(ix,ix+3))[0]
    for i in ii[0]:
        if i == -1: continue
        ixx = np.where(ii==i)
        if len(ixx[0])< 3: continue
        sp = pp[...,ixx[0],ixx[1]]
        vmag = np.linalg.norm(sp[...,0]-sp[...,1],axis=1).mean(axis=0)
        mpred = 2*sp[...,1]-sp[...,0]
        merror = np.linalg.norm(sp[...,2]-mpred,axis=1).mean(axis=0)
        ixx_others = [nxx for nxx,inx in enumerate(ii[2]) if inx not in [-1,i]]
        sp_others = pp[...,2,ixx_others]
        merror_others = np.linalg.norm(sp_others - mpred[...,None],axis=1).mean(axis=0).min()
        mpred_stats.append([vmag,merror,ix,i,merror_others])

mpred_stats = np.array(mpred_stats)
vel_mag_eps = np.percentile(mpred_stats[:,0],90)/10
pred_error_thresh = np.percentile(mpred_stats[:,1]/(mpred_stats[:,0]+vel_mag_eps),75)

def check_motion_link(p1,p2,p3,vel_mag_eps,pred_error_thresh):
    vmag = np.linalg.norm(p1 - p2, axis=1).mean(axis=0)
    mpred = 2 * p1 - p2
    d2pred = np.linalg.norm(mpred[..., None] - p3, axis=1).mean(axis=0)
    match_ndx = None
    if d2pred.min() / (vmag + vel_mag_eps) < pred_error_thresh:
        if d2pred.size == 1:
            # if there is only one, then link
            match_ndx = 0
        else:
            zx = d2pred.copy()
            zx.sort()
            if (zx[1] / zx[0]) > 2:
                # link only if the next closest match is more than 2x away
                match_ndx = np.argmin(d2pred)
    return match_ndx, d2pred

def merge_ids(id1,id2,t0s,t1s,ids):
    use_ndx, overwrite_ndx = [id1, id2] if id1 > id2 else [id2, id1]
    idx_m = ids.where(overwrite_ndx)
    for mx in zip(*idx_m):
        ids.settargetframe(use_ndx, mx[0], mx[1])
    t0s[overwrite_ndx] = -100
    t1s[overwrite_ndx] = -100
    t0s[use_ndx] = min(t0s[overwrite_ndx], t0s[use_ndx])
    t1s[use_ndx] = max(t1s[overwrite_ndx], t1s[use_ndx])
    return use_ndx

cur_ndx = 0
mcount = 0
zcount = 0
fcount = 0
rcount = 0
pqq = []
zqqf = []
zqqr = []
while cur_ndx<nids:
    pqq.append(cur_ndx)
    use_ndx = cur_ndx
    matched = False
    if (t0s[use_ndx]==t1s[use_ndx]) or (t0s[use_ndx]<0):
        cur_ndx +=1
        continue
    idx = ids.where(use_ndx)
    if not np.all(np.diff(idx[1]) == 1):
        idx = list(idx)
        ff = np.argsort(idx[1])
        idx[1] = idx[1][ff]
        idx[0] = idx[0][ff]
    assert(np.all(np.diff(idx[1]) == 1))
    assert(np.min(idx[1])==t0s[cur_ndx])
    assert(np.max(idx[1])==t1s[cur_ndx])
    etrks = np.where(t1s==(t0s[use_ndx]-1))[0]
    strks = np.where(t0s==(t1s[use_ndx]+1))[0]
    if len(etrks)==0 and len(strks)==0:
        zcount+=1
    if len(etrks)>0:
        p1 = trk.gettargetframe(idx[0][0],idx[1][0])[...,0,0]
        p2 = trk.gettargetframe(idx[0][1],idx[1][1])[...,0,0]
        p3 = trk.getframe(t0s[cur_ndx]-1)[...,0,:]
        ids_prev = ids.getframe(t0s[cur_ndx]-1)[0,0]
        sel_prev = [ndx for ndx,ii in enumerate(ids_prev) if ii in etrks]
        assert(len(sel_prev)==len(etrks))
        p3 = p3[...,sel_prev]

        match_idx, z = check_motion_link(p1,p2,p3,vel_mag_eps,pred_error_thresh)
        zqqf.append(z.min())
        if match_idx is not None:
            print('Extending in front')
            match_ndx = ids_prev[sel_prev[match_idx]]
            use_ndx = merge_ids(use_ndx,match_ndx,t0s,t1s,ids)
            matched = True
            mcount += 1
        else:
            fcount +=1
    if len(strks)>0:
        p1 = trk.gettargetframe(idx[0][-1],idx[1][-1])[...,0,0]
        p2 = trk.gettargetframe(idx[0][-2],idx[1][-2])[...,0,0]
        p3 = trk.getframe(t1s[cur_ndx]+1)[...,0,:]
        ids_prev = ids.getframe(t1s[cur_ndx]+1)[0,0]
        sel_prev = [ndx for ndx,ii in enumerate(ids_prev) if ii in strks]
        assert(len(sel_prev)==len(strks))
        p3 = p3[...,sel_prev]

        match_idx,z = check_motion_link(p1,p2,p3,vel_mag_eps,pred_error_thresh)
        zqqr.append(z.min())
        if match_idx is not None:
            print('Extending in rear')
            match_ndx = ids_prev[sel_prev[match_idx]]
            use_ndx = merge_ids(use_ndx,match_ndx,t0s,t1s,ids)
            matched = True
            mcount += 1
        else:
            rcount +=1

    if matched:
        assert(use_ndx>=cur_ndx)
        if use_ndx>cur_ndx:
            cur_ndx += 1
            print(f'{cur_ndx}')
    else:
        cur_ndx+=1






## debugging nails

ims = [dd[0] for dd in data]
rndx = np.random.choice(output1.shape[0])
indx = data_info[rndx,0,1]
mm1 = process_id_ims(ims[indx],confd,False,rescale)
ix1 = data_info[rndx,0,2]
ix2 = data_info[rndx,0,3]
nn1 = torch.tensor(mm1.astype('float32')).cuda()
oo1 = net(nn1)
##
rr1 = output1.detach().cpu().numpy()
rr2 = output2.detach().cpu().numpy()
pp1 = np.concatenate([rr1[None,rndx],rr2[None,rndx]],axis=0)
pp2 = oo1[[ix1,ix2],:].detach().\
    cpu().numpy()
dd1 = np.linalg.norm(pp1-pp2,axis=-1)
dd2 = np.linalg.norm(pp1[0]-pp1[1])




##
trk = TrkFile.Trk(prev_linked[0])
trk_new = TrkFile.Trk(out_files[0])
missing = []
tlen = []
for xx in range(trk.ntargets):
    jj = trk.gettarget(xx)[0, 0, :, 0]
    jj_n = trk_new.gettarget(xx)[0, 0, :, 0]
    clen = np.count_nonzero(~np.isnan(jj))
    clen_n = np.count_nonzero(~np.isnan(jj_n))
    qq = np.ones(jj.shape[0]+2)>0
    qq[1:-1] = np.isnan(jj)

    qq1 = np.where(   qq[:-2]   & (~qq[1:-1]) )[0]
    qq2 = np.where( (~qq[1:-1]) &   qq[2:] )[0]

    missing.extend(list(zip(qq2[:-1],qq1[1:])))
    tlen.append([clen,clen_n])

    kk = trk_new.gettarget(xx)[..., 0]
    dd = np.abs(np.diff(kk, axis=-1)).sum(axis=(0, 1))
    sel = np.isnan(jj[:-1]) & ~np.isnan(dd)
    sel_idx = np.where(sel)[0]
    sort_idx = np.argsort(dd[sel_idx])[::-1]


# alice data
# total # detections in pure - 439057
# total # detections in prev id - 426014. top 10 417000
# total # detections in new id -- 426440 top 10 417190

# Roian data
# pure - 216000
# total # detections in prev id -- 214847, top 2 214817
# total # detections in new id -- 215544, top 2 215514

##
import movies
import TrkFile
from tqdm import tqdm
tp = TrkFile.Trk(trk_pure_files[0])
ti = TrkFile.Trk(out_files[0])
st,en = tp.get_startendframes()
nmax = max(en)
mov = movies.Movie(mov_files[0])

def get_valid(fr):
    valid = ~np.isnan(fr[0,0])
    return fr[:,:,valid]


mismatch = []
for ndx in tqdm(range(nmax)):
    fr1 = tp.getframe(ndx)[:,:,0]
    fr1 = get_valid(fr1)
    fr2 = ti.getframe(ndx)[:,:,0]
    fr2 = get_valid(fr2)

    if fr2.shape[-1] != fr1.shape[-1]:
        if (fr2.shape[-1] ==0) or(fr1.shape[-1]==0): continue
        dd = np.sum(np.abs(fr1[...,None]-fr2[...,None,:]),axis=(0,1))
        miss = np.where(dd.min(axis=1)>0)[0]
        mismatch.append([ndx,miss])

##

from reuse import *
sel = np.random.choice(len(mismatch),5)
f,ax = plt.subplots()
for ix in sel:
    ii,cc = mismatch[ix]
    ax.clear()
    fr = mov.get_frame(ii)[0]
    ax.imshow(fr,'gray')

    fr1 = tp.getframe(ii)[:,:,0]
    fr1 = get_valid(fr1).transpose([2,0,1])
    msctr(fr1[cc],ax=ax)
    plt.title(f'{ix}, {ii}')
    plt.waitforbuttonpress()

##

aa = TrkFile.Trk('/groups/branson/home/kabram/temp/ma_expts/alice/trks/cx_GMR_SS00030_CsChr_RigC_20150826T144616_grone_pure.trk')
bb = TrkFile.Trk('/groups/branson/home/kabram/temp/a.trk')

st,en = bb.get_startendframes()
maxn = max(en)

diff = []
for i in range(5000):
    jj = aa.getframe(i)[0, 0, 0,:]
    jj_n = bb.getframe(i)[0, 0, 0,:]
    n1 = np.count_nonzero(~np.isnan(jj))
    n2 = np.count_nonzero(~np.isnan(jj_n))
    if n1!= n2:
        diff.append(i)


## 08092022

tt = TrkFile.Trk('/groups/branson/home/kabram/temp/ma_expts/alice/trks/cx_GMR_SS00030_CsChr_RigC_20150826T144616_bbox.trk')
ttp =  TrkFile.Trk('/groups/branson/home/kabram/temp/ma_expts/alice/trks/cx_GMR_SS00030_CsChr_RigC_20150826T144616_bbox_tracklet.trk')
##

ss, ee = tt.get_startendframes()

tinfo = []
tlen = []
gaps = []
dist =[]
sz = []
for xx in range(tt.ntargets):
    pp = tt.gettarget(xx)
    jj = pp[0, 0, :, 0]
    clen = np.count_nonzero(~np.isnan(jj))
    qq = np.ones(jj.shape[0]+2)>0
    qq[1:-1] = np.isnan(jj)

    qq1 = np.where(   qq[:-2]   & (~qq[1:-1]) )[0]
    qq2 = np.where( (~qq[1:-1]) &   qq[2:] )[0]
    rr = qq1[1:]-qq2[:-1]-1
    gaps.extend(rr.tolist())
    tlen.append(clen)
    for ndx in range(rr.size):
        a1 = pp[:,:,qq1[ndx+1],0]
        a2 = pp[:,:,qq2[ndx],0]
        dd = a1-a2
        dd = np.linalg.norm(dd,axis=1).mean()
        sz1 = a1.max(axis=0)-a1.min(axis=0)
        sz1 = np.mean(sz1)
        sz2 = a2.max(axis=0)-a2.min(axis=0)
        sz2 = np.mean(sz2)
        csz = (sz1+sz2)/2*rr[ndx]
        if rr[ndx]<4:
            dist.append(dd)
            sz.append(csz)

gaps = np.array(gaps)
dist = np.array(dist)
sz = np.array(sz)


## finding linking costs

import tqdm
thresh = 14
nint = 30
mult = 2
ln_costs = []
frs = []
npts = tt.pTrk.data[0].shape[0]
sf = tt.startframes
ef = tt.endframes
for ndx in tqdm.tqdm(range(tt.ntargets)):
    pp = tt.pTrk.data[ndx]
    dd = np.abs(np.diff(pp,axis=-1)).sum(axis=(0,1))/npts
    ln_costs.append(dd)
    zz1 = np.where(dd>thresh)[0]
    zz = zz1+tt.startframes[ndx]
    for zndx in range(len(zz)):
        z = zz[zndx]
        gg = tt.getframe(np.arange(z-nint,z+nint))
        gg[:,:,:,ndx] = np.nan
        curp = pp[:,:,zz1[zndx]]
        dd_s = np.abs(curp[:,:,None,None]-gg).sum(axis=(0,1))/npts
        if np.any(dd_s<mult*thresh):
            kk1,kk2 = np.where(dd_s<mult*thresh)
            for kndx in range(len(kk2)):
                kid = kk2[kndx]
                if sf[ndx]>ef[kid]: continue
                if ef[ndx]<sf[kid]: continue
                frs.append([ndx+1,z+1,kk1[kndx]-nint])


frs = np.array(frs)
aa = np.argsort(frs[:,1])
frs = frs[aa]
