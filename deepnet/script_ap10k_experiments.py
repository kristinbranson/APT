# script to train and evaluate all the networks on AP10K and AP36K datasets

## For individual animals

import ap36_train as a36
import glob
import re
from reuse import *
import cv2

ap36 = False
if ap36:
    im_dir = '/groups/branson/bransonlab/datasets/APT-36K/data'

    json_files = glob.glob('/groups/branson/bransonlab/datasets/APT-36K/annotations/apt36k_annotations_fixed_train-*.json')
    bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/ap36k'
    animals = [re.search('apt36k_annotations_fixed_train-(.*).json', jf).group(1) for jf in json_files]
else:

    json_files = glob.glob('/groups/branson/bransonlab/datasets/ap-10k/annotations/ap10k-train-split1-*.json')
    im_dir = '/groups/branson/bransonlab/datasets/ap-10k/data'
    bdir = '/groups/branson/bransonlab/mayank/apt_cache_2/ap10k'
    animals = [re.search('ap10k-train-split1-(.*).json', jf).group(1) for jf in json_files]

# grab match patterns from the json files
animals = ['-'+a for a in animals]
animals.append('')

nets = [
        ['mdn_joint_fpn','mdn',{}],
        ['mdn_joint_fpn','mdn_hrnet',{'mdn_use_hrnet':True}],
        ['mmpose','hrnet',{'mmpose_net':'hrnet'}],
         ['mmpose','mspn',{'mmpose_net':'mspn'}],
        ['mmpose','hrformer',{'mmpose_net':'hrformer'}],
        ]


## create datasets for each animal

for animal in animals:
    if ap36:
        cur_train = f'/groups/branson/bransonlab/datasets/APT-36K/annotations/apt36k_annotations_fixed_train{animal}.json'
        cur_val = f'/groups/branson/bransonlab/datasets/APT-36K/annotations/apt36k_annotations_fixed_val{animal}.json'
        cur_test = f'/groups/branson/bransonlab/datasets/APT-36K/annotations/apt36k_annotations_fixed_test{animal}.json'
    else:
        cur_train = f'/groups/branson/bransonlab/datasets/ap-10k/annotations/ap10k-train-split1{animal}.json'
        cur_val = f'/groups/branson/bransonlab/datasets/ap-10k/annotations/ap10k-val-split1{animal}.json'
        cur_test = f'/groups/branson/bransonlab/datasets/ap-10k/annotations/ap10k-test-split1{animal}.json'

    V = pt.json_load(cur_train)
    n_train = len(V['annotations'])
    n_steps = int(np.ceil(n_train/64*210./1000)*1000)
    cura = animal[1:] if animal else ''
    exp_dir = f'{bdir}/ap36k_topdown_{cura}'
    conf = a36.create_conf_sa({'dl_steps':n_steps,'batch_size':64,},exp_dir,'',expname=f'ap36k_{cura}')

    a36.create_dataset_sa(conf,im_dir,cur_train,'train',force=False)


    a36.create_dataset_sa(conf,im_dir,cur_val,'val',force=True)
    a36.create_dataset_sa(conf,im_dir,cur_test,'test',force=True)
    if False:
        vv = os.path.join(conf.cachedir,'test_TF.json')
        vv = pt.json_load(vv)
        ex = np.random.choice(len(vv['images']))
        im = cv2.imread(vv['images'][ex]['file_name'])
        pts = np.array(vv['annotations'][ex]['keypoints']).reshape(-1,3)
        plt.figure()
        plt.imshow(im)
        plt.scatter(pts[:,0],pts[:,1],c='r')

## train on all animals

# nets = [
#         # ['mdn_joint_fpn','mdn',{}],
#         # ['mdn_joint_fpn','mdn_hrnet',{'mdn_use_hrnet':True}],
#         # ['mmpose','hrnet',{'mmpose_net':'hrnet'}],
#         #  ['mmpose','mspn',{'mmpose_net':'mspn'}],
#         ['mmpose','hrformer',{'mmpose_net':'hrformer'}],
#         ]

for animal in animals:
    if ap36:
        cur_train = f'/groups/branson/bransonlab/datasets/APT-36K/annotations/apt36k_annotations_fixed_train{animal}.json'
    else:
        cur_train = f'/groups/branson/bransonlab/datasets/ap-10k/annotations/ap10k-train-split1{animal}.json'

    V = pt.json_load(cur_train)
    n_train = len(V['annotations'])
    n_steps = int(np.ceil(n_train/64*210./1000)*1000)
    cura = animal[1:] if animal else ''
    exp_dir = f'{bdir}/ap36k_topdown_{cura}'
    for net in nets:

        bdict = {'dl_steps':n_steps,'batch_size':64}
        bdict.update(net[2])
        conf = a36.create_conf_sa(bdict,exp_dir,'',expname=f'ap36k_{animal}')
        a36.train_bsub(conf,net[0],net[1],net[1]+animal)

## get status of all animals


iters = []
for animal in animals:
    cura = animal[1:] if animal else ''
    exp_dir = f'{bdir}/ap36k_topdown_{cura}'

    for net in nets:
        exp_str = exp_dir+f'/{net[1]}'
        st = a36.get_status(net[1]+animal,exp_str)
        if (not st[0]) and st[2]<0:
            print(st)
        iters.append([st[2],net[1],cura])


## get results for all animals
res = []
for animal in animals:
    try:
        cura = animal[1:] if animal else ''
        exp_dir = f'{bdir}/ap36k_topdown_{cura}'
        resa = []
        for net in nets:
            tdata_str = exp_dir+f'/ap36k_{animal}_{net[1]}_traindata'
            V = pt.pickle_load(tdata_str)
            perf = a36.compute_perf_sa(V[1],net[0],exp_dir+'/test_TF.json',net[1],exp_dir)
            resa.append([net[1],perf])
        res.append([animal,resa])
    except:
        pass

ms = []
ans = []
for resa in res:
    cur_ms = []
    ans.append(resa[0])
    for r in resa[1]:
        cur_ms.append(r[1][0][0])
    ms.append(cur_ms)

res_dict = {}
net_names = [n[1] for n in nets]
for mm,aa in zip(ms,ans):
    mstr = ','.join([f'{m:.2f}' for m in mm])
    print(f'{aa:10s}, {mstr}')
    if aa.startswith('-'):
        an = aa[1:]
    else:
        an = 'all'
    res_dict[an] = {}
    for ndx,nn in enumerate(net_names):
        res_dict[an][nn] = mm[ndx]


## save results as pickle file
import pickle
astr = 'ap36' if ap36 else 'ap10'
with open(f'/groups/branson/bransonlab/mayank/apt_results/{astr}_results.pkl','wb') as f:
    pickle.dump(res_dict,f)

##
import PoseTools as pt
astr = 'ap36' if ap36 else 'ap10'
res_dict = pt.pickle_load(f'/groups/branson/bransonlab/mayank/apt_results/{astr}_results.pkl')
net_names = [n[1] for n in nets]
print(','.join(net_names))
for k,v in res_dict.items():
    mstr = [f'{v[m]:.2f}' for m in net_names]
    mstr = ','.join(mstr)
    print(f'{k:10s}, {mstr}')




## accuracy vs occlusion plot

counts = []
sela = [r[0] for r in res]
for animal in animals:
    if animal not in sela: continue
    cura = animal[1:] if animal else ''
    exp_dir = f'{bdir}/ap36k_topdown_{cura}'
    J = pt.json_load(exp_dir+'/train_TF.json')
    jj = np.array([jk['keypoints'] for jk in J['annotations']]).reshape([-1,17,3])
    counts.append(np.mean(jj[:,:,2]>0))

ss = np.array([ [rr[1][0]  for rr in r1[1]] for r1 in res])
plt.figure();plt.scatter(ss[:,0,0],counts)


## accuracy of occlusion prediction
# doing it only for all animals

occ_pred = []
ll = resa[0][1][-2]
oo_l = np.isnan(ll[:, :, 0])
for ndx,net in enumerate(nets):
    oo_p = resa[ndx][1][-3]
    occ_pred.append(oo_p)

##
f,ax = plt.subplots(1,len(nets),figsize=(15,5))
nbin = 20
ww = 0.4
for ndx,net in enumerate(nets):
    oo_p = occ_pred[ndx]
    if 'mdn' not in net[1]:
        oo_p = -oo_p
    oo_p = np.clip(oo_p,np.percentile(oo_p,5),np.percentile(oo_p,95))
    histc1,bin_edges = np.histogram(oo_p[oo_l],bins=nbin,density=True)
    histc2,bin_edges2 = np.histogram(oo_p[~oo_l],bins=bin_edges,density=True)
    acc = (histc1[nbin//2:].sum()+histc2[:nbin//2].sum())/histc1.sum()/2
    ax[ndx].bar(np.arange(0,1,1/nbin),histc1/20,width=ww/nbin)
    ax[ndx].bar(np.arange(0,1,1/nbin)+ww/nbin,histc2/20,width=ww/nbin)
    ax[ndx].set_title(f'{net[1]}, {acc:.2f}')
    ax[ndx].set_ylim([0,0.75])
    ax[ndx].axis('on')
    ax[ndx].tick_params(axis='y', which='both', left=False, right=False, labelleft=True)
    ax[ndx].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if ndx>0:
        ax[ndx].set_yticks([])

ax[0].axis('on')
ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=True)
ax[0].set_ylabel('Density')
ax[0].set_xlabel('Occlusion prediction')
ax[-1].legend(['Occluded','Not occluded'])


## mAP as function of x offset
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

assert not ap36

animal = ''
cura = animal[1:] if animal else ''
exp_dir = f'{bdir}/ap36k_topdown_{cura}'
valfile = exp_dir + '/test_TF.json'
J = pt.json_load(valfile)
kk = np.array([jj['keypoints'] for jj in J['annotations']]).reshape([-1, 17, 3])
np.where(np.all(kk[:, :, 2] > 0, axis=1))
J['annotations'] = J['annotations'][40:41]
J['images'] = J['images'][40:41]
import json

out_file_l = '/groups/branson/home/kabram/temp/a10k_single_animal_test.json'
with open(out_file_l, 'w') as f:
    json.dump(J, f)

Jo = J
kk_i = np.array(Jo['annotations'][0]['keypoints']).reshape([17, 3])
out_file_p = '/groups/branson/home/kabram/temp/a10k_single_animal_pred.json'
aps = []
sigmas = [0.025, 0.025, 0.026, 0.035, 0.035, 0.079, 0.072, 0.062, 0.079, 0.072, 0.062, 0.107, 0.087, 0.089, 0.107,
          0.087, 0.089]
aa = Jo['annotations'][0]['area']
mm = np.sqrt( (np.array(sigmas)*2)**2*(aa*2))/10

for x in np.arange(0, 10,0.25):
    Jx = pt.json_load(out_file_l)
    kk = np.array(J['annotations'][0]['keypoints']).reshape([17, 3])
    kk[:,0] += x*mm
    Jx['annotations'][0]['keypoints'] = kk.flatten().tolist()
    Jx['annotations'][0]['score'] = 1.
    with open(out_file_p, 'w') as f:
        json.dump(Jx, f)


    coco_gt = COCO(out_file_l)
    coco_gt.loadAnns()

    coco_dt = COCO(out_file_p)
    coco_dt.loadAnns()

    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    # coco_eval.params.kpt_oks_sigmas = np.array(sigmas)
    coco_eval.sigmas = np.array(sigmas)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap = coco_eval.eval['precision'][:,:,0,0,0].mean()
    aps.append(ap)

##
plt.figure(32)
plt.clf()
plt.plot(np.arange(0, 10,0.25),aps)
plt.xlabel('offset')
plt.ylabel('mAP')
plt.savefig('/groups/branson/home/kabram/temp/mAP_offset.png',bbox_inches='tight',pad_inches=0)
plt.figure(34)
plt.clf()
J = pt.json_load(valfile)
ii = cv2.imread(J['images'][40]['file_name'])
ii = cv2.cvtColor(ii,cv2.COLOR_BGR2RGB)
plt.imshow(ii)
skel = np.array( [[1, 2],
  [1, 3],
  [2, 3],
  [3, 4],
  [4, 5],
  [4, 6],
  [6, 7],
  [7, 8],
  [4, 9],
  [9, 10],
  [10, 11],
  [5, 12],
  [12, 13],
  [13, 14],
  [5, 15],
  [15, 16],
  [16, 17]])-1
dskl(kk_i[:,:2],skel,cc='r')
kk = kk_i[:,:2].copy()
kk[:,0] += 2.25*mm
dskl(kk,skel,cc='b')
kk = kk_i[:,:2].copy()
kk[:,0] += 5.75*mm
dskl(kk,skel,cc='g')
kk = kk_i[:,:2].copy()
kk[:,0] += 8.5*mm
dskl(kk,skel,cc='y')
plt.axis('off')

## scale vs mAP
J = pt.json_load('/groups/branson/bransonlab/datasets/ap-10k/annotations/ap10k-train-split1.json')
skel = np.array(J['categories'][0]['skeleton'])-1
ii =np.array([jj['id'] for jj in J['images']])
aa = np.array([jj['image_id'] for jj in J['annotations']])
ar = np.array([jj['area'] for jj in J['annotations']])
all_a = np.array([jj['keypoints'] for jj in J['annotations']]).reshape([-1,17,3])
n_occ = np.sum(all_a[:,:,2]==0,axis=1)
rr = []
for ix in ii:
    ss = np.where(aa==ix)[0]
    rr.append(ar[ss].max()/ar[ss].min())

sr = np.argsort(rr)[::-1]
selndx = 3122#65
fn = os.path.join('/groups/branson/bransonlab/datasets/ap-10k/data',J['images'][selndx]['file_name'])
im = cv2.imread(fn)
im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
plt.figure(42)
plt.clf()
plt.imshow(im)
cura = []
ss = np.where(aa==ii[selndx])[0]
for s in ss:
    cur = J['annotations'][s]
    cur = np.array(cur['keypoints']).reshape([17,3])
    cura.append(cur)
cura = np.array(cura).astype('float32')
cura[cura[:,:,2]==0,:] = np.nan
mdskl(cura,skel,cc='r')
plt.axis('off')
print(np.sqrt(rr[selndx]))
ratio = 481815/71448
plt.savefig('/groups/branson/home/kabram/temp/mAP_scale.png',bbox_inches='tight',pad_inches=0)

## occlusion vs mAP

J = pt.json_load('/groups/branson/bransonlab/datasets/ap-10k/annotations/ap10k-train-split1.json')
skel = np.array(J['categories'][0]['skeleton'])-1
ii =np.array([jj['id'] for jj in J['images']])
aa = np.array([jj['image_id'] for jj in J['annotations']])
ar = np.array([jj['area'] for jj in J['annotations']])
all_a = np.array([jj['keypoints'] for jj in J['annotations']]).reshape([-1,17,3]).astype('float32')
n_occ = np.sum(all_a[:,:,2]==0,axis=1)
all_a[all_a[:,:,2]==0,:2] = np.nan
sr = np.argsort(n_occ)[::-1]
selndx = 1220
indx = np.where(ii==aa[selndx])[0][0]
fn = os.path.join('/groups/branson/bransonlab/datasets/ap-10k/data',J['images'][indx]['file_name'])
im = cv2.imread(fn)
im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
plt.figure(43)
plt.clf()
plt.imshow(im)
dskl(all_a[selndx,:,:2],skel,cc='r')
sctr(all_a[selndx,:,:2],marker='o',color='r')
plt.axis('off')
plt.savefig('/groups/branson/home/kabram/temp/mAP_occlusion.png',bbox_inches='tight',pad_inches=0)

#
plt.figure(44)
plt.clf()
pt_occ = np.sum(all_a[:,:,2]==0,axis=0)/len(all_a)
plt.plot(pt_occ)
plt.xticks(np.arange(all_a.shape[1]),J['categories'][0]['keypoints'],rotation=45,ha='right')
plt.subplots_adjust(bottom=0.3)
plt.xlabel('Landmark')
plt.ylabel('Fraction occluded')
plt.savefig('/groups/branson/home/kabram/temp/occlusion_fraction.png',bbox_inches='tight',pad_inches=0)

## Occlusion in APT36K
J = pt.json_load('/groups/branson/bransonlab/datasets/APT-36K/annotations/apt36k_annotations_fixed_train.json')
skel = np.array(J['categories'][0]['skeleton'])-1
ii =np.array([jj['id'] for jj in J['images']])
aa = np.array([jj['image_id'] for jj in J['annotations']])
ar = np.array([jj['area'] for jj in J['annotations']])
all_a = np.array([jj['keypoints'] for jj in J['annotations']]).reshape([-1,17,3]).astype('float32')
n_occ = np.sum(all_a[:,:,2]==0,axis=1)
plt.figure(45)
plt.clf()
pt_occ = np.sum(all_a[:,:,2]==0,axis=0)/len(all_a)
plt.plot(pt_occ)
plt.xticks(np.arange(all_a.shape[1]),J['categories'][0]['keypoints'],rotation=45,ha='right')
plt.subplots_adjust(bottom=0.3)
plt.xlabel('Landmark')
plt.ylabel('Fraction occluded')
plt.title('APT36K')
plt.savefig('/groups/branson/home/kabram/temp/occlusion_fraction_apt36k.png',bbox_inches='tight',pad_inches=0)

##

# NOTES: For code to train using mmpose, check /groups/branson/bransonlab/mayank/code/AP-10K/run_test.py (use pycharm 2023.1.2 to open the project)

# !!!!!!!!!!!!!!!!!!!         TOP DOWN               !!!!!!!!!!!!!!!!!!!!!!!!


## AP 10K
# json_file = '/groups/branson/bransonlab/datasets/ap-10k/annotations/ap10k-train-split1-dog.json'
# test_json = '/groups/branson/bransonlab/datasets/ap-10k/annotations/ap10k-val-split1-dog.json'

json_file = '/groups/branson/bransonlab/datasets/ap-10k/annotations/ap10k-train-split1.json'
test_json = '/groups/branson/bransonlab/datasets/ap-10k/annotations/ap10k-val-split1.json'

im_dir = '/groups/branson/bransonlab/datasets/ap-10k/data'
exp_dir = '/groups/branson/bransonlab/mayank/apt_cache_2/ap10k_topdown_full'

ap36 = False

## AP 36K


json_file = '/groups/branson/bransonlab/datasets/APT-36K/annotations/apt36k_annotations_fixed_train.json'
test_json = '/groups/branson/bransonlab/datasets/APT-36K/annotations/apt36k_annotations_fixed_val.json'
im_dir = '/groups/branson/bransonlab/datasets/APT-36K/data'
exp_dir = '/groups/branson/bransonlab/mayank/apt_cache_2/ap36k_topdown'

ap36 = True
##
# create single animal top-down data set

from reuse import *
import multiResData as mrd
from poseConfig import conf

os.makedirs(exp_dir,exist_ok=True)

imsz = [256,256]
conf.imsz = imsz
conf.save_step = 5000
conf.maxckpt = 2
conf.expname = 'ap_36' if ap36 else 'ap_10k'
conf.coco_im_dir = im_dir
conf.db_format = 'coco'
conf.cachedir = exp_dir
conf.n_classes = 17
conf.ht_pts = [2,4]
conf.is_multi=False
conf.horz_flip = True
conf.predict_occluded = True
conf.scale_factor_range = 1.5
conf.rrange = 70
conf.trange = 0
conf.batch_size = 64
conf.dl_steps = 600000
J = pt.json_load(json_file)
n_train = len(J['annotations'])
conf.dl_steps = int(np.ceil(n_train/conf.batch_size*210./1000)*1000)

# conf.trange = 50
mm = [[0,1],[5,8],[6,9],[7,10],[11,14],[12,15],[13,16]]
fm = {}
for m in mm:
    fm[f'{m[0]}'] = m[1]
    fm[f'{m[1]}'] = m[0]
conf.flipLandmarkMatches = fm

skel = [[1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [6, 7], [7, 8], [4, 9], [9, 10], [10, 11], [5, 12],
 [12, 13], [13, 14], [5, 15], [15, 16], [16, 17]]
skel = apt.to_py(skel)
conf.op_affinity_graph = skel
conf.multi_scale_by_bbox = True
conf.trx_align_theta = False
conf.multi_loss_mask = False
conf.check_bounds_distort = False
conf.mmpose_pad = 1.25

net_type = 'mdn_joint_fpn'
train_name = 'ap_full'

# conf.mdn_backbone = 'wide_resnet50_2'
# train_name = 'ap_full_wide'

##
# net_type = 'multi_mdn_joint_torch_split'
# train_name = 'ap_full_split'
##
train_name = 'ap_full_mdn_hrnet_dpred_0trange_lr20'
net_type = 'mdn_joint_fpn'
conf.mdn_use_hrnet = True

##
net_type = 'mmpose'
conf.mmpose_net = 'hrnet'
train_name = 'hrnet_padfix_noshift'

## prediction

conf.multi_scale_by_bbox = False
valfile = os.path.join(exp_dir,'val_TF.json')
preds, labels, info, model_file = apt.classify_db_all(net_type,conf,valfile,name=train_name,fullret=True)

ll = labels
pp = preds['locs']
ll[ll<-1000] = np.nan
dd = np.linalg.norm(ll-pp,axis=-1)

cur_dist = dd

prcs = [50,75,90,95]
vv = cur_dist.copy()
mm = np.nanpercentile(vv,prcs,axis=0,interpolation='nearest')

use_orig = False

# convert the detections to coco format for evaluation
import json
def convert_to_orig(pts,bbox):
    pts= pts.copy()
    bbox = bbox.copy()
    bbox[2:] += bbox[:2]
    psz_y,psz_x = conf.imsz
    scale = mrd.get_scale_bbox(bbox, [psz_y,psz_x])
    pts[...,0] -= psz_x/2-0.5
    pts[...,1] -= psz_y/2-0.5
    pts = pts/scale
    x = (bbox[0]+bbox[2])/2
    y = (bbox[1]+bbox[3])/2
    pts[...,0] += x
    pts[...,1] += y
    return pts

#

n_max_occ=20

n_occ = np.zeros(len(info))
if use_orig:
    gt_json = test_json
    J = pt.json_load(test_json)
    for ndx,i in enumerate(info):
        cura = J['annotations'][i[1]]
        bbox = np.array(cura['bbox']).astype('float64')
        newpts = convert_to_orig(pp[ndx],bbox)
        lpts= np.array(cura['keypoints']).reshape([17,3]).astype('float64')
        lpts[:,:2] = newpts
        cura['keypoints'] = lpts.flatten().tolist()
        assert(False,'Update scores')
        cura['score'] = 1.
else:
    gt_json = valfile
    J = pt.json_load(gt_json)
    for ndx,i in enumerate(info):
        cura = J['annotations'][ndx]
        n_occ[ndx] = np.count_nonzero(np.array(cura['keypoints'][2::3])==0)
        newpts = pp[ndx]
        lpts= np.array(cura['keypoints']).reshape([17,3]).astype('float64')
        lpts[:,:2] = newpts
        cura['keypoints'] = lpts.flatten().tolist()
        if 'occ' not in preds:
            cc = preds['conf'][ndx]
            k_scores = cc[cc > 0.2]
            if len(k_scores) > 0:
                cur_s = np.mean(k_scores)
            else:
                cur_s = 0.
            cura['score'] = cur_s
        else:
            # cc = np.clip(1-preds['occ'][ndx],0,1)
            # cc = np.ones_like(preds['occ'][ndx])
            cc = preds['conf'][ndx]
            occ = preds['occ'][ndx]
            cc = cc.mean()
            # cc = np.exp(-np.clip(cc,0,25).mean()/100)
            # cc = (1/np.clip(cc,1,25)).mean()

            # cc_min = np.percentile(preds['conf'].mean(axis=1),2)
            # cc_max = np.percentile(preds['conf'].mean(axis=1),90)
            # cc =  (cc.mean()-cc_min)/(cc_max-cc_min)
            cc1 = preds['conf_joint'][ndx]
            cc_min = np.percentile(preds['conf_joint'],2)
            cc_max = np.percentile(preds['conf_joint'],90)
            cc1 =  (cc1-cc_min)/(cc_max-cc_min)
            cura['score'] = np.clip(cc,0,1)

Jo = pt.json_load(gt_json)
to_keep = np.where(n_occ<=n_max_occ)[0]
Jo['annotations'] = [a for ndx,a in enumerate(Jo['annotations']) if ndx in to_keep]
J['annotations'] = [a for ndx,a in enumerate(J['annotations']) if ndx in to_keep]


out_file_p = os.path.join(conf.cachedir,'pred_coco.json')
with open(out_file_p,'w') as f:
    json.dump(J,f)

out_file_l = os.path.join(conf.cachedir,'label_coco.json')
with open(out_file_l,'w') as f:
    json.dump(Jo,f)

# coco evaluations

sigmas = [0.025,0.025,0.026,0.035,0.035,0.079,0.072,0.062,0.079,0.072,0.062,0.107,0.087,0.089,0.107,0.087,0.089]
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

coco_gt = COCO(out_file_l)
coco_gt.loadAnns()

coco_dt = COCO(out_file_p)
coco_dt.loadAnns()

coco_eval = COCOeval(coco_gt,coco_dt,'keypoints')
coco_eval.params.useSegm = None
# coco_eval.params.iouThrs = np.linspace(.05, 0.5, int(np.round((0.5 - .05) / .05)) + 1, endpoint=True)
# coco_eval.params.kpt_oks_sigmas = np.array(sigmas)
coco_eval.sigmas =np.array(sigmas)
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

## resutls

# AP 36K

# ap_full_mdn_hrnet_dpred_0trange_100lr
# Average
# Precision(AP) @ [IoU = 0.50:0.95 | area = all | maxDets = 20] =  0.739
# Average
# Precision(AP) @ [IoU = 0.50 | area = all | maxDets = 20] =  0.935
# Average
# Precision(AP) @ [IoU = 0.75 | area = all | maxDets = 20] =  0.817
# Average
# Precision(AP) @ [IoU = 0.50:0.95 | area = medium | maxDets = 20] = -1.000
# Average
# Precision(AP) @ [IoU = 0.50:0.95 | area = large | maxDets = 20] =  0.739
# Average
# Recall(AR) @ [IoU = 0.50:0.95 | area = all | maxDets = 20] =  0.778
# Average
# Recall(AR) @ [IoU = 0.50 | area = all | maxDets = 20] =  0.945
# Average
# Recall(AR) @ [IoU = 0.75 | area = all | maxDets = 20] =  0.842
# Average
# Recall(AR) @ [IoU = 0.50:0.95 | area = medium | maxDets = 20] = -1.000
# Average
# Recall(AR) @ [IoU = 0.50:0.95 | area = large | maxDets = 20] =  0.778
## train

conf.multi_scale_by_bbox = False

train_filename = os.path.join(conf.cachedir, conf.trainfilename)
train_sample = os.path.join(exp_dir,train_name+'_samples.mat')
# apt.gen_train_samples(conf, model_type=net_type, nsamples=90, train_name=train_name, out_file=train_sample,
                  # debug=True)
# print('Done generating training samples...')

module_name = 'Pose_{}'.format(net_type)
pose_module = __import__(module_name)
foo = getattr(pose_module, module_name)
self = foo(conf, name=train_name)

logging.info('Starting training...')
self.train_wrapper()


## create DBs

conf.multi_scale_by_bbox = True

import json
def write_coco_single_animal(ann_file,out_file,out_im_dir,pad,force=False):

    skeleton = [[i, i + 1] for i in range(conf.n_classes - 1)]
    names = ['pt_{}'.format(i) for i in range(conf.n_classes)]
    categories = [{'id': 1, 'skeleton': skeleton, 'keypoints': names, 'super_category': 'fly', 'name': 'fly'},
                  {'id': 2, 'super_category': 'neg_box', 'name': 'neg_box'}]

    tann = {'images': [], 'info': [], 'annotations': [], 'categories': categories}
    tinfo = {'ndx': 0, 'ann_ndx': 0,    'imdir': os.path.join(conf.cachedir, out_im_dir)}
    os.makedirs(tinfo['imdir'], exist_ok=True)


    ann = pt.json_load(ann_file)
    count = 0
    pad = 1- 1/(1+pad)
    for ima in ann['images']:
        id = ima['id']
        im = cv2.imread(os.path.join(im_dir,ima['file_name']), cv2.IMREAD_UNCHANGED)
        if im.ndim > 2:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        for andx,a in enumerate(ann['annotations']):
            if a['image_id'] != id: continue
            pts = np.array(a['keypoints']).reshape([conf.n_classes,3]).astype('float')
            locs = pts[:,:2]
            occ = pts[:,2]
            locs[occ==0,:] = np.nan
            occ = occ<1.5
            bbox = np.array(a['bbox'])
            bbox[2:] += bbox[:2]
            x,y = (bbox[:2]+bbox[2:])/2
            w,h = bbox[2:]-bbox[:2]
            bbox[0] -= pad/2*w
            bbox[2] += pad/2*w
            bbox[1] -= pad/2*h
            bbox[3] += pad/2*h

            cur_patch,newl,scale = mrd.crop_patch_trx(conf,im,x,y,0,locs,bbox)

            pw = pad*conf.imsz[1]/2
            ph = pad*conf.imsz[0]/2
            roi = np.array([[[pw,ph],[conf.imsz[1]-pw,conf.imsz[0]-ph] ]])

            data_out = {'im': cur_patch, 'locs': newl, 'info': [id,andx,count], 'occ': occ,'roi':roi}
            count+=1
            apt.convert_to_coco(tinfo,tann,data_out,conf,force=force)

    with open(out_file + '.json', 'w') as f:
        json.dump(tann, f)

train_filename = os.path.join(conf.cachedir, conf.trainfilename)
val_filename = os.path.join(conf.cachedir, conf.valfilename)

if ap36:
    pad = .25
else:
    pad = 0.25

write_coco_single_animal(json_file,train_filename,'train',pad)
write_coco_single_animal(test_json,val_filename,'val',pad,force=True)


## convert the detections to coco format for evaluation == original size!
import multiResData as mrd
import json
def convert_to_orig(pts,bbox):
    pts= pts.copy()
    bbox = bbox.copy()
    bbox[2:] += bbox[:2]
    psz_y,psz_x = conf.imsz
    scale = mrd.get_scale_bbox(bbox, [psz_y,psz_x])
    pts[...,0] -= psz_x/2-0.5
    pts[...,1] -= psz_y/2-0.5
    pts = pts/scale
    x = (bbox[0]+bbox[2])/2
    y = (bbox[1]+bbox[3])/2
    pts[...,0] += x
    pts[...,1] += y
    return pts

val_filename = os.path.join(conf.cachedir, conf.valfilename) + '.json'
J = pt.json_load(val_filename)

for ndx,i in enumerate(info):
    cura = J['annotations'][ndx]
    bbox = np.array(cura['bbox']).astype('float64')
    newpts = convert_to_orig(pp[ndx],bbox)
    lpts= np.array(cura['keypoints']).reshape([17,3]).astype('float64')
    lpts[:,:2] = newpts
    cura['keypoints'] = lpts.flatten().tolist()
    cura['score'] = 1.


out_file = os.path.join(conf.cachedir,'pred_coco_val.json')
with open(out_file,'w') as f:
    json.dump(J,f)


## !!!!!!!!!!!!!!!!!!           For bottom-up!          !!!!!!!!!!!!!!!!!!!!!!!


# AP 10K
# json_file = '/groups/branson/bransonlab/datasets/ap-10k/annotations/ap10k-train-split1-dog.json'
json_file = '/groups/branson/bransonlab/datasets/ap-10k/annotations/ap10k-train-split1.json'
test_json = '/groups/branson/bransonlab/datasets/ap-10k/annotations/ap10k-val-split1.json'
im_dir = '/groups/branson/bransonlab/datasets/ap-10k/data'
exp_dir = '/groups/branson/bransonlab/mayank/apt_cache_2/ap10k_bottomup'

# AP 36K

# json_file = '/groups/branson/bransonlab/datasets/APT-36K/annotations/apt36k_annotations_fixed_train.json'
# test_json = '/groups/branson/bransonlab/datasets/APT-36K/annotations/apt36k_annotations_fixed_val-dog.json'
# im_dir = '/groups/branson/bransonlab/datasets/APT-36K/data'
# exp_dir = '/groups/branson/bransonlab/mayank/apt_cache_2/ap36k_bottomup'
#

#
from reuse import *

im_sz = [1024,1024]

os.makedirs(exp_dir,exist_ok=True)

ann = pt.json_load(json_file)
im_sz_in = np.array([[a['width'],a['height']] for a in ann['images']])
#plt.scatter(im_sz_in[:,0],im_sz_in[:,1])
im_num = np.array([a['image_id'] for a in ann['annotations']])
x,y = np.unique(im_num,return_counts=True)
max_n_animals = max(y)
n_classes = 17


from poseConfig import conf

tfilename = os.path.join(exp_dir,conf.trainfilename+'.json')
if not os.path.exists(tfilename):
    os.symlink(json_file,tfilename)

ann = pt.json_load(json_file)
im_num = np.array([a['image_id'] for a in ann['annotations']])
x,y = np.unique(im_num,return_counts=True)

conf.expname = 'ap_10k'
conf.imsz = im_sz
conf.coco_im_dir = im_dir
conf.db_format = 'coco'
conf.cachedir = exp_dir
conf.dl_steps = 300000
conf.multi_crop_ims = False
conf.multi_frame_sz = conf.imsz
conf.multi_loss_mask = False
conf.max_n_animals = max(y)
conf.n_classes = 17
conf.ht_pts = [2,4]
conf.is_multi=True
conf.horz_flip = True
conf.rescale = 1.6
conf.predict_occluded = True
conf.scale_factor_range = 2
conf.mdn_joint_layer_num = 1
conf.rrange = 70
conf.trange = 50
conf.check_bounds_distort = False

mm = [[0,1],[5,8],[6,9],[7,10],[11,14],[12,15],[13,16]]
fm = {}
for m in mm:
    fm[f'{m[0]}'] = m[1]
    fm[f'{m[1]}'] = m[0]
conf.flipLandmarkMatches = fm

skel = [[1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [6, 7], [7, 8], [4, 9], [9, 10], [10, 11], [5, 12],
 [12, 13], [13, 14], [5, 15], [15, 16], [16, 17]]
skel = apt.to_py(skel)
conf.op_affinity_graph = skel

net_type = 'multi_mdn_joint_torch'
train_name = 'ap_full_hrnet'

conf.mdn_use_hrnet = True


# classify
preds, labels, info, model_file = apt.classify_db_all(net_type,conf,test_json,name=train_name,fullret=True)

# accuracy
ll = labels
pp = preds['locs']
ll[ll<-1000] = np.nan
dd = np.linalg.norm(ll[:,None]-pp[:,:,None],axis=-1)
dd1 = find_dist_match(dd)
valid_l = np.any(~np.isnan(ll[:,:,:,0]),axis=-1)

cur_dist = dd1[valid_l]

prcs = [50,75,90]
vv = cur_dist.copy()
mm = np.nanpercentile(vv,prcs,axis=0,interpolation='nearest')

# COCO evaluation..
import json
from copy import deepcopy
gt_json = test_json
J = pt.json_load(test_json)
J['categories'] = J['categories'][:1]
for jix in J['annotations']:
    jix['category_id'] = 1
gt_json = os.path.join(exp_dir,'val_nocat.json')
with open(gt_json,'w') as f:
    json.dump(J,f)
ex_a = J['annotations'][0]
J['annotations'] = []
J['categories'] = J['categories'][:1]
for ndx in range(pp.shape[0]):
    for ix in range(pp.shape[1]):
        if np.all(np.isnan(pp[ndx,ix,:,0])): continue

        cura = deepcopy(ex_a)
        bbox = np.array(cura['bbox']).astype('float64')
        ww = J['images'][ndx]['width']
        hh = J['images'][ndx]['height']

        aspect_ratio = ww / hh
        target_aspect_ratio = conf.imsz[1] / conf.imsz[0]

        # Determine the scaling factor
        if aspect_ratio > target_aspect_ratio:
            scale_factor = conf.imsz[1]/ww
        else:
            scale_factor = conf.imsz[0]/hh

        lpts= np.array(cura['keypoints']).reshape([17,3]).astype('float64')
        lpts[:,:2] = pp[ndx,ix]/scale_factor
        cura['keypoints'] = lpts.flatten().tolist()
        assert(False,'Update scores')
        cura['score'] = np.mean(preds['conf'][ndx,0,ix])
        cura['image_id'] = J['images'][ndx]['id']
        cura['id'] = ndx*100+ix
        J['annotations'].append(cura)

out_file = os.path.join(conf.cachedir,'pred_coco.json')
with open(out_file,'w') as f:
    json.dump(J,f)

# coco evaluations

sigmas = [0.025,0.025,0.026,0.035,0.035,0.079,0.072,0.062,0.079,0.072,0.062,0.107,0.087,0.089,0.107,0.087,0.089]
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

coco_gt = COCO(gt_json)
coco_gt.loadAnns()

coco_dt = COCO(out_file)
coco_dt.loadAnns()

coco_eval = COCOeval(coco_gt,coco_dt,'keypoints')
coco_eval.params.useSegm = None
# coco_eval.params.kpt_oks_sigmas = np.array(sigmas)
coco_eval.sigmas =np.array(sigmas)
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Average
# Precision(AP) @ [IoU = 0.50:0.95 | area = all | maxDets = 20] =  0.538
# Average
# Precision(AP) @ [IoU = 0.50 | area = all | maxDets = 20] =  0.825
# Average
# Precision(AP) @ [IoU = 0.75 | area = all | maxDets = 20] =  0.570
# Average
# Precision(AP) @ [IoU = 0.50:0.95 | area = medium | maxDets = 20] =  0.187
# Average
# Precision(AP) @ [IoU = 0.50:0.95 | area = large | maxDets = 20] =  0.544
# Average
# Recall(AR) @ [IoU = 0.50:0.95 | area = all | maxDets = 20] =  0.671
# Average
# Recall(AR) @ [IoU = 0.50 | area = all | maxDets = 20] =  0.899
# Average
# Recall(AR) @ [IoU = 0.75 | area = all | maxDets = 20] =  0.720
# Average
# Recall(AR) @ [IoU = 0.50:0.95 | area = medium | maxDets = 20] =  0.185
# Average
# Recall(AR) @ [IoU = 0.50:0.95 | area = large | maxDets = 20] =  0.679
## show example

J = pt.json_load(test_json)
import cv2
# vv = np.sum(~np.all(np.isnan(ll[...,0]),axis=-1),axis=-1)
# vp = np.sum(~np.all(np.isnan(pp[...,0]),axis=-1),axis=-1)
# sq = np.where(vv!=vp)[0]
# item = np.random.choice(sq)
item = sq[count]
count += 1
im = cv2.imread(os.path.join(im_dir,J['images'][item]['file_name']),cv2.IMREAD_UNCHANGED)
plt.figure(); plt.imshow(im)

curl = np.ones([max_n_animals, n_classes, 3]) * np.nan
lndx = 0
annos = []
all_l = []
for a in J['annotations']:
    if not (a['image_id'] == J['images'][item]['id']):
        continue
    locs = np.array(a['keypoints'])
    if a['num_keypoints'] > 0 and a['area'] > 1:
        locs = np.reshape(locs, [n_classes, 3]).astype('float')
        locs[locs[:,2]==0,:] = np.nan
        # if np.all(locs[:,2]>0.5):
        curl[lndx, ...] = locs
        lndx += 1
    annos.append(a)
mdskl(curl,skel,cc='b')
ww = J['images'][item]['width']
hh = J['images'][item]['height']

aspect_ratio = ww / hh
target_aspect_ratio = conf.imsz[1] / conf.imsz[0]

# Determine the scaling factor
if aspect_ratio > target_aspect_ratio:
    scale_factor = conf.imsz[1] / ww
else:
    scale_factor = conf.imsz[0] / hh

mdskl(pp_g[item]/scale_factor,skel,cc='g')
mdskl(pp_2[item]/scale_factor,skel,cc='r')

## show training example
from reuse import *
import hdf5storage
train_sample = '/groups/branson/bransonlab/mayank/apt_expts/ap10k/ap_dog_samples.mat'
m = hdf5storage.loadmat(train_sample)

idx = np.random.randint(m['ims'].shape[0])
plt.figure()
plt.imshow((m['ims'][idx]*256).astype('uint8'))
mdskl(m['locs'][idx],skel)

## Train

train_sample = os.path.join(exp_dir,train_name+'_samples.mat')
apt.gen_train_samples(conf, model_type=net_type, nsamples=90, train_name=train_name, out_file=train_sample,
                  debug=True)
print('Done generating training samples...')

module_name = 'Pose_{}'.format(net_type)
pose_module = __import__(module_name)
foo = getattr(pose_module, module_name)
self = foo(conf, name=train_name)

logging.info('Starting training...')
self.train_wrapper()

##

import glob
ss = glob.glob('/groups/branson/bransonlab/mayank/apt_cache_2/ap10k_topdown_full/ap_10k_*_traindata.json')
plt.figure()
names = []
for curs in ss:
    V = pt.json_load(curs)
    names.append( curs[80:-15])
    plt.plot(run_avg(V['val_dist'][50:]))

plt.legend(names)

## !!!!!!!!!!!!!!!!!

## object detection ... 2 stage

preds, labels, info, model_file = apt.classify_db_all(net_type,conf,test_json,name=train_name,fullret=True)

## accuracy
pp = preds['locs']

# COCO evaluation..
import json
from copy import deepcopy
gt_json = test_json
J = pt.json_load(test_json)
J['categories'] = J['categories'][:1]
for jix in J['annotations']:
    jix['category_id'] = 1
gt_json = os.path.join(exp_dir,'val_nocat.json')
with open(gt_json,'w') as f:
    json.dump(J,f)
ex_a = J['annotations'][0]
J['annotations'] = []
J['categories'] = J['categories'][:1]
for ndx in range(pp.shape[0]):
    for ix in range(pp.shape[1]):
        if np.all(np.isnan(pp[ndx,ix,:,0])): continue

        cura = deepcopy(ex_a)
        bbox = np.array(cura['bbox']).astype('float64')
        ww = J['images'][ndx]['width']
        hh = J['images'][ndx]['height']

        aspect_ratio = ww / hh
        target_aspect_ratio = conf.imsz[1] / conf.imsz[0]

        # Determine the scaling factor
        if aspect_ratio > target_aspect_ratio:
            scale_factor = conf.imsz[1]/ww
        else:
            scale_factor = conf.imsz[0]/hh

        bb_pred = pp[ndx,ix].flatten()/scale_factor
        bb_pred[2:] -= bb_pred[:2]
        cura['bbox'] = bb_pred.astype('int').tolist()
        cura['score'] = np.mean(preds['conf'][ndx,ix])
        cura['image_id'] = J['images'][ndx]['id']
        cura['id'] = ndx*100+ix
        J['annotations'].append(cura)

out_file = os.path.join(conf.cachedir,'pred_coco.json')
with open(out_file,'w') as f:
    json.dump(J,f)

# coco evaluations

sigmas = [0.025,0.025,0.026,0.035,0.035,0.079,0.072,0.062,0.079,0.072,0.062,0.107,0.087,0.089,0.107,0.087,0.089]
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_gt = COCO(gt_json)
coco_gt.loadAnns()

coco_dt = COCO(out_file)
coco_dt.loadAnns()

coco_eval = COCOeval(coco_gt,coco_dt,'bbox')
coco_eval.params.useSegm = None
# coco_eval.params.kpt_oks_sigmas = np.array(sigmas)
coco_eval.sigmas =np.array(sigmas)
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()



## !!!!!!!!!!!!!!!!!!!!!!!!!

## 2 stage

json_file = '/groups/branson/bransonlab/datasets/ap-10k/annotations/ap10k-train-split1.json'
test_json = '/groups/branson/bransonlab/datasets/ap-10k/annotations/ap10k-val-split1.json'
im_dir = '/groups/branson/bransonlab/datasets/ap-10k/data'

ap36 = False

# AP 36K


# json_file = '/groups/branson/bransonlab/datasets/APT-36K/annotations/apt36k_annotations_fixed_train.json'
# test_json = '/groups/branson/bransonlab/datasets/APT-36K/annotations/apt36k_annotations_fixed_val.json'
# im_dir = '/groups/branson/bransonlab/datasets/APT-36K/data'
# exp_dir = '/groups/branson/bransonlab/mayank/apt_cache_2/ap36k_topdown'
#
# ap36 = True
#

from reuse import *
import multiResData as mrd

# second stage
from poseConfig import conf
import copy
conf = copy.deepcopy(conf)
exp_dir = '/groups/branson/bransonlab/mayank/apt_cache_2/ap10k_topdown_full'

imsz = [256,256]
conf.imsz = imsz
conf.save_step = 5000
conf.maxckpt = 2
if ap36:
    conf.expname = 'ap_36'
else:
    conf.expname = 'ap_10k'
conf.coco_im_dir = im_dir
conf.db_format = 'coco'
conf.cachedir = exp_dir
conf.dl_steps = 75000
conf.n_classes = 17
conf.ht_pts = [2,4]
conf.is_multi=False
conf.horz_flip = True
conf.predict_occluded = True
conf.scale_factor_range = 1.5
conf.rrange = 70
conf.trange = 30; #50
conf.rescale = 1
mm = [[0,1],[5,8],[6,9],[7,10],[11,14],[12,15],[13,16]]
fm = {}
for m in mm:
    fm[f'{m[0]}'] = m[1]
    fm[f'{m[1]}'] = m[0]
conf.flipLandmarkMatches = fm

skel = [[1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [6, 7], [7, 8], [4, 9], [9, 10], [10, 11], [5, 12],
 [12, 13], [13, 14], [5, 15], [15, 16], [16, 17]]
skel = apt.to_py(skel)
conf.op_affinity_graph = skel
conf.multi_scale_by_bbox = True
conf.trx_align_theta = False
conf.multi_loss_mask = False
conf.check_bounds_distort = False
conf.batch_size = 8

conf.use_bbox_trx = True
net_type1 = 'mdn_joint_fpn'
conf.mdn_use_hrnet = True
train_name1 = 'ap_full_mdn_hrnet_dpred_0trange_100lr_64bsz_lr'

net_type1 = 'mmpose'
conf.mmpose_net = 'hrnet'
train_name1 = 'hrnet'

conf1 = copy.deepcopy(conf)


exp_dir = '/groups/branson/bransonlab/mayank/apt_cache_2/ap10k_2stage'

im_sz = [1024,1024]

ann = pt.json_load(json_file)
im_sz_in = np.array([[a['width'],a['height']] for a in ann['images']])
#plt.scatter(im_sz_in[:,0],im_sz_in[:,1])
im_num = np.array([a['image_id'] for a in ann['annotations']])
x,y = np.unique(im_num,return_counts=True)
max_n_animals = max(y)
n_classes = 17


from poseConfig import conf
conf = copy.deepcopy(conf)
import json

ann = pt.json_load(json_file)
im_num = np.array([a['image_id'] for a in ann['annotations']])
x,y = np.unique(im_num,return_counts=True)

conf.expname = 'ap_10k'
conf.imsz = im_sz
conf.coco_im_dir = im_dir
conf.db_format = 'coco'
conf.cachedir = exp_dir
conf.dl_steps = 600000
conf.multi_crop_ims = False
conf.multi_frame_sz = conf.imsz
conf.multi_loss_mask = False
conf.max_n_animals = max(y)
conf.n_classes = 2
conf.ht_pts = []
conf.is_multi=True
conf.horz_flip = True
conf.rescale = 1.6
conf.predict_occluded = True
conf.scale_factor_range = 2
conf.mdn_joint_layer_num = 1
conf.rrange = 70
conf.trange = 50
conf.check_bounds_distort = False

mm = [[0,1],[5,8],[6,9],[7,10],[11,14],[12,15],[13,16]]
fm = {}
for m in mm:
    fm[f'{m[0]}'] = m[1]
    fm[f'{m[1]}'] = m[0]
conf.flipLandmarkMatches = fm

skel = [[1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [6, 7], [7, 8], [4, 9], [9, 10], [10, 11], [5, 12],
 [12, 13], [13, 14], [5, 15], [15, 16], [16, 17]]
skel = apt.to_py(skel)
conf.op_affinity_graph = skel

net_type = 'detect_mmdetect'
train_name = 'ap36'
#
preds, labels, info, model_file = apt.classify_db_all(net_type,conf,test_json,name=train_name,fullret=True,model_type2=net_type1,name2=train_name1,conf2=conf1)

# accuracy
ll = labels
pp = preds['locs']
ll[ll<-1000] = np.nan
dd = np.linalg.norm(ll[:,None]-pp[:,:,None],axis=-1)
dd1 = find_dist_match(dd)
valid_l = np.any(~np.isnan(ll[:,:,:,0]),axis=-1)

cur_dist = dd1[valid_l]

prcs = [50,75,90]
vv = cur_dist.copy()
mm = np.nanpercentile(vv,prcs,axis=0,interpolation='nearest')

# COCO evaluation..
import json
from copy import deepcopy
gt_json = test_json
J = pt.json_load(test_json)
J['categories'] = J['categories'][:1]
for jix in J['annotations']:
    jix['category_id'] = 1
gt_json = os.path.join(exp_dir,'val_nocat.json')
with open(gt_json,'w') as f:
    json.dump(J,f)
ex_a = J['annotations'][0]
J['annotations'] = []
J['categories'] = J['categories'][:1]
for ndx in range(pp.shape[0]):
    for ix in range(pp.shape[1]):
        if np.all(np.isnan(pp[ndx,ix,:,0])): continue

        cura = deepcopy(ex_a)
        bbox = np.array(cura['bbox']).astype('float64')
        ww = J['images'][ndx]['width']
        hh = J['images'][ndx]['height']

        aspect_ratio = ww / hh
        target_aspect_ratio = conf.imsz[1] / conf.imsz[0]

        # Determine the scaling factor
        if aspect_ratio > target_aspect_ratio:
            scale_factor = conf.imsz[1]/ww
        else:
            scale_factor = conf.imsz[0]/hh

        lpts= np.array(cura['keypoints']).reshape([17,3]).astype('float64')
        lpts[:,:2] = pp[ndx,ix]/scale_factor
        cura['keypoints'] = lpts.flatten().tolist()
        assert(False,'Update scores')
        cura['score'] = np.mean(preds['conf'][ndx,0,ix])
        cura['image_id'] = J['images'][ndx]['id']
        cura['id'] = ndx*100+ix
        J['annotations'].append(cura)

out_file = os.path.join(conf.cachedir,'pred_coco.json')
with open(out_file,'w') as f:
    json.dump(J,f)

# coco evaluations

sigmas = [0.025,0.025,0.026,0.035,0.035,0.079,0.072,0.062,0.079,0.072,0.062,0.107,0.087,0.089,0.107,0.087,0.089]
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

coco_gt = COCO(gt_json)
coco_gt.loadAnns()

coco_dt = COCO(out_file)
coco_dt.loadAnns()

coco_eval = COCOeval(coco_gt,coco_dt,'keypoints')
coco_eval.params.useSegm = None
# coco_eval.params.kpt_oks_sigmas = np.array(sigmas)
coco_eval.sigmas =np.array(sigmas)
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
#
# Average
# Precision(AP) @ [IoU = 0.50:0.95 | area = all | maxDets = 20] =  0.575
# Average
# Precision(AP) @ [IoU = 0.50 | area = all | maxDets = 20] =  0.850
# Average
# Precision(AP) @ [IoU = 0.75 | area = all | maxDets = 20] =  0.613
# Average
# Precision(AP) @ [IoU = 0.50:0.95 | area = medium | maxDets = 20] =  0.308
# Average
# Precision(AP) @ [IoU = 0.50:0.95 | area = large | maxDets = 20] =  0.581
# Average
# Recall(AR) @ [IoU = 0.50:0.95 | area = all | maxDets = 20] =  0.676
# Average
# Recall(AR) @ [IoU = 0.50 | area = all | maxDets = 20] =  0.883
# Average
# Recall(AR) @ [IoU = 0.75 | area = all | maxDets = 20] =  0.719
# Average
# Recall(AR) @ [IoU = 0.50:0.95 | area = medium | maxDets = 20] =  0.305
# Average
# Recall(AR) @ [IoU = 0.50:0.95 | area = large | maxDets = 20] =  0.683