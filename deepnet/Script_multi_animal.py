
## Settings for Alice
name = 'alice'
op_af = '\(0,1\),\(0,5\),\(1,2\),\(3,4\),\(3,5\),\(5,6\),\(5,7\),\(5,9\),\(3,16\),\(9,10\),\(10,15\),\(9,14\),\(4,11\),\(7,8\),\(8,12\),\(7,13\)'
lbl_file = '/groups/branson/home/kabram/APT_projects/alice_touch_stripped.lbl'

n_grid = 4
sz = np.round(1024/n_grid).astype('int')
fill_value = 255
bb_ex = 10 # extra pixels around bb
buffer = 60 # roughly half the animal size + bb_ex


## Settings for Roian
name = 'roian'
lbl_file = '/groups/branson/bransonlab/apt/experiments/data/four_points_all_mouse_linux_tracker_updated20200423_new_skl_20200817.lbl_mdn.lbl'
animal_sz = 256
bb_ex = 40
buffer = 170
n_grid = 8
sz = int(2048/n_grid)

## Read the data

import PoseTools
import re
import h5py
import numpy as np

import multiResData
import numpy as np
import movies
import PoseTools as pt

debug = False
lbl = h5py.File(lbl_file,'r')
nmov = lbl['labeledpos'].value.shape[1]
cmap = pt.get_cmap(n_grid**2)

all_data = []
for ndx in range(nmov):
    mov_file = u''.join(chr(c) for c in lbl[lbl['movieFilesAll'][0,ndx]])
    cur_cap = movies.Movie(mov_file)
    cur_pts = multiResData.trx_pts(lbl,ndx)
    allfr = np.where(np.invert(np.all(np.isnan(cur_pts), axis=(0, 2, 3))))[0]
    for fndx,fr in enumerate(allfr):
        curi = cur_cap.get_frame(fr)[0]
        if curi.ndim == 2:
            curi = np.tile(curi[:,:,np.newaxis],[1,1,3])
        curi = np.pad(curi,[[buffer,buffer],[buffer,buffer],[0,0]])
        done_f = np.all(np.isnan(cur_pts[:, fr, ...]), axis=(-1, -2))
        info = []
        for xx in range(n_grid):
            for yy in range(n_grid):
                valid = []
                all_bb = []
                for aa in range(cur_pts.shape[0]):
                    pp = cur_pts[aa,fr,:,:]
                    if np.all(np.isnan(pp)):
                        continue
                    bb1 = np.min(pp,axis=1)
                    bb2 = np.max(pp,axis=1)

                    if not( (bb1[0] > xx*sz-buffer + bb_ex) and (bb2[0]<(xx+1)*sz+buffer-bb_ex) and (bb1[1] > yy*sz-buffer+bb_ex) and (bb2[1] < (yy+1)*sz+buffer-bb_ex)):
                        # The whole bounding box should lie within.
                        continue
                    # print('X',xx,[bb1[0],bb2[0]],xx*sz-buffer,(xx+1)*sz+buffer)
                    # print('Y',yy,[bb1[1],bb2[1]],yy*sz-buffer,(yy+1)*sz+buffer)
                    valid.append(aa)
                    done_f[aa] = True
                    curbb = np.array([[bb1[0],bb2[0]],[bb1[1],bb2[1]]])
                    curbb[0,:] += buffer - xx*sz
                    curbb[1,:] += buffer - yy*sz
                    all_bb.append(curbb)
                if len(valid)> 0:
                    info.append([xx,yy,valid,all_bb])
                    curp = curi[yy*sz:((yy+1)*sz+2*buffer),xx*sz:((xx+1)*sz+2*buffer),:]
                    curl = cur_pts[valid,fr,:,:]
                    curl[:,0,:] = curl[:,0,:] + buffer - xx*sz
                    curl[:,1,:] = curl[:,1,:] + buffer - yy*sz

                    all_data.append([curp,curl,[ndx,fr,xx,yy],all_bb])

                if len(valid)> 0 and debug:
                    plt.figure()
                    plt.imshow(curp)
                    plt.scatter(curl[:,0,:],curl[:,1,:])
        assert np.all(done_f),'Some missing for {} {}'.format(ndx,fr)

        if debug:
            plt.figure()
            plt.imshow(curi[buffer:-buffer,buffer:-buffer,:])
            for ii in info:
                xx = ii[0]
                yy = ii[1]
                for vndx in range(len(ii[2])):
                    curbb = ii[3][vndx]
                    curbb = curbb.copy()
                    curbb[:,0] -= bb_ex
                    curbb[:,1] += bb_ex
                    curbb[0,:] += xx*sz - buffer  #+ 5*xx
                    curbb[1,:] += yy*sz - buffer #+ 5*yy
                    # plt.scatter(curl[:,0,:],curl[:,1,:])
                # for curbb in all_bb:
                    plt.plot([curbb[0,0],curbb[0,1],curbb[0,1],curbb[0,0],curbb[0,0]],[curbb[1,0],curbb[1,0],curbb[1,1],curbb[1,1],curbb[1,0]],c=cmap[xx*n_grid+yy])
                for xx in range(1,n_grid):
                    plt.plot([xx*sz,xx*sz],[0.1,sz*n_grid],c=[0.05,0.05,0.05],alpha=0.3)
                    plt.plot([xx * sz -buffer, xx * sz - buffer], [0.1, sz*n_grid], c=[0.05, 0.05, 0.05], alpha=0.05)
                    plt.plot([xx * sz + buffer, xx * sz + buffer], [0.1, sz*n_grid], c=[0.05, 0.05, 0.05], alpha=0.05)
                for yy in range(1,n_grid):
                    plt.plot([0.1,sz*n_grid],[yy*sz,yy*sz],c=[0.05,0.05,0.05],alpha=0.3)
                    plt.plot([0.1,sz*n_grid],[yy*sz-buffer,yy*sz-buffer],c=[0.05,0.05,0.05],alpha=0.05)
                    plt.plot([0.1,1023],[yy*sz+buffer,yy*sz+buffer],c=[0.05,0.05,0.05],alpha=0.05)




## show a few examples

f,ax = plt.subplots(5,5)
ax = ax.flatten()
for ix in range(25):
    qq = np.random.choice(len(all_data))
    ax[ix].imshow(all_data[qq][0])
    ax[ix].set_title(qq)
    # ax[ix].scatter(all_data[qq][1][:,0,:],all_data[qq][1][:,1,:])
    ii = all_data[qq][-1]
    for vndx in range(len(ii)):
        curbb = ii[vndx]
        curbb = curbb.copy()
        curbb[:,0] -= bb_ex
        curbb[:,1] += bb_ex
        # plt.scatter(curl[:,0,:],curl[:,1,:])
        # for curbb in all_bb:
        ax[ix].plot([curbb[0, 0], curbb[0, 1], curbb[0, 1], curbb[0, 0], curbb[0, 0]],
                 [curbb[1, 0], curbb[1, 0], curbb[1, 1], curbb[1, 1], curbb[1, 0]], c=cmap[xx * n_grid + yy])

## create the DBs

from multiResData import int64_feature, float_feature, bytes_feature,create_envs
import tensorflow as tf
import APT_interface as apt
import os
import json

exp_name = 'deepnet'; val_frac = 0.3
exp_name = 'full'; val_frac = 0

n_multi_extra = 0
#exp_name = 'more_multi' # 'deepnet'
#n_multi_extra = 4
net_type = 'multi_mdn_joint_torch' #'multi_openpose'
def create_mask(bb,sz,bb_ex):
    # sz should be h x w (i.e y first then x)
    mask = np.zeros(sz)
    for c in bb:
        b = np.round(c).astype('int')
        b[:,0] -= bb_ex
        b[:,1] += bb_ex
        b[0,:] = np.clip(b[0,:],0,sz[1])
        b[1,:] = np.clip(b[1,:],0,sz[1])
        mask[b[1,0]:b[1,1],b[0,0]:b[0,1]] = 1
    return mask

conf = apt.create_conf(lbl_file,0,exp_name,net_type=net_type,cache_dir='/nrs/branson/mayank/apt_cache_2')
max_n = max([x[1].shape[0] for x in all_data])
vv = np.unique([x[2][:2] for x in all_data],axis=0)
rr = np.random.uniform(size=vv.shape[0])>val_frac
splits = [vv[rr],vv[np.invert(rr)]]

env, val_env = create_envs(conf,True)
is_val = False

for d in all_data:
    frame_in = d[0]
    cur_n = d[1].shape[0]
    cur_loc = np.ones([max_n,conf.n_classes,2])*(-100000)
    cur_loc[:cur_n,...] = np.transpose(d[1],[0,2,1])
    ndx = d[2][0]
    fnum = d[2][1]
    occ = np.zeros(cur_loc.shape[:-1])

    if np.any(np.all(np.array([ndx,fnum]) == splits[0],axis=1)):
        cur_env = env
        is_val = False
    else:
        cur_env = val_env
        is_val = True

    trx_ndx = d[2][2]*n_grid + d[2][3]
    rows = frame_in.shape[0]
    cols = frame_in.shape[1]
    depth = frame_in.shape[2] if frame_in.ndim > 2 else 1
    mask = create_mask(d[-1],[sz+2*buffer,sz+2*buffer],bb_ex)
    mask = mask.astype('uint8')
    mask_raw = mask.tostring()
    image_raw = frame_in.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': int64_feature(rows),
        'width': int64_feature(cols),
        'depth': int64_feature(depth),
        'trx_ndx': int64_feature(trx_ndx),
        'locs': float_feature(cur_loc.flatten()),
        'expndx': float_feature(ndx),
        'max_n': int64_feature(max_n),
        'ts': float_feature(fnum),
        'mask': bytes_feature(mask_raw),
        'occ': float_feature(occ.flatten()),
        'image_raw': bytes_feature(image_raw)}))
    cur_env.write(example.SerializeToString())
    if not is_val and cur_n > 1:
        for ix in range(n_multi_extra):
            cur_env.write(example.SerializeToString())


env.close(); val_env.close()

with open(os.path.join(conf.cachedir, 'splitdata.json'), 'w') as f:
    json.dump([s.tolist() for s in splits], f)

## create the single animal dbs and train
import h5py
import PoseTools as pt
import json

info_in = '/nrs/branson/mayank/apt_cache_2/multitarget_bubble/multi_openpose/view_0/deepnet/splitdata.json'
multi_split = pt.json_load(info_in)
lbl = h5py.File(lbl_file,'r')

m_ndx = lbl['preProcData_MD_mov'].value[0, :].astype('int') - 1
t_ndx = lbl['preProcData_MD_iTgt'].value[0, :].astype('int') - 1
f_ndx = lbl['preProcData_MD_frm'].value[0, :].astype('int') - 1

single_split = [[],[]]
for ndx in range(len(m_ndx)):
    cur_i = [int(m_ndx[ndx]),int(f_ndx[ndx]),int(t_ndx[ndx])]
    if cur_i[:2] in multi_split[0]:
        single_split[0].append(cur_i)
    else:
        assert cur_i[:2] in multi_split[1], 'missing'
        single_split[1].append(cur_i)

out_file = '/nrs/branson/mayank/apt_cache_2/multitarget_bubble/single_animal_split.json'

with open(out_file,'w') as f:
    json.dump(single_split, f)
##

import APT_interface as apt
import Pose_mdn_joint_fpn

conf = apt.create_conf(lbl_file,0,'multi_compare','/nrs/branson/mayank/apt_cache_2','mdn_joint_fpn')
conf.splitType = 'predefined'
# apt.create_tfrecord(conf,True,out_file,True)

conf.rrange = 15
conf.trange = 5
conf.img_dim = 1
conf.dl_steps = 100000
conf.save_step = 5000
conf.mdn_use_unet_loss = False
conf.horz_flip = True
conf.flipLandmarkMatches = {'11': 16, '16': 11, '1': 2, '2': 1, '3': 4, '4': 3, '7': 9, '9': 7, '8': 10, '10': 8, '12': 15, '15': 12, '13': 14, '14': 13}

self = Pose_mdn_joint_fpn.Pose_mdn_joint_fpn(conf,name='deepnet')
self.train_wrapper()

## Single animal Val
import APT_interface as apt
conf = apt.create_conf(lbl_file,0,'multi_compare','/nrs/branson/mayank/apt_cache_2','mdn_joint_fpn')
aa = apt.classify_db_all('mdn_joint_fpn',conf,'/nrs/branson/mayank/apt_cache_2/multitarget_bubble/mdn_joint_fpn/view_0/multi_compare/val_TF.tfrecords','/nrs/branson/mayank/apt_cache_2/multitarget_bubble/mdn_joint_fpn/view_0/multi_compare/deepnet-200000')
dd = np.linalg.norm(aa[0]-aa[1],axis=-1)
ss = np.percentile(dd,[50,76,90,95,97],axis=0)

ss_p = np.array(
      [[0.42840735, 0.4810704 , 0.45353072, 0.47667045, 0.43482706,
        0.65485645, 0.57873938, 0.58888061, 0.64061295, 0.57605449,
        0.62341821, 0.74522241, 0.68657156, 0.79792583, 0.85144955,
        0.67172788, 0.75864856],
       [0.63211929, 0.72749439, 0.69235898, 0.69997487, 0.65601051,
        0.95167786, 0.85066391, 0.88433295, 0.9494333 , 0.87212001,
        0.94723436, 1.23056336, 1.07450126, 1.40045434, 1.43145354,
        1.03504751, 1.22165465],
       [0.83296097, 0.93825742, 0.94274987, 0.90386443, 0.88350752,
        1.27219873, 1.14472418, 1.18550657, 1.33433393, 1.18925402,
        1.380698  , 2.51935159, 1.77288897, 2.8916157 , 2.94362341,
        1.60514727, 2.44962582],
       [0.9744914 , 1.10669744, 1.11987133, 1.04421034, 1.03507524,
        1.50326877, 1.33868259, 1.49815334, 1.71422052, 1.46375225,
        1.80434256, 3.90917211, 2.8378863 , 4.8086896 , 4.71446444,
        2.62088531, 3.96248446],
       [1.0843334 , 1.23422864, 1.24537087, 1.17385346, 1.17910442,
        1.67553969, 1.52816968, 1.63252459, 1.99759642, 1.64145342,
        2.15951314, 5.31034328, 4.06158659, 5.97704301, 6.53505963,
        3.76201398, 5.36708072]])

## Single animal val mmpose hrnet
# mmpose training code in debug_mmpose. Why? I don't know the answers to all whys.
import APT_interface as apt
lbl_file = '/groups/branson/home/kabram/APT_projects/alice_touch_stripped.lbl'
net = 'mmpose'
tname = 'mmpose_aug_halfr' #'apt_aug_0tbcrange' #
mfile = f'/nrs/branson/mayank/apt_cache_2/multitarget_bubble/{net}/view_0/deepnet/{tname}-100000'
conf = apt.create_conf(lbl_file,0,'deepnet','/nrs/branson/mayank/apt_cache_2',net)
conf.rescale = conf.imsz[0]/768
conf.flipLandmarkMatches = {'11': 16, '16': 11, '1': 2, '2': 1, '3': 4, '4': 3, '7': 9, '9': 7, '8': 10, '10': 8, '12': 15, '15': 12, '13': 14, '14': 13}
aa = apt.classify_db_all('mmpose',conf,'/nrs/branson/mayank/apt_cache_2/multitarget_bubble/mdn_joint_fpn/view_0/multi_compare/val_TF.tfrecords',mfile)
dd = np.linalg.norm(aa[0]-aa[1],axis=-1)
np.percentile(dd,[50,76,90,95,97],axis=0)

## results for above
# mmpose_aug
ss_p = np.array([[
   0.60678062,  0.72401675,  0.69266972,  0.73614369,  0.63038603,
   0.78885669,  0.60267996,  0.71213826,  0.65421493,  0.66869511,
   0.65193726,  0.79540047,  0.76038724,  0.8351834 ,  0.73163715,
   0.69163898,  0.7377945 ],
 [ 0.81025462,  1.01898314,  0.94285418,  0.9974547 ,  0.87995722,
   1.07171201,  0.90385802,  0.99364457,  0.96555011,  0.95331625,
   0.95838583,  1.21592178,  1.16864766,  1.29919216,  1.15046276,
   1.05285898,  1.10395918],
 [ 1.02269067,  1.34106268,  1.2015605 ,  1.24744181,  1.10054479,
   1.35739623,  1.18744747,  1.24250771,  1.32335199,  1.30395163,
   1.31097367,  2.65294372,  1.74338707,  2.13401055,  2.02444784,
   1.56621432,  2.12212215],
 [ 1.1623227 ,  1.62087227,  1.38629284,  1.41550808,  1.2639266 ,
   1.61226656,  1.40801593,  1.47597128,  1.64034196,  1.5890363 ,
   1.70473661,  7.45829637,  3.25971781,  5.27976319,  5.44663423,
   3.47422057,  6.74071957],
 [ 1.2687928 ,  1.72806523,  1.54432477,  1.52666533,  1.35778042,
   1.79039263,  1.60786236,  1.66382308,  1.91845103,  1.91103697,
   2.19652039, 18.67957179,  9.40915188, 10.85032846, 11.61707408,
   7.4876683 , 14.38828816]])

# apt_aug
ss_p = np.array(
[[ 0.65968933,  0.79613052,  0.63977734,  0.67674728,  0.6917282 ,
   0.83142799,  0.66049028,  0.79616175,  0.67561729,  0.65977467,
   0.67558502,  0.82626909,  0.77378994,  0.8598894 ,  0.7462997 ,
   0.7341453 ,  0.74816819],
 [ 0.88618856,  1.09690823,  0.90254063,  0.93310064,  0.96966364,
   1.16149692,  0.95990967,  1.10687722,  1.01936975,  0.97761014,
   1.00406658,  1.35239825,  1.15788261,  1.3289594 ,  1.20786176,
   1.0950882 ,  1.14461453],
 [ 1.10402314,  1.4146774 ,  1.16692478,  1.17177508,  1.21246801,
   1.47423678,  1.26224367,  1.40374623,  1.46535435,  1.31675041,
   1.35338707,  3.78831604,  1.77212009,  2.71535442,  2.18502425,
   1.64777784,  2.93462648],
 [ 1.26525597,  1.65653247,  1.38007368,  1.31593658,  1.39587005,
   1.68738586,  1.5209087 ,  1.65117148,  1.981345  ,  1.61581103,
   1.74507998, 11.42261617,  4.10459351,  8.27113573,  9.76020702,
   4.52716184,  8.5922846 ],
 [ 1.38054383,  1.76613676,  1.53856521,  1.43577488,  1.53980064,
   1.88758081,  1.71175107,  1.81659915,  2.56625633,  1.89246121,
   2.13041491, 22.1506002 , 11.40180743, 15.82514843, 17.493125  ,
   9.55352863, 18.62976534]])

# apt_aug_linear
ss_p = np.array(
[[ 0.60513743,  0.51311197,  0.50520846,  0.5304525 ,  0.54061249,
   0.83992854,  0.70083376,  0.71141541,  0.67374442,  0.63740602,
   0.59822284,  0.74264014,  0.63815094,  0.69753938,  0.6842919 ,
   0.6023353 ,  0.62913579],
 [ 0.88618934,  0.77907711,  0.74591226,  0.7716955 ,  0.77985561,
   1.35862486,  1.12842406,  1.04677807,  1.10339064,  0.96034554,
   0.92620593,  1.4116996 ,  0.98617196,  1.15116393,  1.1350265 ,
   0.94527291,  1.11075938],
 [ 1.16316854,  1.00889582,  0.98892277,  0.98893824,  1.03200025,
   7.0421866 , 15.62164331,  1.3611215 ,  1.89241696,  1.32531158,
   1.45577998,  6.37386379,  1.5825091 ,  2.64891937,  3.49692293,
   1.56068354,  5.89716977],
 [ 1.37266099,  1.19053889,  1.15962724,  1.12856934,  1.19535872,
   16.22534779, 29.32462583,  1.62994516,  3.18870503,  1.66936937,
   2.1217337 , 17.44083319,  5.27230431,  8.50430279, 13.98854183,
   9.89484331, 25.69862955],
 [ 1.67579649,  1.33396424,  1.26549761,  1.25078491,  1.33480071,
   23.31041537, 37.79796532,  1.86931412,  4.57782958,  1.97332505,
   3.23628371, 27.2097361 , 16.38200379, 15.54365172, 24.52120956,
   25.33360063, 37.9377642 ]])

# apt_aug no flip test
ss = np.array(
[[ 0.54665327,  0.51074237,  0.4974397 ,  0.50130289,  0.52915385,
   0.67659697,  0.61563477,  0.68813948,  0.57155936,  0.58633669,
   0.54686773,  0.69326561,  0.61966636,  0.70445778,  0.68495128,
   0.61070081,  0.64103947],
 [ 0.7708433 ,  0.79058574,  0.72871792,  0.73128652,  0.78365901,
   1.01381508,  0.90775571,  1.01555795,  0.93395736,  0.91347324,
   0.85405034,  1.23781112,  0.95267476,  1.16654876,  1.12218953,
   0.92536099,  1.03457257],
 [ 0.99110161,  1.0680421 ,  0.97515966,  0.96201433,  1.02296304,
   1.38180015,  1.17831246,  1.37858743,  1.47126292,  1.23780451,
   1.29438747,  4.602689  ,  1.52083624,  2.96127175,  2.81104364,
   1.54566434,  3.23715742],
 [ 1.15807138,  1.27223577,  1.18423941,  1.09661081,  1.19197401,
   1.66229333,  1.45190842,  1.66090135,  2.12383978,  1.58527803,
   1.68735944, 13.94697285,  4.97679229,  9.38861114, 13.40876075,
   6.7694264 , 12.86249201],
 [ 1.31028191,  1.37463521,  1.28180246,  1.20974828,  1.33045892,
   1.89074816,  1.66246872,  2.00609187,  2.99390741,  1.82314445,
   2.39079832, 25.81876824, 14.20460814, 19.42837882, 25.80212155,
   16.47228712, 28.51006455]])

# mmpose_aug_1rot_prop
ss = np.array(
[[ 0.41273209,  0.475355  ,  0.47104497,  0.47330433,  0.48465899,
   0.58336473,  0.52067021,  0.55869058,  0.52398988,  0.54685883,
   0.52120971,  0.62599219,  0.60670344,  0.64939623,  0.66476035,
   0.57942122,  0.58769433],
 [ 0.61326625,  0.73667521,  0.69772857,  0.70799756,  0.70130226,
   0.88778378,  0.78004956,  0.81533   ,  0.79334535,  0.84026244,
   0.78777807,  1.06903875,  0.92203704,  1.05085849,  1.08559549,
   0.92212245,  0.95490221],
 [ 0.83355431,  0.99162086,  0.94847794,  0.94778577,  0.93032107,
   1.20039494,  1.02531004,  1.10110907,  1.12784935,  1.14180163,
   1.16252589,  2.41443453,  1.4680712 ,  2.01575516,  2.07557796,
   1.41611013,  2.55324689],
 [ 0.99876252,  1.14588088,  1.1337413 ,  1.08897908,  1.07282201,
   1.41986819,  1.23850737,  1.30133998,  1.54598072,  1.36594854,
   1.5910653 ,  7.82148256,  3.62344349,  6.18914874,  7.43988258,
   4.79416022, 10.73121485],
 [ 1.0920174 ,  1.27433405,  1.25265071,  1.22354255,  1.19686001,
   1.66254936,  1.46853709,  1.52640552,  1.89244306,  1.6660107 ,
   1.98502866, 21.95254398,  9.29173142, 12.37688205, 16.00122101,
   14.9536625 , 23.63879151]])

# apt aug rot prob 0.6
ss = np.array(
[[ 0.472906  ,  0.61709928,  0.71558478,  0.81267545,  0.61765035,
   0.73479622,  0.61955484,  0.68730539,  0.698984  ,  0.677136  ,
   0.61541386,  0.75555026,  0.69076107,  0.75235497,  0.72720588,
   0.67876725,  0.71790177],
 [ 0.68834537,  0.95497615,  1.01512711,  1.09898932,  0.8902714 ,
   1.0579848 ,  0.90654558,  1.01668055,  1.05541819,  0.99681618,
   0.95715278,  1.23611664,  1.07940735,  1.22926565,  1.20737407,
   1.05584737,  1.12630234],
 [ 0.90519511,  1.30699837,  1.29407239,  1.32676986,  1.11978306,
   1.38312497,  1.16475055,  1.32682853,  1.49087463,  1.32708495,
   1.43301442,  3.39106539,  1.6433613 ,  2.38523029,  2.48743473,
   1.56946603,  2.8602227 ],
 [ 1.0465222 ,  1.58375439,  1.52758524,  1.50897132,  1.29043958,
   1.6337306 ,  1.38553367,  1.58369717,  1.93141422,  1.6349563 ,
   1.98168825, 10.91848615,  3.67807165,  5.84358823,  9.38764822,
   3.45153664,  9.5395739 ],
 [ 1.19841757,  1.71824677,  1.70693849,  1.6714086 ,  1.40992762,
   1.82041432,  1.622206  ,  1.82655848,  2.57786043,  1.88716294,
   2.42996952, 27.02327344,  9.2154558 , 11.74354145, 17.43093006,
   9.25828066, 20.26719385]])
##
import os
import tensorflow  as tf
def read_and_decode_without_session(filename, n_classes):
    # reads the tf record db. Returns entries at location indices
    # If indices is empty, then it reads the whole database.
    # Instead of conf, n_classes can be also be given

    xx = tf.python_io.tf_record_iterator(filename)
    all_ims = []
    all_locs = []
    all_info = []
    all_occ = []
    all_mask = []
    for ndx, record in enumerate(xx):
        example = tf.train.Example()
        example.ParseFromString(record)
        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        depth = int(example.features.feature['depth'].int64_list.value[0])
        expid = int(example.features.feature['expndx'].float_list.value[0])
        maxn = int(example.features.feature['max_n'].int64_list.value[0])
        t = int(example.features.feature['ts'].float_list.value[0])
        img_string = example.features.feature['image_raw'].bytes_list.value[0]
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((height, width, depth))
        mask_string = example.features.feature['mask'].bytes_list.value[0]
        mask_1d = np.fromstring(mask_string, dtype=np.uint8)
        mask = mask_1d.reshape((height, width))

        locs = np.array(example.features.feature['locs'].float_list.value)
        locs = locs.reshape([maxn, n_classes, 2])
        if 'trx_ndx' in example.features.feature.keys():
            trx_ndx = int(example.features.feature['trx_ndx'].int64_list.value[0])
        else:
            trx_ndx = 0
        if 'occ' in example.features.feature.keys():
            occ = np.array(example.features.feature['occ'].float_list.value)
            occ = occ.reshape([maxn,n_classes,])
        else:
            occ = np.zeros([n_classes,])

        all_ims.append(reconstructed_img)
        all_locs.append(locs)
        all_info.append([expid, t, trx_ndx])
        all_occ.append(occ)
        all_mask.append(mask)

    xx.close()
    return all_ims, all_locs, all_info, all_occ, all_mask

db_file = os.path.join(conf.cachedir,'train_TF.tfrecords')
H = read_and_decode_without_session(db_file,conf.n_classes)

nn = np.random.randint(len(H[0]))
plt.figure()
plt.imshow(H[0][nn]*H[-1][nn][...,np.newaxis])
ll = np.array(H[1][nn])
ll[ll<-1000] = np.nan
plt.scatter(ll[:,:,0],ll[:,:,1])
## Check preprocessing

import PoseTools as pt
allim = []
alllocs = []
for bndx in range(8):
    nn = np.random.randint(len(H[0]))
    im = H[0][nn]*H[-1][nn][...,np.newaxis]
    locs = H[1][nn]
    allim.append(im)
    alllocs.append(locs)

conf.rrange = 180
conf.trange = 50
allim = np.array(allim)
alllocs = np.array(alllocs)
nim,nlocs = pt.preprocess_ims(allim,alllocs,conf,True,1)
f,ax = plt.subplots(1,2)
ax = ax.flatten()
ix = np.random.randint(8)
ax[0].imshow(allim[ix,...])
ll = alllocs.copy()
ll[ll<-10000] = np.nan
ax[0].scatter(ll[ix,:,:,0],ll[ix,:,:,1])

ax[1].imshow(nim[ix,...].astype('uint8'))
nlocs[nlocs<-8000] = np.nan
ax[1].scatter(nlocs[ix,...,0],nlocs[ix,...,1])

## multi animal training
import PoseTools
import re
import h5py
import numpy as np
import APT_interface as apt
import matplotlib
matplotlib.use('TkAgg')

# op_af = '\(0,1\),\(0,5\),\(1,2\),\(3,4\),\(3,5\),\(5,6\),\(5,7\),\(5,9\),\(3,16\),\(9,10\),\(10,15\),\(9,14\),\(4,11\),\(7,8\),\(8,12\),\(7,13\)' chedk
exp_name = 'alice'
exp_name = 'roian'
net_type = 'multi_mdn_joint_torch'
lr_mult = 1.
if exp_name == 'alice':
    lbl_file = '/groups/branson/home/kabram/APT_projects/alice_touch_stripped.lbl'
    n_grid = 4
    sz = np.round(1024 / n_grid).astype('int')
    fill_value = 255
    bb_ex = 10  # extra pixels around bb
    buffer = 60  # roughly half the animal size + bb_ex
    max_n = 6
    af_graph = ((0,1),(1,2),(0,5),(5,3),(3,16),(3,4),(4,11),(5,9),(9,10),(10,15),(5,14),(5,6),(5,13),(5,7),(7,8),(8,12))
    if net_type == 'multi_mdn_joint_torch':
        name = 'fliplandmark'
        batch_size = 6
    elif net_type == 'multi_openpose':
        name= '50k_resnet'
        batch_size = 4
elif exp_name == 'roian':
    lbl_file = '/groups/branson/bransonlab/apt/experiments/data/four_points_all_mouse_linux_tracker_updated20200423_new_skl_20200817.lbl_mdn.lbl'
    n_grid = 8
    sz = np.round(2048 / n_grid).astype('int')
    fill_value = 255
    bb_ex = 40  # extra pixels around bb
    buffer = 170  # roughly half the animal size + bb_ex
    max_n = 2
    af_graph = ((0,1),(0,2),(0,3),(2,3))
    if net_type == 'multi_mdn_joint_torch':
        name = 'try_1'
        batch_size = 6
        lr_mult = 0.2
    elif net_type == 'multi_openpose':
        name= 'try_1'
        batch_size = 6

import os
os.environ['CUDA_VISIBLE_DEVICES']  = '0'
import Pose_multi_mdn_joint
import Pose_multi_openpose
import Pose_multi_mdn_joint_torch

conf = apt.create_conf(lbl_file,0,'deepnet',net_type=net_type,cache_dir='/nrs/branson/mayank/apt_cache_2')
conf.rrange = 180
conf.trange = 50
conf.max_n_animals = max_n
conf.imsz = (sz+2*buffer,sz+2*buffer)
conf.mdn_use_unet_loss = False
conf.img_dim = 3
conf.dl_steps = 100000
conf.op_affinity_graph = af_graph
conf.save_step = 5000
conf.maxckpt = 5
conf.mdn_joint_use_fpn = True
conf.batch_size = batch_size
conf.dl_steps = (conf.dl_steps * 8 // (conf.batch_size*50))*50
conf.learning_rate_multiplier = lr_mult

if exp_name == 'alice':
    conf.horz_flip = True
    conf.flipLandmarkMatches = {'11': 16, '16': 11, '1': 2, '2': 1, '3': 4, '4': 3, '7': 9, '9': 7, '8': 10, '10': 8, '12': 15, '15': 12, '13': 14, '14': 13}

if net_type == 'multi_openpose':
    self = Pose_multi_openpose.Pose_multi_openpose(conf,'50k_resnet')
elif net_type == 'multi_mdn_joint_torch':
    self = Pose_multi_mdn_joint_torch.Pose_multi_mdn_joint_torch(conf,name=name,is_multi=True)
else:
    self = Pose_multi_mdn_joint.Pose_multi_mdn_joint(conf,'50k_low_noise_fpn')

self.train_wrapper()


## multi animal training val
import PoseTools
import re
import h5py
import numpy as np
import APT_interface as apt
import torch
import matplotlib
matplotlib.use('TkAgg')
import os

exp_name = 'alice' #'roian'
net_type = 'multi_mmpose' #'multi_openpose'#'multi_mdn_joint_torch'
scale = 1
flip_idx = {}
if exp_name == 'alice':
    lbl_file = '/groups/branson/home/kabram/APT_projects/alice_touch_stripped.lbl'
    n_grid = 4
    sz = np.round(1024 / n_grid).astype('int')
    fill_value = 255
    bb_ex = 10  # extra pixels around bb
    buffer = 60  # roughly half the animal size + bb_ex
    max_n = 6
    af_graph = ((0,1),(1,2),(0,5),(5,3),(3,16),(3,4),(4,11),(5,9),(9,10),(10,15),(5,14),(5,6),(5,13),(5,7),(7,8),(8,12))
    isz = sz+2*buffer
    flip_idx = {'11': 16, '16': 11, '1': 2, '2': 1, '3': 4, '4': 3, '7': 9, '9': 7, '8': 10, '10': 8, '12': 15, '15': 12, '13': 14, '14': 13}
    if net_type == 'multi_mdn_joint_torch':
        name = '?'
        batch_size = 6
    elif net_type == 'multi_openpose':
        name= '50k_resnet'
        batch_size = 4
    elif net_type == 'multi_mmpose':
        name = 'fixed_lr_mmpose_aug_push10'
        scale = isz/384

elif exp_name == 'roian':
    lbl_file = '/groups/branson/bransonlab/apt/experiments/data/four_points_all_mouse_linux_tracker_updated20200423_new_skl_20200817.lbl_mdn.lbl'
    n_grid = 8
    sz = np.round(2048 / n_grid).astype('int')
    fill_value = 255
    bb_ex = 40  # extra pixels around bb
    buffer = 170  # roughly half the animal size + bb_ex
    max_n = 2
    af_graph = ((0,1),(0,2),(0,3),(2,3))
    isz = sz+2*buffer
    flip_idx = {'2':3,'3':2}
    if net_type == 'multi_mdn_joint_torch':
        name = 'try_1'
        batch_size = 6
    elif net_type == 'multi_openpose':
        name= '50k_resnet'
        batch_size = 6

conf = apt.create_conf(lbl_file,0,'deepnet',net_type=net_type,cache_dir='/nrs/branson/mayank/apt_cache_2')
conf.rrange = 180
conf.trange = 50
conf.max_n_animals = max_n
conf.imsz = (isz,isz)
conf.mdn_use_unet_loss = False
conf.img_dim = 3
conf.op_affinity_graph = af_graph
conf.mdn_joint_use_fpn = True
conf.batch_size = 1
conf.rescale = scale
conf.flipLandmarkMatches = flip_idx

if net_type == 'multi_mmpose':
    db_file = os.path.join(conf.cachedir.replace('multi_mmpose','multi_mdn_joint_torch'), 'val_TF.tfrecords')
else:
    db_file = os.path.join(conf.cachedir,'val_TF.tfrecords')
out = apt.classify_db_all(net_type,conf,db_file,classify_fcn=apt.classify_db_multi,name=name)
torch.cuda.empty_cache()
# net_type = 'multi_openpose'; train_name =  '50k_resnet'
# conf.cachedir = '/nrs/branson/mayank/apt_cache_2/multitarget_bubble/multi_openpose/view_0/deepnet/'
# out1 = apt.classify_db_all(net_type,conf,db_file,classify_fcn=apt.classify_db_multi,name=train_name)

def find_dist_match(dd):
    dout = np.ones_like(dd[:,:,0,:])*np.nan
    yy = np.nanmean(dd,axis=-1)
    for a in range(dd.shape[0]):
        for ndx in range(dd.shape[2]):
            if np.all(np.isnan(yy[a,:,ndx])):
                continue
            r = np.nanargmin(yy[a,:,ndx])
            dout[a,ndx,:] = dd[a,r,ndx,:]
    return dout


pp1 = out[0]
ll1 = out[1]
dd1 = np.linalg.norm(pp1[:,:,np.newaxis,...]-ll1[:,np.newaxis,...],axis=-1)
dd1 = find_dist_match(dd1)
valid = ll1[:,:,0,0]>-1000
dd1_val = dd1[valid,:]
pp2 = out1[0]
ll2 = out1[1]
dd2 = np.linalg.norm(pp2[:,:,np.newaxis,...]-ll2[:,np.newaxis,...],axis=-1)
dd2 = find_dist_match(dd2)
valid = ll2[:,:,0,0]>-1000
dd2_val = dd2[valid,:]
qq2_val = dd2_val.copy()
qq2_val[np.isnan(qq2_val)] = 30.
np.nanpercentile(dd1_val,[50,75,90,95,97],axis=0)
np.nanpercentile(dd2_val,[50,75,90,95,97],axis=0)
np.percentile(qq2_val,[50,75,90,95,97],axis=0)

##
dd1_val_op_res_roian = np.array(
      [[ 1.93155229,  1.44328667,  3.48972097,  3.4300447 ],
       [ 3.26580622,  2.24146619,  5.73207368,  5.44089424],
       [ 5.55472001,  3.5796961 ,  8.30318538,  7.97684309],
       [10.04838788,  6.43453273, 10.99336949, 10.57449935],
       [17.66234382, 30.39128443, 19.27057629, 14.19206188]])

dd1_val_res_roian = np.array(
      [[ 2.11400175,  1.55124713,  3.55102363,  3.64351822],
       [ 3.41642902,  2.46662894,  5.63632769,  5.56673483],
       [ 6.2260891 ,  3.77569077,  7.91711641,  7.74754844],
       [ 8.91358103,  5.63457711, 10.34090886,  9.60476665],
       [11.66999203,  8.26534379, 12.65886563, 11.002267  ]])

dd1_val_res_alice = np.array(
      [[0.37571816, 0.40797099, 0.42281148, 0.44628574, 0.41362639,
        0.55537074, 0.4966842 , 0.54103561, 0.5501459 , 0.53170573,
        0.55939253, 0.63851684, 0.55768218, 0.60753119, 0.60699976,
        0.52606894, 0.59576524],
       [0.55557696, 0.60038199, 0.63021987, 0.65290058, 0.62176412,
        0.82956788, 0.73590624, 0.80554673, 0.85593521, 0.80739034,
        0.8399457 , 1.15038564, 0.88735888, 1.06242606, 1.06848452,
        0.82438128, 1.09069089],
       [0.76082594, 0.80857697, 0.89380999, 0.86254469, 0.84723529,
        1.13587254, 1.02566256, 1.17449165, 1.2557275 , 1.10332146,
        1.33260235, 3.29338253, 1.55085631, 2.85808842, 2.9436011 ,
        1.44422769, 3.22910022],
       [0.90692339, 0.91713381, 1.0813115 , 0.99307531, 1.01939656,
        1.38085315, 1.2122561 , 1.42527569, 1.73907163, 1.34316353,
        1.75428751, 5.50974705, 3.69980195, 5.84442034, 5.96314512,
        3.60767653, 5.37650514],
       [1.02197887, 1.03388241, 1.19648805, 1.12134421, 1.15200746,
        1.55346395, 1.40153364, 1.59385141, 2.16816316, 1.60542645,
        2.15471344, 7.62781143, 6.2059065 , 8.34251727, 8.98594804,
        6.70788806, 7.22719603]])

dd1_val_res_alice_mmpose_multi_4x = np.array(
  [[2.31168254,  2.18079821,  2.19741097,  2.2162855 ,  2.21384254,
    2.13330316,  2.08430547,  2.22649612,  2.31377776,  2.24285992,
    2.28301735,  2.54500892,  2.48847715,  2.52530095,  2.53395845,
    2.39067206,  2.43953998],
  [ 2.959758  ,  2.74827434,  2.73005754,  2.77373468,  2.77394325,
    2.72736219,  2.69065558,  2.87965297,  2.97273466,  2.90070478,
    3.02098812,  3.37120098,  3.3260235 ,  3.46245833,  3.40822568,
    3.23444931,  3.28142832],
  [ 3.50420204,  3.2642192 ,  3.20540946,  3.28695091,  3.27524705,
    3.2806699 ,  3.24096361,  3.45807699,  3.66693846,  3.45463597,
    3.71212903,  4.9175951 ,  4.24302295,  5.1246264 ,  4.95434497,
    4.06667628,  4.8904174 ],
  [ 3.82775517,  3.57035386,  3.47345152,  3.54631505,  3.53006889,
    3.66745447,  3.57135797,  3.82321425,  4.26098431,  3.81139869,
    4.29146675,  7.94482038,  6.02804177,  8.70994202,  7.89355757,
    5.41205647,  8.0632981 ],
  [ 4.06034263,  3.75539379,  3.65198986,  3.7113408 ,  3.66108634,
    3.89928751,  3.79057812,  4.06420915,  4.60830353,  4.00542101,
    4.76583314, 10.42229588,  9.2771437 , 11.76617461, 11.72781675,
    8.91533821, 13.86395213]])

## few examples of both
import multiResData
import PoseTools as pt
import matplotlib
matplotlib.use('TkAgg')
B = multiResData.read_and_decode_without_session_multi(db_file,conf.n_classes)
##

def plot_sk(inp,c=[0.3,0.3,0.3]):
    sk = conf.op_affinity_graph
    for ix in range(inp.shape[0]):
        for s in sk:
            xs = [inp[ix,s[0],0],inp[ix,s[1],0]]
            ys = [inp[ix,s[0],1],inp[ix,s[1],1]]
            plt.plot(xs,ys,c=c)

def plot_rel(inp1,inp2,c=[0.3,0.3,0.3]):
    #inp1 labels
    ss = np.nanmean(np.linalg.norm(inp1[:,np.newaxis,...] - inp2[np.newaxis,...], axis=-1), axis=-1)
    for ix in range(inp1.shape[0]):
        for iy in range(inp1.shape[1]):
            if np.isnan(inp1[ix,iy,0]):
                continue
            pred_match = np.nanargmin(ss[ix, :], 0)
            oo = inp2[pred_match,iy,:]
            ii = inp1[ix,iy,:]
            if np.isnan(oo[0]):
                plt.scatter(ii[0],ii[1],c=c,marker='+',s=70)
            else:
                xs = [inp1[ix,iy,0],oo[0]]
                ys = [inp1[ix,iy,1],oo[1]]
                plt.plot(xs,ys,c=c)


sel = np.where(np.any( ((dd1>30))&(dd1<1000),axis=(1,2)))[0]
# nan_preds = np.isnan(out1[0][:,:,:,0])
# nan_preds = np.any(nan_preds,axis=-1) & (~np.all(nan_preds,axis=-1))
# sel = np.where(np.any( ((dd2>8)|(dd1>8))&(dd1<1000),axis=(1,2)))[0]
# sel = np.where(np.any(nan_preds,axis=1))[0]
ndx = np.random.choice(sel)
cmap = np.tile(pt.get_cmap(conf.n_classes),[max_n,1])
plt.figure()
plt.imshow(B[0][ndx][:,:,0]*B[-1][ndx],'gray')
kk = out[1]
kk[kk<-1000] = np.nan
# plot_rel(kk[ndx,...],out[0][ndx,...],[0,1,0,0.75])
plot_sk(out[0][ndx,...],[0,1,0,0.3])
# plt.scatter(out[0][ndx,:,:,0],out[0][ndx,:,:,1],c=cmap,marker='.')

plot_sk(kk[ndx,...],[0,0,1,0.3])
# plt.scatter(kk[ndx,:,:,0],kk[ndx,:,:,1],c=cmap,marker='+')
# plt.scatter(out1[0][ndx,:,:,0],out1[0][ndx,:,:,1],c=cmap,marker='*')
# plot_sk(out1[0][ndx,...],[1,0,0,0.3])
# plot_rel(kk[ndx,...],out1[0][ndx,...],[1,0,0,0.75])
plt.title('Red:OP, Green:GRoN,Blue:Labels -- {}'.format(ndx))
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()

## multi animal video tracking

import PoseTools
import re
import h5py
import numpy as np
import APT_interface as apt
import matplotlib
matplotlib.use('TkAgg')

# op_af = '\(0,1\),\(0,5\),\(1,2\),\(3,4\),\(3,5\),\(5,6\),\(5,7\),\(5,9\),\(3,16\),\(9,10\),\(10,15\),\(9,14\),\(4,11\),\(7,8\),\(8,12\),\(7,13\)' chedk
lbl_file = '/groups/branson/home/kabram/APT_projects/alice_touch_stripped.lbl'

n_grid = 4
sz = np.round(1024/n_grid).astype('int')
fill_value = 255
bb_ex = 10 # extra pixels around bb
buffer = 60 # roughly half the animal size + bb_ex
max_n = 6

import os
os.environ['CUDA_VISIBLE_DEVICES']  = '0'
import Pose_multi_mdn_joint
import Pose_multi_openpose
import Pose_multi_mdn_joint_torch

# net_type = 'multi_mdn_joint_torch'; name = 'test_fpn_more_noise'
net_type = 'multi_openpose'; name= '50k_resnet'
conf = apt.create_conf(lbl_file,0,'deepnet',net_type=net_type,cache_dir='/nrs/branson/mayank/apt_cache_2')
conf.rrange = 180
conf.trange = 50
conf.max_n_animals = max_n
conf.imsz = (sz+2*buffer,sz+2*buffer)
conf.mdn_use_unet_loss = False
conf.img_dim = 3
conf.dl_steps = 50000
conf.op_affinity_graph = ((0,1),(1,2),(0,5),(5,3),(3,16),(3,4),(4,11),(5,9),(9,10),(10,15),(5,14),(5,6),(5,13),(5,7),(7,8),(8,12))
conf.save_step = 5000
conf.maxckpt = 10
conf.mdn_joint_use_fpn = True

if net_type == 'multi_openpose':
    conf.batch_size = 4
    self = Pose_multi_openpose.Pose_multi_openpose(conf,'50k_resnet')
elif net_type == 'multi_mdn_joint_torch':
    self = Pose_multi_mdn_joint_torch.Pose_multi_mdn_joint_torch(conf,name=name,is_multi=True)
else:
    self = Pose_multi_mdn_joint.Pose_multi_mdn_joint(conf,'50k_low_noise_fpn')


mov = '/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00038_CsChr_RigB_20150729T150617/movie.ufmf'
import movies
cap = movies.Movie(mov)

##
fr = cap.get_frame(33)[0]
n_rep = 3
fr = np.tile(fr[:,:,np.newaxis],[n_rep,n_rep,3])

max_n_track = 10 * n_rep**2
self.conf.batch_size = 1
pfn,cfn,_ = self.get_pred_fn(max_n=max_n_track,imsz=fr.shape[:2])

oo = pfn(fr[np.newaxis,...])['locs']
plt.figure()
plt.imshow(fr[:,:,0],'gray')
import PoseTools as pt
cmap = np.tile(pt.get_cmap(oo.shape[2]),[max_n_track,1])
plt.scatter(oo[0,:,:,0],oo[0,:,:,1],c=cmap)

##

import multiResData
import PoseTools as pt
import matplotlib
matplotlib.use('TkAgg')
import open_pose4 as op
B = multiResData.read_and_decode_without_session_multi(db_file,conf.n_classes)

##

nan_preds = np.isnan(out1[0][:,:,:,0])
nan_preds = np.any(nan_preds,axis=-1) & (~np.all(nan_preds,axis=-1))
sel = np.where(np.any(nan_preds,axis=1))[0]

model = op.model_test(conf.op_imsz_net,
                   backbone=conf.op_backbone,
                   nPAFstg=conf.op_paf_nstage,
                   nMAPstg=conf.op_map_nstage,
                   nlimbsT2=len(conf.op_affinity_graph) * 2,
                   npts=conf.n_classes,
                   doDC=conf.op_hires,
                   nDC=conf.op_hires_ndeconv,
                   fullpred=conf.op_pred_raw)
latest_model_file = PoseTools.get_latest_model_file_keras(conf, train_name)
model.load_weights(latest_model_file)
# thre2 = conf.get('op_param_paf_thres',0.05)
thre_hm = conf.get('op_param_hmap_thres', 0.1)
thre_paf = conf.get('op_param_paf_thres', 0.05)

op_pred_simple = conf.get('op_pred_simple', False)
op_inference_old = conf.get('op_inference_old', False)

## Show OP errors
import tfdatagen
locs_sz = (conf.batch_size, conf.n_classes, 2)
locs_dummy = np.zeros(locs_sz)
ndx = np.random.choice(sel)
all_f = B[0][ndx][np.newaxis,...]*B[4][ndx][np.newaxis,...,np.newaxis]
ims, _ = tfdatagen.ims_locs_preprocess_openpose(all_f, locs_dummy, conf, False, gen_target_hmaps=False)
model_preds = model.predict(ims)
kk = op.do_inference(model_preds[1][0,...],model_preds[0][0,...],conf,thre_hm,thre_paf)
plt.close('all')
dndx = np.where(np.isnan(kk[0,:,0]))[0][0]
aa = np.array(conf.op_affinity_graph)
ee = np.where(aa[:,1]==dndx)[0][0]
plt.figure(); plt.imshow(all_f[0,:,:,0],'gray')
plt.scatter(B[1][ndx][0,dndx,0],B[1][ndx][0,dndx,1])
plt.scatter(B[1][ndx][0,aa[ee,0],0],B[1][ndx][0,aa[ee,0],1])
plt.title('{}'.format(dndx))
plt.figure(); plt.imshow(model_preds[0][0,:,:,ee*2])
plt.figure(); plt.imshow(model_preds[0][0,:,:,ee*2+1])
kk[:2,:,0]

## full image validation

import h5py
import PoseTools as pt
import json
import APT_interface as apt
import Pose_mdn_joint_fpn
import multiResData
import movies
import tensorflow as tf
import os
from multiResData import float_feature, int64_feature,bytes_feature,trx_pts, check_fnum

info_in = '/nrs/branson/mayank/apt_cache_2/multitarget_bubble/multi_openpose/view_0/deepnet/splitdata.json'
multi_split = pt.json_load(info_in)
lbl = h5py.File(lbl_file,'r')

exp_list = lbl['movieFilesAll'][0, :]
local_dirs = [u''.join(chr(c) for c in lbl[jj]) for jj in exp_list]
try:
    for k in lbl['projMacros'].keys():
        r_dir = u''.join(chr(c) for c in lbl['projMacros'][k])
        local_dirs = [s.replace('${}'.format(k), r_dir) for s in local_dirs]
except:
    pass

conf = apt.create_conf(lbl_file,0,'deepnet','/nrs/branson/mayank/apt_cache_2','multi_mdn_joint_torch')

val_db = os.path.join(conf.cachedir, 'val_fullims_TF.tfrecords')
val_env = tf.python_io.TFRecordWriter(val_db)

ii = []
for ndx in range(len(local_dirs)):
    ii.append(apt.trx_pts(lbl, ndx).shape[0])
max_n = max(ii)

for ndx, dir_name in enumerate(local_dirs):
    cap = movies.Movie(dir_name)
    curpts = apt.trx_pts(lbl,ndx,False)
    cur_occ = apt.trx_pts(lbl, ndx, field_name='labeledpostag')
    label_fr = np.where(np.any(~np.isnan(curpts[...,0]),axis=(-1,0)))[0]
    for fr in label_fr:
        if not ([ndx,fr] in multi_split[1]):
            continue
        frame_in = cap.get_frame(fr)[0]
        cur_loc = np.ones([max_n,conf.n_classes,2])*np.nan
        cur_loc[:curpts.shape[0],...] = curpts[:,fr,...].transpose([0,2,1])
        curo = np.zeros([max_n,conf.n_classes])
        curo[:curpts.shape[0],:] = cur_occ[:,fr,:]
        trx_ndx = 0
        info = [ndx,fr]

        rows = frame_in.shape[0]
        cols = frame_in.shape[1]
        depth = frame_in.shape[2] if frame_in.ndim > 2 else 1
        mask = np.ones(frame_in.shape[:2])
        mask = mask.astype('uint8')
        mask_raw = mask.tostring()
        image_raw = frame_in.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': int64_feature(rows),
            'width': int64_feature(cols),
            'depth': int64_feature(depth),
            'trx_ndx': int64_feature(trx_ndx),
            'locs': float_feature(cur_loc.flatten()),
            'expndx': float_feature(ndx),
            'max_n': int64_feature(max_n),
            'ts': float_feature(fr),
            'mask': bytes_feature(mask_raw),
            'occ': float_feature(curo.flatten()),
            'image_raw': bytes_feature(image_raw)}))
        val_env.write(example.SerializeToString())
    cap.close()

val_env.close()
##

import PoseTools
import re
import h5py
import numpy as np
import APT_interface as apt
import torch
import matplotlib
matplotlib.use('TkAgg')

op_af = '\(0,1\),\(0,5\),\(1,2\),\(3,4\),\(3,5\),\(5,6\),\(5,7\),\(5,9\),\(3,16\),\(9,10\),\(10,15\),\(9,14\),\(4,11\),\(7,8\),\(8,12\),\(7,13\)'
lbl_file = '/groups/branson/home/kabram/APT_projects/alice_touch_stripped.lbl'

n_grid = 4
sz = np.round(1024/n_grid).astype('int')
fill_value = 255
bb_ex = 10 # extra pixels around bb
buffer = 60 # roughly half the animal size + bb_ex
max_n = 45

import os
os.environ['CUDA_VISIBLE_DEVICES']  = '0'
import Pose_multi_mdn_joint
import Pose_multi_openpose
# net_type = 'multi_openpose'; train_name =  '50k_resnet'
# net_type = 'multi_mdn_joint'; train_name = '50k_low_noise'
net_type = 'multi_mdn_joint_torch'; train_name = 'test_fpn_more_noise'

conf = apt.create_conf(lbl_file,0,'deepnet',net_type=net_type,cache_dir='/nrs/branson/mayank/apt_cache_2')
conf.rrange = 180
conf.trange = 50
conf.max_n_animals = max_n
conf.imsz = (1024,1024)
conf.mdn_use_unet_loss = False
conf.img_dim = 3
conf.op_affinity_graph = ((0,1),(1,2),(0,5),(5,3),(3,16),(3,4),(4,11),(5,9),(9,10),(10,15),(5,14),(5,6),(5,13),(5,7),(7,8),(8,12))
conf.mdn_joint_use_fpn = True

db_file = os.path.join(conf.cachedir,'val_fullims_TF.tfrecords')
out = apt.classify_db_all(net_type,conf,db_file,classify_fcn=apt.classify_db_multi,name=train_name)
torch.cuda.empty_cache()
net_type = 'multi_openpose'; train_name =  '50k_resnet'
conf.cachedir = '/nrs/branson/mayank/apt_cache_2/multitarget_bubble/multi_openpose/view_0/deepnet/'
conf.batch_size = 1
# out1 = apt.classify_db_all(net_type,conf,db_file,classify_fcn=apt.classify_db_multi,name=train_name)

def find_dist_match(dd):
    dout = np.ones_like(dd[:,:,0,:])*np.nan
    yy = np.nanmean(dd,axis=-1)
    for a in range(dd.shape[0]):
        for ndx in range(dd.shape[2]):
            if np.all(np.isnan(yy[a,:,ndx])):
                continue
            r = np.nanargmin(yy[a,:,ndx])
            dout[a,ndx,:] = dd[a,r,ndx,:]
    return dout


pp1 = out[0]
ll1 = out[1]
dd1 = np.linalg.norm(pp1[:,:,np.newaxis,...]-ll1[:,np.newaxis,...],axis=-1)
dd1 = find_dist_match(dd1)
valid = ll1[:,:,0,0]>-1000
dd1_val = dd1[valid,:]

pp2 = out1[0]
ll2 = out1[1]
dd2 = np.linalg.norm(pp2[:,:,np.newaxis,...]-ll2[:,np.newaxis,...],axis=-1)
dd2 = find_dist_match(dd2)
valid = ll2[:,:,0,0]>-1000
dd2_val = dd2[valid,:]
qq2_val = dd2_val.copy()
qq2_val[np.isnan(qq2_val)] = 1000.
np.nanpercentile(dd1_val,[50,75,90,95,97],axis=0)
np.nanpercentile(dd2_val,[50,75,90,95,97],axis=0)
np.percentile(qq2_val,[50,75,90,95,97],axis=0)

pp3 = out2[0]
ll3 = out2[1]
dd3 = np.linalg.norm(pp3[:,:,np.newaxis,...]-ll3[:,np.newaxis,...],axis=-1)
dd3 = find_dist_match(dd3)
valid = ll3[:,:,0,0]>-1000
dd3_val = dd3[valid,:]
qq3_val = dd3_val.copy()
qq3_val[np.isnan(qq3_val)] = 1000.
np.nanpercentile(dd1_val,[50,75,90,95,97],axis=0)
np.nanpercentile(dd2_val,[50,75,90,95,97],axis=0)
np.percentile(qq2_val,[50,75,90,95,97],axis=0)

##

import multiResData
import PoseTools
db_file = '/nrs/branson/mayank/apt_cache_2/multitarget_bubble/mdn_joint_fpn/view_0/multi_compare/val_TF.tfrecords'
A = multiResData.read_and_decode_without_session(db_file,17,())
ex_im = A[0][0]
ex_loc = np.array(A[1][0])

all_prc = {}
dd1_val_prev_trx = np.array(
      [[0.37571816, 0.40797099, 0.42281148, 0.44628574, 0.41362639,
        0.55537074, 0.4966842 , 0.54103561, 0.5501459 , 0.53170573,
        0.55939253, 0.63851684, 0.55768218, 0.60753119, 0.60699976,
        0.52606894, 0.59576524],
       [0.55557696, 0.60038199, 0.63021987, 0.65290058, 0.62176412,
        0.82956788, 0.73590624, 0.80554673, 0.85593521, 0.80739034,
        0.8399457 , 1.15038564, 0.88735888, 1.06242606, 1.06848452,
        0.82438128, 1.09069089],
       [0.76082594, 0.80857697, 0.89380999, 0.86254469, 0.84723529,
        1.13587254, 1.02566256, 1.17449165, 1.2557275 , 1.10332146,
        1.33260235, 3.29338253, 1.55085631, 2.85808842, 2.9436011 ,
        1.44422769, 3.22910022],
       [0.90692339, 0.91713381, 1.0813115 , 0.99307531, 1.01939656,
        1.38085315, 1.2122561 , 1.42527569, 1.73907163, 1.34316353,
        1.75428751, 5.50974705, 3.69980195, 5.84442034, 5.96314512,
        3.60767653, 5.37650514],
       [1.02197887, 1.03388241, 1.19648805, 1.12134421, 1.15200746,
        1.55346395, 1.40153364, 1.59385141, 2.16816316, 1.60542645,
        2.15471344, 7.62781143, 6.2059065 , 8.34251727, 8.98594804,
        6.70788806, 7.22719603]])

all_prc['trx_gron'] = dd1_val_prev_trx
all_prc['multi_gron'] = np.nanpercentile(dd1_val,[50,75,90,95,97],axis=0)
all_prc['openpose no NaN'] = np.nanpercentile(dd2_val,[50,75,90,95,97],axis=0)
all_prc['openpose with NaN'] = np.percentile(qq2_val,[50,75,90,95,97],axis=0)
# all_prc['mmpose no NaN'] = np.nanpercentile(dd3_val,[50,75,90,95,97],axis=0)
# all_prc['mmpose with NaN'] = np.percentile(qq3_val,[50,75,90,95,97],axis=0)


n_types = len(all_prc)
nc = n_types  # int(np.ceil(np.sqrt(n_types)))
nr = 1  # int(np.ceil(n_types/float(nc)))
nc = int(2)
nr = int(2)
cmap = PoseTools.get_cmap(5, 'cool')
f, axx = plt.subplots(nr, nc, figsize=(12, 8), squeeze=False)
axx = axx.flat
for idx,k  in enumerate(all_prc.keys()):
    ax = axx[idx]
    if ex_im.ndim == 2:
        ax.imshow(ex_im, 'gray')
    elif ex_im.shape[2] == 1:
        ax.imshow(ex_im[:, :, 0], 'gray')
    else:
        ax.imshow(ex_im)

    mm = all_prc[k]
    for pt in range(ex_loc.shape[0]):
        for pp in range(mm.shape[0]):
            c = plt.Circle(ex_loc[pt, :], mm[pp, pt], color=cmap[pp, :], fill=False)
            ax.add_patch(c)
    ttl = '{} '.format(k)
    ax.set_title(ttl)
    ax.axis('off')

f.tight_layout()

## compare centernet to gron on alice

import PoseTools as pt
import multiResData
import APT_interface as apt

exp = 'alice_hg'; npts = 17; pname = 'multitarget_bubble'; op_af_graph = [(0,1),(0,5),(1,2),(3,4),(3,5),(5,6),(5,7),(5,9),(3,16),(9,10),(10,15),(9,14),(4,11),(7,8),(8,12),(7,13)]; lbl_file = '/groups/branson/home/kabram/APT_projects/alice_touch_stripped.lbl'

# exp = 'roian'; npts = 4; pname = 'four_points_180806'; lbl_file = '/groups/branson/bransonlab/apt/experiments/data/four_points_all_mouse_linux_tracker_updated20200423_new_skl_20200817.lbl_mdn.lbl'; op_af_graph  = [[0,1],[0,2],[0,3],[2,3]]

conf = apt.create_conf(lbl_file,0,'deepnet',net_type='mdn',cache_dir='/nrs/branson/mayank/apt_cache_2')
conf.op_affinity_graph = op_af_graph

D = multiResData.read_and_decode_without_session_multi('/nrs/branson/mayank/apt_cache_2/{}/multi_mdn_joint_torch/view_0/deepnet/val_TF.tfrecords'.format(pname),npts)
nmax = D[1][0].shape[0]
nex = len(D[0])

A = pt.json_load('/groups/branson/bransonlab/mayank/code/CenterNet/exp/multi_pose/{}/results.json'.format(exp))

locs = np.ones([nex,nmax,npts,2])*np.nan
exid = [aa['image_id'] for aa in A]
done_ex = np.zeros([nex]).astype('int')
for aa in A:
    id = aa['image_id']
    if done_ex[id]>=nmax:
        continue
    if aa['score'] > 0.1:
        cc = np.reshape(aa['keypoints'],[npts,3])
        locs[id,done_ex[id],:,:] = cc[:,:2]
        done_ex[id] += 1


def find_dist_match(dd):
    dout = np.ones_like(dd[:,:,0,:])*np.nan
    yy = np.nanmean(dd,axis=-1)
    for a in range(dd.shape[0]):
        for ndx in range(dd.shape[2]):
            if np.all(np.isnan(yy[a,:,ndx])):
                continue
            r = np.nanargmin(yy[a,:,ndx])
            dout[a,ndx,:] = dd[a,r,ndx,:]
    return dout


ll1 = np.array(D[1])
dd1 = np.linalg.norm(locs[:,:,np.newaxis,...]-ll1[:,np.newaxis,...],axis=-1)
dd1 = find_dist_match(dd1)

valid = ll1[:,:,0,0]>-1000
dd1_val = dd1[valid,:]
qq1_val = dd1_val.copy()
qq1_val[np.isnan(qq1_val)] = 30.
np.nanpercentile(dd1_val,[50,75,90,95,97],axis=0)

c_result_roian_dla = np.array(
      [[ 2.23831939,  2.12836239,  3.56787174,  3.77595307],
       [ 3.66429912,  3.0600084 ,  5.71371946,  5.40865326],
       [ 6.3579376 ,  4.41793902,  8.16115064,  7.57496198],
       [ 9.20927934,  6.23117073, 10.52501399,  9.61082781],
       [13.33453599,  8.24206581, 12.56785246, 11.33315241]])

c_result_alice_hg = np.array([[
    0.42452105,  0.57163937,  0.58948338,  0.65482018,  0.61912023,
    0.6414152 ,  0.52744085,  0.60217338,  0.58686243,  0.57859589,
    0.60845803,  0.78675463,  0.65796309,  0.80786939,  0.85656303,
    0.64879231,  0.76044465],
  [ 0.64571809,  0.98196348,  1.05389202,  1.14444503,  1.11539536,
    0.97638182,  0.80639031,  0.91338946,  0.9697707 ,  0.88434371,
    0.96669426,  1.56283533,  1.14602571,  1.5891312 ,  1.61698126,
    1.00673281,  1.4735257 ],
  [ 0.93031874,  1.73495499,  1.62646151,  1.81581743,  1.77698935,
    1.35404999,  1.15702064,  1.2828805 ,  1.54864412,  1.30273486,
    1.54949345,  3.63199161,  2.08244296,  3.71387576,  4.04830881,
    1.94610988,  3.58881572],
  [ 1.13077864,  2.22823912,  2.08272843,  2.3241055 ,  2.28804197,
    1.63121107,  1.48734709,  1.55404458,  2.08671475,  1.66769491,
    2.01426688,  5.86250337,  4.11106801,  6.78195731,  7.77633905,
    3.76331868,  5.84498575],
  [ 1.31136543,  2.56204075,  2.27455851,  2.53161581,  2.67714002,
    1.91310023,  1.75877217,  1.8798918 ,  2.74566017,  1.99038072,
    2.67787528,  8.35709419,  6.0484933 , 10.3690371 , 10.32800727,
    6.77214867,  8.04431186]])
##

def plot_sk(inp,c=[0.3,0.3,0.3]):
    sk = conf.op_affinity_graph
    for ix in range(inp.shape[0]):
        for s in sk:
            xs = [inp[ix,s[0],0],inp[ix,s[1],0]]
            ys = [inp[ix,s[0],1],inp[ix,s[1],1]]
            plt.plot(xs,ys,c=c)

def plot_rel(inp1,inp2,c=[0.3,0.3,0.3]):
    #inp1 labels
    ss = np.nanmean(np.linalg.norm(inp1[:,np.newaxis,...] - inp2[np.newaxis,...], axis=-1), axis=-1)
    for ix in range(inp1.shape[0]):
        for iy in range(inp1.shape[1]):
            if np.isnan(inp1[ix,iy,0]):
                continue
            pred_match = np.nanargmin(ss[ix, :], 0)
            oo = inp2[pred_match,iy,:]
            ii = inp1[ix,iy,:]
            if np.isnan(oo[0]):
                plt.scatter(ii[0],ii[1],c=c,marker='+',s=70)
            else:
                xs = [inp1[ix,iy,0],oo[0]]
                ys = [inp1[ix,iy,1],oo[1]]
                plt.plot(xs,ys,c=c)


sel = np.where(np.any( ((dd1>30))&(dd1<1000),axis=(1,2)))[0]
# nan_preds = np.isnan(out1[0][:,:,:,0])
# nan_preds = np.any(nan_preds,axis=-1) & (~np.all(nan_preds,axis=-1))
# sel = np.where(np.any( ((dd2>8)|(dd1>8))&(dd1<1000),axis=(1,2)))[0]
# sel = np.where(np.any(nan_preds,axis=1))[0]
ndx = np.random.choice(sel)
ndx = np.random.choice(range(dd1.shape[0]))
cmap = np.tile(pt.get_cmap(17),[nmax,1])
plt.figure()
plt.imshow(D[0][ndx][:,:,0]*D[-1][ndx],'gray')
kk = ll1
kk[kk<-1000] = np.nan
# plot_rel(kk[ndx,...],locs[ndx,...],[0,1,0,0.75])
plot_sk(locs[ndx,...],[0,1,0,0.3])
# plt.scatter(out[0][ndx,:,:,0],out[0][ndx,:,:,1],c=cmap,marker='.')

plot_sk(kk[ndx,...],[0,0,1,0.3])
# plt.scatter(kk[ndx,:,:,0],kk[ndx,:,:,1],c=cmap,marker='+')
# plt.scatter(out1[0][ndx,:,:,0],out1[0][ndx,:,:,1],c=cmap,marker='*')
# plot_sk(out1[0][ndx,...],[1,0,0,0.3])
# plot_rel(kk[ndx,...],out1[0][ndx,...],[1,0,0,0.75])
plt.title('Red:OP, Green:GRoN,Blue:Labels -- {}'.format(ndx))
# mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
plt.show()

## Generate MSPN dataset
pt.tfrecord_to_coco('/groups/branson/bransonlab/mayank/apt_cache_2/multitarget_bubble/mdn_joint_fpn/view_0/apt_expt/train_TF.tfrecords',17,'/groups/branson/bransonlab/mayank/code/MSPN/dataset/APT/alice/train','/groups/branson/bransonlab/mayank/code/MSPN/dataset/APT/alice/train_annotations.json',out_size=(192,192),scale=4)
pt.tfrecord_to_coco('/nrs/branson/mayank/apt_cache_2/multitarget_bubble/mdn_joint_fpn/view_0/multi_compare/val_TF.tfrecords',17,'/groups/branson/bransonlab/mayank/code/MSPN/dataset/APT/alice/val','/groups/branson/bransonlab/mayank/code/MSPN/dataset/APT/alice/val_annotations.json',out_size=(192,192),scale=4)


## compare MSPN on trx touch alice

import multiResData
import PoseTools as pt
npts = 17
db_file = '/nrs/branson/mayank/apt_cache_2/multitarget_bubble/mdn_joint_fpn/view_0/multi_compare/val_TF.tfrecords'
D = multiResData.read_and_decode_without_session(db_file,npts,())
nex = len(D[0])
labels = np.array(D[1])

A = pt.json_load('/groups/branson/bransonlab/mayank/code/MSPN/model_logs/kabram/mspn_apt/test_dir/results.json')

locs = np.ones([nex,npts,2])*np.nan
exid = [aa['image_id'] for aa in A]
for aa in A:
    id = aa['image_id']
    cc = np.reshape(aa['keypoints'],[npts,3])
    locs[id,:,:] = cc[:,:2]

locs = locs/4
dd = np.linalg.norm(labels-locs,axis=-1)
res = np.percentile(dd,[50,76,90,95,97],axis=0)

res_mspn_touch = np.array(
      [[3.60901082, 2.96437957, 2.61135852, 2.6162504 , 3.1433405 ,
        3.48950909, 3.18417849, 3.18435523, 3.3883829 , 3.34790884,
        3.20125733, 3.52350268, 3.12599089, 3.15357812, 3.31116302,
        3.27867244, 3.32913092],
       [4.09836307, 3.47739892, 3.36126241, 3.22234868, 3.69543699,
        4.04474346, 3.77206427, 3.83164464, 3.98584584, 3.90208611,
        3.83666471, 4.26424979, 3.77530696, 3.80710223, 4.11445949,
        3.94748099, 4.0474106 ],
       [4.46283689, 3.9045543 , 3.86265606, 3.65265912, 4.07312331,
        4.46604299, 4.19194213, 4.3476288 , 4.44527971, 4.4519147 ,
        4.32530642, 4.88614791, 4.33594927, 4.44966767, 4.89149688,
        4.52826267, 4.81770363],
       [4.72568883, 4.17534227, 4.16542228, 4.03816721, 4.29775329,
        4.76213138, 4.50089712, 4.72654555, 4.85519994, 4.79897602,
        4.68061405, 5.57379369, 4.80204168, 5.50112875, 6.0622686 ,
        4.98576904, 5.59061406],
       [4.8642368 , 4.34025935, 4.41453382, 4.33548257, 4.41615825,
        4.95160886, 4.69767496, 4.91355755, 5.10580455, 5.01212075,
        4.93404589, 6.16480284, 5.37515876, 6.84087747, 7.61421033,
        5.4356143 , 6.84915304]])

## plot mspn vs gron
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.ion()
import PoseTools

import multiResData
H = multiResData.read_and_decode_without_session('/nrs/branson/mayank/apt_cache_2/multitarget_bubble/mdn_joint_fpn/view_0/multi_compare/val_TF.tfrecords',17,())
ex_im = H[0][0]
ex_loc = H[1][0]
all_prc = {}
all_prc['MSPN 4x'] = res
all_prc['MSPN 1x'] = res_mspn_touch
all_prc['GRONe'] = ss

n_types = len(all_prc)
nc = n_types  # int(np.ceil(np.sqrt(n_types)))
nr = 1  # int(np.ceil(n_types/float(nc)))
# nc = int(np.ceil(np.sqrt(n_types)))
# nr = int(np.ceil(n_types / float(nc)))
cmap = PoseTools.get_cmap(6, 'cool')
f, axx = plt.subplots(nr, nc, figsize=(12, 8), squeeze=False)
axx = axx.flat
for idx,k  in enumerate(all_prc.keys()):
    ax = axx[idx]
    if ex_im.ndim == 2:
        ax.imshow(ex_im, 'gray')
    elif ex_im.shape[2] == 1:
        ax.imshow(ex_im[:, :, 0], 'gray')
    else:
        ax.imshow(ex_im)

    mm = all_prc[k]
    for pt in range(ex_loc.shape[0]):
        for pp in range(mm.shape[0]):
            c = plt.Circle(ex_loc[pt, :], mm[pp, pt], color=cmap[pp, :], fill=False)
            ax.add_patch(c)
    ttl = '{} '.format(k)
    ax.set_title(ttl)
    ax.axis('off')

f.tight_layout()


## plot center vs gron-multi
c_result_alice_hg = np.array([[
    0.42452105,  0.57163937,  0.58948338,  0.65482018,  0.61912023,
    0.6414152 ,  0.52744085,  0.60217338,  0.58686243,  0.57859589,
    0.60845803,  0.78675463,  0.65796309,  0.80786939,  0.85656303,
    0.64879231,  0.76044465],
[ 0.64571809,  0.98196348,  1.05389202,  1.14444503,  1.11539536,
  0.97638182,  0.80639031,  0.91338946,  0.9697707 ,  0.88434371,
  0.96669426,  1.56283533,  1.14602571,  1.5891312 ,  1.61698126,
   1.00673281,  1.4735257 ],
  [ 0.93031874,  1.73495499,  1.62646151,  1.81581743,  1.77698935,
    1.35404999,  1.15702064,  1.2828805 ,  1.54864412,  1.30273486,
    1.54949345,  3.63199161,  2.08244296,  3.71387576,  4.04830881,
    1.94610988,  3.58881572],
  [ 1.13077864,  2.22823912,  2.08272843,  2.3241055 ,  2.28804197,
    1.63121107,  1.48734709,  1.55404458,  2.08671475,  1.66769491,
    2.01426688,  5.86250337,  4.11106801,  6.78195731,  7.77633905,
    3.76331868,  5.84498575],
  [ 1.31136543,  2.56204075,  2.27455851,  2.53161581,  2.67714002,
    1.91310023,  1.75877217,  1.8798918 ,  2.74566017,  1.99038072,
    2.67787528,  8.35709419,  6.0484933 , 10.3690371 , 10.32800727,
    6.77214867,  8.04431186]])

dd1_val_res_alice = np.array(
    [[0.37571816, 0.40797099, 0.42281148, 0.44628574, 0.41362639,
      0.55537074, 0.4966842 , 0.54103561, 0.5501459 , 0.53170573,
      0.55939253, 0.63851684, 0.55768218, 0.60753119, 0.60699976,
      0.52606894, 0.59576524],
     [0.55557696, 0.60038199, 0.63021987, 0.65290058, 0.62176412,
      0.82956788, 0.73590624, 0.80554673, 0.85593521, 0.80739034,
      0.8399457 , 1.15038564, 0.88735888, 1.06242606, 1.06848452,
      0.82438128, 1.09069089],
     [0.76082594, 0.80857697, 0.89380999, 0.86254469, 0.84723529,
      1.13587254, 1.02566256, 1.17449165, 1.2557275 , 1.10332146,
      1.33260235, 3.29338253, 1.55085631, 2.85808842, 2.9436011 ,
      1.44422769, 3.22910022],
     [0.90692339, 0.91713381, 1.0813115 , 0.99307531, 1.01939656,
      1.38085315, 1.2122561 , 1.42527569, 1.73907163, 1.34316353,
      1.75428751, 5.50974705, 3.69980195, 5.84442034, 5.96314512,
      3.60767653, 5.37650514],
     [1.02197887, 1.03388241, 1.19648805, 1.12134421, 1.15200746,
      1.55346395, 1.40153364, 1.59385141, 2.16816316, 1.60542645,
      2.15471344, 7.62781143, 6.2059065 , 8.34251727, 8.98594804,
      6.70788806, 7.22719603]])

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.ion()
import PoseTools

import multiResData
H = multiResData.read_and_decode_without_session('/nrs/branson/mayank/apt_cache_2/multitarget_bubble/mdn_joint_fpn/view_0/multi_compare/val_TF.tfrecords',17,())
ex_im = H[0][0]
ex_loc = H[1][0]
all_prc = {}
all_prc['centernet'] = c_result_alice_hg
all_prc['GRONe'] = dd1_val_res_alice
all_prc['mmpose'] = dd1_val_res_alice_mmpose_multi_4x

n_types = len(all_prc)
nc = n_types  # int(np.ceil(np.sqrt(n_types)))
nr = 1  # int(np.ceil(n_types/float(nc)))
# nc = int(np.ceil(np.sqrt(n_types)))
# nr = int(np.ceil(n_types / float(nc)))
cmap = PoseTools.get_cmap(5, 'cool')
f, axx = plt.subplots(nr, nc, figsize=(12, 8), squeeze=False)
axx = axx.flat
for idx,k  in enumerate(all_prc.keys()):
    ax = axx[idx]
    if ex_im.ndim == 2:
        ax.imshow(ex_im, 'gray')
    elif ex_im.shape[2] == 1:
        ax.imshow(ex_im[:, :, 0], 'gray')
    else:
        ax.imshow(ex_im)

    mm = all_prc[k]
    for pt in range(ex_loc.shape[0]):
        for pp in range(mm.shape[0]):
            c = plt.Circle(ex_loc[pt, :], mm[pp, pt], color=cmap[pp, :], fill=False)
            ax.add_patch(c)
    ttl = '{} '.format(k)
    ax.set_title(ttl)
    ax.axis('off')

f.tight_layout()


## create trn pack for Alice's touch dataset to test it out.
name = 'alice'
op_af = '\(0,1\),\(0,5\),\(1,2\),\(3,4\),\(3,5\),\(5,6\),\(5,7\),\(5,9\),\(3,16\),\(9,10\),\(10,15\),\(9,14\),\(4,11\),\(7,8\),\(8,12\),\(7,13\)'
lbl_file = '/groups/branson/home/kabram/APT_projects/alice_touch_stripped.lbl'
split_file = '/nrs/branson/mayank/apt_cache_2/multitarget_bubble/multi_mdn_joint/view_0/deepnet/splitdata.json'
outdir = '/nrs/branson/mayank/apt_cache_2/multitarget_bubble/multi_mdn_joint_torch/view_0/deepnet'
import h5py
import PoseTools as pt
import movies
import multiResData
import os
import cv2
import json
lbl = h5py.File(lbl_file,'r')
nmov = lbl['labeledpos'].value.shape[1]
sdata = pt.json_load(split_file)

imdir = os.path.join(outdir,'im')
loc_file = os.path.join(outdir,'loc.json')
imgid = 0
out_info = []
for ndx in range(nmov):
    mov_file = u''.join(chr(c) for c in lbl[lbl['movieFilesAll'][0,ndx]])
    cur_cap = movies.Movie(mov_file)
    cur_pts = multiResData.trx_pts(lbl,ndx)
    allfr = np.where(np.invert(np.all(np.isnan(cur_pts), axis=(0, 2, 3))))[0]
    for fndx,fr in enumerate(allfr):
        curi = cur_cap.get_frame(fr)[0]
        curl = cur_pts[:,fr,...]
        img_path = os.path.join(outdir,'im','{:08d}.png'.format(imgid))
        cv2.imwrite(img_path,curi)
        sel = np.where(np.invert(np.all(np.isnan(curl), axis=(1, 2))))[0]
        out_locs = []
        out_occ = []
        out_ts = []
        for s in sel:
            out_locs.append(curl[s,...].flatten().tolist())
            out_occ.append(np.zeros([17]).tolist())

        cursplit = 0 if [ndx,fr] in sdata[0] else 1
        cur_out = {'pabs':out_locs,'occ':out_occ,'id':imgid,'img':img_path,'frm':int(fr),'imov':ndx,'itgt':[1],'ts':[1,],'roi':[],'ntgt':len(sel),'split':cursplit}
        out_info.append(cur_out)
        imgid +=1

out_info = {'locdata':out_info,'splitnames':['train','val']}
with open(loc_file,'w') as f:
    json.dump(out_info,f)


## train mmpose on allen's trnpack

import APT_interface as apt
import poseConfig
from Pose_multi_mmpose import Pose_multi_mmpose

lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'
conf = apt.create_conf(lbl_file,0,'apt','/nrs/branson/mayank/apt_cache_2','multi_mmpose')

conf.db_format = 'coco'
conf.dl_steps = 500
conf.nviews = 1
conf.view = 0
conf.n_classes = 17
conf.is_multi = True
conf.mmpose_net = 'higherhrnet'
conf.json_trn_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/loc.json'
conf.set_exp_name('alice')

apt.setup_ma(conf)
# apt.create_coco_db(conf,True)
self = Pose_multi_mmpose(conf,'higherhr')
self.train_wrapper()

## cross-validation for single animal touch label for alice using mdn_unet. Nov 26 2020


name = 'alice'
op_af = '\(0,1\),\(0,5\),\(1,2\),\(3,4\),\(3,5\),\(5,6\),\(5,7\),\(5,9\),\(3,16\),\(9,10\),\(10,15\),\(9,14\),\(4,11\),\(7,8\),\(8,12\),\(7,13\)'
lbl_file = '/groups/branson/home/kabram/APT_projects/alice_touch_stripped_20200811.lbl'


import PoseTools
import re
import h5py
import numpy as np

import multiResData
import numpy as np
import movies
import PoseTools as pt

lbl = h5py.File(lbl_file,'r')
nmov = lbl['labeledpos'].value.shape[1]

splits = [[],[],[]]
for ndx in range(nmov):
    mov_file = u''.join(chr(c) for c in lbl[lbl['movieFilesAll'][0,ndx]])
    cur_cap = movies.Movie(mov_file)
    cur_pts = multiResData.trx_pts(lbl,ndx)
    allfr = np.where(np.invert(np.all(np.isnan(cur_pts), axis=(0, 2, 3))))[0]
    np.random.shuffle(allfr)
    ss = [allfr[::3], allfr[1::3], allfr[2::3]]
    for sndx in range(3):
        for curs in ss[sndx]:
            curt = np.where(np.invert(np.all(np.isnan(cur_pts[:,curs,...]), axis=(1, 2))))[0]
            for tt in curt:
                splits[sndx].append([ndx,int(curs),int(tt)])

import os, json
import APT_interface as apt
out_dir = '/nrs/branson/mayank/apt_cache_2/multitarget_bubble'
for ndx in range(3):
    st = splits[(ndx+1)%3] + splits[(ndx+2)%3]
    sv = splits[ndx]
    out_file = os.path.join(out_dir,'cv_split_touch_framewise_{}.json'.format(ndx))
    with open(out_file,'w') as f:
        json.dump([st,sv],f)
    conf = apt.create_conf(lbl_file,0,'alice_touch_xv_{}'.format(ndx),'/nrs/branson/mayank/apt_cache_2',net_type='mdn_unet')
    conf.splitType = 'predefined'
    apt.create_tfrecord(conf,True,split_file=out_file)


## script to run the jobs.
import APT_interface as apt
import Pose_mdn_unet

ndx = 0
lbl_file = '/groups/branson/home/kabram/APT_projects/alice_touch_stripped_20200811.lbl'
conf = apt.create_conf(lbl_file, 0, 'alice_touch_xv_{}'.format(ndx), '/nrs/branson/mayank/apt_cache_2',
                       net_type='mdn_unet')

conf.rrange = 15
conf.trange = 5
conf.img_dim = 1
conf.dl_steps = 100000
conf.save_step = 5000
conf.mdn_use_unet_loss = False
conf.horz_flip = True
conf.flipLandmarkMatches = {'11': 16, '16': 11, '1': 2, '2': 1, '3': 4, '4': 3, '7': 9, '9': 7, '8': 10, '10': 8, '12': 15, '15': 12, '13': 14, '14': 13}

self = Pose_mdn_unet.Pose_mdn_unet(conf,name='deepnet')
self.train_wrapper()


## xv results.

import APT_interface as apt
import Pose_mdn_unet

lbl_file = '/groups/branson/home/kabram/APT_projects/alice_touch_stripped_20200811.lbl'
pp = []; ll = []; ii = [];
for ndx in range(3):
    conf = apt.create_conf(lbl_file, 0, 'alice_touch_xv_{}'.format(ndx), '/nrs/branson/mayank/apt_cache_2', net_type='mdn_unet')

    conf.rrange = 15
    conf.trange = 5
    conf.img_dim = 1
    conf.dl_steps = 100000
    conf.save_step = 5000
    conf.mdn_use_unet_loss = False
    conf.horz_flip = True
    conf.flipLandmarkMatches = {'11': 16, '16': 11, '1': 2, '2': 1, '3': 4, '4': 3, '7': 9, '9': 7, '8': 10, '10': 8, '12': 15, '15': 12, '13': 14, '14': 13}

    import APT_interface as apt
    mfile = f'/{conf.cachedir}/deepnet-100000'
    aa = apt.classify_db_all('mdn_unet',conf,f'{conf.cachedir}/val_TF.tfrecords',mfile)
    pp.append(aa[0])
    ll.append(aa[1])
    ii.append(aa[2])

sp = [np.ones(p.shape[0])*(ndx+1) for ndx, p in enumerate(pp) ]
pp1 = np.concatenate(pp,0)
ll1 = np.concatenate(ll,0)
ii1 = apt.to_mat(np.concatenate(ii,0))
sp1 = np.concatenate(sp,0)
dd = np.linalg.norm(pp1-ll1,axis=-1)
ss = np.percentile(dd,[50,75,90,95,97],axis=0)

from scipy import io as sio
sio.savemat('/groups/branson/home/kabram/temp/alice_xv_touch_results_20201221.mat',{'preds':pp1,'labels':ll1,'info':ii1,'split':sp1})

## Training with new touch labels 20201211
# create splits in the trn json file.
trn = '/nrs/branson/mayank/apt_cache_2/alice_ma/loc.json'
split_trn = '/nrs/branson/mayank/apt_cache_2/alice_ma/loc_split.json'
# trn = '/nrs/branson/mayank/apt_cache_2/four_points_180806/loc.json'
# split_trn = '/nrs/branson/mayank/apt_cache_2/four_points_180806/loc_split.json'

import PoseTools as pt
import json
T = pt.json_load(trn)
T['splitnames'] = ['trn','val']
split_ratio = 0.3
for t in T['locdata']:
    if np.random.rand() < split_ratio:
        t['split'] = 1


with open(split_trn,'w') as f:
  json.dump(T,f)

## MA Training command 20201211
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from importlib import reload
import APT_interface as apt
reload(apt)
lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'
split_trn = '/nrs/branson/mayank/apt_cache_2/alice_ma/loc_split.json'
cmd = f'{lbl_file} -json_trn_file {split_trn} -conf_params dl_steps 100000 is_multi True db_format \"coco\" max_n_animals 7 -type multi_mmpose -cache /nrs/branson/mayank/apt_cache_2 train -skip_db'
apt.main(cmd.split())

higherhrnet_2x_cmd = 'python APT_interface.py /nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl -json_trn_file /nrs/branson/mayank/apt_cache_2/alice_ma/loc_split.json -conf_params dl_steps 100000 is_multi True db_format \"coco\" max_n_animals 7 mmpose_net \"higherhrnet_2x\" -train_name higher_2x -type multi_mmpose -cache /nrs/branson/mayank/apt_cache_2 train -skip_db'
## classify db 20201214
#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from importlib import reload
import APT_interface as apt
reload(apt)
import torch
lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'

net_type = 'multi_mmpose' #'multi_mmpose' #
train_name = 'deepnet'

run_name = 'alice_neg_split_mmpose_multi'
# run_name = 'alice_maskloss_split_crop_ims_grone_multi'
conf = apt.create_conf(lbl_file,0,run_name,net_type=net_type,cache_dir='/nrs/branson/mayank/apt_cache_2')
# conf.batch_size = 4 if net_type == 'multi_openpose' else 8
# db_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/val_split_fullims/val_TF.json'
db_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/alice_neg_split_full_ims_20210304/val_TF.json'
conf.db_format = 'coco'
conf.max_n_animals = 10
conf.imsz = (1024,1024) #(288,288)
conf.img_dim = 3
conf.mmpose_net = 'higherhrnet' #'higherhrnet_2x'#
conf.is_multi = True
# conf.op_affinity_graph = ((0,1),(0,5),(1,2),(3,4),(3,5),(5,6),(5,7),(5,9),(3,16),(9,10),(10,15),(9,14),(4,11),(7,8),(8,12),(7,13))
conf.op_affinity_graph = ((0,1),(0,5),(0,2),(5,4),(3,5),(5,6),(5,7),(5,9),(5,16),(9,10),(10,15),(5,14),(5,11),(7,8),(8,12),(5,13))

out = apt.classify_db_all(net_type,conf,db_file,classify_fcn=apt.classify_db_multi,name=train_name)
torch.cuda.empty_cache()

def find_dist_match(dd):
    dout = np.ones_like(dd[:,:,0,:])*np.nan
    yy = np.nanmean(dd,axis=-1)
    for a in range(dd.shape[0]):
        for ndx in range(dd.shape[2]):
            if np.all(np.isnan(yy[a,:,ndx])):
                continue
            r = np.nanargmin(yy[a,:,ndx])
            dout[a,ndx,:] = dd[a,r,ndx,:]
    return dout

import pickle
import PoseTools as pt
with open(os.path.join(conf.cachedir,'val_results_'+train_name+'_'+pt.get_datestr()),'wb') as f:
    pickle.dump({'out':out,'conf':conf},f)

pp1 = out[0]
ll1 = out[1]
dd1 = np.linalg.norm(pp1[:,:,np.newaxis,...]-ll1[:,np.newaxis,...],axis=-1)
dd1 = find_dist_match(dd1)
valid = ll1[:,:,0,0]>-1000
dd1_val = dd1[valid,:]

np.nanpercentile(dd1_val,[50,75,90,95,97],axis=0)


## ht classify db
#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from importlib import reload
import APT_interface as apt
reload(apt)
import torch
lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'

net_type = 'multi_openpose' #'multi_mmpose' #
train_name = 'deepnet'

run_name = 'alice_neg_ht_crop'
conf = apt.create_conf(lbl_file,0,run_name,net_type=net_type,cache_dir='/nrs/branson/mayank/apt_cache_2')
db_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/alice_neg_ht_test_no_crop/val_TF.json'
conf.db_format = 'coco'
conf.max_n_animals = 10
conf.imsz = (1024,1024) #(288,288)
conf.img_dim = 3
conf.mmpose_net = 'higherhrnet' #'higherhrnet_2x'#
conf.is_multi = True
conf.ht_pts = (0,6)
conf.multi_only_ht = True
conf.rescale = 2
conf.n_classes = 2
conf.op_affinity_graph = ((0,1),)

out = apt.classify_db_all(net_type,conf,db_file,classify_fcn=apt.classify_db_multi,name=train_name)
torch.cuda.empty_cache()

def find_dist_match(dd):
    dout = np.ones_like(dd[:,:,0,:])*np.nan
    yy = np.nanmean(dd,axis=-1)
    for a in range(dd.shape[0]):
        for ndx in range(dd.shape[2]):
            if np.all(np.isnan(yy[a,:,ndx])):
                continue
            r = np.nanargmin(yy[a,:,ndx])
            dout[a,ndx,:] = dd[a,r,ndx,:]
    return dout

import pickle
import PoseTools as pt
with open(os.path.join(conf.cachedir,'val_results_'+train_name+'_'+pt.get_datestr()),'wb') as f:
    pickle.dump({'out':out,'conf':conf},f)

pp1 = out[0]
ll1 = out[1]
dd1 = np.linalg.norm(pp1[:,:,np.newaxis,...]-ll1[:,np.newaxis,...],axis=-1)
dd1 = find_dist_match(dd1)
valid = ll1[:,:,0,0]>-1000
dd1_val = dd1[valid,:]

np.nanpercentile(dd1_val,[50,75,90,95,97,99],axis=0)

##

res_mmpose_higherhr = np.array(
[[ 1.59025148,  1.57246142,  1.61618328,  1.61044092,  1.54820717,
   1.5833667 ,  1.71653502,  1.62968975,  1.63938534,  1.60167714,
   1.68509588,  1.70851295,  1.64898595,  1.72950686,  1.68878916,
   1.64920816,  1.72607833],
 [ 1.91658338,  1.88815945,  1.92700416,  1.93016214,  1.84278747,
   1.9580876 ,  2.08009631,  2.01451654,  2.09185323,  1.98840793,
   2.1336131 ,  2.26060487,  2.07608914,  2.2925405 ,  2.26166151,
   2.08128144,  2.24606321],
 [ 2.21409738,  2.18780826,  2.23584268,  2.2169981 ,  2.14721593,
   2.32650326,  2.46391023,  2.4235862 ,  2.61345091,  2.35222775,
   2.65581814,  3.86159165,  2.65422023,  3.99815763,  4.02509844,
   2.63499015,  3.73648833],
 [ 2.43265573,  2.40819486,  2.44577496,  2.41593548,  2.33950703,
   2.57411833,  2.71968702,  2.71051682,  3.08334222,  2.60343952,
   3.06713587,  6.04246233,  3.59490789,  7.00536606,  6.7537968 ,
   3.93151148,  5.78402375],
 [ 2.57647466,  2.5731014 ,  2.58325681,  2.54330834,  2.48049758,
   2.72671861,  2.91511774,  2.86473097,  3.59306262,  2.81289814,
   3.39542885,  7.96037672,  5.30467274, 10.49585832,  9.64348857,
   5.41232654,  7.55973543]])

# with flip test
res_mmpose_higherhr_80k = np.array(
[[ 1.28600199,  1.28404059,  1.24717243,  1.26758502,  1.24945427,
   1.31853269,  1.34491897,  1.28940813,  1.27296108,  1.27331348,
   1.31685311,  1.39518966,  1.28289141,  1.37291796,  1.34436316,
   1.28487391,  1.38650967],
 [ 1.58195216,  1.60833751,  1.54528402,  1.58256662,  1.54475478,
   1.67107409,  1.70574614,  1.6682908 ,  1.69408509,  1.69432865,
   1.75143627,  2.01631188,  1.7268426 ,  1.96061942,  1.90535635,
   1.71199126,  1.9228693 ],
 [ 1.84700179,  1.91421524,  1.81691864,  1.87256271,  1.80692869,
   2.0511133 ,  2.09638607,  2.04310904,  2.2446744 ,  2.09947931,
   2.29864303,  3.68380626,  2.37262488,  3.76644634,  3.5413647 ,
   2.36775979,  3.57136171],
 [ 2.03012076,  2.1333211 ,  2.01832259,  2.04019113,  2.00349652,
   2.28593596,  2.31726398,  2.34199934,  2.77317001,  2.37162825,
   2.77369851,  6.29919424,  3.43077572,  7.19248912,  6.29739689,
   3.76548002,  5.4985839 ],
 [ 2.15954964,  2.28809191,  2.16009081,  2.15799936,  2.10629193,
   2.47751166,  2.51632747,  2.51907654,  3.2469796 ,  2.5875642 ,
   3.05693614,  8.03030626,  5.13688962, 11.01978112,  8.10900644,
   5.20921272,  7.39779361]])

##
res_grone = np.array(
[[ 0.51825876,  0.53640457,  0.523971  ,  0.52818559,  0.52422258,
   0.68611694,  0.68923818,  0.68579398,  0.80827945,  0.67700339,
   0.84266775,  0.98304195,  0.78003791,  1.0017143 ,  0.98363547,
   0.72675727,  0.8920803 ],
 [ 0.7784654 ,  0.80229563,  0.78455876,  0.78044602,  0.78091015,
   1.03413408,  1.02518601,  1.0116536 ,  1.28622757,  1.02127056,
   1.31671241,  2.2444052 ,  1.35186466,  2.29438933,  2.16561188,
   1.24454576,  2.19241612],
 [ 1.0961171 ,  1.11895176,  1.07916868,  1.07334173,  1.08834052,
   1.43120791,  1.4266494 ,  1.43239227,  2.06909066,  1.42250529,
   1.9914484 ,  5.89534169,  3.65795757,  6.4590554 ,  6.97413842,
   3.44183753,  5.64203704],
 [ 1.31151266,  1.40291   ,  1.35339841,  1.284168  ,  1.32687514,
   1.72493709,  1.74463758,  1.80780595,  2.91220518,  1.74948291,
   2.59646542,  9.12465826,  7.65392865, 11.54285562, 11.11391139,
   7.23967383,  8.35388393],
 [ 1.54328995,  1.68424391,  1.5783073 ,  1.44765421,  1.53076493,
   2.03460826,  2.23832274,  2.27369234,  3.70883576,  2.159851  ,
   3.29307106, 11.49959654, 11.70188101, 15.60458571, 14.59941366,
   10.25722381, 10.88584497]])

res_mmpose_higherhr_2x = np.array(
[[ 0.8920856 ,  0.89480332,  0.87342909,  0.86497631,  0.86767505,
   0.97916882,  0.96745863,  0.92142408,  0.96919232,  0.95443933,
   1.01764616,  1.1159877 ,  1.00692657,  1.10323764,  1.08745648,
   0.93536743,  1.01320421],
 [ 1.17873906,  1.1889788 ,  1.16176621,  1.16865238,  1.14584881,
   1.3491497 ,  1.31176573,  1.26780101,  1.36895948,  1.30887596,
   1.4267259 ,  1.73509155,  1.40038061,  1.72362991,  1.65051565,
   1.31433187,  1.557348  ],
 [ 1.46674256,  1.4878163 ,  1.46044762,  1.43990492,  1.44895097,
   1.74363388,  1.6673035 ,  1.61576967,  1.90272125,  1.69922399,
   1.92748548,  3.90477574,  2.06597381,  4.02762537,  3.97151256,
   1.91293537,  3.74722248],
 [ 1.66972602,  1.68320558,  1.6549672 ,  1.61090914,  1.64426032,
   2.02075368,  1.93425025,  1.91489757,  2.46988238,  2.01646693,
   2.405567  ,  6.46867149,  3.53084615,  7.84045566,  7.70660707,
   3.80722926,  6.52910778],
 [ 1.84053966,  1.83040291,  1.80013699,  1.72409954,  1.77489244,
   2.24191145,  2.14253974,  2.14829588,  3.12470149,  2.27225283,
   2.7852572 ,  8.59306191,  6.56859893, 11.42711834, 10.89980942,
   5.84109636,  8.03244694]])

res_mmpose_higherhr_fullims_mask_loss = np.array(
[[ 1.86378998,  1.82932391,  1.80815844,  1.7948106 ,  1.7567717 ,
   1.92708917,  1.76973321,  1.87603868,  1.97681903,  1.83744725,
   1.95823429,  1.99875811,  1.91119869,  2.09754751,  2.07865273,
   1.88755936,  1.98028446],
 [ 2.2816747 ,  2.21648517,  2.21400803,  2.16875315,  2.0991835,
   2.42880289,  2.19002665,  2.31412683,  2.57104661,  2.33340469,
   2.61136449,  2.95236689,  2.53866349,  3.08284848,  2.94977993,
   2.37230709,  2.78991269],
 [ 2.70704579,  2.61958891,  2.62908091,  2.56984319,  2.673729 ,  2.90266761,  2.61391938,  2.81619663,  3.37849522,  2.3676242,
   3.38253915,  6.6572158 ,  4.5216547 ,  9.58672127,  8.55167025,
   4.00789868,  5.77563598],
 [ 3.05759035,  2.88957411,  2.99820643,  2.79738511,  2.5290924,
   3.29322007,  3.0450663 ,  3.16395615,  4.23791681,  3.22684536,
   4.02685229, 14.04435817, 10.02852856, 19.77940377, 19.01278426,
   11.36506503, 11.53236775],
 [ 3.37664836,  3.24329351,  3.34583606,  3.07948759,  3.00825363,
   3.50468925,  3.40412718,  3.57359008,  5.2699353 ,  3.51849766,
   4.6659969 , 20.68311989, 19.83110497, 39.23067786, 34.31006099,
   28.6249871 , 22.86439127]])

res_mmpose_higherhr_fullims_mask_im = np.array(
[[  2.72281307,   2.48651865,   2.53799062,   2.52833073,
    2.14836262,   4.55376454,   2.8874044 ,   3.97788846,
    5.58799077,   5.21678137,   6.94687443,  33.23032114,
    33.62704092,  37.43112077,  40.93849402,  33.75711507,
    24.97922286],
  [ 43.02373041,  34.8797601 ,  37.38764122,  39.9468718 ,
    35.3184365 ,  50.38179981,  58.04153271,  39.63408422,
    38.97176848,  55.96305619,  65.58825174,  61.00434949,
    49.42792172,  59.34615428,  60.41923135,  55.45305695,
    53.21251942],
   [172.47628551, 149.63983526, 142.85436038, 155.92358357,
    146.62035572, 179.53638061, 149.40747532, 151.33626297,
    144.90784378, 209.0008151 , 252.49037731, 223.40011892,
    144.68888581, 162.25749822, 150.16838643, 144.49430888,
    157.82603568],
   [258.98070079, 225.37524187, 215.18571348, 232.1410462 ,
    215.68280713, 275.75383056, 231.31586152, 226.0876158 ,
    218.38013611, 372.34390108, 502.83887526, 459.62931247,
    215.76492499, 237.68054035, 225.95743427, 209.20056495,
    232.5208645 ],
   [359.323717  , 271.75115812, 268.98122445, 285.92240393,
    261.3305535 , 385.93250986, 278.85270175, 275.78077115,
    254.03009188, 643.66206974, 735.20790739, 750.8270152 ,
    259.99221931, 289.61275695, 274.46486058, 268.64797744,
    295.39544729]])

res_mmpose_higherhr_mask_loss = np.array(
[[  1.84638703,   1.77889457,   1.69334361,   1.77829953,
    1.70758687,   1.80919987,   1.93328005,   1.90542352,
    2.23728981,   2.08498892,   2.14545499,   2.440377  ,
    2.24276269,   2.52403789,   2.64572902,   2.13282936,
    2.29004487],
 [  2.42716673,   2.2875475 ,   2.14665939,   2.23160298,
    2.22907619,   2.49230221,   2.53744286,   2.63194114,
    3.09572669,   2.74590087,   3.1132674 ,   4.83718407,
    3.22385684,   6.99764742,   6.00172865,   3.10542548,
    4.28243225],
 [  3.32667659,   2.98311999,   2.83569236,   2.85937794,
    2.9264328 ,   3.74193229,   4.28365592,   3.97066236,
    5.84890622,   4.30808103,   5.55113872,  37.11290622,
    18.72790544,  48.72660671,  50.48462593,  21.7052866 ,
    25.4698981 ],
  [ 47.06427364,  36.96279238,  22.00495965,  33.99483705,
    41.99848645,  49.4094596 ,  76.02302961,  56.12590386,
    59.7914009 ,  64.25174063,  59.00362319,  90.06953878,
    63.0194346 , 101.00014228,  97.39421617,  61.39529135,
    73.99922691],
  [ 83.57990204,  68.07318081,  52.9727217 ,  65.10161481,
    77.08267879,  83.34738573,  96.86099697,  94.50728186,
    88.68886401,  93.43581385,  82.58247465, 116.54550253,
    91.66289702, 125.76442646, 121.03525078,  92.92953846,
    96.86705736]])

res_mmpose_higherhr_mask_im = np.array(
[[  2.05108052,   1.60573762,   1.75596447,   1.64430286,
    1.66409641,   1.74773043,   1.98711592,   1.6997191 ,
    1.94687753,   1.78900346,   2.38063151,   5.51295116,
    5.06528723,   4.40214024,   4.51530753,   5.3676712 ,
    4.60733054],
 [  2.56137412,   2.21340661,   2.22120756,   2.08183976,
    2.24927865,   2.22748866,   2.73388498,   2.27537398,
    2.72351704,   2.28342137,   3.47851563,  17.7303501 ,
    10.98466667,  12.11690637,  13.61340599,  12.30262371,
    11.81929747],
 [  3.18289557,   2.82260582,   2.74269632,   2.59118987,
    2.80455257,   2.78866817,   6.82990878,   2.95708126,
    4.35326039,   2.91340121,   6.91899511,  57.78825865,
    30.74497046,  47.29993523,  38.72173772,  36.84791548,
    36.73040738],
 [  5.67538071,   3.55712011,   3.79687117,   3.60887985,
    3.52629526,   3.49446766,  32.30570438,   4.50309775,
    10.93992664,   3.87975082,  32.38696632, 128.32635275,
    99.12673838, 144.96074726,  89.85035995, 101.31374493,
    70.13688836],
  [ 62.29970516,  45.61449429,  49.30316578,  47.78257913,
    47.32146737,  57.54112383,  82.78400255,  54.4310502 ,
    57.50385791,  50.51670172,  74.95385543, 163.36022566,
    154.26892309, 222.01260183, 129.659557  , 197.69592432,
    115.85488878]])


res_grone_mask_im = np.array(
[[ 0.52925804,  0.56327437,  0.5458152 ,  0.55541665,  0.53800312,
   0.70928437,  0.676464  ,  0.67728926,  0.86142119,  0.6509753 ,
   0.86429836,  1.14725354,  0.83991379,  1.21139861,  1.22052075,
   0.79726783,  1.06958927],
[ 0.77696745,  0.82813789,  0.80027814,  0.78939695,  0.78060101,
  1.05074856,  0.99632601,  1.00812218,  1.3501264 ,  0.98304191,
  1.3555424 ,  2.82727428,  1.45734527,  3.27432411,  2.82775349,
  1.32566336,  2.76147115],
[ 1.07761104,  1.1342592 ,  1.10677511,  1.05652809,  1.07952541,
  1.49365333,  1.40353093,  1.38932107,  2.10012048,  1.42272307,
  2.07707189,  7.64653046,  4.8745687 ,  8.74712099,  8.60810446,
  3.83741011,  7.51459581],
[ 1.36738618,  1.35607219,  1.33062548,  1.24873909,  1.29081953,
  1.84569648,  1.75858883,  1.76655868,  2.96564111,  1.86958114,
  2.77747917, 11.12558181,  9.40522329, 13.70277038, 13.36958007,
  8.21880652, 11.42492082],
[ 1.73533946,  1.52258982,  1.5440721 ,  1.39410185,  1.4393606 ,
  2.06340166,  2.08344834,  2.08277842,  3.83996443,  2.15919431,
  3.63236395, 14.37028349, 14.14804486, 16.92548454, 18.83129903,
  12.62299376, 13.9749357 ]])

res_openpose_maskim = np.array(
[[ 0.66119854,  0.6619866 ,  0.65403937,  0.65189814,  0.60880075,
   0.72872549,  0.71720055,  0.71966763,  0.88917565,  0.72337974,
   0.87637675,  1.38163275,  1.06472721,  1.50286624,  1.67086494,
   1.15378331,  1.29029151],
 [ 0.93193101,  0.95853472,  0.96296705,  0.91286889,  0.88963424,
   1.05802101,  1.0437572 ,  1.03247884,  1.33192121,  1.08920301,
   1.33928781,  2.39724172,  1.71995814,  2.69668714,  2.76966277,
   1.76512861,  2.10586319],
[ 1.21305717,  1.27024373,  1.23960722,  1.16792871,  1.17404099,
  1.41543882,  1.40037031,  1.41675286,  1.90331439,  1.49869321,
  1.93190006,  4.8088216 ,  3.2580568 ,  6.49553434,  6.26238158,
  2.8091625 ,  4.4856982 ],
[ 1.40251272,  1.4485769 ,  1.42512077,  1.33581986,  1.3818785 ,
  1.65995868,  1.64888808,  1.68828927,  2.50506119,  1.78482933,
  2.39049115,  8.05712895,  5.71741327, 10.67567699, 11.78594716,
  4.56599742,  7.75393521],
[ 1.53922957,  1.57227016,  1.57896114,  1.4293556 ,  1.49478988,
  1.89623004,  1.82139778,  1.91394187,  3.18548649,  1.98940054,
  2.82792176, 10.69128628,  8.86018342, 14.03672142, 15.43054906,
  7.43147934, 11.11918523]])

# lots of nans
res_openpose_maskloss = np.array(
[[ 0.59972172,  0.60872955,  0.6372762 ,  0.61001726,  0.60003426,
   0.71540745,  0.7061171 ,  0.71634283,  0.81676436,  0.69253638,
   0.81734149,  0.93102854,  0.8131992 ,  0.96909097,  0.95627584,
   0.79681566,  0.86675734],
[ 0.85776309,  0.87972159,  0.94239452,  0.87863123,  0.85041857,
  1.02864158,  1.01161494,  1.03600932,  1.21872995,  1.01725765,
  1.25227977,  1.72534666,  1.31566981,  1.76119243,  1.68334429,
  1.21214888,  1.55886998],
[ 1.1779924 ,  1.19583332,  1.26216367,  1.14261509,  1.1536813 ,
  1.40613995,  1.35251733,  1.44239479,  1.92252272,  1.41428313,
  1.92073099,  4.38487004,  2.34313635,  4.22902404,  4.1589624 ,
  2.1654285 ,  4.35677173],
[ 1.44650394,  1.4095342 ,  1.50790756,  1.3195588 ,  1.37942079,
  1.72557962,  1.63530726,  1.77426313,  2.71766317,  1.69089995,
  2.4126185 ,  7.86650146,  5.20565304,  7.44135746,  6.92206162,
  4.5037466 ,  8.61558433],
[ 1.69230802,  1.57680655,  1.68272694,  1.5183835 ,  1.54574248,
  2.04471061,  1.91724478,  2.05781836,  3.46370351,  1.98375682,
  2.98479131, 11.64178328,  7.45493532, 10.48567304,  9.46923409,
  7.93504498, 13.69101484]])

## Multi-animal experiments commands

## comparing masking strategies

import PoseTools as pt
import re

lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'
sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'
gpu_q = 'gpu_rtx' #'gpu_tesla' #
simg = '/groups/branson/bransonlab/mayank/singularity/pytorch_mmpose.sif'

cmd = 'APT_interface.py /nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl -json_trn_file /nrs/branson/mayank/apt_cache_2/alice_ma/loc_split.json -conf_params dl_steps 100000 is_multi True db_format \\"coco\\" max_n_animals 7 trange 30 rrange 180 op_affinity_graph \\(\\(0,1\\),\\(0,5\\),\\(1,2\\),\\(3,4\\),\\(3,5\\),\\(5,6\\),\\(5,7\\),\\(5,9\\),\\(3,16\\),\\(9,10\\),\\(10,15\\),\\(9,14\\),\\(4,11\\),\\(7,8\\),\\(8,12\\),\\(7,13\\)\\) {} -type multi_mmpose -name val_split -cache /nrs/branson/mayank/apt_cache_2 train -skip_db'

# n = 'higher_2rescale'
# cur_cmd = cmd.replace('is_multi True','is_multi True mmpose_net \\"higherhrnet\\" rescale 0.5')
# cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
# print(cur_cmd)
# print()
# pt.submit_job('alice_multi_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'higherhr_maskim'
cur_cmd = cmd.replace('is_multi True','is_multi True multi_use_mask True multi_loss_mask False mmpose_net \\"higherhrnet\\"')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
cur_cmd = cur_cmd.replace(' -skip_db','')
cur_cmd = cur_cmd.replace('-name val_split', '-name maskim_split')
print(cur_cmd)
print()
pt.submit_job('alice_multi_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'higherhr_maskloss'
cur_cmd = cmd.replace('is_multi True','is_multi True multi_use_mask False multi_loss_mask True mmpose_net \\"higherhrnet\\"')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
cur_cmd = cur_cmd.replace(' -skip_db','')
print(cur_cmd)
print()
pt.submit_job('alice_multi_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'grone_maskim'
cur_cmd = cmd.replace('is_multi True','is_multi True multi_use_mask True multi_loss_mask False mmpose_net \\"higherhrnet\\"')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
cur_cmd = cur_cmd.replace('-name val_split', '-name maskim_split')
cur_cmd = cur_cmd.replace(' -type multi_mmpose',' -type multi_mdn_joint_torch')
cur_cmd = cur_cmd.replace(' -skip_db','')
print(cur_cmd)
print()
pt.submit_job('alice_multi_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'grone_maskloss'
cur_cmd = cmd.replace('is_multi True','is_multi True multi_use_mask False multi_loss_mask True mmpose_net \\"higherhrnet\\"')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
cur_cmd = cur_cmd.replace(' -type multi_mmpose',' -type multi_mdn_joint_torch')
cur_cmd = cur_cmd.replace(' -skip_db','')
print(cur_cmd)
print()
pt.submit_job('alice_multi_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'openpose_maskim'
cur_cmd = cmd.replace('is_multi True','is_multi True multi_use_mask True multi_loss_mask False mmpose_net \\"higherhrnet\\"')
cur_cmd = cur_cmd.replace("coco","tfrecord")
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
cur_cmd = cur_cmd.replace(' -type multi_mmpose',' -type multi_openpose')
cur_cmd = cur_cmd.replace(' -skip_db',' -use_cache')
cur_cmd = cur_cmd.replace('-name val_split', '-name maskim_split')
print(cur_cmd)
print()
pt.submit_job('alice_multi_{}'.format(n),cur_cmd,sdir,queue='gpu_tesla',sing_image=simg,numcores=4)

n = 'openpose_maskloss'
cur_cmd = cmd.replace('is_multi True','is_multi True multi_use_mask False multi_loss_mask True mmpose_net \\"higherhrnet\\"')
cur_cmd = cur_cmd.replace("coco","tfrecord")
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
cur_cmd = cur_cmd.replace(' -type multi_mmpose',' -type multi_openpose')
cur_cmd = cur_cmd.replace(' -skip_db',' -use_cache')
print(cur_cmd)
print()
pt.submit_job('alice_multi_{}'.format(n),cur_cmd,sdir,queue='gpu_tesla',sing_image=simg,numcores=4)

# Full ims

n = 'higherhr_maskim_fullims'
cur_cmd = cmd.replace('is_multi True','is_multi True multi_use_mask True multi_loss_mask False multi_crop_ims False mmpose_net \\"higherhrnet\\"')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
cur_cmd = cur_cmd.replace('-name val_split', '-name maskim_split_fullims')
cur_cmd = cur_cmd.replace(' -skip_db','')
print(cur_cmd)
print()
pt.submit_job('alice_multi_{}'.format(n),cur_cmd,sdir,queue='gpu_tesla',sing_image=simg,numcores=4)

n = 'higherhr_maskloss_fullims'
cur_cmd = cmd.replace('is_multi True','is_multi True multi_use_mask False multi_loss_mask True multi_crop_ims False mmpose_net \\"higherhrnet\\"')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
cur_cmd = cur_cmd.replace('-name val_split', '-name val_split_fullims')
cur_cmd = cur_cmd.replace(' -skip_db','')
print(cur_cmd)
print()
pt.submit_job('alice_multi_{}'.format(n),cur_cmd,sdir,queue='gpu_tesla',sing_image=simg,numcores=4)

n = 'grone_maskim_fullims'
cur_cmd = cmd.replace('is_multi True','is_multi True multi_use_mask True multi_loss_mask False multi_crop_ims False mmpose_net \\"higherhrnet\\"')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
cur_cmd = cur_cmd.replace('dl_steps 100000', 'dl_steps 200000 batch_size 4')
cur_cmd = cur_cmd.replace('-name val_split', '-name maskim_split_fullims')
cur_cmd = cur_cmd.replace(' -type multi_mmpose',' -type multi_mdn_joint_torch')
cur_cmd = cur_cmd.replace(' -skip_db','')
print(cur_cmd)
print()
pt.submit_job('alice_multi_{}'.format(n),cur_cmd,sdir,queue='gpu_tesla',sing_image=simg,numcores=4)

n = 'grone_maskloss_fullims'
cur_cmd = cmd.replace('is_multi True','is_multi True multi_use_mask False multi_loss_mask True multi_crop_ims False mmpose_net \\"higherhrnet\\"')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
cur_cmd = cur_cmd.replace('dl_steps 100000', 'dl_steps 200000 batch_size 4')
cur_cmd = cur_cmd.replace('-name val_split', '-name val_split_fullims')
cur_cmd = cur_cmd.replace(' -type multi_mmpose',' -type multi_mdn_joint_torch')
cur_cmd = cur_cmd.replace(' -skip_db','')
print(cur_cmd)
print()
pt.submit_job('alice_multi_{}'.format(n),cur_cmd,sdir,queue='gpu_tesla',sing_image=simg,numcores=4)

n = 'openpose_maskim_fullims'
cur_cmd = cmd.replace('is_multi True','is_multi True multi_use_mask True multi_loss_mask False multi_crop_ims False mmpose_net \\"higherhrnet\\"')
cur_cmd = cur_cmd.replace("coco","tfrecord")
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
cur_cmd = cur_cmd.replace('dl_steps 100000', 'dl_steps 400000 batch_size 2')
cur_cmd = cur_cmd.replace('-name val_split', '-name maskim_split_fullims')
cur_cmd = cur_cmd.replace(' -type multi_mmpose',' -type multi_openpose')
cur_cmd = cur_cmd.replace(' -skip_db',' -use_cache')
print(cur_cmd)
print()
pt.submit_job('alice_multi_{}'.format(n),cur_cmd,sdir,queue='gpu_tesla',sing_image=simg,numcores=4)

n = 'openpose_maskloss_fullims'
cur_cmd = cmd.replace('is_multi True','is_multi True multi_use_mask False multi_loss_mask True multi_crop_ims False mmpose_net \\"higherhrnet\\"')
cur_cmd = cur_cmd.replace("coco","tfrecord")
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
cur_cmd = cur_cmd.replace('dl_steps 100000', 'dl_steps 400000 batch_size 2')
cur_cmd = cur_cmd.replace('-name val_split', '-name val_split_fullims')
cur_cmd = cur_cmd.replace(' -type multi_mmpose',' -type multi_openpose')
cur_cmd = cur_cmd.replace(' -skip_db',' -use_cache')
print(cur_cmd)
print()
pt.submit_job('alice_multi_{}'.format(n),cur_cmd,sdir,queue='gpu_tesla',sing_image=simg,numcores=4)


## Create single split for the label file 20201221
#  !!! movie order in alice_ma doesn't match touch_20200811. Sigh
import h5py
import PoseTools as pt
import json
import os

lbl_file = '/groups/branson/home/kabram/APT_projects/alice_touch_stripped_20200811.lbl'
lbl = h5py.File(lbl_file,'r')
exp_list = lbl['movieFilesAll'][0, :]
mov_single = [u''.join(chr(c) for c in lbl[jj]) for jj in exp_list]

info_in = '/groups/branson/home/kabram/nrs/apt_cache_2/alice_ma/loc_split.json'
out_dir = '/groups/branson/home/kabram/nrs/apt_cache_2/alice_ma'
multi_info = pt.json_load(info_in)
mov_multi = multi_info['movies']
multi_split = [[],[]]
for curi in multi_info['locdata']:
    ss = curi['split'] if type(curi['split']) is int else curi['split'][0]
    single_mov_ndx = mov_single.index(multi_info['movies'][curi['imov']-1])
    multi_split[ss].append([single_mov_ndx,curi['frm']-1])


m_ndx = lbl['preProcData_MD_mov'].value[0, :].astype('int') - 1
t_ndx = lbl['preProcData_MD_iTgt'].value[0, :].astype('int') - 1
f_ndx = lbl['preProcData_MD_frm'].value[0, :].astype('int') - 1

single_split = [[],[]]
for ndx in range(len(m_ndx)):
    cur_i = [int(m_ndx[ndx]),int(f_ndx[ndx]),int(t_ndx[ndx])]
    if cur_i[:2] in multi_split[0]:
        single_split[0].append(cur_i)
    else:
        assert cur_i[:2] in multi_split[1], 'missing'
        single_split[1].append(cur_i)

# For tf records file
out_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/single_animal_split.json'


with open(out_file,'w') as f:
    json.dump(single_split, f)

import APT_interface as apt
import Pose_mdn_joint_fpn

conf = apt.create_conf(lbl_file,0,'touch_20200811','/nrs/branson/mayank/apt_cache_2','mdn_joint_fpn')
conf.splitType = 'predefined'
apt.create_tfrecord(conf,True,out_file,True)

## convert the tf record to coco db for mmpose
import PoseTools  as pt
import APT_interface as apt
import os
lbl_file = '/groups/branson/home/kabram/APT_projects/alice_touch_stripped_20200811.lbl'
conf = apt.create_conf(lbl_file,0,'touch_20200811','/nrs/branson/mayank/apt_cache_2','mdn_joint_fpn')
confm = apt.create_conf(lbl_file,0,'touch_20200811','/nrs/branson/mayank/apt_cache_2','mmpose')

db_file = os.path.join(conf.cachedir,conf.trainfilename + '.tfrecords')
out_im_dir =  os.path.join(confm.cachedir,'train')
os.makedirs(out_im_dir,exist_ok=True)
out_json = os.path.join(confm.cachedir,conf.trainfilename + '.json')
pt.tfrecord_to_coco(db_file,conf.n_classes,out_im_dir,out_json,out_size=[192,192])

db_file = os.path.join(conf.cachedir,conf.valfilename + '.tfrecords')
out_im_dir =  os.path.join(confm.cachedir,'val')
os.makedirs(out_im_dir,exist_ok=True)
out_json = os.path.join(confm.cachedir,conf.valfilename + '.json')
pt.tfrecord_to_coco(db_file,conf.n_classes,out_im_dir,out_json,out_size=[192,192])

## Single animal commands

import PoseTools
import re
sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'
gpu_q = 'gpu_rtx' #'gpu_tesla' #
simg = '/groups/branson/bransonlab/mayank/singularity/test_tf_torch.sif'
cmd = 'APT_interface.py /groups/branson/home/kabram/APT_projects/alice_touch_stripped_20200811.lbl -name touch_20200811 -cache /nrs/branson/mayank/apt_cache_2 -conf_params rrange 15 trange 5 scale_factor_range 1.2 dl_steps 100000 db_format \\"coco\\" {} -type mdn train -skip_db -use_cache'


n = 'base_mmpose'
cur_cmd = cmd.replace('type mdn','type mmpose')
cur_cmd = cur_cmd.format("imsz \(192,192\) -train_name {}".format(n))
print(cur_cmd)
print()
PoseTools.submit_job('alice_single_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'base_mdn'
cur_cmd = cmd.replace('type mdn','type mdn_joint_fpn')
cur_cmd = cur_cmd.replace('db_format \\"coco\\"','db_format \\"tfrecord\\"')
cur_cmd = cur_cmd.format(" -train_name {}".format(n))
print(cur_cmd)
print()
PoseTools.submit_job('alice_single_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

##
import PoseTools
import os
import glob
import APT_interface as apt
import apt_expts
import re
import run_apt_expts as rae
import multiResData
from importlib import reload

reload(apt_expts)
import PoseUNet_resnet
reload(PoseUNet_resnet)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
run_names = ['base_mmpose', 'base_mdn']
out_dir = '/groups/branson/home/kabram/temp'

out = {}
proj_name = 'multitarget_bubble'
for n in run_names:
    if 'mmpose' in n:
        ntype = 'mmpose'
    else:
        ntype = 'mdn_joint_fpn'

    if ntype == 'mdn_joint_fpn':
        db_file = '/nrs/branson/mayank/apt_cache_2/multitarget_bubble/mdn_joint_fpn/view_0/touch_20200811/val_TF.tfrecords'
        cdir = os.path.dirname(db_file)
    else:
        cdir = '/nrs/branson/mayank/apt_cache_2/multitarget_bubble/{}/view_0/touch_20200811'.format(ntype)
        db_file = '/nrs/branson/mayank/apt_cache_2/multitarget_bubble/mmpose/view_0/touch_20200811/val_TF.json'
    if ntype == 'deeplabcut':
        tfile = os.path.join(cdir, '{}_traindata'.format(n))
    elif n == 'deepnet':
        tfile = os.path.join(cdir, 'traindata')
    else:
        tfile = os.path.join(cdir, '{}_{}_traindata'.format(proj_name,n))

    if not os.path.exists(tfile):
        continue
    A = PoseTools.pickle_load(tfile)
    conf = A[1]
    # conf.rescale = 181./conf.imsz[0]
    # conf.imsz = [181,181]

    if 'mmpose' in n:
        files = glob.glob(os.path.join(cdir, "{}-[0-9]*").format(n))
        files.sort(key=os.path.getmtime)
        aa = [int(re.search('-(\d*)', f).groups(0)[0]) for f in files]
        conf.img_dim = 3
    else:
        files = glob.glob(os.path.join(cdir, "{}-[0-9]*.index").format(n))
        files.sort(key=os.path.getmtime)
        aa = [int(re.search('-(\d*).index', f).groups(0)[0]) for f in files]
    aa = [b - a for a, b in zip(aa[:-1], aa[1:])]
    if any([a < 0 for a in aa]):
        bb = int(np.where(np.array(aa) < 0)[0]) + 1
        files = files[bb:]
    files = [f.replace('.index', '') for f in files]
    files = files[-1:]
    # if len(files) > 12:
    #     gg = len(files)
    #     sel = np.linspace(0, len(files) - 1, 12).astype('int')
    #     files = [files[s] for s in sel]

    mdn_out = apt.classify_db_all(ntype, conf, db_file, model_file=files[-1], name=n)
    out[n] = [mdn_out]

H = multiResData.read_and_decode_without_session(db_file, conf)
ex_ims = np.array(H[0][0])
ex_locs = np.array(H[1][0])
import matplotlib
matplotlib.use('TkAgg')
f = rae.plot_hist([out,ex_ims,ex_locs],[50,75,90,95,97])
# f2 = rae.plot_results(out)

## create full touch datasets for training/tracking .

# alice

import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from importlib import reload
import APT_interface as apt
reload(apt)
import PoseTools as pt
lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'
json_trn = '/nrs/branson/mayank/apt_cache_2/alice_ma/loc.json'
gpu_q = 'gpu_rtx' #'gpu_tesla' #
simg = '/groups/branson/bransonlab/mayank/singularity/pytorch_mmpose.sif'
sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'

cmd = 'APT_interface.py /nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl -json_trn_file /nrs/branson/mayank/apt_cache_2/alice_ma/loc.json -conf_params dl_steps 100000 is_multi True multi_use_mask False  multi_loss_mask True db_format \\"coco\\" max_n_animals 7 {} -type multi_mmpose -name full_touch_20200811 -cache /nrs/branson/mayank/apt_cache_2 train -skip_db'

n = 'higherhrnet'
cur_cmd = cmd.replace('is_multi True','is_multi True mmpose_net \\"higherhrnet\\"')
cur_cmd = cur_cmd.replace('-skip_db','')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
print(cur_cmd)
print()
pt.submit_job('alice_multi_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'higherhrnet_fullims'
cur_cmd = cmd.replace('is_multi True','is_multi True mmpose_net \\"higherhrnet\\" multi_crop_ims False')
cur_cmd = cur_cmd.replace('-skip_db','')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
cur_cmd = cur_cmd.replace('-name full_touch_20200811','-name full_touch_20200811_fullims')
print(cur_cmd)
print()
pt.submit_job('alice_multi_{}'.format(n),cur_cmd,sdir,queue='gpu_tesla',sing_image=simg,numcores=4)

n = 'grone'
cur_cmd = cmd.replace('type multi_mmpose','type multi_mdn_joint_torch')
cur_cmd = cur_cmd.replace('-skip_db','')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
print(cur_cmd)
print()
pt.submit_job('alice_multi_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

## Tracking Alice's movies.
import PoseTools as pt
import os
mov = '/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00030_CsChr_RigC_20150826T144616/movie.ufmf'
exp = os.path.split(os.path.split(mov)[0])[1]
gpu_q = 'gpu_rtx' #'gpu_tesla' #
# simg = '/groups/branson/bransonlab/mayank/singularity/test_tf_torch.sif'
simg = '/groups/branson/bransonlab/mayank/singularity/pytorch_mmpose.sif'

sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'

cmd = 'APT_interface.py /nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl -name full_touch_20200811 -type {} -train_name {} -conf_params has_trx_file False imsz \(1024,1024\) batch_size 1 max_n_animals 12 is_multi True mmpose_net \\"higherhrnet\\" -cache /nrs/branson/mayank/apt_cache_2 track -mov {} -out /groups/branson/home/kabram/temp/alice_multi/{}.trk'

n = 'higherhrnet'
cur_cmd = cmd.format('multi_mmpose',n,mov,exp+'_'+n)
print(cur_cmd)
print()
pt.submit_job('alice_track_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'grone'
cur_cmd = cmd.format('multi_mdn_joint_torch',n,mov,exp+'_'+n)
print(cur_cmd)
print()
pt.submit_job('alice_track_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'higherhrnet_fullims'
cur_cmd = cmd.format('multi_mmpose',n,mov,exp+'_'+n)
print(cur_cmd)
print()
pt.submit_job('alice_track_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)


## Full training for Roian.

import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from importlib import reload
import PoseTools as pt
import APT_interface as apt
reload(apt)
lbl_file = '/nrs/branson/mayank/apt_cache_2/four_points_180806/20201225T042233_20201225T042235.lbl'
json_trn = '/nrs/branson/mayank/apt_cache_2/four_points_180806/loc.json'
gpu_q = 'gpu_rtx' #'gpu_tesla' #
# simg = '/groups/branson/bransonlab/mayank/singularity/test_tf_torch.sif'
simg = '/groups/branson/bransonlab/mayank/singularity/pytorch_mmpose.sif'
sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'

cmd = f'APT_interface.py {lbl_file} -json_trn_file {json_trn} -conf_params dl_steps 200000 is_multi True db_format \\"coco\\" batch_size 4 max_n_animals 2 {{}} -type multi_mmpose -name full_dataset -cache /nrs/branson/mayank/apt_cache_2 train'


# !!!! NOTE the images have been overwritten without masks!!!
n = 'higherhrnet'
cur_cmd = cmd.replace('is_multi True','is_multi True mmpose_net \\"higherhrnet\\"')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
print(cur_cmd)
print()
pt.submit_job('roian_multi_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'grone'
cur_cmd = cmd.replace('type multi_mmpose','type multi_mdn_joint_torch')
# cur_cmd = cur_cmd.replace('-skip_db','')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
print(cur_cmd)
print()
pt.submit_job('roian_multi_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=6)

n = 'higherhrnet_nomask'
cur_cmd = cmd.replace('is_multi True','is_multi True mmpose_net \\"higherhrnet\\" multi_use_mask False')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
print(cur_cmd)
print()
pt.submit_job('roian_multi_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'grone_nomask'
cur_cmd = cmd.replace('type multi_mmpose','type multi_mdn_joint_torch')
cur_cmd = cur_cmd.replace('is_multi True','is_multi True mmpose_net \\"higherhrnet\\" multi_use_mask False')
cur_cmd = cur_cmd.replace('-skip_db','')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
print(cur_cmd)
print()
pt.submit_job('roian_multi_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=6)

n = 'grone_nomask_bn'
cur_cmd = cmd.replace('type multi_mmpose','type multi_mdn_joint_torch')
cur_cmd = cur_cmd.replace('is_multi True','is_multi True mmpose_net \\"higherhrnet\\" multi_use_mask False pretrain_freeze_bnorm False')
cur_cmd = cur_cmd.replace('-skip_db','')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
print(cur_cmd)
print()
pt.submit_job('roian_multi_{}'.format(n),cur_cmd,sdir,queue='gpu_tesla',sing_image=simg,numcores=6)


n = 'grone_nomask_lr'
cur_cmd = cmd.replace('type multi_mmpose','type multi_mdn_joint_torch')
cur_cmd = cur_cmd.replace('is_multi True','is_multi True mmpose_net \\"higherhrnet\\" multi_use_mask False learning_rate_multiplier 0.1')
# cur_cmd = cur_cmd.replace('-skip_db','')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
print(cur_cmd)
print()
pt.submit_job('roian_multi_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=6)

n = 'grone_nomask_bn_lr'
cur_cmd = cmd.replace('type multi_mmpose','type multi_mdn_joint_torch')
cur_cmd = cur_cmd.replace('is_multi True','is_multi True mmpose_net \\"higherhrnet\\" multi_use_mask False pretrain_freeze_bnorm False learning_rate_multiplier 0.1')
# cur_cmd = cur_cmd.replace('-skip_db','')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
print(cur_cmd)
print()
pt.submit_job('roian_multi_{}'.format(n),cur_cmd,sdir,queue='gpu_tesla',sing_image=simg,numcores=6)




## Full trakcing on Roian's movie

import PoseTools as pt
import os
mov = '/groups/branson/home/kabram/temp/roian_multi/200918_m170234vocpb_m170234_odor_m170232_f0180322.mjpg'
exp = os.path.splitext(os.path.split(mov)[1])[0]
gpu_q = 'gpu_rtx' #'gpu_tesla' #
# simg = '/groups/branson/bransonlab/mayank/singularity/test_tf_torch.sif'
simg = '/groups/branson/bransonlab/mayank/singularity/pytorch_mmpose.sif'

sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'

cmd = 'APT_interface.py /nrs/branson/mayank/apt_cache_2/four_points_180806/20201225T042233_20201225T042235.lbl -name full_dataset -type {} -train_name {} -conf_params has_trx_file False imsz \(2048,2048\) batch_size 1 max_n_animals 3 min_n_animals 2 is_multi True mmpose_net \\"higherhrnet\\" -cache /nrs/branson/mayank/apt_cache_2 track -mov {} -out /groups/branson/home/kabram/temp/roian_multi/{}.trk'

n = 'higherhrnet'
cur_cmd = cmd.format('multi_mmpose',n,mov,exp+'_'+n)
print(cur_cmd)
print()
pt.submit_job('roian_track_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'grone'
cur_cmd = cmd.format('multi_mdn_joint_torch',n,mov,exp+'_'+n)
print(cur_cmd)
print()
pt.submit_job('roian_track_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'higherhrnet_nomask'
cur_cmd = cmd.format('multi_mmpose',n,mov,exp+'_'+n)
print(cur_cmd)
print()
pt.submit_job('roian_track_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)


## Roian xv results 20210204

import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from importlib import reload
import APT_interface as apt
reload(apt)
import torch
lbl_file = '/nrs/branson/mayank/apt_cache_2/four_points_180806/20201225T042233_20201225T042235.lbl'

n_pairs = [['multi_openpose','openpose'],]#[['multi_mdn_joint_torch','grone'],['multi_mmpose','mmpose']]#,
for curp in n_pairs:
    net_type = curp[0] #'multi_mdn_joint_torch' #'multi_mmpose' #
    train_name = 'deepnet'

    run_name = f'roian_split_crop_ims_{curp[1]}_multi'
    conf = apt.create_conf(lbl_file,0,run_name,net_type=net_type,cache_dir='/nrs/branson/mayank/apt_cache_2')
    # conf.batch_size = 4 if net_type == 'multi_openpose' else 8
    db_file = '/nrs/branson/mayank/apt_cache_2/four_points_180806/multi_mdn_joint_torch/view_0/roian_split_full_ims_grone_multi/val_TF.json'
    conf.db_format = 'coco'
    conf.max_n_animals = 2
    conf.imsz = (1024*2,1024*2) #(288,288)
    conf.img_dim = 3
    conf.mmpose_net = 'higherhrnet' #'higherhrnet_2x'#
    conf.is_multi = True
    conf.op_affinity_graph = ((0,1),(0,2),(0,3))
    conf.batch_size = 2
    conf.rescale = 1

    out = apt.classify_db_all(net_type,conf,db_file,classify_fcn=apt.classify_db_multi,name=train_name)
    torch.cuda.empty_cache()

    import pickle
    import PoseTools as pt
    with open(os.path.join(conf.cachedir,'val_results_'+train_name+'_'+pt.get_datestr()),'wb') as f:
        pickle.dump({'out':out},f)

    def find_dist_match(dd):
        dout = np.ones_like(dd[:,:,0,:])*np.nan
        yy = np.nanmean(dd,axis=-1)
        for a in range(dd.shape[0]):
            for ndx in range(dd.shape[2]):
                if np.all(np.isnan(yy[a,:,ndx])):
                    continue
                r = np.nanargmin(yy[a,:,ndx])
                dout[a,ndx,:] = dd[a,r,ndx,:]
        return dout


    pp1 = out[0]
    ll1 = out[1]
    dd1 = np.linalg.norm(pp1[:,:,np.newaxis,...]-ll1[:,np.newaxis,...],axis=-1)
    dd1 = find_dist_match(dd1)
    valid = ll1[:,:,0,0]>-1000
    dd1_val = dd1[valid,:]

    np.nanpercentile(dd1_val,[50,75,90,95,97],axis=0)


##
import multiResData
import PoseTools
db_file = '/nrs/branson/mayank/apt_cache/four_points_180806/mdn/view_0/cv_split_0/train_TF.tfrecords'
A = multiResData.read_and_decode_without_session(db_file,4,())
db_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/mdn_joint_fpn/view_0/alice_neg/train_TF.tfrecords'
A = multiResData.read_and_decode_without_session(db_file,17,())

ex_im = A[0][0]
ex_loc = np.array(A[1][0])

all_prc = {}

# val_results = {'gron':'/nrs/branson/mayank/apt_cache_2/four_points_180806/multi_mdn_joint_torch/view_0/roian_split_crop_ims_grone_multi/val_results_deepnet_20210301',
#                'op mask loss':'/nrs/branson/mayank/apt_cache_2/four_points_180806/multi_openpose/view_0/roian_split_crop_ims_openpose_multi/val_results_fixed_masks_20210308',
#                'op mask output':'/nrs/branson/mayank/apt_cache_2/four_points_180806/multi_openpose/view_0/roian_split_crop_ims_openpose_multi/val_results_deepnet_20210302',
#                'mmpose':'/nrs/branson/mayank/apt_cache_2/four_points_180806/multi_mmpose/view_0/roian_split_crop_ims_mmpose_multi/val_results_deepnet_20210301'}

val_results = {'gron crop':'/nrs/branson/mayank/apt_cache_2/four_points_180806/multi_mdn_joint_torch/view_0/roian_split_crop_ims_grone_multi/val_results_deepnet_20210301',
               'gron full mask':'/nrs/branson/mayank/apt_cache_2/four_points_180806/multi_mdn_joint_torch/view_0/roian_split_full_ims_grone_multi/val_results_deepnet_20210304',
               'gron full no mask':'/nrs/branson/mayank/apt_cache_2/four_points_180806/multi_mdn_joint_torch/view_0/roian_split_full_maskless_grone_multi/val_results_deepnet_20210303'
               }
prcs = [50, 75, 90, 95, 97]

# ALice HT
# val_results = {'grone':'/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/alice_neg_ht_test/val_results_deepnet_20210413',
#                'openpose':'/nrs/branson/mayank/apt_cache_2/alice_ma/multi_openpose/view_0/alice_neg_ht_crop/val_results_deepnet_20210414'}
# prcs = [50, 75, 90, 95, 97,99]
# pts = [0,6]
# ex_loc = ex_loc[pts,:]

val_results = {'grone':'/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/alice_neg_split_grone_multi/val_results_deepnet_20210420',
'openpose':'/nrs/branson/mayank/apt_cache_2/alice_ma/multi_openpose/view_0/alice_neg_split_openpose_multi/val_results_deepnet_20210421'}

plot_nans = True
for k in val_results.keys():
    out = PoseTools.pickle_load(val_results[k])['out']
    pp1 = out[0]
    ll1 = out[1]
    dd1 = np.linalg.norm(pp1[:,:,np.newaxis,...]-ll1[:,np.newaxis,...],axis=-1)
    dd1 = find_dist_match(dd1)
    valid = ll1[:,:,0,0]>-1000
    dd1_val = dd1[valid,:]
    qq1_val = dd1_val.copy()
    qq1_val[np.isnan(qq1_val)] = 400

    if plot_nans:
        all_prc[k + ' without NaN'] = np.nanpercentile(dd1_val,prcs,axis=0)
        all_prc[k + ' with NaN'] = np.nanpercentile(qq1_val,prcs,axis=0)
    else:
        all_prc[k] = np.nanpercentile(dd1_val,prcs,axis=0)


n_types = len(all_prc)
nc = n_types  # int(np.ceil(np.sqrt(n_types)))
nr = 1  # int(np.ceil(n_types/float(nc)))
if plot_nans:
    nc = int(2)
    nr = int(len(val_results))
else:
    nc = len(val_results)
    nr = 1
cmap = PoseTools.get_cmap(len(prcs), 'cool')
f, axx = plt.subplots(nr, nc, figsize=(12, 8), squeeze=False)
axx = axx.flat
for idx,k  in enumerate(all_prc.keys()):
    ax = axx[idx]
    if ex_im.ndim == 2:
        ax.imshow(ex_im, 'gray')
    elif ex_im.shape[2] == 1:
        ax.imshow(ex_im[:, :, 0], 'gray')
    else:
        ax.imshow(ex_im)

    mm = all_prc[k]
    for pt in range(ex_loc.shape[0]):
        for pp in range(mm.shape[0]):
            c = plt.Circle(ex_loc[pt, :], mm[pp, pt], color=cmap[pp, :], fill=False)
            ax.add_patch(c)
    ttl = '{} '.format(k)
    ax.set_title(ttl)
    ax.axis('off')

f.tight_layout()


## Alice head tail training

import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from importlib import reload
import APT_interface as apt
reload(apt)
lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma_ht/20201231T071216_20201231T071218.lbl'
json_trn = '/nrs/branson/mayank/apt_cache_2/alice_ma_ht/loc.json'
gpu_q = 'gpu_rtx' #'gpu_tesla' #
simg = '/groups/branson/bransonlab/mayank/singularity/pytorch_mmpose.sif'
sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'

cmd = 'APT_interface.py /nrs/branson/mayank/apt_cache_2/alice_ma_ht/20201231T071216_20201231T071218.lbl -json_trn_file /nrs/branson/mayank/apt_cache_2/alice_ma_ht/loc.json -conf_params dl_steps 100000 is_multi True multi_crop_ims False multi_use_mask False db_format \\"coco\\" max_n_animals 11 rescale 2 {} -type multi_mmpose -name full_ht_20200811 -cache /nrs/branson/mayank/apt_cache_2 train'

n = 'higherhrnet'
cur_cmd = cmd.replace('is_multi True','is_multi True mmpose_net \\"higherhrnet\\"')
cur_cmd = cur_cmd.replace('-skip_db','')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
print(cur_cmd)
print()
pt.submit_job('alice_ht_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'grone'
cur_cmd = cmd.replace('type multi_mmpose','type multi_mdn_joint_torch')
cur_cmd = cur_cmd.replace('-skip_db','')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
print(cur_cmd)
print()
pt.submit_job('alice_ht_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

## HT tracking.

import PoseTools as pt
import os
mov = '/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00030_CsChr_RigC_20150826T144616/movie.ufmf'
exp = os.path.split(os.path.split(mov)[0])[1]
gpu_q = 'gpu_rtx' #'gpu_tesla' #
# simg = '/groups/branson/bransonlab/mayank/singularity/test_tf_torch.sif'
simg = '/groups/branson/bransonlab/mayank/singularity/pytorch_mmpose.sif'

sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'

cmd = 'APT_interface.py /nrs/branson/mayank/apt_cache_2/alice_ma_ht/20201231T071216_20201231T071218.lbl -name full_ht_20200811 -type {} -train_name {} -conf_params has_trx_file False imsz \(1024,1024\) batch_size 1 rescale 2 max_n_animals 12 is_multi True mmpose_net \\"higherhrnet\\" -cache /nrs/branson/mayank/apt_cache_2 track -mov {} -out /groups/branson/home/kabram/temp/alice_multi/{}.trk'

n = 'higherhrnet'
cur_cmd = cmd.format('multi_mmpose',n,mov,exp+'_'+n + '_ht')
print(cur_cmd)
print()
pt.submit_job('alice_track_ht_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'grone'
cur_cmd = cmd.format('multi_mdn_joint_torch',n,mov,exp+'_'+n +'_ht')
print(cur_cmd)
print()
pt.submit_job('alice_track_ht_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)


## Alice HT using labels only from the flies that have all the 17 landmarks. Not using CTrax
## create the json file
import PoseTools as pt
in_loc = '/nrs/branson/mayank/apt_cache_2/alice_ma/loc_split_neg.json'
out_locs = '/nrs/branson/mayank/apt_cache_2/alice_ma/loc_split_neg_ht.json'
# in_loc = '/nrs/branson/mayank/apt_cache_2/alice_ma/loc_split.json'
# out_locs = '/nrs/branson/mayank/apt_cache_2/alice_ma/loc_ht_split.json'

A = pt.json_load(in_loc)

for cur_t in A['locdata']:
    p = cur_t['pabs']
    ntgt = cur_t['ntgt']
    cur_locs = np.array(cur_t['pabs'])
    cur_locs = cur_locs.reshape([1, 2, 17, ntgt])
    cur_locs = cur_locs[...,[0,6],:].flatten().tolist()
    cur_t['pabs'] = cur_locs
    cur_occ = np.array(cur_t['occ'])
    cur_occ = cur_occ.reshape([1, 17, ntgt])
    cur_occ = cur_occ[:,[0,6], :].flatten().tolist()
    cur_t['occ'] = cur_occ
    cur_occ = np.array(cur_t['ts'])
    cur_occ = cur_occ.reshape([1, 17, ntgt])
    cur_occ = cur_occ[:,[0,6], :].flatten().tolist()
    cur_t['ts'] = cur_occ

import json
with open(out_locs,'w') as f:
    json.dump(A,f)

## Train

import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from importlib import reload
import APT_interface as apt
import PoseTools as pt
reload(apt)
lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'
json_trn = '/nrs/branson/mayank/apt_cache_2/alice_ma/loc_ht_split.json'
gpu_q = 'gpu_rtx' #'gpu_tesla' #
simg = '/groups/branson/bransonlab/mayank/singularity/pytorch_mmpose.sif'
sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'

cmd = 'APT_interface.py /nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl -json_trn_file /nrs/branson/mayank/apt_cache_2/alice_ma/loc_ht_split.json -conf_params dl_steps 100000 is_multi True multi_crop_ims False db_format \\"coco\\" max_n_animals 11 rescale 2 n_classes 2 flipLandmarkMatches \{{\}} {} -type multi_mmpose -name {} -cache /nrs/branson/mayank/apt_cache_2 train -skip_db'

n = 'higherhrnet_maskim'
cur_cmd = cmd.replace('is_multi True','is_multi True mmpose_net \\"higherhrnet\\" multi_use_mask True multi_loss_mask False')
cur_cmd = cur_cmd.replace('-skip_db','')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n),'ht_maskim')
print(cur_cmd)
print()
pt.submit_job('alice_ht_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'higherhrnet_maskloss'
cur_cmd = cmd.replace('is_multi True','is_multi True mmpose_net \\"higherhrnet\\" multi_use_mask False multi_loss_mask True')
cur_cmd = cur_cmd.replace('-skip_db','')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n),'ht_nomask')
print(cur_cmd)
print()
pt.submit_job('alice_ht_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'grone_maskim'
cur_cmd = cmd.replace('is_multi True','is_multi True mmpose_net \\"higherhrnet\\" multi_use_mask True multi_loss_mask False')
cur_cmd = cur_cmd.replace('type multi_mmpose','type multi_mdn_joint_torch')
cur_cmd = cur_cmd.replace('-skip_db','')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n),'ht_maskim')
print(cur_cmd)
print()
pt.submit_job('alice_ht_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'grone_maskloss'
cur_cmd = cmd.replace('is_multi True','is_multi True mmpose_net \\"higherhrnet\\" multi_use_mask False multi_loss_mask True')
cur_cmd = cur_cmd.replace('type multi_mmpose','type multi_mdn_joint_torch')
cur_cmd = cur_cmd.replace('-skip_db','')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n),'ht_maskloss')
print(cur_cmd)
print()
pt.submit_job('alice_ht_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

## ht classify db 20210220
#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from importlib import reload
import APT_interface as apt
reload(apt)
import torch
lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'

net_type = 'multi_openpose' #'multi_mmpose' #
train_name = 'deepnet'

run_name = 'alice_ht_split_openpose_multi'
conf = apt.create_conf(lbl_file,0,run_name,net_type=net_type,cache_dir='/nrs/branson/mayank/apt_cache_2')
# conf.batch_size = 4 if net_type == 'multi_openpose' else 8
# db_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/val_split_fullims/val_TF.json'
db_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/alice_ht_split_grone_multi/val_TF.json'
conf.db_format = 'coco'
conf.max_n_animals = 10
conf.imsz = (1024,1024) #(288,288)
conf.img_dim = 3
conf.mmpose_net = 'higherhrnet' #'higherhrnet_2x'#
conf.is_multi = True
conf.rescale = 2
conf.n_classes = 2
conf.op_affinity_graph = ((0,1),)
conf.flipLandmarkMatches = {}

out = apt.classify_db_all(net_type,conf,db_file,classify_fcn=apt.classify_db_multi,name=train_name)
torch.cuda.empty_cache()

def find_dist_match(dd):
    dout = np.ones_like(dd[:,:,0,:])*np.nan
    yy = np.nanmean(dd,axis=-1)
    for a in range(dd.shape[0]):
        for ndx in range(dd.shape[2]):
            if np.all(np.isnan(yy[a,:,ndx])):
                continue
            r = np.nanargmin(yy[a,:,ndx])
            dout[a,ndx,:] = dd[a,r,ndx,:]
    return dout

import pickle
import PoseTools as pt
with open(os.path.join(conf.cachedir,'val_results_'+train_name+'_'+pt.get_datestr()),'wb') as f:
    pickle.dump({'out':out,'conf':conf},f)

pp1 = out[0]
ll1 = out[1]
dd1 = np.linalg.norm(pp1[:,:,np.newaxis,...]-ll1[:,np.newaxis,...],axis=-1)
dd1 = find_dist_match(dd1)
valid = ll1[:,:,0,0]>-1000
dd1_val = dd1[valid,:]

np.nanpercentile(dd1_val,[50,75,90,95,99],axis=0)


## Roian Full image training -- quick surrogate for HT.

import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from importlib import reload
import PoseTools as pt
import APT_interface as apt
reload(apt)
lbl_file = '/nrs/branson/mayank/apt_cache_2/four_points_180806/20201225T042233_20201225T042235.lbl'
json_trn = '/nrs/branson/mayank/apt_cache_2/four_points_180806/loc.json'
gpu_q = 'gpu_rtx' #'gpu_tesla' #
# simg = '/groups/branson/bransonlab/mayank/singularity/test_tf_torch.sif'
simg = '/groups/branson/bransonlab/mayank/singularity/pytorch_mmpose.sif'
sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'

cmd = f'APT_interface.py {lbl_file} -json_trn_file {json_trn} -conf_params dl_steps 200000 is_multi True multi_crop_ims False multi_use_mask False db_format \\"coco\\" batch_size 8 save_step 10000 rescale 4 max_n_animals 2 {{}} -type multi_mmpose -name full_ims -cache /nrs/branson/mayank/apt_cache_2 train'

n = 'higherhrnet'
cur_cmd = cmd.replace('is_multi True','is_multi True mmpose_net \\"higherhrnet\\"')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
print(cur_cmd)
print()
pt.submit_job('roian_multi_ht_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=4)

n = 'grone'
cur_cmd = cmd.replace('type multi_mmpose','type multi_mdn_joint_torch')
# cur_cmd = cur_cmd.replace('-skip_db','')
cur_cmd = cur_cmd.format(' -train_name {}'.format(n))
print(cur_cmd)
print()
pt.submit_job('roian_multi_ht_{}'.format(n),cur_cmd,sdir,queue=gpu_q,sing_image=simg,numcores=6)

## Show tracking results
import TrkFile
import movies
mov = '/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00030_CsChr_RigC_20150826T144616/movie.ufmf'
trk_file = '/groups/branson/home/kabram/temp/alice_multi/cx_GMR_SS00030_CsChr_RigC_20150826T144616_grone.trk.part'

trk = TrkFile.load_trk(trk_file)
p = np.transpose(trk['pTrk'], (1, 0, 3, 2))

cap = movies.Movie(mov)

##
fr_num = np.random.choice(4000)
fr = cap.get_frame(fr_num)[0]
plt.imshow(fr,'gray')
plt.scatter(p[0,:,:,fr_num],p[1,:,:,fr_num])


## overlapping augmentation

import shapely
trn_coco = '/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/alice_maskloss_split_crop_ims_grone_multi/train_TF.json'
# trn_coco = '/nrs/branson/mayank/apt_cache_2/four_points_180806/multi_mdn_joint_torch/view_0/roian_maskloss_split_crop_ims_grone_multi/train_TF.json'
T = pt.json_load(trn_coco)
nims = len(T['images'])
rois = [[] for n in range(nims)]
anns = [[] for n in range(nims)]
for a in T['annotations']:
    rois[a['image_id']].append(np.array(a['segmentation'][0]).reshape([4,2]))
    anns[a['image_id']].append(a)

##
from scipy.ndimage.morphology import distance_transform_edt

done_ims = np.zeros(nims)
data = []
margin = 10
while np.any(done_ims==0):
    missing = np.where(done_ims==0)[0]
    selndx = np.random.choice(missing)
    im = cv2.imread(T['images'][selndx]['file_name'],cv2.IMREAD_UNCHANGED)
    mask = apt.create_mask(rois[selndx],im.shape[:2])
    dt = distance_transform_edt(1 - mask)
    mask = dt<10
    selids = [selndx,]
    do_merge = np.random.rand()>0.3
    print(do_merge)
    for x in np.random.permutation(missing)[:50]:
        if not do_merge:
            continue
        if x == selndx:
            continue
        cmask = apt.create_mask(rois[x],im.shape[:2])
        dt = distance_transform_edt(1-cmask)
        cmask = dt < margin
        if np.any(cmask.flatten() & mask.flatten()):
            continue
        selids.append(x)
        mask = mask | cmask

    #
    joint_im = None
    all_ims = []
    all_mask = []
    all_dt = []
    joint_anns = []
    for x in selids:
        cur_im = cv2.imread(T['images'][x]['file_name'], cv2.IMREAD_UNCHANGED)
        cmask = apt.create_mask(rois[x],im.shape[:2])
        dt = distance_transform_edt(1-cmask)
        dt = 1/(dt+1)
        all_dt.append(dt)
        if joint_im is None:
            joint_im = cur_im.astype('float64')*dt
            jd = dt
        else:
            joint_im += cur_im*dt
            jd += dt
        all_ims.append(cur_im)
        all_mask.append(cmask)
        joint_anns.extend(anns[x])

    joint_im = joint_im/jd
    joint_im = joint_im.astype('uint8')
    data.append([joint_im,joint_anns,selids])
    done_ims[selids] = 1
    print(f'{np.count_nonzero(done_ims)} out of {nims} done')

# ff();plt.imshow(joint_im,'gray')
# ff(); plt.imshow(joint_im*prev_dt,'gray')
##
import json
outdir = '/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mmpose/view_0/merged_2'
outfile = os.path.join(outdir,'train_TF.json')
imdir = os.path.join(outdir,'train')

outd = {'info':T['info'],'categories':T['categories']}
h = T['images'][0]['height']
w = T['images'][0]['width']
im_array = []
ann_array = []
for ix,d in enumerate(data):
    curid = d[2]
    imfile = os.path.join(imdir,f'{ix:08d}.png')
    mov = [T['images'][i]['movid'] for i in curid]
    frm = [T['images'][i]['frm'] for i in curid]
    patch = [T['images'][i]['patch'] for i in curid]
    idict = {'id':ix,'width':w,'height':h,'file_name':imfile,'movid':mov,'patch':patch,'frm':frm}
    im_array.append(idict)
    for a in d[1]:
        kk = a.copy()
        kk['image_id'] = ix
        ann_array.append(kk)

    if d[0].ndim<3:
        cv2.imwrite(imfile,d[0])
    elif d[0].shape[2] == 1:
        cv2.imwrite(imfile,d[0])
    else:
        cur_im = cv2.cvtColor(cur_im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(imfile,d[0])

outd['images'] = im_array
outd['annotations'] = ann_array
with open(outfile,'w') as f:
    json.dump(outd,f)

## Add negatives

import PoseTools as pt
import json
import h5py
from scipy import io as sio
import os
import multiResData
import cv2

# trn = '/nrs/branson/mayank/apt_cache_2/alice_ma/loc.json'
in_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/loc_split.json'
lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'
# trn = '/nrs/branson/mayank/apt_cache_2/four_points_180806/loc.json'
# split_trn = '/nrs/branson/mayank/apt_cache_2/four_points_180806/loc_split.json'
jj = os.path.splitext(in_file)
out_file = jj[0] + '_neg' + jj[1]

sz = (1024,1024)
boxsz = 80
negbox = 240
debug = False
bdir,_ = os.path.split(in_file)
T = pt.json_load(in_file)
lbl = h5py.File(lbl_file,'r')
done_ix = []
num_negs = 150
for cc in range(num_negs):
    ix = np.random.choice(len(T['locdata']))
    done_ix.append(ix)
    curp = T['locdata'][ix]
    tt = os.path.split(T['movies'][curp['imov']-1])
    trx = sio.loadmat(os.path.join(tt[0],'registered_trx.mat'))['trx'][0]
    ntrx = len(trx)
    fnum = curp['frm']
    all_mask = np.zeros(sz)
    boxes = []
    for tndx in range(ntrx):
        cur_trx = trx[tndx]
        if fnum > cur_trx['endframe'][0, 0] - 1:
            continue
        if fnum < cur_trx['firstframe'][0, 0] - 1:
            continue
        x, y, theta = multiResData.read_trx(cur_trx, fnum)
        x = int(round(x)); y = int(round(y))
        x_min = max(0,x-boxsz)
        x_max = min(sz[1],x+boxsz)
        y_min = max(0,y-boxsz)
        y_max = min(sz[0],y+boxsz)
        all_mask[y_min:y_max,x_min:x_max] = 1
        boxes.append([x_min,x_max,y_min,y_max])

    if debug:
        im = cv2.imread(os.path.join(bdir,curp['img'][0]),cv2.IMREAD_UNCHANGED)
        plt.imshow(im,'gray')
        for b in boxes:
            plt.plot([b[0],b[1],b[1],b[0],b[0]],[b[2],b[2],b[3],b[3],b[2]])

    done = False
    selb = []
    for count in range(20):
        negx_min = np.random.randint(sz[1]-negbox)
        negy_min = np.random.randint(sz[0]-negbox)
        if np.any(all_mask[negy_min:negy_min+negbox,negx_min:negx_min+negbox]>0):
            continue
        done = True
        selb = [negx_min,negx_min+negbox,negy_min,negy_min+negbox]
        break

    if debug:
        cc = all_mask.copy()
        cc[selb[2]:selb[3],selb[0]:selb[1]] = -1
        ff(); imshow(cc)

    if done:
        print(f'Adding Roi for {ix}')
        curp['extra_roi'] = [negx_min,negx_min,negx_min+negbox,negx_min+negbox,negy_min,negy_min+negbox,negy_min+negbox,negy_min]

with open(out_file,'w') as f:
  json.dump(T,f)

## convert HT aligned bboxes to axis aligned

import PoseTools as pt
import json
import h5py
from scipy import io as sio
import os
import multiResData
import cv2

in_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/loc_split_neg.json'
lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'

jj = os.path.splitext(in_file)
out_file = jj[0] + '_axis_aligned' + jj[1]

# sz = (1024,1024)
T = pt.json_load(in_file)
# lbl = h5py.File(lbl_file,'r')
sfactor = 1.0
rad = 0 #5.
for h in T['locdata']:
    pts = np.array(h['pabs']).reshape([2,17,-1])
    pts_min = np.min(pts,axis=-2)
    pts_max = np.max(pts,axis=-2)
    pts_ctr = (pts_min+pts_max)/2
    pts_sz = (pts_max-pts_min)/2

    new_min = pts_ctr - pts_sz*sfactor - rad
    new_max = pts_ctr + pts_sz*sfactor+rad

    obb = np.array(h['roi']).reshape([2,4,-1])
    obb[0,:2,:] = new_min[0]
    obb[0,2:,:] = new_max[0]
    obb[1,[0,3],:] = new_min[1]
    obb[1,[1,2],:] = new_max[1]

    obb = obb.reshape([8,-1])
    out_roi = [oo.tolist() for oo in obb]
    h['roi'] = out_roi

with open(out_file,'w') as f:
  json.dump(T,f)


## ht crop vs full training

sdir= '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'
simg = '/groups/branson/bransonlab/mayank/singularity/pytorch_mmpose.sif'

crop_cmd = 'APT_interface.py /nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl -conf_params db_format \\"coco\\" mmpose_net \\"higherhrnet\\" dl_steps 200000 save_step 50000 op_affinity_graph \(\(0,1\),\) multi_use_mask False multi_loss_mask True multi_crop_ims True rrange 180 trange 30 is_multi True max_n_animals 7 ht_pts \(0,6\) multi_only_ht True rescale 2 -json_trn_file /nrs/branson/mayank/apt_cache_2/alice_ma/loc_split_neg.json -type multi_mdn_joint_torch -name alice_neg_ht_test -cache /nrs/branson/mayank/apt_cache_2 -no_except train -use_cache'

full_cmd = 'APT_interface.py /nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl -conf_params db_format \\"coco\\" mmpose_net \\"higherhrnet\\" dl_steps 200000 save_step 50000 op_affinity_graph \(\(0,1\),\) multi_use_mask False multi_loss_mask True multi_crop_ims False rrange 180 trange 30 is_multi True max_n_animals 7 ht_pts \(0,6\) multi_only_ht True rescale 2 -json_trn_file /nrs/branson/mayank/apt_cache_2/alice_ma/loc_split_neg.json -type multi_mdn_joint_torch -name alice_neg_ht_test_no_crop -cache /nrs/branson/mayank/apt_cache_2 -no_except train -use_cache'

pt.submit_job('alice_ht_crop',crop_cmd,sdir,sing_image=simg)
pt.submit_job('alice_ht_full',full_cmd,sdir,sing_image=simg)

## ht val db for validating single animal second stage
from reuse import *
import multiResData
import tensorflow as tf
ht_dir = '/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/alice_neg_ht_test_no_crop'
ht_val_res = '/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/alice_neg_ht_test/val_results_deepnet_20210406'
ht_db = f'{ht_dir}/val_TF.json'
ht_val = f'{ht_dir}/val'
all_db = '/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/alice_neg_split_full_ims/val_TF.json'
lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'

net = 'mdn_joint_fpn'
tname = 'alice_neg' #'apt_aug_0tbcrange' #
conf = apt.create_conf(lbl_file,0,tname,'/nrs/branson/mayank/apt_cache_2',net)
conf.imsz = (192,192)
conf.trx_align_theta = True
H = pt.json_load(all_db)
out = pt.pickle_load(ht_val_res)['out']
T = pt.json_load(ht_db)
pack_dir = ht_val
out_db = os.path.join(conf.cachedir,'val_ht.tfrecords')
env = tf.python_io.TFRecordWriter(out_db)
cur_out = lambda data: env.write(apt.tf_serialize(data))

indx = np.array([t['image_id'] for t in T['annotations']])

for selndx, cur_t in enumerate(T['images']):

    cur_frame = cv2.imread(cur_t['file_name'], cv2.IMREAD_UNCHANGED)
    if cur_frame.ndim == 2:
        cur_frame = cur_frame[..., np.newaxis]

    cur_locs = []
    all_locs = []
    for ix in np.where(indx==selndx)[0]:
        cur_locs.append(T['annotations'][ix]['keypoints'])
        all_locs.append(H['annotations'][ix]['keypoints'])
    cur_locs = np.array(cur_locs)
    cur_locs = np.reshape(cur_locs,[-1,2,3])
    cur_locs = cur_locs[...,:2].copy()

    all_locs = np.array(all_locs)
    all_locs = np.reshape(all_locs,[-1,17,3])
    all_occ = all_locs[...,2].copy()
    all_locs = all_locs[...,:2].copy()

    for ndx in range(len(cur_locs)):
        info = [cur_t['movid'], cur_t['frm'], ndx]
        ht_locs_orig = cur_locs[ndx, ...].copy()
        dd_mat = np.linalg.norm(ht_locs_orig[None,...]-out[0][selndx],axis=-1)
        if np.all(np.isnan(dd_mat)):
            continue
        pndx = np.nanargmin(dd_mat.sum(-1))
        ht_locs = out[0][selndx,pndx]
        ht_ctr = ht_locs.mean(axis=0)
        theta = np.arctan2(ht_locs[0, 1] - ht_locs[1, 1], ht_locs[0, 0] - ht_locs[1, 0])
        curl = all_locs[ndx].copy()
        cur_patch, curl = multiResData.crop_patch_trx(conf, cur_frame, ht_ctr[0], ht_ctr[1], theta, curl)


        data_out = {'im': cur_patch, 'locs': curl, 'info': info, 'occ': all_occ[ndx,]}
        cur_out(data_out)



env.close()

## top down 2 stage accuracy

import APT_interface as apt
lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'
net = 'mdn_joint_fpn'
tname = 'alice_neg' #'apt_aug_0tbcrange' #
mfile = f'/nrs/branson/mayank/apt_cache_2/alice_ma/mdn_joint_fpn/view_0/alice_neg/deepnet-100000'
conf = apt.create_conf(lbl_file,0,tname,'/nrs/branson/mayank/apt_cache_2',net)
conf.imsz = (192,192)
conf.flipLandmarkMatches = {'11': 16, '16': 11, '1': 2, '2': 1, '3': 4, '4': 3, '7': 9, '9': 7, '8': 10, '10': 8, '12': 15, '15': 12, '13': 14, '14': 13}
aa = apt.classify_db_all(net,conf,'/nrs/branson/mayank/apt_cache_2/alice_ma/mdn_joint_fpn/view_0/alice_neg/val_ht.tfrecords',mfile)
import pickle
import PoseTools as pt
with open(os.path.join(conf.cachedir,'val_results_ht'+'_'+pt.get_datestr()),'wb') as f:
    pickle.dump({'out':aa,'conf':conf},f)

dd = np.linalg.norm(aa[0]-aa[1],axis=-1)
np.percentile(dd,[50,76,90,95,97],axis=0)

## check how many unlabeled flies lie inside the mask.
lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'
locfile = '/nrs/branson/mayank/apt_cache_2/alice_ma/20210203/loc.json'
import multiResData
import PoseTools as pt
import APT_interface as apt
from scipy import io as sio
import tqdm

mfiles = multiResData.find_local_dirs(lbl_file)
tfiles = [m.replace('movie.ufmf','registered_trx.mat') for m in mfiles]

T = pt.json_load(locfile)
prev_trx = -1
trx = None
ht_ndx = [0,6]
in_mask = []
part_mask = []
for _,t in tqdm.tqdm(enumerate(T['locdata'])):
    cur_trx_ndx = t['imov']-1
    cur_fr = t['frm'] -1
    if cur_trx_ndx != prev_trx:
        cur_trx = sio.loadmat(tfiles[cur_trx_ndx])['trx'][0]
        ntrx = len(cur_trx)
        prev_trx = cur_trx_ndx

    cur_pts = []
    for tndx in range(ntrx):
        if (cur_fr > cur_trx[tndx]['endframe'][0, 0] - 1) or (cur_fr < cur_trx[tndx]['firstframe'][0, 0] - 1):
            continue
        trx_fnum = cur_fr - int(cur_trx[tndx]['firstframe'][0, 0]) + 1
        x = cur_trx[tndx]['x'][0, trx_fnum] - 1
        y = cur_trx[tndx]['y'][0, trx_fnum] - 1
        # -1 for 1-indexing in matlab and 0-indexing in python
        theta = cur_trx[tndx]['theta'][0, trx_fnum]
        a = cur_trx[tndx]['a'][0, trx_fnum]
        b = cur_trx[tndx]['b'][0, trx_fnum]

        R = np.array([[np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2), 0],
             [- np.sin(theta + np.pi / 2), np.cos(theta + np.pi / 2), 0],
             [0, 0, 1]])

        hh = np.array([[0,-a * 2,1],[0,a*2,1]])
        ht_pts = np.matmul(hh,R)[:,:2]
        ht_pts[:,0] += x
        ht_pts[:,1] += y
        cur_pts.append(ht_pts)
    cur_pts = np.array(cur_pts)
    ss = np.array(t['pabs']).reshape([2, 17, t['ntgt']]).transpose([2, 1, 0])[:, ht_ndx, :]
    dd = np.linalg.norm(ss[:, None, ...] - cur_pts[None, ...], axis=-1).sum(-1)
    to_keep = np.where(dd.min(axis=0) > 15)[0]
    cur_pts = cur_pts[to_keep]

    roi = np.array(t['roi']).reshape([2,4,-1]).transpose([2,1,0])
    mm = apt.create_mask(roi,[1024,1024])

    cin_mask = 0
    cpart_mask = 0
    for p in cur_pts:
        gg = p.astype('int')
        if np.all(mm[gg[:,1],gg[:,0]]):
            cin_mask += 1
        elif np.any(mm[gg[:,1],gg[:,0]]):
            cpart_mask += 1
    in_mask.append(cin_mask)
    part_mask.append(cpart_mask)


# 94,507 out of 7208 new lbl, aligned roi
# 202,767 out of 7197 for 20210203

## Show open pose errors

import h5py
from reuse import *
from Pose_multi_openpose import Pose_multi_openpose
res_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/val_results/alice_split_ht_openpose_full_ims_multi_0.mat'
trn_json = '/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/alice_split_ht_grone_full_ims_multi/val_TF.json'
tdata = '/nrs/branson/mayank/apt_cache_2/alice_ma/multi_openpose/view_0/alice_split_ht_openpose_full_ims_multi/traindata'

res_file = '/nrs/branson/mayank/apt_cache_2/four_points_180806/val_results/roian_split_full_maskless_openpose_pose_multi_0.mat'
trn_json = '/nrs/branson/mayank/apt_cache_2/four_points_180806/multi_mdn_joint_torch/view_0/roian_split_full_maskless_grone_pose_multi/val_TF.json'
tdata = '/nrs/branson/mayank/apt_cache_2/four_points_180806/multi_openpose/view_0/roian_split_full_maskless_openpose_pose_multi/traindata'

conf = pt.pickle_load(tdata)[1]
K = h5py.File(res_file, 'r')
ll = K['labeled_locs'][()].T
# pp = K['pred_locs']['locs_top'][()].T
# ll = ll[...,conf.ht_pts,:]
pp = K['pred_locs']['locs'][()].T
ll[ll < -1000] = np.nan
dd = np.linalg.norm(ll[:, None] - pp[:, :, None], axis=-1)
dd1 = find_dist_match(dd)
K.close()


##

from Pose_multi_openpose import Pose_multi_openpose
import PoseTools as pt
sel = 21 # For alice HT error


H = pt.json_load(trn_json)
im = cv2.imread(H['images'][sel]['file_name'],cv2.IMREAD_UNCHANGED)
conf.batch_size = 1
J = Pose_multi_openpose(conf)
O = J.diagnose(np.tile(im[None,...,None],[1,1,1,conf.img_dim]))
paf = O['pred_hmaps'][0][0][0]
paf = np.sqrt(paf[...,::2]**2 + paf[...,1::2]**2)
hmap = O['pred_hmaps'][0][1][0]

##
pt.show_stack(paf.transpose([2,0,1]),1,3,'jet')
pt.show_stack(hmap.transpose([2,0,1]),2,2,'jet')
ff();imshow(im,'gray'); mdskl(pp[sel],[[0,1],[0,2],[0,3]])
mdskl(ll[sel],[[0,1],[0,2],[0,3]],cc=[0,1,0])

##
f,ax = plt.subplots(1,3,sharex=True,sharey=True)
ax = ax.flatten()
ax[0].imshow(im,'gray')
hm = cv2.resize(hmap.sum(-1),im.shape)
ax[0].imshow(hm,alpha=0.3)
ax[0].set_title('Landmark heatmap')
pr = cv2.resize(paf.sum(-1),im.shape)
ax[1].imshow(im,'gray')
ax[1].imshow(pr,alpha=0.3)
ax[1].set_title('Affinity Field')

ax[2].imshow(im,'gray')
ax[2].plot(pp[sel,...,0].T,pp[sel,...,1].T,color='b')
ax[2].set_title('Prediction')

f.set_size_inches([12,4])
for aa in ax:
    aa.axis('tight')
    aa.axis('off')
# ax[0].set_xlim([650,950])
# ax[1].set_ylim([650,350])

# savefig('/groups/branson/home/kabram/temp/openpose_error.png',ax_type=[])

## Debug ma dlc
## create the dataset
from reuse import *
in_dir = '/nrs/branson/mayank/apt_cache_2/four_points_180806/multi_mdn_joint_torch/view_0/roian_split_crop_ims_grone_pose_multi'
tfile = os.path.join(in_dir,'train_TF.json')
out_dir = '/nrs/branson/mayank/apt_cache_2/four_points_180806/multi_dlc/view_0/roian_split_crop_ims_dlc_pose_multi/'

H = pt.json_load(tfile)
iidx = np.array([h['image_id'] for h in H['annotations']])
outd = []
for h in H['images']:
    oo = {}
    oo['image'] = h['file_name']
    oo['size'] = np.array((1,512,512))
    jj = np.where(iidx==h['id'])[0]
    joints = {}
    for ndx,j in enumerate(jj):
        curj = np.zeros([4,3])
        curj[:,0] = np.arange(4)
        curj[:,1:3] = np.array(H['annotations'][j]['keypoints']).reshape(4,3)[:,:2]
        joints[f'{ndx}'] = curj
    oo['joints'] = joints
    outd.append(oo)

os.makedirs(out_dir,exist_ok=True)
import pickle
with open(os.path.join(out_dir,'data.h5'),'wb') as f:
    pickle.dump(outd,f)

## compute the DLC results
import PoseTools as pt
from scipy.optimize import linear_sum_assignment

out_file = '/nrs/branson/mayank/apt_cache_2/four_points_180806/multi_dlc/view_0/roian_split_crop_ims_dlc_pose_multi/val_results.p'
val_file = '/nrs/branson/mayank/apt_cache_2/four_points_180806/multi_mdn_joint_torch/view_0/roian_split_full_ims_grone_pose_multi/val_TF.json'
vdat = pt.json_load(val_file)
ijxx = np.array([v['image_id'] for v in vdat['annotations']])
dres = pt.pickle_load(out_file)

op_gr = [[0,1],[0,2],[0,3]]
n_pts = 4

preds = np.ones([len(dres),5,4,2])*np.nan
labels = np.ones([len(dres),2,4,2])*np.nan
for ndx in range(len(dres)):
    pts = dres[ndx][0]['coordinates'][0]
    curp = [[p] for p in pts[0]]
    for ix,ee in enumerate(op_gr):
        cc = dres[ndx][0]['costs'][ix]['m1'].copy()
        for cndx in range(cc.shape[0]):
            if np.all(np.isnan(cc)):
                break
            xx,yy = np.unravel_index(np.nanargmax(cc), cc.shape)
            curp[xx].append(pts[ee[1]][yy])
            cc[xx,:] = np.nan
            cc[:,yy] = np.nan

    for ax,p in enumerate(curp):
        for bx,v in enumerate(p):
            preds[ndx,ax,bx,:] = v

    ss = np.where(ijxx==ndx)[0]
    for lndx,sndx in enumerate(ss):
        if vdat['annotations'][sndx]['area']<3:
            continue
        curk = np.array(vdat['annotations'][sndx]['keypoints']).reshape([4,3])[:,:2]
        labels[ndx,lndx] = curk

dd = np.linalg.norm(preds[:,:,None]-labels[:,None],axis=-1)
dd1 = find_dist_match(dd)

dd1 = np.reshape(dd1,[-1,4])
np.nanpercentile(dd1,[50,75,90,95,98],axis=0)
