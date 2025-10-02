import os
#os.environ['CUDA_VISIBLE_DEVICES'] ='0'
cmd = '/groups/branson/home/kabram/.apt/tpefaab1af_1c8f_4d33_8452_b770bd22e77b/train00/20250226T025722_20250226T025725.json -name 20250226T025722 -json_trn_file /groups/branson/home/kabram/.apt/tpefaab1af_1c8f_4d33_8452_b770bd22e77b/train00/loc.json -conf_params -type multi_mdn_joint_torch -ignore_local 1 -cache /groups/branson/home/kabram/.apt/tpefaab1af_1c8f_4d33_8452_b770bd22e77b train -use_cache'

# cmd =  ['-name', '20220629T224821', '-view', '1', '-cache', '/groups/branson/home/kabram/APT_bugs/tp657c1885_2aa3_49ac_b34a_57baabbaff11', '-conf_params','op_affinity_graph','((1,0),)','-json_trn_file', '/groups/branson/home/kabram/APT_bugs/tp657c1885_2aa3_49ac_b34a_57baabbaff11/APTproject/loc.json', '-type', 'multi_openpose', '/groups/branson/home/kabram/APT_bugs/tp657c1885_2aa3_49ac_b34a_57baabbaff11/APTproject/20220705T203931_20220705T203934.lbl', 'train', '-use_cache']


#cmd = '/groups/branson/home/kabram/temp/ma_expts/alice/trn_packdir_23022022/2stageBBox/conf_nocrop.json -name 2stageBBox_nocrop_mask_first_23022022 -json_trn_file /groups/branson/home/kabram/temp/ma_expts/alice/trn_packdir_23022022/2stageBBox/loc_neg.json -conf_params multi_loss_mask True link_id True -cache /groups/branson/bransonlab/mayank/apt_cache_2 -stage multi -model_files2 /groups/branson/bransonlab/mayank/apt_cache_2/alice_ma/mdn_joint_fpn/view_0/2stageBBox_nocrop_second_23022022/deepnet-100000 -conf_params2  -type2 mdn_joint_fpn -name2 2stageBBox_nocrop_second_23022022 -type detect_mmdetect track -mov /groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00030_CsChr_RigC_20150826T144616/movie.ufmf -out /groups/branson/home/kabram/temp/ma_expts/alice/trks/cx_GMR_SS00030_CsChr_RigC_20150826T144616_1_bbox.trk -trx /groups/branson/home/kabram/temp/ma_expts/alice/trks/aa.trk -track_type only_predict -start_frame 48800 -end_frame 49000'

cmd = '/groups/branson/home/kabram/.apt/tp1c29b469_3c98_4ddd_a1d2_b610a734c36f/BottomViewsMA/20250805T033405_20250805T033410.json -name 20250805T033405 -type multi_mdn_joint_torch -model_files /groups/branson/home/kabram/.apt/tp1c29b469_3c98_4ddd_a1d2_b610a734c36f/BottomViewsMA/multi_mdn_joint_torch/view_0/20250805T033405/deepnet-2000 -ignore_local 1 -cache /groups/branson/home/kabram/.apt/tp1c29b469_3c98_4ddd_a1d2_b610a734c36f track -config_file /groups/branson/home/kabram/.apt/tp1c29b469_3c98_4ddd_a1d2_b610a734c36f/BottomViewsMA/multi_mdn_joint_torch/view_0/20250805T033405/trk/trkconfig_image_cam_1_date_2025_04_08_time_18_35_20_v001_crop_col0to1331_rot90_0adecc_trn20250805T033405_view0_iter2000_20250805T050345.json -track_type only_predict -out /groups/branson/home/kabram/.apt/tp1c29b469_3c98_4ddd_a1d2_b610a734c36f/BottomViewsMA/multi_mdn_joint_torch/view_0/20250805T033405/trk/image_cam_1_date_2025_04_08_time_18_35_20_v001_crop_col0to1331_rot90_0adecc_trn20250805T033405_view0_iter2000_20250805T050345.trk -mov /groups/branson/bransonlab/aniket/APT/3D_labeling_project/movie_output_dir_combined_views/exp_32/image_cam_1_date_2025_04_08_time_18_35_20_v001_crop_col0to1331_rot90.ufmf -start_frame 1 -end_frame 282'

cmd = '/groups/branson/home/kabram/.apt/tp0c2a761a_f5a9_42a6_9adb_790655439828/combinedSideViewMA/20250905T022700_20250905T022705.json -name 20250905T022700 -type multi_mdn_joint_torch -model_files /groups/branson/home/kabram/.apt/tp0c2a761a_f5a9_42a6_9adb_790655439828/combinedSideViewMA/multi_mdn_joint_torch/view_0/20250905T022700/deepnet-51000 -ignore_local 1 -cache /groups/branson/home/kabram/.apt/tp0c2a761a_f5a9_42a6_9adb_790655439828 track -config_file /groups/branson/home/kabram/.apt/tp0c2a761a_f5a9_42a6_9adb_790655439828/combinedSideViewMA/multi_mdn_joint_torch/view_0/20250905T022700/trk/trkconfig_image_cam_0_date_2025_06_11_time_18_40_40_v001_crop_col1167to1919_rot90_c7a992_trn20250905T022700_view0_iter51000_20250905T061458.json -track_type only_predict -out /groups/branson/home/kabram/.apt/tp0c2a761a_f5a9_42a6_9adb_790655439828/combinedSideViewMA/multi_mdn_joint_torch/view_0/20250905T022700/trk/image_cam_0_date_2025_06_11_time_18_40_40_v001_crop_col1167to1919_rot90_c7a992_trn20250905T022700_view0_iter51000_20250905T061458.trk -mov /groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/exp_41/fly_images/cropped_uniform_sizes/image_cam_0_date_2025_06_11_time_18_40_40_v001_crop_col1167to1919_rot90.ufmf -start_frame 21708 -end_frame 21791'

cmd = '/groups/branson/home/kabram/.apt/tp0c2a761a_f5a9_42a6_9adb_790655439828/combinedSideViewMA/20250905T022700_20250905T022705.json -name 20250905T022700 -json_trn_file /groups/branson/home/kabram/.apt/tp0c2a761a_f5a9_42a6_9adb_790655439828/combinedSideViewMA/loc.json -conf_params -type multi_mdn_joint_torch -ignore_local 1 -cache /groups/branson/home/kabram/.apt/tp0c2a761a_f5a9_42a6_9adb_790655439828 train -use_cache -continue -skip_db'

cmd = ['/groups/branson/home/kabram/.apt/tpdb2c3173_f192_4328_9185_fab6369cb9fd/unmarkedMice/20250612T153510_20250612T153510.json', '-name', '20250612T153510', '-json_trn_file', '/groups/branson/home/kabram/.apt/tpdb2c3173_f192_4328_9185_fab6369cb9fd/unmarkedMice/loc.json', '-stage', 'first', '-conf_params', 'mmdetect_net','"frcnn"','mmdetect_use_default_sz','False','-type', 'detect_mmdetect', '-ignore_local', '1', '-cache', '/groups/branson/home/kabram/.apt/tpdb2c3173_f192_4328_9185_fab6369cb9fd', 'train', '-use_cache','-skip_db']

cmd = '/groups/branson/home/kabram/.apt/tp5314232a_cd75_468f_bb50_39cad4848880/combinedSideViewMA/20250930T055312_20250930T055317.json -name 20250930T055312 -err_file /groups/branson/home/kabram/.apt/tp5314232a_cd75_468f_bb50_39cad4848880/combinedSideViewMA/multi_mdn_joint_torch/view_0/20250930T055312/trk/track_20251001T014542_mov1_vw1.err -log_file /groups/branson/home/kabram/.apt/tp5314232a_cd75_468f_bb50_39cad4848880/combinedSideViewMA/multi_mdn_joint_torch/view_0/20250930T055312/trk/track_20251001T014542_mov1_vw1.log -type multi_mdn_joint_torch -model_files /groups/branson/home/kabram/.apt/tp5314232a_cd75_468f_bb50_39cad4848880/combinedSideViewMA/multi_mdn_joint_torch/view_0/20250930T055312/deepnet-200000 -ignore_local 1 -cache /groups/branson/home/kabram/.apt/tp5314232a_cd75_468f_bb50_39cad4848880 track -config_file /groups/branson/home/kabram/.apt/tp5314232a_cd75_468f_bb50_39cad4848880/combinedSideViewMA/multi_mdn_joint_torch/view_0/20250930T055312/trk/trkconfig_image_cam_0_date_2025_04_08_time_18_35_18_v001_crop_col1167to1919_rot90_3cc838_trn20250930T055312_view0_iter200000_20251001T014542.json -track_type only_predict -out /groups/branson/home/kabram/.apt/tp5314232a_cd75_468f_bb50_39cad4848880/combinedSideViewMA/multi_mdn_joint_torch/view_0/20250930T055312/trk/image_cam_0_date_2025_04_08_time_18_35_18_v001_crop_col1167to1919_rot90_3cc838_trn20250930T055312_view0_iter200000_20251001T014542.trk -mov /groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/exp_32/fly_images/cropped_uniform_sizes/image_cam_0_date_2025_04_08_time_18_35_18_v001_crop_col1167to1919_rot90.ufmf -start_frame 107 -end_frame 270'

cmd = ['/groups/branson/home/kabram/.apt/tp2f16fcfd_b4b7_4638_80a7_31cbb8280296/RatCityFullSizeRT/20251002T033239_20251002T033245.json', '-name', '20251002T033239','-model_files','/groups/branson/home/kabram/.apt/tp2f16fcfd_b4b7_4638_80a7_31cbb8280296/RatCityFullSizeRT/multi_mdn_joint_torch/view_0/20251002T033239/deepnet-9000','-json_trn_file', '/groups/branson/home/kabram/.apt/tp2f16fcfd_b4b7_4638_80a7_31cbb8280296/RatCityFullSizeRT/loc.json', '-conf_params', '-type', 'multi_mdn_joint_torch', '-ignore_local', '1', '-cache', '/groups/branson/home/kabram/.apt/tp2f16fcfd_b4b7_4638_80a7_31cbb8280296', 'train', '-use_cache','-skip_db']

if type(cmd)!=list:
    cmd = cmd.replace('\\','')
#     cmd = cmd.replace('"','')
#     cmd = cmd.replace("'","")

from reuse import *
# from APT_track import main
# cmd = cmd.replace('"/','/')
# cmd = cmd.replace('" ',' ')
if __name__ == '__main__':
    if type(cmd) == list:
        apt.main(cmd)
    else:
        # cmd = cmd.replace('\\', '')
        apt.main(cmd.split())




##
## convert ufmf to h.264

import movies
import cv2
from reuse import *
in_mov = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigA_20201212T163531//movie.ufmf'
out_mov = '/groups/branson/home/kabram/temp/nochr_TrpA65F12_Unknown_RigA_20201212T163531.mp4'
cap = movies.Movie(in_mov)

fps = 30
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter(out_mov, fourcc, fps, (1024,1024))

for ndx in range(cap.get_n_frames()):
    img = cap.get_frame(ndx)[0]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out.write(img)

out.release()

##
# 3244
# /groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00168_CsChr_RigD_20150909T111218/movie.ufmf
# frame: 24687

# [    7, 23048,     0]

##
cmd = '/groups/branson/home/kabram/.apt/tp72b99fab_dd31_4d25_8177_821315aebd59/floss/20220609T054112_20220609T054115.lbl -name 20220609T054112 -json_trn_file /groups/branson/home/kabram/.apt/tp72b99fab_dd31_4d25_8177_821315aebd59/floss/loc.json -stage first -ignore_local 1 -type multi_mdn_joint_torch -cache /groups/branson/home/kabram/.apt/tp72b99fab_dd31_4d25_8177_821315aebd59 train -use_cache'

from reuse import *
# cmd = cmd.replace('"/','/')
# cmd = cmd.replace('" ',' ')
if type(cmd) == list:
    apt.main(cmd)
else:
    cmd = cmd.replace('\\', '')
    apt.main(cmd.split())

## creating conf two methods
view = 0
cache_dir = '/groups/branson/home/kabram/.apt/tpe983c269_9bad_4910_84ec_9f1d18bd2535'
net_type = 'mdn_joint_fpn'
first_stage = False
second_stage = False
name = 'test'
lbl_file = '/groups/branson/home/kabram/.apt/tpe983c269_9bad_4910_84ec_9f1d18bd2535/more_than_two_mice/20210928T151013_20210928T151036.lbl'
json_conf_file = '/groups/branson/home/kabram/.apt/tpe983c269_9bad_4910_84ec_9f1d18bd2535/more_than_two_mice/20210928T151013_20210928T151036.json'
json_trn_file = None
conf_params = None
quiet = False

conf1 = apt.create_conf(lbl_file, view, name, cache_dir=cache_dir,net_type=net_type,first_stage=first_stage,second_stage=second_stage)

A = pt.json_load(json_conf_file)
net_names_dict = {'mdn':'MDN',
                  'dpk':'DeepPoseKit',
                  'openpose':'OpenPose',
                  'multi_openpose':'MultiAnimalOpenPose',
                  'sb':'',
                  'unet':'Unet',
                  'deeplabcut':'DeepLabCut',
                  'leap':'LEAP',
                  'detect_mmpose':'MMDetect',
                  'mdn_joint_fpn':'GRONe',
                  'multi_mdn_joint_torch':'MultiAnimalGRONe',
                  'mmpose':'MSPN',
                  }

import poseConfig
import re
from APT_interface import to_py
import ast

conf = poseConfig.config()
proj_name = A['ProjName']
conf.set_exp_name(proj_name)
cc = A['Config']
conf.nviews = cc['NumViews']
conf.n_classes = cc['NumLabelPoints']
conf.selpts = np.arange(conf.n_classes)
conf.project_file = ''   ## TODO
conf.is_multi = cc['MultiAnimal']>0.5
conf.img_dim = cc['NumChans']
has_crops = cc['HasCrops']

conf.cachedir = os.path.join(cache_dir, proj_name, net_type, 'view_{}'.format(view), name)
if not os.path.exists(conf.cachedir):
    os.makedirs(conf.cachedir)

if first_stage:
    conf.stage = 'first'
elif second_stage:
    conf.stage = 'second'
else:
    conf.stage = None

dt_params = A['TrackerData']['sPrmAll']['ROOT']
if second_stage:
    # Find out whether head-tail or bbox detector
    if dt_params['MultiAnimal']['Detect']['multi_only_ht']:
        conf.use_ht_trx = True
    else:
        conf.use_bbox_trx = True

# If the project has trx file then we use the crop locs
# specified by the user. If the project doesnt have trx files
# then we use the crop size specified by user else use the whole frame.
if conf.has_trx_file or conf.use_ht_trx or conf.use_bbox_trx:
    width = dt_params['MultiAnimal']['TargetCrop']['Radius']*2
    conf.imsz = (width,width)
elif has_crops:
    conf.imsz = (-1,-1)   # TODO
else:
    conf.imsz = (A['MovieInfo']['NumRows'],A['MovieInfo']['NumCols'])

conf.labelfile = json_conf_file
conf.sel_sz = min(conf.imsz)
conf.multi_animal_crop_sz = dt_params['MultiAnimal']['TargetCrop']['Radius']*2
conf.trx_align_theta = dt_params['MultiAnimal']['TargetCrop']['AlignUsingTrxTheta']

def set_all(conf, cur_set, flatten=False):
    for k in cur_set:
        if type(cur_set[k]) is not dict:
            conf.__dict__[k] = cur_set[k]

set_all(conf, dt_params['MultiAnimal'])
set_all(conf, dt_params['DeepTrack']['Saving'])
set_all(conf, dt_params['DeepTrack']['ImageProcessing'])
set_all(conf, dt_params['DeepTrack']['GradientDescent'])
set_all(conf, dt_params['DeepTrack']['DataAugmentation'])
set_all(conf, dt_params['DeepTrack']['LossFunction'])
set_all(conf, dt_params['MultiAnimal']['TrackletStitch'])

conf.rescale = conf.scale
conf.ht_pts = to_py(dt_params['MultiAnimal']['Detect']['ht_pts'])

net_conf = dt_params['DeepTrack'][net_names_dict[net_type]]
set_all(conf, net_conf)

try:
    conf.op_affinity_graph = poseConfig.parse_aff_graph(dt_params['DeepTrack']['OpenPose']['affinity_graph'])
except KeyError:
    pass

f_str = conf.flipLandmarkMatches
graph = {}
if f_str:
    f_str = f_str.split(',')
    for b in f_str:
        mm = re.search('(\d+)\s+(\d+)', b)
        n1 = int(mm.groups()[0]) - 1
        n2 = int(mm.groups()[1]) - 1
        graph['{}'.format(n1)] = n2
        graph['{}'.format(n2)] = n1
        # The keys have to be strings so that they can be saved in the trk file
conf.flipLandmarkMatches = graph
conf.mdn_groups = [(i,) for i in range(conf.n_classes)]

if conf_params is not None:
    cc = conf_params
    assert len(cc) % 2 == 0, 'Config params should be in pairs of name value'
    for n, v in zip(cc[0::2], cc[1::2]):
        if not quiet:
            print('Overriding param %s <= ' % n, v)
        setattr(conf, n, ast.literal_eval(v))

# overrides for each network
if net_type == 'sb':
    sb.update_conf(conf)
# elif net_type == 'openpose':
#     op.update_conf(conf)
elif net_type == 'dpk':
    if conf.dpk_use_op_affinity_graph:
        apt_dpk.update_conf_dpk_from_affgraph_flm(conf)
    else:
        assert conf.dpk_skel_csv is not None
        apt_dpk.update_conf_dpk_skel_csv(conf, conf.dpk_skel_csv)

# elif net_type == 'deeplabcut':
#     conf.batch_size = 1
elif net_type == 'unet':
    conf.use_pretrained_weights = False

conf.unet_rescale = conf.rescale
conf.leap_rescale = conf.rescale

assert not (conf.vert_flip and conf.horz_flip), 'Only one type of flipping, either horizontal or vertical is allowed for augmentation'

import PoseTools as pt
ks = list(conf1.__dict__.keys())
for k in ks:
    if k.startswith('DeepTrack_') or k.startswith('MultiAnimal_') or k[0].isupper():
        conf1.__dict__.pop(k)

pt.compare_conf(conf1,conf)



##



J = TrkFile.Trk('/groups/branson/home/kabram/temp/roian_190423_m2f0_vocpbm164564_m164564odor_m164301_f163284_ds2_grone_id_raw.trk');
pred_locs = J.getfull()[0] ;
pred_locs = np.transpose(pred_locs,[2,3,0,1]) ;
pred_conf = J.pTrkConf.getdense()[0];
pred_conf = np.transpose(pred_conf,[1,2,0]);
pred_animal_conf = None


## Training with neg APT
from importlib import reload
import Pose_detect_mmdetect2x as mmdetect_file
reload(mmdetect_file)
from Pose_detect_mmdetect2x import Pose_detect_mmdetect
from poseConfig import conf
conf.mmpose_use_epoch_runner = True
conf.mmdetect_net = 'test'
conf.cachedir = '/groups/branson/bransonlab/mayank/APT/deepnet/mmdetection/test_apt_neg'
conf.dl_steps = 24*550
self = Pose_detect_mmdetect(conf, 'deepnet')
self.cfg.seed = 3
self.cfg.model.train_cfg.rpn.assigner.type = 'APTMaxIoUAssigner'
self.cfg.model.train_cfg.rpn.assigner.ignore_iof_thr = 0.2
self.cfg.data.train.ann_file = '/nrs/branson/mayank/apt_cache_2/four_points_180806/detection_cache/trn_neg.json'
self.cfg.train_pipeline[-1]['keys'].append('gt_bboxes_ignore')
# self.cfg.model.train_cfg.rpn_proposal.nms.iou_threshold=0.85

# self.cfg.model.train_cfg.rpn.assigner.neg_iou_thr=(0.15,0.25)

# self.cfg.load_from = '/groups/branson/bransonlab/mayank/APT/deepnet/mmdetection/test_apt1/epoch_4.pth'
# self.cfg.model.rpn_head.anchor_generator.strides = [19,25,32,43,56]
self.train_wrapper()

##
import numpy as np
in_im = all_f[0,...,0]
cols = all_f.shape[1]
rows = all_f.shape[2]
out_ims = []
for rangle in range(-12,13):
    ang = np.deg2rad(rangle)
    rot = [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]
    mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), rangle, 1)
    ii = in_im.copy()
    ii = cv2.warpAffine(ii, mat, (int(cols), int(rows)), flags=cv2.INTER_CUBIC)  # ,borderMode=cv2.BORDER_REPLICATE)
    if ii.ndim == 2:
        ii = ii[..., np.newaxis]
    out_ims.append(ii)

out_ims = np.array(out_ims)

out_l  = []
for rxx in range(out_ims.shape[0]//8):
    zz = pred_fn(out_ims[rxx*8:(rxx+1)*8,...])
    out_l.append(zz['locs'])

out_l = np.concatenate(out_l,axis=0)
dd = np.diff(out_l,axis=0)
dd = np.linalg.norm(dd,axis=-1)

##
import APT_interface as apt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cmd = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl -conf_params db_format \"coco\" mmpose_net \"mspn\" dl_steps 100000 rrange 30 trange 20 imsz \(192,192\) trx_align_theta True img_dim 1 ht_pts \(0,6\) use_ht_trx True -json_trn_file /nrs/branson/mayank/apt_cache_2/alice_ma/loc_split_neg.json -type mmpose -name alice_neg -cache /nrs/branson/mayank/apt_cache_2  train -use_cache -skip_db'
cmd = cmd.replace('\\','')
#cmd = cmd.replace('"','')
apt.main(cmd.split())

##
aa = [np.array(yy['keypoints']).reshape([-1,3]) for yy in Y['annotations']]
negs = [np.all(np.isnan(a[:,:2])) for a in aa]
nx = [i for i,x in enumerate(negs) if x]

##
f,ax = plt.subplots(5,5)
ax = ax.flatten()
for ndx,sel in enumerate(nx):
    iid = Y['annotations'][sel]['image_id']
    im = cv2.imread(Y['images'][iid]['file_name'])
    ax[ndx].imshow(im)
    bb = np.array(Y['annotations'][sel]['segmentation']).reshape(4,2)
    # mask = apt.create_mask([bb],[320,320])
    ax[ndx].plot(bb[:,0],bb[:,1])

## single animal ht
# import APT_interface as apt
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# cmd = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl -conf_params db_format \"tfrecord\" mmpose_net \"higherhrnet\" dl_steps 100000 op_affinity_graph \(\(0,1\),\(0,5\),\(1,2\),\(3,4\),\(3,5\),\(5,6\),\(5,7\),\(5,9\),\(3,16\),\(9,10\),\(10,15\),\(9,14\),\(4,11\),\(7,8\),\(8,12\),\(7,13\)\) multi_use_mask False multi_loss_mask True  multi_crop_ims True rrange 30 trange 30 is_multi False max_n_animals 7 imsz \(192,192\) use_ht_trx True ht_pts \(0,6\) -json_trn_file /nrs/branson/mayank/apt_cache_2/alice_ma/loc_split_neg.json -type openpose -name alice_ht_test -cache /nrs/branson/mayank/apt_cache_2 -no_except train -use_cache'
# cmd = cmd.replace('\\','')
# apt.main(cmd.split())

##
import APT_interface as apt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cmd = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl -conf_params db_format \"coco\" mmpose_net \"higherhrnet\" dl_steps 100000 op_affinity_graph \(\(0,1\),\(0,5\),\(1,2\),\(3,4\),\(3,5\),\(5,6\),\(5,7\),\(5,9\),\(3,16\),\(9,10\),\(10,15\),\(9,14\),\(4,11\),\(7,8\),\(8,12\),\(7,13\)\) multi_use_mask False multi_loss_mask True multi_crop_ims True rrange 180 trange 30 is_multi True max_n_animals 7 ht_pts \(0,6\) multi_only_ht True -json_trn_file /nrs/branson/mayank/apt_cache_2/alice_ma/loc_split_neg.json -type multi_mdn_joint_torch -name alice_neg_ht_test -cache /nrs/branson/mayank/apt_cache_2 -no_except train -use_cache'
cmd = cmd.replace('\\','')
apt.main(cmd.split())

##
# %run reuse

import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from importlib import reload
import APT_interface as apt
reload(apt)
import torch
lbl_file = '/nrs/branson/mayank/apt_cache_2/four_points_180806/20201225T042233_20201225T042235.lbl'

n_pairs = [['multi_openpose','openpose']]
curp = n_pairs[0]
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
conf.background_mask_sel_rate = 1.1
info =[3,3,3]
T = pt.json_load('/nrs/branson/mayank/apt_cache_2/four_points_180806/loc_split.json')
im = cv2.imread('/nrs/branson/mayank/apt_cache_2/four_points_180806/'+T['locdata'][0]['img'][0],cv2.IMREAD_UNCHANGED)
cur_pts = np.array(T['locdata'][0]['pabs']).reshape([2,4,2]).transpose([2,1,0])
occ = np.array(T['locdata'][0]['occ']).reshape([4,-1]).transpose([1,0])


##
mcase = 'far2'
debug = True
plt.close('all')
if mcase == 'overlap':
    extra_roi = np.array([[
        [ 820.36679812, 286.26984163],
        [ 820.36679812, 2587.13256204],
        [2173.98770493, 2587.13256204],
        [2173.98770493, 286.26984163]]])
elif mcase =='full':
    extra_roi = np.array( [[[0, 0], [0, 2048], [2048, 2048], [2048, 0]]])
elif mcase =='fit':
    extra_roi = np.array( [[[1290, 190],
                            [1290, 500],
                            [1400, 500],
                            [1400, 190]]])
elif mcase == 'far':
    extra_roi = np.array( [[[1190, 1090],
                            [1190, 1200],
                            [500, 1200],
                            [500, 1090]]])

elif mcase == 'far2':
    extra_roi = np.array( [[[1190, 1090],
                            [1190, 1200],
                            [500, 1200],
                            [500, 1090]],
                           [[1190, 1290],
                            [1190, 1400],
                            [500, 1400],
                            [500, 1290]]
                           ])

roi = np.array(T['locdata'][0]['roi']).reshape([2,4,-1]).transpose([2,1,0])
conf.multi_frame_sz = [2048,2048]
conf.imsz = [640,640]

if debug:
    roim = apt.create_mask(roi,conf.multi_frame_sz)
    eroim = apt.create_mask(extra_roi,conf.multi_frame_sz)
    f = plt.figure();
    plt.imshow(im*(roim|eroim),'gray')

all_data = apt.create_ma_crops(conf,np.tile(im[...,None],[1,1,3]),cur_pts,info,occ,roi,extra_roi)

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.ion()

if debug:
    f1,ax = plt.subplots(int(np.ceil(len(all_data)/2)),2)
    ax = ax.flatten()
    for ndx, a in enumerate(all_data):
        mm = apt.create_mask(a['roi'],conf.imsz)
        if a['extra_roi'] is not None:
            mm = mm| apt.create_mask(a['extra_roi'],conf.imsz)
        ax[ndx].imshow(a['im']*mm[:,:,None])
        ax[ndx].axis('off')

    plt.figure(f.number)
    plt.axis('off')
    for a in all_data:
        xx = [a['x_left'],a['x_left'],a['x_left']+conf.imsz[1],a['x_left']+conf.imsz[1],a['x_left']]
        yy = [a['y_top'],a['y_top']+conf.imsz[0],a['y_top']+conf.imsz[0],a['y_top'],a['y_top']]
        plt.plot(xx,yy)

##
ix = 1127
im_file = os.path.join(os.path.dirname(db_file),'val',f'{ix:08}.png')
im = cv2.imread(im_file,cv2.IMREAD_UNCHANGED)
im = np.tile(im[None,...,None],[1,1,1,3])
conf.batch_size = 1
rr = 8
pl = []
plt.cla()
imshow(im[0,...,0],'gray')
for xx in range(-rr,rr+1,4):
    xl = []
    for yy in range(-rr,rr+1,4):
        oim = np.pad(im,[[0,0],[rr,rr],[rr,rr],[0,0]])
        oim = oim[:,(rr+yy):(rr+yy+1024),(rr+xx):(rr+xx+1024),:]
        dfile = os.path.join('/tmp/',f'diagnose_val_{ix}_wt_offset_5.p')
        import Pose_multi_mdn_joint_torch
        pp = Pose_multi_mdn_joint_torch.Pose_multi_mdn_joint_torch(conf,name='wt_offset_5')
        pp.diagnose(oim,dfile)
        A = pt.pickle_load(dfile)
        A = A['ret_dict']
        # curl = A['locs'][0]
        curl = A['raw_locs'][0]['joint'][0]
        curl[...,0] += xx
        curl[...,1] += yy
        xl.append(curl.copy())
        mdskl(curl,conf.op_affinity_graph)
    pl.append(xl)
mdskl(np.clip(ll1[ix,...],0,10000),conf.op_affinity_graph,cc=[0,0,1.])
aa = np.array(pl)
plt.title(f'{ix}')
##
layers  = list(model.named_modules())
layers = [l for l in layers if 'conv' in l[0]]
train_dict = {}
for l in layers:
    train_dict[l[0]] = None

def save_outputs_hook(layer_id: str):
    def fn(_, __, output):
        oo = output.detach().cpu().numpy().copy()
        if train_dict[layer_id] is None:
            train_dict[layer_id] = [[oo.sum(axis=(0,2,3))],[(oo**2).sum(axis=(0,2,3))]]
        else:
            train_dict[layer_id][0].append(oo.sum(axis=(0,2,3)))
            train_dict[layer_id][1].append((oo**2).sum(axis=(0, 2, 3)))
    return fn

for l in layers:
    if 'conv' not in l[0]:
        continue
    l[1].register_forward_hook(save_outputs_hook(l[0]))

import cv2
for ix in range(0,500,10):
    im_file = os.path.join(conf.cachedir, 'train', f'{ix:08}.png')
    im = cv2.imread(im_file, cv2.IMREAD_UNCHANGED)
    im = np.tile(im[None, ..., None], [1, 1, 1, 3])
    sims, _ = PoseTools.preprocess_ims(im, locs_dummy[:1], conf, False, conf.rescale)
    with torch.no_grad():
        preds = model({'images': torch.tensor(sims).permute([0, 3, 1, 2]) / 255.})

for ix in range(0,500,10):
    im_file = os.path.join(conf.cachedir, 'val', f'{ix:08}.png')
    im = cv2.imread(im_file, cv2.IMREAD_UNCHANGED)
    im = np.tile(im[None, ..., None], [1, 1, 1, 3])
    sims, _ = PoseTools.preprocess_ims(im, locs_dummy[:1], conf, False, conf.rescale)
    with torch.no_grad():
        preds = model({'images': torch.tensor(sims).permute([0, 3, 1, 2]) / 255.})

##
import pickle
k = 'module.locs_ref.conv1'
rr = [np.array(t) for t in train_dict[k]]
k1 = 'module.locs_joint.conv1'
rr1 = [np.array(t) for t in train_dict[k1]]

with open('/groups/branson/home/kabram/temp/grone_out_offset5.p','wb') as f:
    pickle.dump({k:rr,k1:rr1},f)

##

k = 'module.locs_ref.conv1'
# rr = [np.array(t) for t in train_dict[k]]
rr = A[k]
f,ax = plt.subplots(1,2)
ax = ax.flatten()
rm = rr[0][:50].sum(axis=0).std()
ax[0].scatter(rr[0][:50].sum(axis=0)/rm,rr[0][50:].sum(axis=0)/rm,marker='.')
rm = rr[1][:50].sum(axis=0).std()
ax[1].scatter(rr[1][:50].sum(axis=0)/rm,rr[1][50:].sum(axis=0)/rm,marker='.')
# rr = A[k]
rm = rr[0][:50].sum(axis=0).std()
ax[0].scatter(rr[0][:50].sum(axis=0)/rm,rr[0][50:].sum(axis=0)/rm,marker='.')
rm = rr[1][:50].sum(axis=0).std()
ax[1].scatter(rr[1][:50].sum(axis=0)/rm,rr[1][50:].sum(axis=0)/rm,marker='.')
ax[0].axis('equal')
ax[1].axis('equal')
k = 'module.locs_joint.conv1'
# rr = [np.array(t) for t in train_dict[k]]
rr = A[k]
f,ax = plt.subplots(1,2)
ax = ax.flatten()
rm = rr[0][:50].sum(axis=0).std()
ax[0].scatter(rr[0][:50].sum(axis=0)/rm,rr[0][50:].sum(axis=0)/rm,marker='.')
rm = rr[1][:50].sum(axis=0).std()
ax[1].scatter(rr[1][:50].sum(axis=0)/rm,rr[1][50:].sum(axis=0)/rm,marker='.')
# rr = A[k]
rm = rr[0][:50].sum(axis=0).std()
ax[0].scatter(rr[0][:50].sum(axis=0)/rm,rr[0][50:].sum(axis=0)/rm,marker='.')
rm = rr[1][:50].sum(axis=0).std()
ax[1].scatter(rr[1][:50].sum(axis=0)/rm,rr[1][50:].sum(axis=0)/rm,marker='.')
ax[0].axis('equal')
ax[1].axis('equal')

## diagnose grone

import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from importlib import reload
import APT_interface as apt
reload(apt)
import torch
lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'

net_type = 'multi_mdn_joint_torch' #'multi_mmpose' #
# train_name = 'grone_maskim' # 'higher'# 'deepnet' #
run_name = 'val_split'
train_name = 'deepnet'

run_name = 'alice_maskim_split_crop_ims_grone_multi'
# train_name = 'grone_maskloss' # 'higher'# 'deepnet' #
# run_name = 'val_split'

# net_type = 'multi_mmpose' #'multi_mmpose' #
# train_name = 'higherhr_maskloss' # 'higher'# 'deepnet' #
# run_name = 'val_split'

# train_name = 'higherhr_maskim' # 'higher'# 'deepnet' #
# run_name = 'maskim_split'

# net_type = 'multi_openpose' #'multi_mmpose' #
# train_name = 'openpose_maskloss' # 'higher'# 'deepnet' #
# run_name = 'val_split'

# db_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/val_split/val_TF.json'
# use whole unmasked images for validation
conf = apt.create_conf(lbl_file,0,run_name,net_type=net_type,cache_dir='/nrs/branson/mayank/apt_cache_2')
# conf.batch_size = 4 if net_type == 'multi_openpose' else 8
db_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/val_split_fullims/val_TF.json'
conf.db_format = 'coco'
conf.max_n_animals = 10
conf.imsz = (1024,1024) #(288,288)
conf.img_dim = 3
conf.mmpose_net = 'higherhrnet' #'higherhrnet_2x'#
conf.is_multi = True
conf.op_affinity_graph = ((0,1),(0,5),(1,2),(3,4),(3,5),(5,6),(5,7),(5,9),(3,16),(9,10),(10,15),(9,14),(4,11),(7,8),(8,12),(7,13))

import Pose_multi_mdn_joint_torch
import cv2
ix = 862
im_file = os.path.join(os.path.dirname(db_file),'val',f'{ix:08}.png')
im = cv2.imread(im_file,cv2.IMREAD_UNCHANGED)
im = np.tile(im[None,...,None],[1,1,1,3])
import PoseTools as pt
A = pt.pickle_load('/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/alice_maskim_split_crop_ims_grone_multi/diagnose_20210201')
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.ion()
plt.imshow(im[0,:,:,0],'gray')
A = A['ret_dict']
kk = A['preds'][0]
kk1 = A['preds'][1]
jj1 =  A['raw_locs'][1]['ref'][0] + 16
jj =  A['raw_locs'][0]['ref'][0]
ff =  jj[-1,...]
ff1 = jj1[4,...]
ff-ff1
mm = kk[0][0,...,0,23,23]
mm.round()
mm1 = kk1[0][0,...,0,23,23]
mm1.round()
## masking loss
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import APT_interface as apt
cmd = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl -json_trn_file /nrs/branson/mayank/apt_cache_2/alice_ma/loc_split.json -conf_params dl_steps 110000 is_multi True multi_use_mask False multi_loss_mask True mmpose_net \"higherhrnet\" db_format \"coco\" max_n_animals 7  -train_name higherhr_maskloss -type multi_mmpose -name val_split -cache /nrs/branson/mayank/apt_cache_2 train -continue -skip_db'
cmd = cmd.replace('\\','')
apt.main(cmd.split())

##
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

cmd = '/nrs/branson/mayank/apt_cache_2/four_points_180806/20201225T042233_20201225T042235.lbl -json_trn_file /nrs/branson/mayank/apt_cache_2/four_points_180806/loc.json -conf_params dl_steps 200000 pretrain_freeze_bnorm False is_multi True mmpose_net "higherhrnet" multi_use_mask False db_format "coco" batch_size 2 max_n_animals 2 save_step 10000 -train_name grone_nomask_bn -type multi_mdn_joint_torch -name full_dataset -cache /nrs/branson/mayank/apt_cache_2 train -skip_db'


import APT_interface as apt
apt.main(cmd.split())


##
nims = len(A['images'])
aas = np.random.rand(nims,11,2,3)*-1000
for im in range(nims):
    lndx = 0
    for a in A['annotations']:
        if not (a['image_id'] == im):
            continue
        locs = np.array(a['keypoints'])
        locs = np.reshape(locs, [2, 3])
        aas[im,lndx, ...] = locs
        lndx += 1

dd = np.ones([nims,11,11])*10000
for ix in range(2):
    for iy in range(2):
        cdd = np.linalg.norm(aas[:,:,np.newaxis,ix,:2]-aas[:,np.newaxis,:,iy,:2],axis=-1)
        dd = np.minimum(cdd,dd)
dd = dd.reshape([nims,-1])
dd[:,::12] = 1000
dd = dd.reshape([nims,11,11])
##
kk = np.where(dd<20)
sndx = np.random.randint(len(kk[0]))
plt.figure(210)
plt.cla()
plt.plot(aas[kk[0][sndx],:,:,0].T,aas[kk[0][sndx],:,:,1].T)
plt.show()
##
import h5py
lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'
lbl = h5py.File(lbl_file,'r')
exp_list = lbl['movieFilesAll'][0, :]
local_dirs = [u''.join(chr(c) for c in lbl[jj]) for jj in exp_list]
try:
    for k in lbl['projMacros'].keys():
        r_dir = u''.join(chr(c) for c in lbl['projMacros'][k])
        local_dirs = [s.replace('${}'.format(k), r_dir) for s in local_dirs]
except:
    pass

vals = []
from scipy import io as sio
for ndx in range(len(local_dirs)):
    pfile = local_dirs[ndx].replace('movie.ufmf','perframe/dnose2ell.mat')
    Z = sio.loadmat(pfile)
    for im in range(ims):
        if A['images']['movid'] != ndx:
            continue
        fr = A['images']['frm']



##

rep = np.zeros(p.shape[-1])
flip = np.zeros(p.shape[-1])

for t in range(p.shape[-1]):
    cp = p[...,t]
    dd1 = np.linalg.norm(cp[:,:,np.newaxis,:]-cp[:,:,:,np.newaxis],axis=0).sum(0)
    dd1.flat[::dd1.shape[0]+1] = 1000
    cpi = cp[:,[1,0],...]
    dd2 = np.linalg.norm(cp[:,:,np.newaxis,:]-cpi[...,np.newaxis],axis=0).sum(0)
    while np.any(dd1<14):
        rep[t] = 1
        zz = np.where(dd1<14)[0][-1]
        p[:,:,zz,t] = np.nan
        cp = p[...,t]
        dd1 = np.linalg.norm(cp[:,:,np.newaxis,:]-cp[:,:,:,np.newaxis],axis=0).sum(0)
        dd1.flat[::dd1.shape[0]+1] = 1000
    if np.any(dd2<14):
        flip[t] = 1



## Debugging errors in Alice's MA tracking
pp = newtrk['pTrk']
t = 100
for t1 in range(t,t+800):
    id1 = np.nonzero(~np.isnan(pp[0,0,t1,:]))[0]
    id2 = np.nonzero(~np.isnan(pp[0,0,t1+1,:]))[0]
    if not np.array_equal(id1,id2):
        print(t1,id1.shape[0],id2.shape[0])
        print(id1)
        print(id2)
        break

#
pp = newtrk['pTrk']
t = t1
id1 = np.nonzero(~np.isnan(pp[0,0,t,:]))
id2 = np.nonzero(~np.isnan(pp[0,0,t+1,:]))
print(id1)
print(id2)
if not np.array_equal(id1,id2):
    mov = '/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00030_CsChr_RigC_20150826T144616/movie.ufmf'
    # mov = '/groups/branson/home/kabram/temp/roian_multi/200918_m170234vocpb_m170234_odor_m170232_f0180322.mjpg'
    import movies

    cap = movies.Movie(mov)
    fr = cap.get_frame(t)[0]
    fr1 = cap.get_frame(t+1)[0]
    f = plt.figure(234,frameon=False)
    plt.cla()
    plt.imshow(np.minimum(fr,fr1),'gray')
    plt.plot(pp[:,0,t,:],pp[:,1,t,:],c='r')
    # plt.scatter(pp[0, 0, t, :], pp[0, 1, t, :], c='r',marker='^')
    plt.plot(pp[:,0,t+1,:]+3,pp[:,1,t+1,:]+3,c='b')
    # plt.scatter(pp[0, 0, t+1, :]+3, pp[0, 1, t+1, :]+3, c='b',marker='^')
    plt.plot(p[0,:,:,t]+6,p[1,:,:,t]+6,c='g')
    plt.plot(p[0,:,:,t+1]-3,p[1,:,:,t+1]-3,c='k')

##
import APT_interface as apt
import Pose_multi_mdn_joint_torch
import torch
import numpy as np
import PoseTools
lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'
conf = apt.create_conf(lbl_file,0,'full_touch_20200811','/nrs/branson/mayank/apt_cache_2','multi_mdn_joint_torch')
conf.has_trx_file = False
conf.imsz = (1024, 1024)
conf.batch_size = 1
conf.max_n_animals = 12
conf.is_multi = True
conf.mmpose_net = "higherhrnet"
model_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/full_touch_20200811/grone-100000'
self = Pose_multi_mdn_joint_torch.Pose_multi_mdn_joint_torch(conf)
model = self.create_model()
model = torch.nn.DataParallel(model)

self.model = model
self.restore(model_file, model)
model.to('cuda')
model.eval()
conf = self.conf
match_dist = 4

##
t = 1140
fr = cap.get_frame(t)[0]
ims = np.tile(fr[np.newaxis,...,np.newaxis],[1,1,1,3])
locs_sz = (conf.batch_size, conf.n_classes, 2)
locs_dummy = np.zeros(locs_sz)

ims, _ = PoseTools.preprocess_ims(ims, locs_dummy, conf, False, conf.rescale)
with torch.no_grad():
    preds = model({'images': torch.tensor(ims).permute([0, 3, 1, 2]) / 255.})

# do prediction on half grid cell size offset images. o is for offset
hsz = 16
oims = np.pad(ims, [[0, 0], [0, hsz], [0, hsz], [0, 0]])[:, hsz:, hsz:, :]
with torch.no_grad():
    opreds = model({'images': torch.tensor(oims).permute([0, 3, 1, 2]) / 255.})
locs = self.get_joint_pred(preds)
olocs = self.get_joint_pred(opreds)

plt.figure(322)
plt.cla()
plt.imshow(fr,'gray')
plt.scatter(locs['ref'][...,0],locs['ref'][...,1])
plt.scatter(olocs['ref'][...,0]+16,olocs['ref'][...,1]+16)

plt.figure(333)
plt.cla()
plt.imshow(opreds[1][0,2,:,:].cpu().numpy().reshape([32,32]))

matched = {}
for dkeys in ['ref','joint']:
    olocs_orig = olocs[dkeys] + hsz
    locs_orig = locs[dkeys]
    cur_pred = np.ones_like(olocs_orig) * np.nan
    dd = olocs_orig[:,:,np.newaxis,...] - locs_orig[:,np.newaxis,...]
    dd = np.linalg.norm(dd,axis=-1).mean(-1)
    matched_ndx = 0
    # match predictions from offset pred and normal preds
    for b in range(dd.shape[0]):
        done_offset = np.zeros(dd.shape[1])
        done_locs = np.zeros(dd.shape[1])
        for ix in range(dd.shape[1]):
            if np.all(np.isnan(dd[b,:,ix])):
                continue
            olocs_ndx = np.nanargmin(dd[b,:,ix])
            if dd[b,olocs_ndx,ix] < match_dist:
                cc = (olocs_orig[b,olocs_ndx,...] + locs_orig[b,ix,...])/2
                done_offset[olocs_ndx] = 1
                done_locs[ix] = 1
                print(f'Matched {ix} with {olocs_ndx}')
            else:
                cc = locs_orig[b,ix,...]
                done_locs[ix] = 1
            cur_pred[b,matched_ndx,...] = cc
            matched_ndx += 1
        for ix in np.where(done_offset<0.5)[0]:
            if np.all(np.isnan(dd[b,ix,:])):
                continue
            if matched_ndx >= conf.max_n_animals:
                break
            cc = olocs_orig[b,ix,...]
            cur_pred[b,matched_ndx,...] = cc
            matched_ndx += 1
    matched[dkeys] = cur_pred
matched['ref'][0,:,0,0]
## Roian -no mask
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
import objgraph
cmd = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl -name full_touch_20200811 -type multi_mmpose -train_name higherhrnet -conf_params has_trx_file False imsz (1024,1024) batch_size 1 max_n_animals 12 is_multi True mmpose_net "higherhrnet" -cache /nrs/branson/mayank/apt_cache_2 track -mov /groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00030_CsChr_RigC_20150826T144616/movie.ufmf -out /groups/branson/home/kabram/temp/alice_multi/cx_GMR_SS00030_CsChr_RigC_20150826T144616_higherhrnet.trk'
import APT_interface as apt
apt.main(cmd.split())

##

from mmcv import Config, DictAction
import APT_interface as apt
import os
import numpy as np
import poseConfig
import matplotlib
matplotlib.use('TkAgg')
from Pose_mmpose import Pose_mmpose

lbl_file = '/groups/branson/home/kabram/APT_projects/alice_touch_stripped.lbl'
conf = apt.create_conf(lbl_file,0,'deepnet','/nrs/branson/mayank/apt_cache_2','mmpose')
conf.batch_size = 3
conf.imsz = [768,768]#(sz+2*buffer,sz+2*buffer)
conf.rescale = 1.
conf.save_step = 10000
conf.dl_steps = 100000
conf.brange = [0,0]
conf.crange =[1,1]
conf.horz_flip = True
conf.flipLandmarkMatches = {'11': 16, '16': 11, '1': 2, '2': 1, '3': 4, '4': 3, '7': 9, '9': 7, '8': 10, '10': 8, '12': 15, '15': 12, '13': 14, '14': 13}
conf.mmpose_use_apt_augmentation = False
ss = Pose_mmpose(conf,'mmpose_aug')
# ss.cfg.model.pretrained='/nrs/branson/mayank/apt_cache_2/multitarget_bubble/mmpose/view_0/deepnet/mmpose_aug-100000'
ss.train_wrapper(False)


## debugging topk for mmpose

import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from importlib import reload
import APT_interface as apt
reload(apt)
import torch
lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'
db_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/apt/val_TF.json'

net_type = 'multi_mmpose' #'multi_mdn_joint_torch' #
train_name = 'higher_2x' # 'higher'# 'deepnet' #
conf = apt.create_conf(lbl_file,0,'apt',net_type=net_type,cache_dir='/nrs/branson/mayank/apt_cache_2')
conf.db_format = 'coco'
conf.max_n_animals = 8
conf.imsz = (288,288)
conf.img_dim = 3
conf.mmpose_net = 'higherhrnet_2x'#'higherhrnet'
conf.is_multi = True
conf.min_n_animals = 2
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


pp1 = out[0]
ll1 = out[1]
dd1 = np.linalg.norm(pp1[:,:,np.newaxis,...]-ll1[:,np.newaxis,...],axis=-1)
dd1 = find_dist_match(dd1)
valid = ll1[:,:,0,0]>-1000
dd1_val = dd1[valid,:]
np.nanpercentile(dd1_val,[50,75,90,95,97],axis=0)


##
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
conf.mmpose_net = 'higherhrnet_2x'
conf.json_trn_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/loc.json'
conf.max_n_animals = 7
conf.mmpose_use_apt_augmentation = False
conf.set_exp_name('alice')

apt.setup_ma(conf)
# apt.create_coco_db(conf,True)
self = Pose_multi_mmpose(conf,'higherhr_2x')
self.train_wrapper()


##
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from importlib import reload
import APT_interface as apt
reload(apt)
lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'
split_trn = '/nrs/branson/mayank/apt_cache_2/alice_ma/loc_split.json'
cmd = f'{lbl_file} -json_trn_file {split_trn} -conf_params dl_steps 100000 is_multi True db_format \"coco\" max_n_animals 7 -type multi_mdn_joint_torch -cache /nrs/branson/mayank/apt_cache_2 train -skip_db'
apt.main(cmd.split())


##
cmd = '-name 20200925T080001 -view 1 -cache /groups/branson/home/kabram/.apt/tp3fdd7f66_1a7e_4213_b390_47a7e8798800 -type mdn /groups/branson/home/kabram/.apt/tp3fdd7f66_1a7e_4213_b390_47a7e8798800/alice_test/20200925T080001_20200925T080130.lbl train -use_cache -skip_db'
## Roian Tracking
# cmd = '-name 20201110T005848 -view 1 -cache /groups/branson/home/kabram/.apt/tp8480fd4e_f20c_4592_8eb6_29a1a0ff4564 -debug -type mdn /groups/branson/home/kabram/.apt/tp8480fd4e_f20c_4592_8eb6_29a1a0ff4564/test1/20201110T005848_20201110T010044.lbl train -use_cache -skip_db'
cmd = '/groups/branson/bransonlab/apt/experiments/data/four_points_all_mouse_linux_tracker_updated20200423_new_skl_20200817.lbl_mdn.lbl -name full -type multi_mdn_joint_torch -no_except -conf_params has_trx_file False imsz (2048,2048) batch_size 1 max_n_animals 2 is_multi True -cache /nrs/branson/mayank/apt_cache_2 track -mov /groups/branson/home/kabram/temp/roian_multi/200918_m170234vocpb_m170234_odor_m170232_f0180322.mjpg -out /groups/branson/home/kabram/temp/roian_multi/200918_m170234vocpb_m170234_odor_m170232_f0180322.trk1 -start_frame 481 -end_frame 482'
import APT_interface as apt
apt.main(cmd.split())

##
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
        name = 'fixed_lr_mmpose_aug'
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


## mmpose single animal
lbl_file = '/groups/branson/home/kabram/APT_projects/alice_touch_stripped.lbl'
import APT_interface as apt
net = 'mmpose'
tname = 'mmpose_aug'
mfile = f'/nrs/branson/mayank/apt_cache_2/multitarget_bubble/{net}/view_0/deepnet/{tname}-100000'
conf = apt.create_conf(lbl_file,0,'deepnet','/nrs/branson/mayank/apt_cache_2',net)
conf.rescale = conf.imsz[0]/768
conf.flipLandmarkMatches = {'11': 16, '16': 11, '1': 2, '2': 1, '3': 4, '4': 3, '7': 9, '9': 7, '8': 10, '10': 8, '12': 15, '15': 12, '13': 14, '14': 13}
aa = apt.classify_db_all('mmpose',conf,'/nrs/branson/mayank/apt_cache_2/multitarget_bubble/mdn_joint_fpn/view_0/multi_compare/val_TF.tfrecords',mfile)
import numpy as np
dd = np.linalg.norm(aa[0]-aa[1],axis=-1)
ss = np.percentile(dd,[50,76,90,95,97],axis=0)




##
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import APT_interface as apt
import poseConfig
from Pose_multi_mmpose import Pose_multi_mmpose
from Pose_multi_mdn_joint_torch import Pose_multi_mdn_joint_torch

lbl_file = '/nrs/branson/mayank/apt_cache_2/alice_ma/alice_ma.lbl_multianimal.lbl'
conf = apt.create_conf(lbl_file,0,'deepnet','/nrs/branson/mayank/apt_cache_2','multi_mdn_fpn_torch')

conf.cachedir = '/nrs/branson/mayank/apt_cache_2/alice_ma/'
conf.db_format = 'coco'
conf.dl_steps = 100000
conf.nviews = 1
conf.view = 0
conf.n_classes = 17
conf.is_multi = True
conf.max_n_animals = 6
conf.set_exp_name('alice')

apt.setup_ma(conf)
# apt.create_coco_db(conf,True)
# self = Pose_multi_mmpose(conf,'test')
self = Pose_multi_mdn_joint_torch(conf,name='test')
self.train_wrapper()


##
import APT_interface as apt
net = 'mmpose'
tname = 'mmpose_aug'
mfile = f'/nrs/branson/mayank/apt_cache_2/multitarget_bubble/{net}/view_0/deepnet/{tname}-100000'
lbl_file = '/groups/branson/home/kabram/APT_projects/alice_touch_stripped.lbl'
conf = apt.create_conf(lbl_file,0,'deepnet','/nrs/branson/mayank/apt_cache_2',net)
conf.rescale = 183/768
conf.flipLandmarkMatches = {'11': 16, '16': 11, '1': 2, '2': 1, '3': 4, '4': 3, '7': 9, '9': 7, '8': 10, '10': 8, '12': 15, '15': 12, '13': 14, '14': 13}
aa = apt.classify_db_all('mmpose',conf,'/nrs/branson/mayank/apt_cache_2/multitarget_bubble/mdn_joint_fpn/view_0/multi_compare/val_TF.tfrecords',mfile)
dd = np.linalg.norm(aa[0]-aa[1],axis=-1)
ss = np.percentile(dd,[50,76,90,95,97],axis=0)

##
import APT_interface as apt
import poseConfig
from Pose_multi_mmpose import Pose_multi_mmpose
conf = poseConfig.conf

conf.labelfile = '/groups/branson/bransonlab/apt/ma/trnpack_20201123/bub_wking_2movs_20201112.lbl_multianimal.lbl'
conf.cachedir = '/groups/branson/home/kabram/temp/mapack'
conf.db_format = 'coco'
conf.dl_steps = 500
conf.nviews = 1
conf.view = 0
conf.n_classes = 17
conf.is_multi = True
conf.set_exp_name('alice')

apt.setup_ma(conf)
apt.create_coco_db(conf,True)
self = Pose_multi_mmpose(conf,'test')
self.train_wrapper()

##

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
    if net_type == 'multi_mdn_joint_torch':
        name = '?'
        batch_size = 6
    elif net_type == 'multi_openpose':
        name= '50k_resnet'
        batch_size = 4
    elif net_type == 'multi_mmpose':
        name = 'test'
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

if net_type == 'multi_mmpose':
    db_file = os.path.join(conf.cachedir.replace('multi_mmpose','multi_mdn_joint_torch'), 'val_TF.tfrecords')
else:
    db_file = os.path.join(conf.cachedir,'val_TF.tfrecords')
out = apt.classify_db_all(net_type,conf,db_file,classify_fcn=apt.classify_db_multi,name=name)


## joining trajectories
import numpy as np
import link_trajectories as lnk
import movies
cap = movies.Movie('/groups/branson/home/kabram/temp/roian_multi/200918_m170234vocpb_m170234_odor_m170232_f0180322.mjpg')
import h5py
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.ion()
A = h5py.File('/groups/branson/home/kabram/temp/roian_multi/200918_m170234vocpb_m170234_odor_m170232_f0180322.trk','r')
ll = A['pTrk'].value

plt.imshow(cap.get_frame(0)[0])
plt.scatter(ll[0,:,0,0],ll[0,:,1,0])
##

import PoseTools
import re
import h5py
import numpy as np
import APT_interface as apt
import torch
import matplotlib
matplotlib.use('TkAgg')
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import Pose_coco_mdn_joint
name = 'rescale_3'
self = Pose_coco_mdn_joint.Pose_coco_mdn_joint('/nrs/branson/mayank/apt_cache_2/coco',name=name,rescale=3)
self.conf.learning_rate_multiplier = 0.1
import PoseTools
# with PoseTools.GuruMeditation():
self.train_wrapper(restore=True)

exp_name = 'roian'
net_type = 'multi_mdn_joint_torch'
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
        name = '?'
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
        batch_size = 8
    elif net_type == 'multi_openpose':
        name= 'try_1'
        batch_size = 3

conf = apt.create_conf(lbl_file,0,'deepnet',net_type=net_type,cache_dir='/nrs/branson/mayank/apt_cache_2')
conf.rrange = 180
conf.trange = 50
conf.max_n_animals = max_n
conf.imsz = (sz+2*buffer,sz+2*buffer)
conf.mdn_use_unet_loss = False
conf.img_dim = 3
conf.op_affinity_graph = af_graph
conf.mdn_joint_use_fpn = True
conf.batch_size = 1

db_file = os.path.join(conf.cachedir,'val_TF.tfrecords')
out = apt.classify_db_all(net_type,conf,db_file,classify_fcn=apt.classify_db_multi,name=name)
# [  90,  393,  601,  757,  825,  826,  860, 1012, 1047, 1049, 1092,
#        1104, 1105, 1106, 1107, 1326, 1344, 1427, 1661]
##
from matplotlib import pyplot as plt
def pp(x):
    plt.figure()
    plt.imshow(x.detach().cpu().numpy())


##
import run_apt_expts_2 as rae
from importlib import reload
reload(rae)
rae.setup('romain')
rae.all_models = [m for m in rae.all_models if 'orig' not in m]
dstr = '20200912'
rae.get_cv_results(dstr=dstr,queue='gpu_tesla',db_from_mdn_dir=True)



##
cmd = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20200317_stripped20200403_new_skl_20200817.lbl -name apt_expt -cache /groups/branson/bransonlab/mayank/apt_cache_2 -conf_params mdn_use_unet_loss False  dl_steps 100000  decay_steps 50000  save_step 5000  batch_size 8  maxckpt 200  ignore_occluded False  pretrain_freeze_bnorm True  step_lr False  lr_drop_step 0.15  normalize_loss_batch False  use_scale_factor_range True  predict_occluded False  use_leap_preprocessing False  leap_val_size 0.15  leap_preshuffle True  leap_filters 64  leap_val_batches_per_epoch 10  leap_reduce_lr_factor 0.1  leap_reduce_lr_patience 3  leap_reduce_lr_min_delta 1e-05  leap_reduce_lr_cooldown 0  leap_reduce_lr_min_lr 1e-10  leap_amsgrad False  leap_upsampling False  dlc_intermediate_supervision False  dlc_intermediate_supervision_layer 12  dlc_location_refinement True  dlc_locref_huber_loss True  dlc_locref_loss_weight 0.05  dlc_locref_stdev 7.2801  dlc_use_apt_preprocess True learning_rate_multiplier 3. save_time 20  -type unet  -view 1  -train_name lr_mult_3 train -skip_db -use_cache'
import APT_interface as apt
apt.main(cmd.split())


## debug opnpose multi
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

net_type = 'multi_mdn_joint_torch'; name = 'test_time'
# net_type = 'multi_openpose'; name= '50k_resnet'
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
    conf.dl_steps = 100000
    self = Pose_multi_openpose.Pose_multi_openpose(conf,'50k_resnet')
elif net_type == 'multi_mdn_joint_torch':
    self = Pose_multi_mdn_joint_torch.Pose_multi_mdn_joint_torch(conf,name=name,is_multi=True)
else:
    self = Pose_multi_mdn_joint.Pose_multi_mdn_joint(conf,'50k_low_noise_fpn')

self.train_wrapper()

##
import PoseTools as pt
import Pose_mdn_joint_fpn
import multiResData
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
A = pt.pickle_load('/groups/branson/bransonlab/mayank/apt_cache_2/multitarget_bubble/mdn_joint_fpn/view_0/apt_expt/multitarget_bubble_deepnet_20200706_traindata')
conf = A[1]
conf.batch_size = 1
self = Pose_mdn_joint_fpn.Pose_mdn_joint_fpn(conf,name='deepnet_20200706')
pfn = self.get_pred_fn()

db = '/groups/branson/bransonlab/mayank/apt_cache_2/multitarget_bubble/mdn_joint_fpn/view_0/apt_expt/train_TF.tfrecords'

B = multiResData.read_and_decode_without_session(db,17,())
ii = B[0][33]
## for
isz = conf.imsz
rmat = cv2.getRotationMatrix2D((isz[1]/2,isz[0]/2),0,1)
allo = []
alli = []
for x in np.arange(-20,-10,0.5):
    rmat[0,2] = x
    curi = cv2.warpAffine(ii,rmat,(int(isz[1]),int(isz[0])))
    curi = np.tile(curi[np.newaxis,...,np.newaxis],[conf.batch_size,1,1,conf.img_dim])
    xs, _ = pt.preprocess_ims(curi, in_locs=np.zeros([1, self.conf.n_classes, 2]), conf=self.conf, distort=False, scale=self.conf.rescale)

    self.fd[self.inputs[0]] = xs
    self.fd[self.ph['phase_train']] = False
    self.fd[self.ph['learning_rate']] = 0
    out_list = [self.pred, self.inputs]
    out = self.sess.run(out_list, self.fd)
    allo.append(out[0])
    alli.append(xs)

##
import PoseTools as pt
import Pose_multi_mdn_joint_torch
import multiResData
import cv2
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
A = pt.pickle_load('/nrs/branson/mayank/apt_cache_2/multitarget_bubble/multi_mdn_joint_torch/view_0/deepnet/multitarget_bubble_test_fpn_more_conv_traindata')
conf = A[1]
conf.batch_size = 1
self = Pose_multi_mdn_joint_torch.Pose_multi_mdn_joint_torch(conf,name='test_fpn_more_conv')
pfn = self.get_pred_fn()

db = '/nrs/branson/mayank/apt_cache_2/multitarget_bubble/multi_mdn_joint_torch/view_0/deepnet/val_TF.tfrecords'

B = multiResData.read_and_decode_without_session_multi(db,17)
ii = B[0][33]
## for
isz = conf.imsz
rmat = cv2.getRotationMatrix2D((isz[1]/2,isz[0]/2),0,1)
allo = []
alli = []
alll = []
for x in np.arange(-5,5,0.5):
    rmat[0,2] = x
    curi = cv2.warpAffine(ii,rmat,(int(isz[1]),int(isz[0])))
    curi = np.tile(curi[np.newaxis,...],[conf.batch_size,1,1,1])
    xs, _ = pt.preprocess_ims(curi, in_locs=np.zeros([1, self.conf.n_classes, 2]), conf=self.conf, distort=False, scale=self.conf.rescale)

    out = self.model({'images':torch.tensor(xs).permute([0,3,1,2])/255.})
    alll.append(self.get_joint_pred(out))
    allo.append([oo.detach().cpu().numpy() for oo in out])
    alli.append(xs)


##
# import run_apt_expts_2 as rae
# import sys
# if sys.version_info.major > 2:
#     from importlib import reload
# reload(rae)
# # rae.all_models = ['openpose']
# rae.setup('stephen')
# dstr = '20200706' #'20200411'
# rae.get_normal_results(dstr=dstr)

##
import run_apt_expts_2 as rae
import APT_interface as apt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cmd = '/groups/branson/bransonlab/apt/experiments/data/leap_dataset_gt_stripped_new_skl_20200820.lbl -name apt_expt -cache /groups/branson/bransonlab/mayank/apt_cache_2 -conf_params mdn_use_unet_loss False  dl_steps 100000  decay_steps 25000  save_step 5000  batch_size 8  maxckpt 200  ignore_occluded False  pretrain_freeze_bnorm True  step_lr True  lr_drop_step 0.15  normalize_loss_batch False  predict_occluded False  use_leap_preprocessing False  leap_val_size 0.15  leap_preshuffle True  leap_filters 64  leap_val_batches_per_epoch 10  leap_reduce_lr_factor 0.1  leap_reduce_lr_patience 3  leap_reduce_lr_min_delta 1e-05  leap_reduce_lr_cooldown 0  leap_reduce_lr_min_lr 1e-10  leap_amsgrad False  leap_upsampling False  dlc_intermediate_supervision False  dlc_intermediate_supervision_layer 12  dlc_location_refinement True  dlc_locref_huber_loss True  dlc_locref_loss_weight 0.05  dlc_locref_stdev 7.2801  dlc_use_apt_preprocess True  use_real_leap False save_time 20  -type leap_orig  -view 1  -train_name deepnet_test train -skip_db -use_cache'
apt.main(cmd.split())


##
from tfrecord.torch.dataset import TFRecordDataset
from PoseCommon_pytorch import decode_augment
import multiResData

conf.batch_size = 1
titer = multiResData.tf_reader(conf, db_file, False, is_multi=True)
qq = []
for ndx in range(titer.N):
    qq.append(titer.next())

##
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
aa = np.where( np.any((dd1<1e4) & (dd1>10) ,axis=(1,2)))[0]
plt.figure()
sel = np.random.choice(aa)
plt.imshow(qq[sel][0][0,:,:,0],'gray')
ll = qq[sel][1]
ll[ll<-1000] = np.nan
plt.scatter(ll[0,:,:,0],ll[0,:,:,1],marker='+')
plt.scatter(pp1[sel,:,:,0],pp1[sel,:,:,1],marker='.')
plt.scatter(pp2[sel,:,:,0],pp2[sel,:,:,1],marker='*')

plt.show()


## OP single animal centered tracking
import PoseTools
import os
import glob
import APT_interface as apt
import apt_expts
import re
import run_apt_expts as rae
import multiResData
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

out = {}
db_file = '/nrs/branson/mayank/apt_cache/multitarget_bubble/mdn/view_0/alice_compare_touch/val_TF.tfrecords'
cdir = os.path.dirname(db_file)
cdir = cdir.replace('mdn','openpose')
ntype = 'openpose'
n = 'openpose_test'
tfile = os.path.join(cdir, 'multitarget_bubble_{}_traindata'.format(n))

A = PoseTools.pickle_load(tfile)
conf = A[1]

files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format(n))
files.sort(key=os.path.getmtime)
aa = [int(re.search('-(\d*)', f).groups(0)[0]) for f in files]
aa = [b - a for a, b in zip(aa[:-1], aa[1:])]
if any([a < 0 for a in aa]):
    bb = int(np.where(np.array(aa) < 0)[0]) + 1
    files = files[bb:]
files = [f.replace('.index', '') for f in files]
files = files[-1:]
conf.op_pred_simple = False
conf.op_inference_old = False
mdn_out = apt_expts.classify_db_all(conf, db_file, files, ntype, name=n)
# conf.op_inference_old = True
# mdn_out1 = apt_expts.classify_db_all(conf, db_file, files, ntype, name=n)
conf.op_pred_simple = True
mdn_out2 = apt_expts.classify_db_all(conf, db_file, files, ntype, name=n)


## OP single animal centered training
import APT_interface as apt

cmd = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_dlstripped.lbl -name alice_compare_touch -cache /nrs/branson/mayank/apt_cache -conf_params mdn_use_unet_loss False rrange 10 trange 5 img_dim 1 imsz (183,183) dl_steps 50000 step_lr True op_affinity_graph (0,1),(0,5),(1,2),(3,4),(3,5),(5,6),(5,7),(5,9),(3,16),(9,10),(10,15),(9,14),(4,11),(7,8),(8,12),(7,13)  -train_name openpose_test -type openpose train -skip_db -use_cache'

apt.main(cmd.split())


##
lbl_file = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20200317_stripped20200403.lbl'
cmd = '-no_except {} -name alice_compare_touch -cache /nrs/branson/mayank/apt_cache -conf_params rrange 10 trange 5 img_dim 1 dl_steps 10000  -type openpose train -skip_db -use_cache'.format(lbl_file)
import APT_interface as apt
apt.main(cmd.split())

##
import PoseTools
import os
import glob
import APT_interface as apt
import apt_expts
import re
import run_apt_expts as rae
import numpy  as np
from importlib import reload

import APT_interface as apt


sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'

lbl_file = '/groups/branson/bransonlab/apt/experiments/data/romain_dlstripped.trn606.lbl'
view = 0


reload(apt_expts)
import PoseUNet_resnet
reload(PoseUNet_resnet)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
mdn_names = ['mdn_joint_step_more_noise_less',
             # 'mdn_joint_step_more_noise_less_wasp',
             # 'mdn_joint_step_more_noise_less_wasp_dil_2',
             'mdn_joint_step_more_noise_less_wasp_dil_2_skip',
             'mdn_joint_step_more_noise_less_fpn',
             # 'mdn_joint_step_more_noise_less_wasp_skip_fpn',
             'mdn_joint_step_more_noise_less_wasp_skip_fpn_nonorm',
             'dlc_noapt']

out_dir = '/groups/branson/home/kabram/temp'

out = {}
db_file = '/nrs/branson/mayank/apt_cache/romainTrackNov18/mdn/view_0/romain_compare/val_TF.tfrecords'
proj_name = 'romainTrackNov18'
for n in mdn_names:
    if 'dlc' in n:
        ntype = 'deeplabcut'
    elif 'resunet' in n:
        ntype = 'resnet_unet'
    elif 'mdn_unet' in n:
        ntype = 'mdn_unet'
    elif 'mdn_joint' in n:
        ntype = 'mdn_joint'
    else:
        ntype = 'mdn'

    if ntype == 'mdn':
        cdir = os.path.dirname(db_file)
    else:
        cdir = '/nrs/branson/mayank/apt_cache/romainTrackNov18/{}/view_0/romain_compare/'.format(ntype)

    if ntype == 'deeplabcut':
        tfile = os.path.join(cdir, '{}_traindata'.format(n))
    elif n == 'deepnet':
        tfile = os.path.join(cdir, 'traindata')
    else:
        tfile = os.path.join(cdir, '{}_{}_traindata'.format(proj_name,n))

    if not os.path.exists(tfile):
        continue
    A = PoseTools.pickle_load(tfile)
    if ntype == 'deeplabcut':
        conf = apt.create_conf(lbl_file, view, 'romain_compare', cache_dir='/nrs/branson/mayank/apt_cache',
                               net_type='deeplabcut')

        conf.dlc_locref_stdev =        7.2801
        conf.dlc_locref_loss_weight =        0.05
        conf.dlc_location_refinement =        True
        conf.dlc_intermediate_supervision_layer =         12
        conf.maxckpt =        20
        conf.dlc_intermediate_supervision =        False
        conf.dlc_locref_huber_loss =         True
        conf.dlc_use_apt_preprocess =        True
        conf.use_scale_factor_range =        True
        conf.scale_factor_range =        1.3
        conf.batch_size = 1
    else:
        conf = A[1]

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

    mdn_out = apt_expts.classify_db_all(conf, db_file, files, ntype, name=n)
    out[n] = mdn_out


##


cmd = '-no_except /groups/branson/bransonlab/apt/experiments/data/romain_dlstripped.trn606.lbl -name romain_compare -cache /nrs/branson/mayank/apt_cache -conf_params rrange 10 trange 5 scale_factor_range 1.2 mdn_use_unet_loss False img_dim 1 dl_steps 100000 batch_size 4 save_step 5000 learning_rate_multiplier 1 step_lr True maxckpt 100 normalize_loss_batch False mdn_pred_dist True -train_name debug -type mdn train -skip_db -use_cache'
import APT_interface as apt
apt.main(cmd.split())

##
import run_apt_expts_2 as rae
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
if sys.version_info.major > 2:
    from importlib import reload
reload(rae)
rae.setup('alice')
rae.all_models = 'deeplabcut_orig'
dstr = '20200604' # '20200410'
rae.get_normal_results(dstr=dstr) # queue = 'gpu_tesla'
##
import test.test_apt
test.test_apt.main()


##
import APT_interface as apt
lbl_file = '/groups/branson/home/kabram/.apt/tpb0c90511_c3bc_47e1_8231_9424a04ae6ff/alice_test/20200515T060036_20200515T060230.lbl'
conf = apt.create_conf(lbl_file,0,'20200514T081006','/groups/branson/home/kabram/.apt/tpbe0cc40c_6a69_44df_b80f_f777e5008a1b','deeplabcut')
conf.batch_size = 1
A = apt.classify_db_all('deeplabcut',conf,'/groups/branson/home/kabram/.apt/tpbe0cc40c_6a69_44df_b80f_f777e5008a1b/alice_test/deeplabcut/view_0/20200514T081006/train_data.p','/groups/branson/home/kabram/.apt/tpbe0cc40c_6a69_44df_b80f_f777e5008a1b/alice_test/deeplabcut/view_0/20200514T081006/dlc-models/iteration-0/aptMayYay-trainset95shuffle1/train/snapshot-1000')
##
cmd = '-no_except  -name 20200514T081006 -view 1 -cache /groups/branson/home/kabram/.apt/tpbe0cc40c_6a69_44df_b80f_f777e5008a1b -conf_params dl_steps 1000 -type deeplabcut /groups/branson/home/kabram/.apt/tpb0c90511_c3bc_47e1_8231_9424a04ae6ff/alice_test/20200515T060036_20200515T060230.lbl train -use_cache -skip_db'
import APT_interface as apt
apt.main(cmd.split())


##
from deeplabcut.pose_estimation_tensorflow.train import  train as train_dlc
train_dlc('/groups/branson/bransonlab/mayank/apt_expts/deepcut_orig2/examples/openfield-Pranav-2018-10-30/dlc-models/iteration-0/openfieldOct30-trainset95shuffle1/train/pose_cfg_apt.yaml',displayiters=100,saveiters=5000,maxiters=10000)
##
from deeplabcut.pose_estimation_tensorflow import training
training.train_network('/groups/branson/bransonlab/mayank/apt_expts/deepcut_orig2/examples/openfield-Pranav-2018-10-30/config_apt.yaml',displayiters=100,saveiters=5000,maxiters=10000,shuffle=1)
## leap
cmd = '-name 20200512T050857 -view 1 -cache /groups/branson/home/kabram/.apt/tpbe0cc40c_6a69_44df_b80f_f777e5008a1b -type leap /groups/branson/home/kabram/.apt/tpbe0cc40c_6a69_44df_b80f_f777e5008a1b/alice_test/20200512T050857_20200512T051052.lbl train -use_cache -skip_db'
import APT_interface as apt
apt.main(cmd.split())

##
cmd = '-name 20200318T094825 -conf_params dl_steps 100 -cache /groups/branson/home/kabram/.apt/tp17f8408c_b91a_48a8_89d8_39c54aa5fa9f -type mdn /groups/branson/home/bransonk/.apt/tp7784a5ec_74be_4503_a288_4fadc2ab78e5/sh4992/20200325T160019_20200325T160107.lbl train -use_cache'

##
import run_apt_expts as rae
import APT_interface as apt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cmd = '/groups/branson/bransonlab/apt/experiments/data/wheel_rig_tracker_DEEP_cam0_20200318_compress20200327.lbl_mdn.lbl -name cv_split_0 -cache /groups/branson/bransonlab/mayank/apt_cache_2 -conf_params learning_rate_multiplier 1.0  batch_size 2  dl_steps 100  save_step 5000  maxckpt 200  mdn_use_unet_loss False  decay_steps 25000  -type mdn  -view 1  -train_name test_occ train -skip_db -use_cache'
apt.main(cmd.split())

##
from importlib import reload
reload(rae)
rae.setup('roian','')
rae.cv_train_from_mat(dstr='20200430',skip_db=False,run_type='submit',create_splits=True)
rae.cv_train_from_mat(dstr='20200430',queue='gpu_tesla',run_type='submit')


##
import PoseTools as pt
import run_apt_expts_2 as rae

rae.all_models = [m for m in rae.all_models if 'orig' not in m]

cam = 1
for split in range(7): #(5): #(3): #
    f, ax = plt.subplots(2, 5)
    ax = ax.flatten()
    for ndx,m in enumerate(rae.all_models):
        if m =='deeplabcut':
            # tf = '/groups/branson/bransonlab/mayank/apt_cache_2/wheel_rig_tracker_feb_2017_cam{}/{}/view_0/cv_split_{}/deepnet_20200710_traindata'.format(cam,m,split)
            # tf ='/groups/branson/bransonlab/mayank/apt_cache_2/four_points_180806/{}/view_0/cv_split_{}/deepnet_20200712_traindata'.format(m,split)
            tf ='/groups/branson/bransonlab/mayank/apt_cache_2/Test/{}/view_0/cv_split_{}/Test_deepnet_tesla_20200804_traindata'.format(m,split)
        else:
            # tf = '/groups/branson/bransonlab/mayank/apt_cache_2/wheel_rig_tracker_feb_2017_cam{}/{}/view_0/cv_split_{}/wheel_rig_tracker_feb_2017_cam{}_deepnet_20200710_traindata'.format(cam,m,split,cam)
            # tf ='/groups/branson/bransonlab/mayank/apt_cache_2/four_points_180806/{}/view_0/cv_split_{}/four_points_180806_deepnet_20200712_traindata'.format(m,split)
            tf = '/groups/branson/bransonlab/mayank/apt_cache_2/Test/{}/view_0/cv_split_{}/Test_deepnet_tesla_20200804_traindata'.format(
            m, split)
        A = pt.pickle_load(tf)
        ax[ndx].plot(A[0]['step'][50:],A[0]['val_dist'][50:])
        ax[ndx].plot(A[0]['step'][50:],A[0]['train_dist'][50:])
        ax[ndx].set_title(m)


##
import sys
if sys.version_info.major > 2:
    from importlib import reload
import run_apt_expts_2 as rae
reload(rae)
rae.setup('alice')
#rae.create_normal_dbs()
rae.get_normal_results(dstr='20200409',queue='gpu_tesla') #run_type = 'submit' to actually submit jobs.

##

##
import sys
if sys.version_info.major > 2:
    from importlib import reload
import sys
sys.path.insert(0,'/groups/branson/home/leea30/git/dpk')
sys.path.insert(0,'/groups/branson/home/leea30/git/imgaug')
import run_apt_expts_2 as rae
reload(rae)
rae.all_models = ['mdn_unet']
rae.setup('alice')
rae.create_normal_dbs()
rae.run_normal_training(run_type = 'dry') # to actually submit jobs.


##
import run_apt_expts as rae
reload(rae)
rae.setup('leap_fly')
rae.create_gt_db()

##

##
import run_apt_expts as rae
reload(rae)
rae.setup('romain',0)
rae.get_cv_results(num_splits=6)



##
import PoseTools
import os
import glob
import APT_interface as apt
import apt_expts
import re
import run_apt_expts as rae
import multiResData
import numpy as np

reload(apt_expts)
import PoseUNet_resnet

reload(PoseUNet_resnet)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# mdn_names = ['separate_groups', 'joint_groups' ]
mdn_names = ['explicit_offsets', 'implicit_offsets' ]
mdn_names = ['normal', 'True' ]
out_dir = '/groups/branson/home/kabram/temp'

out = {}
db_file = '/nrs/branson/mayank/apt_cache/Test/mdn/view_0/larva_compare/val_TF.tfrecords'
for n in mdn_names:
    cdir = os.path.dirname(db_file)
    if n == 'deepnet':
        tfile = os.path.join(cdir, 'traindata')
    else:
        tfile = os.path.join(cdir, 'Test_{}_traindata'.format(n))

    if not os.path.exists(tfile):
        continue
    A = PoseTools.pickle_load(tfile)
    conf = A[1]

    files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*.index").format(n))
    files.sort(key=os.path.getmtime)
    aa = [int(re.search('-(\d*).index', f).groups(0)[0]) for f in files]
    aa = [b - a for a, b in zip(aa[:-1], aa[1:])]
    if any([a < 0 for a in aa]):
        bb = int(np.where(np.array(aa) < 0)[0]) + 1
        files = files[bb:]
    files = [f.replace('.index', '') for f in files]
    files = files[-1:]
    # if len(files) > 8:
    #     gg = len(files)
    #     sel = np.linspace(0, len(files) - 1, 8).astype('int')
    #     files = [files[s] for s in sel]
    #
    mdn_out = apt_expts.classify_db_all(conf, db_file, files, 'mdn', name=n)
    out[n] = mdn_out

H = multiResData.read_and_decode_without_session(db_file, conf)
ex_ims = np.array(H[0][0])
ex_locs = np.array(H[1][0])
f = rae.plot_hist([out,ex_ims,ex_locs],[50,75,90,95,97])


##
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import PoseTools as pt
import PoseUNet_resnet

a = pt.pickle_load('/nrs/branson/mayank/apt_cache/Test/mdn/view_0/larva_compare/Test_normal_traindata')
a = pt.pickle_load('/nrs/branson/mayank/apt_cache/Test/mdn/view_0/cv_split_0/traindata')

conf = a[1]
self = PoseUNet_resnet.PoseUMDN_resnet(conf,name='deepnet')
self.train_data_name = 'traindata'
self.train_umdn(True)




##
data_type = 'roian'

import APT_interface as apt
import h5py
import PoseTools
import os
import time
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import apt_expts
import os
import ast
import apt_expts
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = ''

gt_lbl = None
lbl_file = '/groups/branson/bransonlab/apt/experiments/data/roian_apt_dlstripped.lbl'
op_af_graph = '\(0,1\),\(0,2\),\(0,3\),\(1,2\),\(1,3\),\(2,3\)'

lbl = h5py.File(lbl_file,'r')
proj_name = apt.read_string(lbl['projname'])
nviews = int(apt.read_entry(lbl['cfg']['NumViews']))
lbl.close()
cache_dir = '/nrs/branson/mayank/apt_cache'
all_models = ['openpose']

gpu_model = 'GeForceRTX2080Ti'
sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'
n_splits = 3


common_conf = {}
common_conf['rrange'] = 10
common_conf['trange'] = 5
common_conf['mdn_use_unet_loss'] = True
common_conf['dl_steps'] = 100000
common_conf['decay_steps'] = 20000
common_conf['save_step'] = 5000
common_conf['batch_size'] = 8
common_conf['maxckpt'] = 20
cache_dir = '/nrs/branson/mayank/apt_cache'
train_name = 'deepnet'


assert gt_lbl is None
all_view = []
for view in range(nviews):
    out_exp = {}
    for tndx in range(len(all_models)):
        train_type = all_models[tndx]

        out_split = None
        for split in range(n_splits):
            exp_name = 'cv_split_{}'.format(split)
            mdn_conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, 'mdn')
            conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)

            if op_af_graph is not None:
                conf.op_affinity_graph = ast.literal_eval(op_af_graph.replace('\\', ''))
            files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format(train_name))
            files.sort(key=os.path.getmtime)
            files = [f for f in files if os.path.splitext(f)[1] in ['.index', '']]
            aa = [int(re.search('-(\d*)',f).groups(0)[0]) for f in files]
            aa = [b-a for a,b in zip(aa[:-1],aa[1:])]
            if any([a<0 for a in aa]):
                bb = int(np.where(np.array(aa)<0)[0])+1
                files = files[bb:]
            n_max = 10
            if len(files)> n_max:
                gg = len(files)
                sel = np.linspace(0,len(files)-1,n_max).astype('int')
                files = [files[s] for s in sel]

            afiles = [f.replace('.index', '') for f in files]
            afiles = afiles[-1:]
            db_file = os.path.join(mdn_conf.cachedir,'val_TF.tfrecords')
            mdn_out = apt_expts.classify_db_all(conf,db_file,afiles,train_type,name=train_name)

        out_exp[train_type] = out_split
    all_view.append(out_exp)



##

in_file = '/nrs/branson/mayank/apt_cache/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402/leap/view_0/stephen_randsplit_round_4/leap_train.h5'
out_file = '/nrs/branson/mayank/apt_cache/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402/leap/view_0/stephen_randsplit_round_4_leap_orig/leap_train.h5'

import h5py
H = h5py.File(in_file,'r')
locs = H['joints'][:]
hf = h5py.File(out_file,'w')

hmaps = PoseTools.create_label_images(locs, conf.imsz[:2], 1, 5)
hmaps += 1
hmaps /= 2  # brings it back to [0,1]

hf.create_dataset('box', data=H['box'][:])
hf.create_dataset('confmaps', data=hmaps)
hf.create_dataset('joints', data=locs)

hf.close()
H.close()

##
import multiResData
A = multiResData.read_and_decode_without_session('/nrs/branson/mayank/apt_cache/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402/gtdata/gtdata_view0.tfrecords',5,())
ims = np.array(A[0])
locs = np.array(A[1])

out_file = '/nrs/branson/mayank/apt_cache/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402/leap/view_0/stephen_randsplit_round_4_leap_orig/leap_gt.h5'

import h5py
hf = h5py.File(out_file,'w')

hf.create_dataset('box', data=ims)
hf.create_dataset('joints', data=locs)

hf.close()

##

gt_file = '/nrs/branson/mayank/apt_cache/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402/leap/view_0/stephen_randsplit_round_4_leap_orig/leap_gt.h5'
out_file = '/nrs/branson/mayank/apt_cache/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402/leap/view_0/stephen_randsplit_round_4_leap_orig/gt_preds_002.mat/leap_gt.h5'

a = h5py.File(gt_file,'r')
b = h5py.File(out_file,'r')
locs = a['joints'][:]
preds = np.transpose(b['positions_pred'][:],[0,2,1])
hmaps = b['conf_pred'][:]

dd = np.sqrt(np.sum((locs-preds)**2,axis=-1))
np.percentile(dd,[50,75,90],axis=0)

##

import os
import APT_interface as apt
import glob
import re
import numpy as np
import multiResData
import math
import h5py
import PoseTools
import json

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

lbl_file = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_dlstripped.lbl'
cache_dir = '/nrs/branson/mayank/apt_cache'
exp_name = 'apt_expt_leap_original'
train_name = 'deepnet'
view = 0
train_type = 'leap'

lbl = h5py.File(lbl_file,'r')
proj_name = apt.read_string(lbl['projname'])
lbl.close()

gt_file = os.path.join(cache_dir, proj_name, 'gtdata', 'gtdata_view{}.tfrecords'.format(view))

conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)

split = False
use_cache = True
train_data = []
val_data = []

# collect the images and labels in arrays
out_fns = [lambda data: train_data.append(data), lambda data: val_data.append(data)]
splits, __ = apt.db_from_cached_lbl(conf, out_fns, split, None)

# save the split data
try:
    with open(os.path.join(conf.cachedir, 'splitdata.json'), 'w') as f:
        json.dump(splits, f)
except IOError:
    logging.warning('SPLIT_WRITE: Could not output the split data information')

for ndx in range(2):
    if not split and ndx == 1:  # nothing to do if we dont split
        continue

    if ndx == 0:
        cur_data = train_data
        out_file = os.path.join(conf.cachedir, 'leap_train.h5')
    else:
        cur_data = val_data
        out_file = os.path.join(conf.cachedir, 'leap_val.h5')

    ims = np.array([i[0] for i in cur_data])
    locs = np.array([i[1] for i in cur_data])
    info = np.array([i[2] for i in cur_data])
    hmaps = PoseTools.create_label_images(locs, conf.imsz[:2], 1, 3)
    hmaps += 1
    hmaps /= 2  # brings it back to [0,1]

    if info.size > 0:
        hf = h5py.File(out_file, 'w')
        hf.create_dataset('box', data=ims)
        hf.create_dataset('confmaps', data=hmaps)
        hf.create_dataset('joints', data=locs)
        hf.create_dataset('exptID', data=info[:, 0])
        hf.create_dataset('framesIdx', data=info[:, 1])
        hf.create_dataset('trxID', data=info[:, 2])
        hf.close()

##

import os
import APT_interface as apt
import glob
import re
import numpy as np
import multiResData
import math
import h5py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

lbl_file = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_dlstripped.lbl'
cache_dir = '/nrs/branson/mayank/apt_cache'
exp_name = 'apt_expt'
train_name = 'deepnet'
view = 0
train_type = 'mdn'

lbl = h5py.File(lbl_file,'r')
proj_name = apt.read_string(lbl['projname'])
lbl.close()

gt_file = os.path.join(cache_dir, proj_name, 'gtdata', 'gtdata_view{}.tfrecords'.format(view))

conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
conf.normalize_img_mean = False
files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format(train_name))
files.sort(key=os.path.getmtime)
files = [f for f in files if os.path.splitext(f)[1] in ['.index', '']]
aa = [int(re.search('-(\d*)', f).groups(0)[0]) for f in files]
aa = [b - a for a, b in zip(aa[:-1], aa[1:])]
if any([a < 0 for a in aa]):
    bb = int(np.where(np.array(aa) < 0)[0]) + 1
    files = files[bb:]
n_max = 6
if len(files) > n_max:
    gg = len(files)
    sel = np.linspace(0, len(files) - 1, n_max).astype('int')
    files = [files[s] for s in sel]

out_file = os.path.join(conf.cachedir, train_name + '_results.p')
afiles = [f.replace('.index', '') for f in files]

for m in afiles[-1:]:
    tf_iterator = multiResData.tf_reader(conf, gt_file, False)
    tf_iterator.batch_size = 1
    read_fn = tf_iterator.next
    pred_fn, close_fn, _ = apt.get_pred_fn(train_type, conf, m, name=train_name)
    bsize = conf.batch_size
    all_f = np.zeros((bsize,) + conf.imsz + (conf.img_dim,))
    n = tf_iterator.N
    pred_locs = np.zeros([n, conf.n_classes, 2])
    unet_locs = np.zeros([n, conf.n_classes, 2])
    mdn_locs = np.zeros([n, conf.n_classes, 2])
    n_batches = int(math.ceil(float(n) / bsize))
    labeled_locs = np.zeros([n, conf.n_classes, 2])
    all_ims = np.zeros([n, conf.imsz[0], conf.imsz[1], conf.img_dim])

    info = []
    for cur_b in range(n_batches):
        cur_start = cur_b * bsize
        ppe = min(n - cur_start, bsize)
        for ndx in range(ppe):
            next_db = read_fn()
            all_f[ndx, ...] = next_db[0]
            labeled_locs[cur_start + ndx, ...] = next_db[1]
            info.append(next_db[2])
        # base_locs, hmaps = pred_fn(all_f)
        ret_dict = pred_fn(all_f)
        base_locs = ret_dict['locs']
        ulocs = ret_dict['locs_unet']
        hmaps = ret_dict['hmaps']

        for ndx in range(ppe):
            pred_locs[cur_start + ndx, ...] = base_locs[ndx, ...]
            unet_locs[cur_start + ndx, ...] = ulocs[ndx, ...]
            mdn_locs[cur_start + ndx, ...] = ret_dict['locs_mdn'][ndx, ...]
            all_ims[cur_start + ndx, ...] = all_f[ndx, ...]

    close_fn()


##

import APT_interface as apt
import h5py
import PoseTools
import os
import time
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import apt_expts
import os
import ast
import apt_expts
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

lbl_file = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402_dlstripped.lbl'
gt_lbl = lbl_file
op_af_graph = '\(0,1\),\(0,2\),\(2,3\),\(1,3\),\(0,4\),\(1,4\)'

lbl = h5py.File(lbl_file,'r')
proj_name = apt.read_string(lbl['projname'])
nviews = int(apt.read_entry(lbl['cfg']['NumViews']))
lbl.close()

cache_dir = '/nrs/branson/mayank/apt_cache'

train_type = 'mdn'
exp_name = 'apt_exp'
for view in range(nviews):
    conf = apt.create_conf(gt_lbl, view, exp_name, cache_dir, train_type)
    gt_file = os.path.join(cache_dir,proj_name,'gtdata','gtdata_view{}.tfrecords'.format(view))
    apt.create_tfrecord(conf,False,None,False,True,[gt_file])



##
import APT_interface as apt
import os
import glob
import apt_expts
import h5py

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cache_dir = '/nrs/branson/mayank/apt_cache'
exp_name = 'apt_expt'
train_name = 'deepnet'
view =0

lbl_file = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_dlstripped.lbl'

lbl = h5py.File(lbl_file,'r')
proj_name = apt.read_string(lbl['projname'])
lbl.close()

train_type = 'leap'
conf = apt.create_conf(lbl_file, view, exp_name, cache_dir, train_type)
gt_file = os.path.join(cache_dir, proj_name, 'gtdata', 'gtdata_view{}.tfrecords'.format(view))
files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*").format(train_name))
mdn_out = apt_expts.classify_db_all(conf ,gt_file ,files ,train_type ,name=train_name)

##
import cv2
cap = cv2.VideoCapture('/nrs/branson/longterm/files_for_working_with_apt/20160214T111910_1_hour_segment_02.mjpg')


##

import PoseTools
import os
import glob
import APT_interface as apt
import apt_expts
reload(apt_expts)
import PoseUNet_resnet
reload(PoseUNet_resnet)
import re
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

db_file = '/nrs/branson/mayank/apt_cache/sh_trn4523_gt080618_made20180627_cacheddata/mdn/view_1/sh_compare/val_TF.tfrecords'
bsz = 4
lr_mul = 1
name = 'bsz_{}_lr_{}'.format(bsz,int(lr_mul*10))
cdir = os.path.dirname(db_file)
tfile = os.path.join(cdir,'sh_trn4523_gt080618_made20180627_cacheddata_{}_traindata'.format(name))

A = PoseTools.pickle_load(tfile)
conf = A[1]

files = glob.glob(os.path.join(conf.cachedir, "{}-[0-9]*.index").format(name))
files.sort(key=os.path.getmtime)
aa = [int(re.search('-(\d*).index',f).groups(0)[0]) for f in files]
aa = [b-a for a,b in zip(aa[:-1],aa[1:])]
if any([a<0 for a in aa]):
    bb = int(np.where(np.array(aa)<0)[0])+1
    files = files[bb:]
files = [f.replace('.index','') for f in files]
files = files[-1:]

mdn_out = apt_expts.classify_db_all(conf,db_file,files,'mdn',name=name)


##
cmd = '-name 20190129T180959 -view 1 -cache /home/mayank/temp/apt_cache -err_file /home/mayank/temp/apt_cache/multitarget_bubble/mdn/view_0/20190129T180959/trk/movie_trn20190129T180959_iter20000_20190208T141629.err -model_files /home/mayank/temp/apt_cache/multitarget_bubble/mdn/view_0/20190129T180959/deepnet-20000 -type mdn /home/mayank/temp/apt_cache/multitarget_bubble/20190129T180959_20190129T181147.lbl track -mov /home/mayank/work/FlySpaceTime/cx_GMR_SS00038_CsChr_RigB_20150729T150617/movie.ufmf -out /home/mayank/temp/apt_cache/multitarget_bubble/mdn/view_0/20190129T180959/trk/movie_trn20190129T180959_iter20000_20190208T141629.trk -start_frame 8496 -end_frame 8696 -trx /home/mayank/work/FlySpaceTime/cx_GMR_SS00038_CsChr_RigB_20150729T150617/registered_trx.mat -trx_ids 3'
##
# debug postprocessing
import APT_interface as apt
import numpy as np

lbl_file = '/home/mayank/temp/apt_cache/multitarget_bubble/20190207T121622_20190207T121731.lbl'
conf = apt.create_conf(lbl_file,0,'20190207T121622','/home/mayank/temp/apt_cache','mdn')

import multiResData
A = multiResData.read_and_decode_without_session('/home/mayank/temp/apt_cache/multitarget_bubble/mdn/view_0/20190207T121622/train_TF.tfrecords',conf,())
ims = np.array(A[0])
locs = np.array(A[1])
import PoseTools
reload(PoseTools)
a,b = PoseTools.randomly_affine(ims[:10,...],locs[:10,...],conf)

##
# debug postprocessing
import APT_interface as apt
import RNN_postprocess

lbl_file = '/home/mayank/temp/apt_cache/multitarget_bubble/20190207T121622_20190207T121731.lbl'
conf = apt.create_conf(lbl_file,0,'20190207T121622','/home/mayank/temp/apt_cache','mdn')
self = RNN_postprocess.RNN_pp(conf,'deepnet',
                              name = 'rnn_pp',
                              data_name='rnn_pp_groups_augfix')
self.rnn_pp_hist = 8
self.train_rep = 3
self.create_db(split_file = '/home/mayank/temp/apt_cache/multitarget_bubble/mdn/view_0/20190129T153403/splitdata.json')


##
import APT_interface as apt
apt.main(cmd.split())
