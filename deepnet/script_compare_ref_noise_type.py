### Anikets bottom view dataset

# commands derived from cmd =['/groups/branson/home/kabram//bransonlab/apt_cache_2/aniket_bottom/combinedSideViewMA/20250808T015629_20250808T015637.json', '-name', '20250808T015629', '-train_name','gaussian_noise','-json_trn_file', '/groups/branson/home/kabram/bransonlab/apt_cache_2/aniket_bottom/combinedSideViewMA/loc.json', '-conf_params', 'mdn_joint_ref_noise_type','gaussian','-type', 'multi_mdn_joint_torch', '-ignore_local', '1', '-cache', '/groups/branson/home/kabram/.apt/tpb0ca5fd1_d8a0_46b3_a5d7_d3747ca30921', 'train', '-use_cache','-skip_db']

conf_file = '/groups/branson/home/kabram/bransonlab/apt_cache_2/aniket_bottom/combinedSideViewMA/20250808T015629_20250808T015637.json'
trn_file = '/groups/branson/home/kabram/bransonlab/apt_cache_2/aniket_bottom/combinedSideViewMA/loc.json'
cache_dir = '/groups/branson/home/kabram/bransonlab/apt_cache_2/aniket_bottom/'

from reuse import *
conf = apt.create_conf(conf_file, 0, 'apt_expt', cache_dir, 'multi_mdn_joint_torch')

import ap36_train as a36

conf.mdn_joint_ref_noise_type = 'gaussian'
# a36.train(conf, 'multi_mdn_joint_torch', 'gaussian_noise')
a36.train_bsub(conf, 'multi_mdn_joint_torch', 'gaussian_noise','aniket_bottom_gaussian_noise')

conf.mdn_joint_ref_noise_type = 'laplacian'
# a36.train(conf, 'multi_mdn_joint_torch', 'laplacian_noise')
a36.train_bsub(conf, 'multi_mdn_joint_torch', 'laplacian_noise','aniket_bottom_laplacian_noise')

## Track the movies

mov_file = '/groups/branson/bransonlab/aniket/APT/3D_labeling_project/movie_output_dir_combined_views/exp_43/image_cam_0_date_2025_06_17_time_11_20_12_v001_crop_col1167to1919_rot90.ufmf'

for tt in ['gaussian_noise','laplacian_noise']:
    out_file = f'/groups/branson/home/kabram/temp/aniket_sideview_{tt}.trk'
    cmd =['/groups/branson/home/kabram//bransonlab/apt_cache_2/aniket_bottom/combinedSideViewMA/20250808T015629_20250808T015637.json', '-name', 'apt_expt', '-train_name',tt,'-json_trn_file', '/groups/branson/home/kabram/bransonlab/apt_cache_2/aniket_bottom/combinedSideViewMA/loc.json', '-conf_params', 'mdn_joint_ref_noise_type','"gaussian"','-type', 'multi_mdn_joint_torch', '-ignore_local', '1', '-cache', '/groups/branson/home/kabram/bransonlab/apt_cache_2/aniket_bottom/', 'track','-mov', mov_file, '-out',out_file,'-track_type','only_predict']
    apt.main(cmd)


## Alices dataset

from reuse import *
K = pt.pickle_load('/groups/branson/bransonlab/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/grone_crop_mask_12072023/alice_ma_noise_scaled_p25_traindata')

conf = K[1]
import ap36_train as a36

conf.mdn_joint_ref_noise_type = 'gaussian'
# a36.train(conf, 'multi_mdn_joint_torch', 'gaussian_noise')
a36.train_bsub(conf, 'multi_mdn_joint_torch', 'gaussian_noise','alice_gaussian_noise',sing_img='/groups/branson/home/kabram/bransonlab/singularity/hopper_pycharm.sif')

conf.mdn_joint_ref_noise_type = 'laplacian'
# a36.train(conf, 'multi_mdn_joint_torch', 'laplacian_noise')
a36.train_bsub(conf, 'multi_mdn_joint_torch', 'laplacian_noise','alice__laplacian_noise',sing_img='/groups/branson/home/kabram/bransonlab/singularity/hopper_pycharm.sif')


##

movs = ['/groups/branson/home/robiea/Projects_data/Labeler_APT/Nan_labelprojects_touchinglabels/CourtshipData/nochr_TrpA65F12_Unknown_RigA_20201212T163531/movie.ufmf', '/groups/branson/home/robiea/Projects_data/Labeler_APT/socialCsChr_JRC_SS56987_CsChrimson_RigD_20190910T163910/movie.ufmf'
]

from reuse import *
K = pt.pickle_load('/groups/branson/bransonlab/mayank/apt_cache_2/alice_ma/multi_mdn_joint_torch/view_0/grone_crop_mask_12072023/alice_ma_noise_scaled_p25_traindata')

conf = K[1]
conf.imsz = [1024,1024]

for ndx,mov in enumerate(movs):
    for tt in ['gaussian_noise','laplacian_noise']:
        out_file = f'/groups/branson/home/kabram/temp/alice_mov_{ndx}_{tt}.trk'
        apt.classify_movie_all('multi_mdn_joint_torch',conf=conf,mov_file=mov,out_file=out_file,name='grone_crop_mask_12072023',train_name=tt,model_file=None,no_except=False)
##

