from __future__ import division

import logging
from builtins import object
#from past.utils import old_div
from operator import floordiv as old_div
import os
import re
# import localSetup
import numpy as np
import copy

class config(object):
    # ----- Names
    DATAAUG_FLDS = {
        'adjust_contrast': ['adjust_contrast', 'clahe_grid_size'],
        'scale_images': ['rescale'],
        'flip': ['horz_flip', 'vert_flip', 'flipLandmarkMatches'],
        'affine': ['use_scale_factor_range', 'scale_range', 'scale_factor_range',
                   'rrange', 'trange', 'rescale', 'check_bounds_distort'],
        'adjust': ['brange', 'crange', 'imax'],
        'normalize': ['normalize_img_mean', 'img_dim', 'perturb_color', 'imax', 'normalize_batch_mean'],
    }

    # ----- Network parameters
    def __init__(self):
        self.rescale = 1  # how much to downsize the base image.
        self.label_blur_rad = 3.  # 1.5
        self.imsz = [100,100] # h x w -- this is the same convention as what we could get when we do np.shape(img)
        self.n_classes = 10
        self.img_dim = 3
        self.has_crops = False

        self.batch_size = 8
        self.view = 0
        self.gamma = 0.1
        self.display_step = 50
        self.num_test = 8
        self.dl_steps = 60000 # number of training iters
        self.decay_steps = 25000
        self.learning_rate = 0.0001
        self.step_lr = True
        # rate will be reduced by gamma every decay_step iterations.

        # range for contrast, brightness and rotation adjustment
        self.has_trx_file = False
        self.trx_align_theta = True
        self.horz_flip = False
        self.vert_flip = False
        self.brange = [-0.2, 0.2]
        self.crange = [0.7, 1.3]
        self.rrange = 30
        self.rot_prob = 0.6
        self.trange = 10
        self.scale_range = 0.1
        self.scale_factor_range = 1.1
        # KB 20191218 - if scale_factor_range is read in, use that
        # otherwise, if scale_range is read in, use that
        self.use_scale_factor_range = True
        self.imax = 255.
        self.check_bounds_distort = False
        self.adjust_contrast = False
        self.clahe_grid_size = 20
        self.normalize_img_mean = False
        self.normalize_batch_mean = False
        self.perturb_color = False
        self.flipLandmarkMatches = {}
        self.learning_rate_multiplier = 1.
        self.predict_occluded = False
        self.nan_as_occluded = False
        self.use_openvino = False
        self.flip_test = False
        self.imresize_expand = False # if True, rescale the images to fit the conf.imsz. Mainly used for testing on public datasets.
        self.pad_images = False # If the images don't match conf.imsz, whether to pad them or resize them. True is pad images, False is resize images. Default value is False here, because all the crowdpose and other experiments are done with False. This should be set to true for projects created in front-end from here on. Doesn't affect older projects because they all had same sized images.

        # ----- Data parameters
        # l1_cropsz = 0
        self.splitType = 'frame'
        self.valdatafilename = 'valdata'
        self.valratio = 0.3
        self.holdoutratio = 0.8
        self.flipud = False
        self.json_trn_file = None
        self.db_format = 'tfrecord' # other option is coco
        #self.db_format = 'coco' # other option is coco

        if self.db_format == 'tfrecord':
            self.trainfilename = 'train_TF'
            self.fulltrainfilename = 'fullTrain_TF'
            self.valfilename = 'val_TF'
        else:
            self.trainfilename = 'traindata'
            self.fulltrainfilename = 'fulltraindata'
            self.valfilename = 'valdata'

        # ----- UNet params
        self.unet_rescale = 1
        #self.unet_steps = 20000
        self.unet_keep_prob = 1.0 # essentially don't use it.
        self.unet_use_leaky = False #will change it to True after testing.
        self.use_pretrained_weights = True

        # ----- MDN params
        self.mdn_min_sigma = 3. # this should just be maybe twice the cell size??
        self.mdn_max_sigma = 4.
        self.mdn_logit_eps_training = 0.001
        self.mdn_extra_layers = 1
        self.mdn_use_unet_loss = True
        self.mdn_pred_dist = True
        self.mdn_joint_thres = -3.
        self.pretrain_freeze_bnorm = True

        # ----- OPEN POSE PARAMS
        self.op_label_scale = 8
        self.op_im_pady = None  # computed at runtime
        self.op_im_padx = None  # "
        self.op_imsz_hires = None  # "
        self.op_imsz_lores = None  # "
        self.op_imsz_net = None  # "
        self.op_imsz_pad = None  # "
        self.op_backbone = 'resnet50_8px'
        self.op_backbone_weights = 'imagenet'
        self.op_map_lores_blur_rad = 1.0
        self.op_map_hires_blur_rad = 2.0
        self.op_paf_lores_tubewidth = 0.95 # not used if tubeblur=True
        self.op_paf_lores_tubeblur = False
        self.op_paf_lores_tubeblursig = 0.95
        self.op_paf_lores_tubeblurclip = 0.05
        self.op_paf_nstage = 5
        self.op_map_nstage = 1
        self.op_hires = True
        self.op_hires_ndeconv = 2
        self.op_base_lr = 4e-5  # Gines 5e-5
        self.op_weight_decay_kernel = 5e-4
        self.op_hmpp_floor = 0.1
        self.op_hmpp_nclustermax = 1
        self.op_pred_raw = False
        self.n_steps = 4.41

        # ---
        #self.sb_rescale = 1
        self.sb_n_transition_supported = 5  # sb network in:out size can be up to 2**<this> (as factor of 2). this
            # is for preproc/input pipeline only; see sb_output_scale for actual ratio
        self.sb_im_pady = None  # computed at runtime
        self.sb_im_padx = None  # "
        self.sb_imsz_net = None  # "
        self.sb_imsz_pad = None  # "
        self.sb_base_lr = 4e-5
        self.sb_weight_decay_kernel = 5e-4
        self.sb_backbone = 'ResNet50_8px'
        self.sb_backbone_weights = 'imagenet'
        self.sb_num_deconv = 3
        self.sb_deconv_num_filt = 512
        self.sb_output_scale = None  # output heatmap dims relative to imszuse (network input size), computed at runtime
        self.sb_upsamp_chan_handling = 'direct_deconv'  # or 'reduce_first'
        self.sb_blur_rad_input_res = 3.0  # target hmap blur rad @ input resolution
        self.sb_blur_rad_output_res = None  # runtime-computed
        self.sb_hmpp_floor = 0.1
        self.sb_hmpp_nclustermax = 1

        # ------ Leap params
        self.leap_net_name = "leap_cnn"
        self.leap_val_size = 0.15
        self.leap_preshuffle = True
        self.leap_filters = 64
        self.leap_val_batches_per_epoch = 10
        self.leap_reduce_lr_factor =0.1
        self.leap_reduce_lr_patience =3
        self.leap_reduce_lr_min_delta = 1e-5
        self.leap_reduce_lr_cooldown = 0
        self.leap_reduce_lr_min_lr = 1e-10
        self.leap_amsgrad =False
        self.leap_upsampling =False
        self.use_leap_preprocessing = False

        # ----- Deep Lab Cut
        self.dlc_train_img_dir = 'train'
        self.dlc_train_data_file = 'train_data.p'
        self.dlc_augment = True
        self.dlc_override_dlsteps = False

        # ---- dpk
        # "early" here is eg after initial setup in APT_interface
        self.dpk_skel_csv = None
        self.dpk_max_val_batches = 1       # maximum number of validation batches
        self.dpk_downsample_factor = 2      # (immutable after early) integer downsample                                            *power* for output shape
        self.dpk_n_stacks = 2
        self.dpk_growth_rate = 48
        self.dpk_use_pretrained = True
        self.dpk_n_outputs = 1              # (settable at TGTFR._call_-time)
        self.dpk_use_augmenter = False      # if true, use dpk_augmenter if distort=True
        self.dpk_augmenter_type = None      # dict for iaa construction
        self.dpk_augmenter = None           # actual iaa object; constructed at TGTFR-init time
        self.dpk_n_transition_min = 5       # target n_transition=this; in practice could be more if imsz is perfect power of 2 etc
        self.dpk_im_pady = None             # auto-computed
        self.dpk_im_padx = None             # auto-computed
        self.dpk_imsz_net = None            # auto-computed
        self.dpk_imsz_pad = None            # auto-computed
        self.dpk_use_graph = True           # (immutable after early) bool
        self.dpk_graph = None               # (immutable after early)
        self.dpk_swap_index = None          # (immutable after early)
        self.dpk_graph_scale = 1.0          # (immutable after early) float, scale factor                                           applied to grp/limb/global confmaps
        self.dpk_output_shape = None        # (computed at TGTFR/init) conf map output shape
        self.dpk_output_sigma = None        # (computed at TGTFR/init) target hmap gaussian                                         sd in output coords
        self.dpk_input_sigma = 5.0          # (immutable after early) target hmap gaussian                                          sd in input coords
        self.dpk_base_lr_factory = .001
        self.dpk_base_lr_used = None        # typically auto-computed at compile-time; actual base lr used
        self.dpk_reduce_lr_on_plat = True   # DEPRECATED in favor of dpk_train_style
                                            # True is as published for dpk, using K cbk (starting from dpk_base_lr_used);
                                            # False is APT-style scheduled (using learning_rate, decay_steps, gamma)
        self.dpk_reduce_lr_style = "__UNUSED__"  # either 'ppr' or 'ipynb'
        self.dpk_early_stop_style = "__UNUSED__"  # either 'ppr' or 'ipynb'
        self.dpk_epochs_used = None         # set at train-time; actual no of epochs used
        self.dpk_use_tfdata = True
        self.dpk_tfdata_num_para_calls_parse = 5
        self.dpk_tfdata_num_para_calls_dataaug = 8
        self.dpk_train_style = 'apt'        # 'dpk' for dpk-orig-style or 'apt' for apt-style
        self.dpk_val_batch_size = 0        # use 0 when dpk_train_style='apt' to not do valdist loggin
        self.dpk_tfdata_shuffle_bsize = 5000       # buffersize for tfdata shuffle
        self.dpk_auto_steps_per_epoch = True  # if True, set .display_step=ntrn/bsize. If False, use .display_step as provided.
        self.dpk_use_op_affinity_graph = True # if True, use affinity graph for dpk skel.
                                              # if False, dpk_skel_csv must be set

        # ============== MULTIANIMAL ==========
        self.is_multi = False
        self.max_n_animals_user = 1 # this is the maximum number of animals that the user enters in the front-end.
        self.max_n_animals = 1 # this is the actual number of detections to do while doing inference. this is set to 1.25 times the max_n_animals_user
        self.min_n_animals = 0
        self.multi_bb_ex = 10 # extra margin to keep around annotations while generating masks
        self.multi_n_grid = 1 # Number of cells to split the image into for multianimal
        self.multi_link_cost = 5 # cost for linking trajectory. 5 is roughly the max movement in pixels per landmark that will not lead to death and birth of new trajectories.
        # actual frame size
        self.multi_frame_sz = []
        self.multi_animal_crop_sz = None
        # multi_use_mask is whether to mask the image or not. Shouldn't be used anymore
        self.multi_use_mask = False
        # whether to mask the loss or not
        self.multi_loss_mask = True
        # crop images during training
        self.multi_crop_ims = True
        # For NMS for pose. Suppress poses whose avg matching distance is less than this percentage of the bounding box edge size.
        self.multi_match_dist_factor = .2
        self.multi_scale_by_bbox = False
        self.multi_pad = 1.25 # if scaling by bbox, pad the bbox by this factor
        self.multi_background_coverage_ratio = 0.5 # ratio of the background to cover while training
        self.multi_background_sample_ratio = 0.5 # ratio of the background samples to training samples

        # ============= TOP-DOWN =================

        # For top-down networks, use these points as head tail
        self.ht_pts = []
        # Train-Track only ht points -- for top-down networks
        self.multi_only_ht = False
        # multi_only_ht is for first stage. And if multi_only_ht is used for first stage then use_ht_trx should be true for the second-stage else use_bbox_trx
        # Use ht points as trx surrogate for top-down network
        self.use_ht_trx = False
        # Use bbox as trx surrogate for top-down networks.
        self.use_bbox_trx = False
        self.stage = None
        self.ht_2stage_n_classes = 2

        # ============== LINKING ===============
        self.link_stage = 'second'
        self.link_maxcost_heuristic = 'secondorder'
        self.link_maxcost_framesfit = 5
        self.link_maxcost_mult = 2.0
        self.link_maxcost_prctile = 95.
        self.link_maxframes_delete = 10
        self.link_maxframes_missed = 10
        self.link_minconf_delete = 0.5
        self.link_maxcost_secondorder_thresh = 1.
        self.link_strict_match_thres = 2.

        self.link_id = False
        # self.link_id_cropsz = None
        self.link_id_cropsz_width = None
        self.link_id_cropsz_height = None
        self.link_id_training_iters = 100000
        self.link_id_tracklet_samples = 25
        self.link_id_rescale = 1
        self.link_id_min_tracklet_len = 6 # should be greater than link_maxcost_framesfit
        self.link_id_mining_steps = 10
        self.link_id_min_train_track_len = 10
        self.link_id_keep_all_preds = False
        self.link_id_batch_size = 16
        self.link_id_ignore_far = False
        self.link_id_motion_link = False
        self.link_id_save_int_wts = False

        # ============= MMPOSE =================
        self.mmpose_net = 'multi_hrnet'
        self.multi_mmpose_detection_threshold = 0.5

        # ============== EXTRA ================

        # ----- Time parameters
        self.time_window_size = 1
        self.do_time = False

        # ------ RNN Parameters
        self.rnn_before = 9
        self.rnn_after = 0

        # ------------ ATTention parameters
        self.att_hist = 128
        self.att_layers = [1] # use layer this far from the middle (top?) layers.

        # ----- Save parameters

        self.save_time = None
        self.save_step = 2000
        self.save_td_step = 100
        self.maxckpt = 2
        self.cachedir = ''
        self.project_file = ''

        # ----- Legacy
        # self.scale = 2
        # self.numscale = 3
        # self.pool_scale = 4
        # self.pool_size = 3
        # self.pool_stride = 2
        # self.cos_steps = 2 #number of times the learning rate is decayed
        # self.step_size = 100000 # not used anymore


    def set_exp_name(self, exp_name):
        self.expname = exp_name
        # self.baseoutname = self.expname + self.baseName
        # self.baseckptname = self.baseoutname + 'ckpt'
        # self.basedataname = self.baseoutname + 'traindata'
        # self.fineoutname = self.expname + self.fineName
        # self.fineckptname = self.fineoutname + 'ckpt'
        # self.finedataname = self.fineoutname + 'traindata'
        # self.mrfoutname = self.expname + self.mrfName
        # self.mrfckptname = self.mrfoutname + 'ckpt'
        # self.mrfdataname = self.mrfoutname + 'traindata'


    def getexpname(self, dirname):
        return os.path.basename(os.path.dirname(dirname)) + '_' + os.path.splitext(os.path.basename(dirname))[0]

    def getexplist(self, L):
        return L['movieFilesAll'][self.view,:]

    def get(self,name,default,silent=False):
        if hasattr(self,name):
            if not silent:
                logging.info('OVERRIDE: Using {} with value {} from config '.format(name,getattr(self,name)))
        else:
            if not silent:
                logging.info('DEFAULT: For {} using with default value {}'.format(name, default))
            setattr(self,name,default)
        return getattr(self,name,default)

    def print_dataaug_flds(self, printfcn=None):
        printfcn = logging.info if printfcn is None else printfcn
        for cat, flds in self.DATAAUG_FLDS.items():
            printfcn('## {} ##'.format(cat))
            for f in flds:
                printfcn('  {}: {}'.format(f, getattr(self, f, '<DNE>')))

conf = config()

def parse_aff_graph(aff_graph_str):
    '''
    Parse an afinity-graph str (comma-sep) into a list of edges
    :param aff_graph_str: eg '1 2,1 3,1 4,3 4'
    :return: eg [[0,1],[0,2],[0,3],[2,3]]
    '''
    graph = []
    aff_graph_str = aff_graph_str.split(',')
    if len(aff_graph_str)==1 and aff_graph_str[0] == '':
        return graph
    for b in aff_graph_str:
        mm = re.search('(\d+)\s+(\d+)', b)
        n1 = int(mm.groups()[0]) - 1
        n2 = int(mm.groups()[1]) - 1
        graph.append([n1, n2])

    return graph


