import os

import numpy as np

import deepposekit as dpk

import PoseTools as pt
import TrainingGeneratorTFRecord as TGTFR
import apt_dpk
# import open_pose_data as opd
import apt_dpk_exps as ade
import tfdatagen

EROOT = '/groups/branson/home/leea30/blapt/dl.al.2020/cache/leap_dset/dpksdn/view_0'
edir_tfr = os.path.join(EROOT, 'dpkorig_20200512T180752_run4_tfr_03')
edir_0 = os.path.join(EROOT, 'dpkorig_20200512T180752_run4')
DSET = 'dpkfly'

# make a TG and set its trn/val indices
h5dset = ade.dbs[DSET]['h5dset']
dg = dpk.io.DataGenerator(h5dset)
# create an APT conf. this is used for
# - file/cache locs
# - params (mostly if not all dpk_* params)
#
# For the leapfly, the projname in the slbl is 'leap_dset'
# Note though that the DPK version of the leap dset has
# locs off-by-one
conf = ade.exp1orig_create_base_conf(edir_0, ade.alcache, DSET)
# this conf-updating is actually prob unnec but prob doesnt hurt
conf = apt_dpk.update_conf_dpk(conf,
                               dg.graph,
                               dg.swap_index,
                               n_keypoints=dg.n_keypoints,
                               imshape=dg.compute_image_shape(),
                               useimgaug=True,
                               imgaugtype=DSET)
conf.batch_size = 16
conf.dpk_use_tfdata = True
apt_dpk.print_dpk_conf(conf)

VALSPLIT = 0.1
assert conf.dpk_downsample_factor == 2
assert conf.dpk_input_sigma == 5.0
assert conf.dpk_graph_scale == 1.0
tg = dpk.TrainingGenerator(generator=dg,
                           downsample_factor=conf.dpk_downsample_factor,
                           augmenter=conf.dpk_augmenter,
                           use_graph=True,
                           shuffle=False, # xxx usually True; here False to compare
                           sigma=conf.dpk_input_sigma,
                           validation_split=VALSPLIT,
                           graph_scale=conf.dpk_graph_scale,
                           random_seed=0)

picf0 = os.path.join(edir_0, 'conf.pickle')
pic0 = pt.pickle_load(picf0)

tg.val_index = pic0['tg']['val_index']
tg.train_index = pic0['tg']['train_index']

tg.val_index.sort()
tg.train_index.sort()


# instantiate train, val generators as in training
sdn_n_outputs = 2
bsize = 16
trn_generator = tg(sdn_n_outputs, bsize, validation=False, confidence=True)
val_generator = tg(sdn_n_outputs, bsize, validation=True, confidence=True)

# reseed augmenters jic
RNGSEED = 17
trn_generator.augmenter.reseed(17)
val_generator.augmenter.reseed(17)

tgtfr = TGTFR.TrainingGeneratorTFRecord(conf)
tgtfr.conf.dpk_augmenter.reseed(17)
dstrn = tgtfr(sdn_n_outputs,
              bsize,
              validation=False,
              confidence=True,
              shuffle=False,  # ordinarily true
              infinite=True)
dsval = tgtfr(sdn_n_outputs,
              bsize,
              validation=True,
              confidence=True,
              shuffle=False,
              infinite=False)

# get some batches
imstgts_dpk_trn = [trn_generator[x] for x in range(3)]
for i in range(3):
    imstgts_dpk_trn[i] = (imstgts_dpk_trn[i][0], imstgts_dpk_trn[i][1][0])  # n_outputs = 2
imsdpktrn, tgtsdpktrn = tfdatagen.xylist2xyarr(imstgts_dpk_trn)
print(imsdpktrn.shape, tgtsdpktrn.shape)

imstgts_dpk_val = [val_generator[x] for x in range(3)]
for i in range(3):
    imstgts_dpk_val[i] = (imstgts_dpk_val[i][0], imstgts_dpk_val[i][1][0])  # n_outputs = 2
imsdpkval, tgtsdpkval = tfdatagen.xylist2xyarr(imstgts_dpk_val)
print(imsdpkval.shape, tgtsdpkval.shape)

resDS = tfdatagen.read_ds_idxed(dstrn, range(3))
for i in range(3):
    resDS[i] = (resDS[i][0], resDS[i][1][0])  # n_outputs = 2
imsDStrn, tgtsDStrn = tfdatagen.xylist2xyarr(resDS)
print(imsDStrn.shape, tgtsDStrn.shape)

resDS = tfdatagen.read_ds_idxed(dsval, range(3))
for i in range(3):
    resDS[i] = (resDS[i][0], resDS[i][1][0])  # n_outputs = 2
imsDSval, tgtsDSval = tfdatagen.xylist2xyarr(resDS)
print(imsDSval.shape, tgtsDSval.shape)

print(np.array_equal(imsdpktrn, imsDStrn), np.array_equal(tgtsdpktrn, tgtsDStrn) )
print(np.array_equal(imsdpkval, imsDSval), np.array_equal(tgtsdpkval, tgtsDSval) )

print(np.allclose(imsdpktrn, imsDStrn), np.allclose(tgtsdpktrn, tgtsDStrn) )
print(np.allclose(imsdpkval, imsDSval), np.allclose(tgtsdpkval, tgtsDSval) )
