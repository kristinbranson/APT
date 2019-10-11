import APT_interface as apt
import os
# Alice's dataset

name = 'alice'
val_ratio = 0.1
lbl_file = '/home/kabram/Dropbox (HHMI)/temp/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_dlstripped.lbl'
nviews = 1

for view in range(nviews):
    conf = apt.create_conf(lbl_file,0,'tfds','/home/kabram/temp','mdn')
    conf.cachedir = '/home/kabram/temp/tfds_{}_view{}'.format(name,view)
    conf.valratio = val_ratio
    os.makedirs(conf.cachedir,exist_ok=True)
    apt.create_tfrecord(conf, split=True, split_file=None, use_cache=True, on_gt=False)

