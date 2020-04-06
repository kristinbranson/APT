from __future__ import division
from __future__ import print_function

##

from builtins import range
from past.utils import old_div
import romainLegConfig
reload(romainLegConfig)
from romainLegConfig import bottomconf as confb
from romainLegConfig import side1conf as conf1
from romainLegConfig import side2conf as conf2
from multiResData import *

L = h5py.File(conf1.labelfile,'r')

create_val_data(conf1, force=True)
isval, localdirs, seldirs = load_val_data(conf1)
pts = np.array(L['labeledpos'])

view = conf1.view
count = 0;
valcount = 0

tr_count = 0
val_count = 0
fr_count = []
for ndx, dirname in enumerate(localdirs):

    expname = conf1.getexpname(dirname)
    curpts = np.array(L[pts[0, ndx]])
    zz = curpts.reshape([curpts.shape[0], -1])
    frames = np.where(np.invert(np.any(np.isnan(curpts[:, :, :]), axis=(1, 2))))[0]
    print(frames.size,dirname)

    fr_count.append(frames.size)
    if isval.count(ndx):
        val_count += frames.size
    else:
        tr_count += frames.size

fr_count = np.array(fr_count)
print(val_count,tr_count)

##

sel = []
for ndx in range(10):
    val_count = 0
    tot_count = fr_count.sum()
    tr = 0.3
    tr_high = 0.40
    kk = np.random.permutation(len(localdirs))
    idx = 0

    while(old_div(float(val_count),tot_count) < tr):
        val_count += fr_count[kk[idx]]
        idx+=1
    if old_div(float(val_count),tot_count) <= tr_high:
        sel.append(kk[:idx])
        print(val_count, kk[:idx])

isval = list(sel[0])
##

for conf in [conf1,conf2,confb]:
    outfile = os.path.join(conf.cachedir, conf.valdatafilename)
    localdirs, seldirs = find_local_dirs(conf)
    with open(outfile, 'w') as f:
        pickle.dump([isval, localdirs, seldirs], f)

