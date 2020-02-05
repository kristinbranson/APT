import multiResData as mrd
import PoseTools as pt
import numpy as np
import open_pose4 as op
import run_apt_expts_al as rae
import util
import matplotlib.pyplot as plt
import heatmap as hm
import skimage
import sb1 as sb
import os

net = 'sb'
view = 0
rae.setup('alice')
exp_name = rae.expname_dict_normaltrain[net]
c1 = rae.create_conf_help(net, view, exp_name)
badrows = [[611,15], [1681,16], [1791,16]] # bub
#badrows = [[281,1], [562,1], [1075,1], [59,2], [125,2], [992,2], [125,5], [136,5]] # shvw1,op
#badrows = [[146,3], [146,4]]  # shvw2, op
#badrows = [[256,1], [90,2], [584,2], [992,2], [36,4], [37,4], [138,4], [392,5], [588,5], [833, 5], [839,5],[844,5],[995,5]] shvw1,sb
#badrows = [[476,1], [487,2], [929,2], [935,2], [1020,2], [637,4], [987,4]] # shvw2,sb

nbads = len(badrows)
print("{} bad rows".format(nbads))
#projname = 'sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402'

tdf = os.path.join(c1.cachedir,'traindata')
tfr = os.path.join(c1.cachedir,'train_TF.tfrecords')
gtf = os.path.join('/nrs/branson/al/cache',c1.expname,'gtdata/gtdata_view{}.tfrecords'.format(view))
mdl = os.path.join(c1.cachedir,'deepnet-50000')
#resf = '/nrs/branson/al/cache/multitarget_bubble/openpose/view_0/apt_expt2/deepnet_results.p'

td = pt.pickle_load(tdf)
c0 = td[-1]

if net=='openpose':
    c1.op_pred_raw = True
    pfcn, closefcn, _ = op.get_pred_fn(c1, model_file=mdl, name='deepnet')
elif net=='sb':
    pfcn, closefcn, _ = sb.get_pred_fn(c1, model_file=mdl, name='deepnet',retrawpred=True)


ims, locs, info = mrd.read_and_decode_without_session(gtf, c1, indices=[])
# % 26/297/14. both pk and pkcmpts 1st opt is best! predloc must have failed b/c ncluster=1.

imsa = np.stack(ims)
#imset = imsa[296:296+8,:,:,:]

hms = []
plt.figure()
naxcol = nbads #int(np.ceil(nbads/2))
for idx,info1b in enumerate(badrows):
    irow = info1b[0]-1
    ipt = info1b[1]-1
    imset = imsa[irow:irow+c1.batch_size,:,:,:]
    preds = pfcn(imset, retrawpred=True)
    predhm = preds['pred_hmaps']
    predam = preds['locs_argmax']
    predwc0 = preds['locs_wgtcnt0']
    if net=='openpose':
        predhm = predhm[-1]
    predhm = predhm[0,:,:,ipt]
    im = imset[0,:,:,0]
    loc = locs[irow][ipt,:]
    locam = predam[0,ipt,:]
    locwc0 = predwc0[0,ipt,:]

    ax = plt.subplot(2,naxcol,idx+1)
    plt.imshow(predhm,cmap='jet')
    plt.colorbar()
    plt.title('row{}pt{}'.format(irow,ipt))
    ax = plt.subplot(2,naxcol,idx+1+naxcol)
    plt.imshow(im)
    plt.hold('on')
    plt.plot(loc[0],loc[1],'rx')
    plt.plot(locam[0],locam[1],'w+')
    plt.plot(locwc0[0],locwc0[1],'r+')

plt.show()
'''
a, mu, sig = hm.compactify_hmap(predhm13,
                                floor=0.1,
                                nclustermax=1)
a2, mu2, sig2 = hm.compactify_hmap(predhm13,
                                   floor=0.1,
                                   nclustermax=2)

assert np.all(predhm13 >= 0.)
floor = 0.1
predhm13floor = predhm13.copy()
predhm13floor[predhm13floor < floor] = 0.0

hmbw = predhm13floor > 0.
lbls = skimage.measure.label(hmbw, connectivity=1)
rp = skimage.measure.regionprops(lbls, intensity_image=predhm13floor)
rp.sort(key=lambda x: x.area, reverse=True)

plt.figure()
ax1=plt.subplot(2,2,1)
plt.imshow(predhm13floor, cmap='jet')

def read_resultsp(pfile, ptiles=[50,90,95,97]):
    res = pt.pickle_load(pfile)
    rd = res[0][0][0]
    rd['lbl'], rd['mft'], rd['mdl'], _, rd['mdlts'] = res[0][0][1:]
    dd = np.sqrt(np.sum((rd['locs'] -rd['lbl']) ** 2, axis=-1))
    ptiles = np.percentile(dd, ptiles, axis=0)
    rd['err'] = dd
    rd['errptls'] = ptiles
    return rd
'''