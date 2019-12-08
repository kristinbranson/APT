from __future__ import print_function

import sys
import time
import timeit
import h5py
from scipy import stats
import scipy.io as sio
import skimage.measure
import numpy as np
import matplotlib.pyplot as plt

def compactify_hmap_gaussian_test():
    sig = np.array([[10,4],[4,7]])
    mu = np.array([33,44])
    mvn = multivariate_normal(mu, sig)

    xgv = range(1,101)
    ygv = range(1,81)
    xg, yg = np.meshgrid(xgv, ygv)
    xy = np.concatenate((xg[:, :, np.newaxis], yg[:, :, np.newaxis]), axis=2)
    hm = mvn.pdf(xy)

    print("hm max is {}".format(np.max(hm)))


    jitterthresh = .01
    tf = hm<jitterthresh
    jitter = .001*np.random.rand(*np.shape(hm))
    jitter[tf] = 0
    hmnoise = hm + jitter

    muobs,sigobs,A,err = compactify_hmap_gaussian(hm)
    muobs2,sigobs2,A2,err2 = compactify_hmap_gaussian(hmnoise)
    muobs3,sigobs3,A3,err3 = compactify_hmap_gaussian(hmnoise,meanismax=True)
    return ((muobs,sigobs,A,err),(muobs2,sigobs2,A2,err2),(muobs3,sigobs3,A3,err3))

def compactify_hmap(hm, floor=0.0, nclustermax=5):
    '''

    :param hm:
    :param floor:
    :param nclustermax:
    :return:
         mu: (2, nclustermax). Each col is (row,col) weighted centroid, 1-based

    '''

    assert np.all(hm >= 0.)
    if floor > 0.0:
        hm[hm < floor] = 0.0

    hmbw = hm > 0.
    lbls = skimage.measure.label(hmbw, connectivity=1)
    rp = skimage.measure.regionprops(lbls, intensity_image=hm)
    rp.sort(key=lambda x: x.area, reverse=True)

    a = np.zeros(nclustermax)
    mu = np.zeros((2,nclustermax))
    sig = np.zeros((2,2,nclustermax))

    nclusters = min(nclustermax,len(rp))
    for ic in range(nclusters):
        a[ic] = rp[ic].weighted_moments[0, 0]
        mu[:, ic] = np.array(rp[ic].weighted_centroid) + 1.0  # transform to 1-based
        wmc = rp[ic].weighted_moments_central
        xxcov = wmc[2,0]/a[ic]
        yycov = wmc[0,2]/a[ic]
        xycov = wmc[1,1]/a[ic]
        sig[:, :, ic] = np.array([[xxcov, xycov], [xycov, yycov]])

    return a, mu, sig

def compactify_hmap_arr(hmagg,offset=1.0,floor=0.0):
    npt, nrtrans, nctrans, nfrm = hmagg.shape
    print("{} frames, {} pts".format(nfrm, npt))

    nclustermax = 5

    As = np.zeros((nclustermax,npt, nfrm), dtype=hmagg.dtype)
    mus = np.zeros((2, nclustermax, npt, nfrm), dtype=hmagg.dtype)
    sigs = np.zeros((2, 2, nclustermax, npt, nfrm), dtype=hmagg.dtype)

    tic = time.time()

    for ipt in range(npt):
        for f in range(nfrm):
            if f%50 == 0:
                print("frame {}".format(f))
            hm = hmagg[ipt, :, :, f]
            hm = hm + offset
            As[:, ipt, f], mus[:, :, ipt, f], sigs[:, :, :, ipt, f] = \
                compactify_hmap(hm, floor, nclustermax)

    toc = time.time() - tic
    print("Took {} s to compactify".format(toc))

    return mus, sigs, As

def get_weighted_centroids(hm, floor=0.0, nclustermax=5):
    '''
    Get predicted loc from hmap as weighted centroid (vs argmax).


    :param hm: (bsize, nr, nc, npts)
    :param floor: see compactify_hmap
    :param nclustermax: see compactify_hmap
    :return: (bsize, npts, 2) (x,y) weighted centroids, 0-based
    '''

    assert np.all(hm >= 0.0), "Heatmap must be positive semi-def"
    bsize, nr, nc, npts = hm.shape

    hmmu = np.zeros((bsize, npts, 2))
    for ib in range(bsize):
        for ipt in range(npts):
            _, mutmp, _ = compactify_hmap(hm[ib, :, :, ipt],
                                          floor=floor,
                                          nclustermax=nclustermax)
            # Convert to (x,y), 0-based
            assert nclustermax==1  # well this is confused
            hmmu[ib, ipt, :] = mutmp[::-1].flatten() - 1.0

    return hmmu

def create_label_hmap(locs, imsz, sigma, clip=0.05):
    """
    Create/return target hmap for parts

    This is a 2d isotropic gaussian with tails clipped to 0.
    Notes:
    - Normalization. hmap is in [0,1], but np.sum(hm) may can vary slightly depending on ctr loc
                    relative to "bin centers"; and may vary greatly if ctr loc approaches edge of im
    - Gaussian has subpx placement and is not centered precisely at a pixel center

    locs: (nbatch x npts x 2) (x,y) locs, 0-based. (0,0) is the center of the upper-left pixel.
    imsz: [2] (nr, nc) size of heatmap to return
    """

    bsize, npts, d = locs.shape
    assert d == 2
    out = np.zeros([bsize, imsz[0], imsz[1], npts])
    for cur in range(bsize):
        for ndx in range(npts):
            x, y = np.meshgrid(range(imsz[1]), range(imsz[0]))
            x0 = locs[cur, ndx, 0]
            y0 = locs[cur, ndx, 1]
            assert not (np.isnan(x0) or np.isnan(y0) or np.isinf(x0) or np.isinf(y0))

            dx = x - x0
            dy = y - y0
            d2 = dx**2 + dy**2
            exp = -d2 / 2.0 / sigma / sigma
            out[cur, :, :, ndx] = np.exp(exp)
    out[out < clip] = 0.

    return out

def hmap_cmp_viz(hm1, hm2, figsz=(1400,1800), figfaceclr=(0.5,0.5,0.5)):
    f, ax = plt.subplots(1, 2)
    m = plt.get_current_fig_manager()
    m.resize(*figsz)
    f.set_facecolor(figfaceclr)

    plt.axes(ax[0])
    plt.cla()
    plt.imshow(hm1)
    plt.colorbar()

    plt.axes(ax[1])
    plt.cla()
    plt.imshow(hm2)
    plt.colorbar()

def main(argv):
    inmatfile = argv[0]
    outmatfile = argv[1]

    tic = time.time()
    hm = h5py.File(inmatfile,'r')
    hmagg = hm['hmagg']
    toc = time.time() - tic
    #print "Took {} s to load".format(toc)

    tic = time.time()
    hmagg = np.array(hmagg)
    toc = time.time() - tic
    #print "Took {} s to get hm array".format(toc)

    tic = timeit.default_timer()
    mus, sigs, As = compactify_hmap_arr(hmagg,floor=.015)
    toc = timeit.default_timer() - tic
    #print "Took {} s to compactify".format(toc)

    sio.savemat(outmatfile, {'mu': mus, 'sig': sigs, 'A': As})
    #print "Saved {}".format(outmatfile)


if __name__ == "__main__":
    main(sys.argv[1:])
