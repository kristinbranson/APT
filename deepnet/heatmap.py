import sys
import time
import timeit
import h5py
from scipy.stats import multivariate_normal
import scipy.io as sio
import skimage.measure
import numpy as np

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
    # mu: for romain, if hm is (720,540), then mu[0] is x

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

def main(argv):
    inmatfile = argv[0]
    outmatfile = argv[1]

    tic = time.time()
    hm = h5py.File(inmatfile,'r')
    hmagg = hm['hmagg']
    toc = time.time() - tic
    print(" Took {} s to load".format(toc))

    tic = time.time()
    hmagg = np.array(hmagg)
    toc = time.time() - tic
    print("Took {} s to get hm array".format(toc))

    tic = timeit.default_timer()
    mus, sigs, As = compactify_hmap_arr(hmagg,floor=.015)
    toc = timeit.default_timer() - tic
    print("Took {} s to compactify".format(toc))

    sio.savemat(outmatfile, {'mu': mus, 'sig': sigs, 'A': As})
    print("Saved {}".format(outmatfile))


if __name__ == "__main__":
    main(sys.argv[1:])
