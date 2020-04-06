from __future__ import division
from __future__ import print_function

import numpy as np

import PoseTools as pt

def lblimg2uint8(lblimg):
    im = lblimg[..., [1]]
    im = np.round(255 * (im + 1.0) / 2.0)
    im = im.astype(np.uint8)
    return im


testim = np.zeros((1, 8, 16, 1), dtype=np.uint8)
#testim2 = testim.copy()
testlocs = np.array([[[2., 2.], [11., 5.]]])

bsize, npts, d = testlocs.shape
assert bsize == 1 and d == 2
for ipt in range(npts):
    x, y = testlocs[0, ipt, :]
    testim[0, int(y), int(x), :] = 255
    #testim2[0, (int(y),int(y)+2), (int(x), int(x)+2), :] = 255

testim2 = pt.create_label_images(testlocs, testim.shape[1:3], 1.0, 2.0)
testim2 = lblimg2uint8(testim2)

testim3 = pt.create_label_images(testlocs, testim.shape[1:3], 2.0, 2.0)
testim3 = testim3[..., [0]]
testim3 = np.round(255*(testim3+1.0)/2.0)
testim3 = testim3.astype(np.uint8)

testim_rs2, testlocs_rs2 = pt.scale_images(testim, testlocs, 2, None)
testim_rs3, testlocs_rs3 = pt.scale_images(testim, testlocs, 3, None)
testim_rs4, testlocs_rs4 = pt.scale_images(testim, testlocs, 4, None)

testim2_rs2, testlocs2_rs2 = pt.scale_images(testim2, testlocs, 2, None)
testim2_rs3, testlocs2_rs3 = pt.scale_images(testim2, testlocs, 3, None)
testim2_rs4, testlocs2_rs4 = pt.scale_images(testim2, testlocs, 4, None)

lblim_rs2 = pt.create_label_images(testlocs, testim.shape[1:3], 2.0, 2.0)
lblim_rs3 = pt.create_label_images(testlocs, testim.shape[1:3], 3.0, 2.0)
lblim_rs4 = pt.create_label_images(testlocs, testim.shape[1:3], 4.0, 2.0)
lblim_rs2 = lblimg2uint8(lblim_rs2)
lblim_rs3 = lblimg2uint8(lblim_rs3)
lblim_rs4 = lblimg2uint8(lblim_rs4)



#testim2_rs3, testlocs2_rs3 = pt.scale_images(testim2, testlocs, 3, None)

ax = pt.show_result(testim, [0], testlocs, mrkrsz=45, fignum=11)
ax[0].get_images()[0].set_clim(0,255)
ax = pt.show_result(testim_rs2, [0], testlocs_rs2, mrkrsz=45, fignum=12)
ax[0].get_images()[0].set_clim(0,255)
ax = pt.show_result(testim_rs3, [0], testlocs_rs3, mrkrsz=45, fignum=13)
ax[0].get_images()[0].set_clim(0,255)
ax = pt.show_result(testim_rs4, [0], testlocs_rs4, mrkrsz=45, fignum=14)
ax[0].get_images()[0].set_clim(0,255)

ax = pt.show_result(testim2_rs2, [0], testlocs2_rs2, mrkrsz=45, fignum=15)
#ax[0].get_images()[0].set_clim(0,255)
ax = pt.show_result(testim2_rs3, [0], testlocs2_rs3, mrkrsz=45, fignum=16)
#ax[0].get_images()[0].set_clim(0,255)
ax = pt.show_result(testim2_rs4, [0], testlocs2_rs4, mrkrsz=45, fignum=17)
#ax[0].get_images()[0].set_clim(0,255)

ax = pt.show_result(testim2, [0], testlocs, mrkrsz=45, fignum=18)
ax = pt.show_result(lblim_rs2, [0], testlocs_rs2, mrkrsz=45, fignum=19)
ax = pt.show_result(lblim_rs3, [0], testlocs_rs3, mrkrsz=45, fignum=20)
ax = pt.show_result(lblim_rs4, [0], testlocs_rs4, mrkrsz=45, fignum=21)




