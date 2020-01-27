import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import APT_interface
import poseConfig
import PoseTools

conf = poseConfig.config()
conf.brange = [0,0]
conf.crange = [1,1]
conf.rrange = 30
conf.trange = 10
conf.scale_factor_range = 4.
print('brange = '+str(conf.brange))
print('crange = '+str(conf.crange))
print('rrange = '+str(conf.rrange))
print('trange = '+str(conf.trange))

print('scale_factor_range = %f'%conf.scale_factor_range)

#lblfile = '20191218T135035_20191218T135248.lbl'

imsz = (20,20) #height, width
patchsz = 1
imszxy = (imsz[1],imsz[0])
conf.imsz = imsz

im = np.zeros(imsz+(3,))
nparts = 3
pts = np.floor(np.array(imszxy)/4.+np.random.rand(nparts,2)*np.array(imszxy)/2.)
print('pts = ' + str(pts))
for c in range(3):
    for xoff in range(-patchsz,patchsz+1):
        for yoff in range(-patchsz,patchsz+1):
            im[pts[:,1].astype(int)+yoff,pts[:,0].astype(int)+xoff,c] = 1.

imt,ptst = PoseTools.randomly_affine(im.reshape((1,)+im.shape),pts.reshape((1,nparts,2)),conf)
imt.shape = im.shape
ptst.shape = pts.shape

plt.subplot(121)
plt.imshow(im)
plt.plot(pts[:,0],pts[:,1],'ro',fillstyle='none')
plt.title('original image and coords')

plt.subplot(122)
plt.imshow(imt)
plt.plot(ptst[:,0],ptst[:,1],'ro',fillstyle='none')
plt.title('transformed image and coords')

plt.show()

