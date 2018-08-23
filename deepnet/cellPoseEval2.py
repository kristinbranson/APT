from __future__ import division
from __future__ import print_function

# coding: utf-8

# In[20]:

from builtins import zip
from builtins import range
from past.utils import old_div
import poseEval2 
reload(poseEval2)
import romainLegConfig
reload(romainLegConfig)
from romainLegConfig import bottomconf as conf
import tensorflow as tf

tf.reset_default_graph()
#%lprun -f poseEval2.poseEvalTrain poseEval2.poseEvalTrain(conf)
poseEval2.poseEvalTrain(conf,restore=False)


# In[1]:

import tensorflow as tf
sess = tf.InteractiveSession()


# In[1]:

# Interactive plots from
# http://matplotlib.1069221.n5.nabble.com/how-to-create-interactive-plots-in-jupyter-python3-notebook-td46804.html
get_ipython().magic(u'pylab notebook')
import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='.*/ipykernel/.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='.*/widgets/.*')


# In[1]:

import poseShape
reload(poseShape)
import romainLegConfig
reload(romainLegConfig)
from romainLegConfig import bottomconf as conf
import tensorflow as tf

tf.reset_default_graph()
poseShape.pose_shape_train(conf,restore=False)


# In[1]:

import poseEval2
reload(poseEval2)
from poseEval2 import *
import tensorflow as tf

from romainLegConfig import bottomconf as conf

tf.reset_default_graph()

restore = False

ph,feed_dict,out,queue,out_dict = poseEvalNetInit(conf)
feed_dict[ph['phase_train']] = False
evalSaver = createEvalSaver(conf) 

loss = tf.nn.l2_loss(out-ph['y'])
correct_pred = tf.cast(tf.equal(out>0.5,ph['y']>0.5),tf.float32)
accuracy = tf.reduce_mean(correct_pred)


#     tf.summary.scalar('cross_entropy',loss)
#     tf.summary.scalar('accuracy',accuracy)

opt = tf.train.AdamOptimizer(learning_rate=ph['learning_rate']).minimize(loss)

merged = tf.summary.merge_all()


sess = tf.InteractiveSession()
data,coord,threads = createCursors(sess,queue,conf)
nlocs,locs,xs = updateFeedDict(conf,'train',distort=True,sess=sess,data=data,feed_dict=feed_dict,ph=ph)

evalstartat = restoreEval(sess,evalSaver,restore,conf,feed_dict)


# In[2]:

count = 50
ll = np.zeros((count,)+feed_dict[ph['y']].shape)
oo = np.zeros((count,)+feed_dict[ph['y']].shape)

ims = []
ims1 = []
alllocs = []
for ndx in range(count):
    nlocs,locs,xs = updateFeedDict(conf,'train',distort=True,sess=sess,data=data,feed_dict=feed_dict,ph=ph)
    ll[ndx,...] = feed_dict[ph['y']]
    oo[ndx,...] = sess.run(out,feed_dict=feed_dict)
    alllocs.append(locs)
    ims.append(feed_dict[ph['X'][0][0]])
    ims1.append(feed_dict[ph['X'][1][0]])
   

 
ims = np.array(ims)
ims = np.reshape(ims,[count*8,128,128])
ims1 = np.reshape(np.array(ims1),[count*8,128,128])
locstrain = np.array(alllocs).reshape(count*8,18,2)

ll = np.reshape(ll,[count*8,8])
oo = np.reshape(oo,[count*8,8])

lltrain = ll
ootrain = oo
qqtrain = np.mean(np.sign(lltrain-0.5)==np.sign(ootrain-0.5),axis=0)
vvtrain = np.mean(lltrain,axis=0)
iitrain = np.mean(ootrain>0.5,axis=0)
imstrain = ims
print(vvtrain)

count = 50
ll = np.zeros((count,)+feed_dict[ph['y']].shape)
oo = np.zeros((count,)+feed_dict[ph['y']].shape)

ims = []
alllocs = []
acc = 0.
for ndx in range(count):
    nlocs,locs,xs = updateFeedDict(conf,'val',distort=False,sess=sess,data=data,feed_dict=feed_dict,ph=ph)
    ll[ndx,...] = feed_dict[ph['y']]
    oo[ndx,...],curacc = sess.run([out,accuracy],feed_dict=feed_dict)
    acc += curacc
    ims.append(feed_dict[ph['X'][0][0]])
    alllocs.append(locs)

print(old_div(acc,count)) 
ims = np.array(ims)
ims = np.reshape(ims,[count*8,128,128])
locsval = np.array(alllocs).reshape(count*8,18,2)

ll = np.reshape(ll,[count*8,8])
oo = np.reshape(oo,[count*8,8])

llval = ll
ooval = oo
qqval = np.mean(np.sign(llval-0.5)==np.sign(ooval-0.5),axis=0)
vvval = np.mean(llval,axis=0)
iival = np.mean(ooval>0.5,axis=0)
imsval = ims
print()
print(vvval)


# In[8]:

pptrain = np.where(lltrain[:,3]==1)[0]
nntrain = np.where(lltrain[:,4]==1)[0]

ppval = np.where(llval[:,3]==1)[0]
nnval = np.where(llval[:,4]==1)[0]

cpptrain = np.random.choice(pptrain,20)
cnntrain = np.random.choice(nntrain,20)
cppval = np.random.choice(ppval,20)
cnnval = np.random.choice(nnval,20)

figtrain = plt.figure()
figval = plt.figure()
for ndx in range(20):
    ax = figtrain.add_subplot(4,5,ndx+1)
    ax.imshow(imstrain[cpptrain[ndx],:,:],cmap='gray',vmax=255)
    curl = locstrain[cpptrain[ndx],:,:]
    curl = curl[0,:]-curl[6,:]+64
    ax.scatter(curl[0],curl[1])
    ax.axis('off')
    ax = figval.add_subplot(4,5,ndx+1)
    ax.imshow(imsval[cppval[ndx],:,:],cmap='gray',vmax=255)
    curl = locsval[cppval[ndx],:,:]
    curl = curl[0,:]-curl[6,:]+64
    ax.scatter(curl[0],curl[1])
    ax.axis('off')

figtrain = plt.figure()
figval = plt.figure()
for ndx in range(20):
    ax = figtrain.add_subplot(4,5,ndx+1)
    ax.imshow(imstrain[cnntrain[ndx],:,:],cmap='gray',vmax=255)
    curl = locstrain[cnntrain[ndx],:,:]
    curl = curl[0,:]-curl[6,:]+64
    ax.scatter(curl[0],curl[1])
    ax.axis('off')
    ax = figval.add_subplot(4,5,ndx+1)
    ax.imshow(imsval[cnnval[ndx],:,:],cmap='gray',vmax=255)
    curl = locsval[cnnval[ndx],:,:]
    curl = curl[0,:]-curl[6,:]+64
    ax.scatter(curl[0],curl[1])
    ax.axis('off')

    


# In[4]:

print(vvval)
print()
print(qqval)
print()
print(iival)
print() 
print(vvtrain)
print() 
rates = np.zeros([2,2,8])
for lndx in range(2):
    for ondx in range(2):
        hh1 = np.sign(llval-0.5)== np.sign(lndx-0.5)
        hh2 = np.sign(ooval-0.5)== np.sign(ondx-0.5)
        zzval = np.mean(hh1&hh2,axis=0)
        print(np.sign(lndx-0.5),np.sign(ondx-0.5), zzval[:3])
        rates[lndx,ondx,:] = zzval
        
prec = old_div(rates[1,1,:],(rates[1,1,:]+rates[0,1,:]))
rec = old_div(rates[1,1,:],(rates[1,1,:]+rates[1,0,:]))
fsc = 2*prec*rec/(prec+rec)
print(fsc[:3])


# In[17]:


# ppval = np.where( np.sum(sign(llval-0.5)==sign(ooval-0.5),axis=1)==8)[0]
# nnval = np.where( np.sum(sign(llval-0.5)==sign(ooval-0.5),axis=1)<8)[0]
selpt = 4
radpt = 0
ppval = np.where( (np.sign(llval[:,selpt]-0.5)==1)&(np.sign(ooval[:,selpt]-0.5)!=1))[0]
nnval = np.where( (np.sign(llval[:,selpt]-0.5)!=1)&(np.sign(ooval[:,selpt]-0.5)==1))[0]

selpt1 = 6
selpt2 = selpt1-6

curl = locsval
kk = curl[:,selpt2,:]-curl[:,selpt1,:]
aacur = np.arctan2( kk[:,1], kk[:,0]+1e-5 )*180/np.pi + 180
ddcur = np.sqrt(np.sum(kk**2,axis=1))

cppval = np.random.choice(ppval,20)
cnnval = np.random.choice(nnval,20)


figval = plt.figure(figsize=[10,8])
for ndx in range(20):
    ax = figval.add_subplot(4,5,ndx+1)
    ax.imshow(imsval[cppval[ndx],:,:],cmap='gray')
    ax.axis('off')
    curid = cppval[ndx]
    ax.set_title('{},{:.2f}\n{:.0f},{:.0f}'.format(llval[curid,selpt],
                ooval[curid,selpt],aacur[curid],ddcur[curid]))
    ax.scatter(kk[curid,0]+64,kk[curid,1]+64)

figval = plt.figure(figsize=[10,8])
for ndx in range(20):
    ax = figval.add_subplot(4,5,ndx+1)
    ax.imshow(imsval[cnnval[ndx],:,:],cmap='gray')
    ax.axis('off')
    curid = cnnval[ndx]
    ax.set_title('{},{:.2f}\n{:.0f},{:.0f}'.format(llval[curid,selpt],
                ooval[curid,selpt],aacur[curid],ddcur[curid]))
    ax.scatter(kk[curid,0]+64,kk[curid,1]+64)
print(ppval.shape)
print(nnval.shape)
    
    


# In[16]:

ndx = 15-1
print(ooval[cnnval[ndx],:])


# In[ ]:

float_formatter = lambda x: "%+3.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})


# In[4]:

kk = locstrain[:,0,:]-locstrain[:,6,:]
dd = np.sqrt(np.sum(kk**2,axis=1))
plt.figure()
plt.hist(dd,normed=True,rwidth=0.3)
kk = locsval[:,0,:]-locsval[:,6,:]
dd = np.sqrt(np.sum(kk**2,axis=1))
plt.hist(dd,normed=True,rwidth=0.3)


# In[92]:


selpt1 = 6
selpt2 = selpt1-6

curl = locsval
kk = curl[:,selpt2,:]-curl[:,selpt1,:]
aacur = np.arctan2( kk[:,1], kk[:,0]+1e-5 )*180/np.pi + 180
ddcur = np.sqrt(np.sum(kk**2,axis=1))
ppval = np.where(ddcur>70)[0]
cppval = np.random.choice(ppval,20)

figval = plt.figure(figsize=[10,8])
for ndx in range(20):
    ax = figval.add_subplot(4,5,ndx+1)
    ax.imshow(imsval[cppval[ndx],:,:],cmap='gray')
    ax.axis('off')
    curid = cppval[ndx]
    ax.set_title('{},{:.2f}\n{:.0f},{:.0f},{:.2f}'.format(llval[curid,selpt,radpt],
                ooval[curid,selpt,radpt],aacur[curid],ddcur[curid],ooval[curid,selpt,1-radpt]))
    ax.scatter(kk[curid,0]+64,kk[curid,1]+64)

    


# In[16]:

import re
vv = tf.global_variables()
aa = [v for v in vv if not re.search('Adam|batch_norm|beta|scale[1-2]|scale0_[1-9][0-9]*|fc_[1-9][0-9]*|L[6-7]_[1-9][0-9]*|biases',v.name)]

gg = sess.run(tf.gradients(loss,aa),feed_dict=feed_dict)

kk = sess.run(aa,feed_dict=feed_dict)

ss = [g.std() for g in gg]
ww = [g.std() for g in kk]

rr = [old_div(s,w) for s,w in zip(ss,ww)]



bb = [[r,n.name] for r,n in zip(rr,aa)]
for b,k,g in zip(bb,ss,ww):
    print(b,k,g)


# In[29]:

import re
vv = tf.global_variables()
aa = [v for v in vv if not re.search('Adam|batch_norm|beta|scale[1-2]|scale0_[1-9][0-9]*|fc_[1-9][0-9]*|L[6-7]_[1-9][0-9]*|biases',v.name)]

gg = sess.run(tf.gradients(loss,aa),feed_dict=feed_dict)

kk = sess.run(aa,feed_dict=feed_dict)

ss = [g.std() for g in gg]
ww = [g.std() for g in kk]

rr = [old_div(s,w) for s,w in zip(ss,ww)]



bb = [[r,n.name] for r,n in zip(rr,aa)]
for b,k,g in zip(bb,ss,ww):
    print(b,k,g)


# In[41]:

aas = []
for k in list(out_dict['base_dict_array'][0][0].keys()):
    aas.append(out_dict['base_dict_array'][0][0][k])
    
gg = sess.run(aas,feed_dict=feed_dict)    

kk = list(out_dict['base_dict_array'][0][0].keys())
for k,g in zip(kk,gg):
    f = old_div(float(np.count_nonzero(g.flatten())),g.size)
    print(k,f)


# In[ ]:

zz = np.sum(np.sign(ll-0.5)==np.sign(oo-0.5),axis=(1,2))
pp = np.where(zz>=32)[0]
nn = np.where(zz<30)[0]

cpp = np.random.choice(pp,20)
cnn = np.random.choice(nn,20)

figp = plt.figure()
fign = plt.figure()
for ndx in range(20):
    ax = figp.add_subplot(4,5,ndx+1)
    ax.imshow(ims[cpp[ndx],:,:],cmap='gray')
    ax = fign.add_subplot(4,5,ndx+1)
    ax.imshow(ims[cnn[ndx],:,:],cmap='gray')


# In[11]:

qq = np.mean(np.sign(ll-0.5)==np.sign(oo-0.5),axis=0)
vv = np.mean(ll,axis=0)
ii = np.mean(oo>0.5,axis=0)
print(qq[:,:])
print()
print(vv[:,:])
print()
print(ii[:,:])
print()
print(qq+vv)
print()
print(np.mean( (np.sign(ll-0.5)>0) &(np.sign(oo-0.5)>0),axis=0))
print()
print(np.mean( (np.sign(ll-0.5)<0) &(np.sign(oo-0.5)<0),axis=0))

lla = ll.reshape([50,8,8,4])
ooa = oo.reshape([50,8,8,4])
lla = np.sum(lla,axis=3)
ooa = np.sum(ooa,axis=3)
qq = np.mean(np.sign(lla-0.5)==np.sign(ooa-0.5),axis=(0,1))
vv = np.mean(lla,axis=(0,1))
ii = np.mean(ooa>0.5,axis=(0,1))
print(qq)
print()
print(vv)
print()
print(ii)


# In[9]:

lla = np.sum(np.sign(np.reshape(ll,[50,8,18,8,4])),axis=4)
ooa = np.sum(np.sign(np.reshape(oo,[50,8,18,8,4])),axis=4)
qq = np.mean(np.sign(ll)==np.sign(oo),axis=(0,1))
qq = qq.reshape([18,8,4])
vv = np.mean(ll,axis=(0,1))
vv = vv.reshape([18,8,4])
ii = np.mean(oo>0,axis=(0,1))
ii = ii.reshape([18,8,4])
ff = np.mean(ll,axis=(0,1))
ff = np.mean(np.sign(ff)==np.sign(ll),axis=(0,1))
ff = ff.reshape([18,8,4])
spt = 6
print(qq[spt,:,:])
print(vv[spt,:,:])
print(ii[spt,:,:])
print(ff[spt,:,:])

qq = np.mean(lla==ooa,axis=(0,1))
vv = np.mean(lla,axis=(0,1))
ii = np.mean(ooa,axis=(0,1))
print(qq[spt,:])
print(vv[spt,:])
print(ii[spt,:])


# In[ ]:

for ndx in range(20):
    nlocs,locs,xs = updateFeedDict(conf,'val',distort=True,sess=sess,data=data,feed_dict=feed_dict,ph=ph)


# In[ ]:

nlocs,locs,xs = updateFeedDict(conf,'train',distort=True,sess=sess,data=data,feed_dict=feed_dict,ph=ph)
inCur = feed_dict[ph['y']]
outCur = sess.run(out,feed_dict=feed_dict)

sim = np.random.randint(8)
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

spt = 6
fig = plt.figure()
plt.imshow(xs[sim,0,:,:],cmap='gray')
plt.scatter(locs[sim,:,0],locs[sim,:,1])
iii = inCur[sim,:].reshape([8,4])
jjj = outCur[sim,:].reshape([8,4])
plt.title('{}'.format(sim))
print(iii[:,:])
print() 
print(jjj[:,:])
ddd = np.sqrt(((locs[sim,0,:] - locs[sim,6,:])**2).sum())
vvv = locs[sim,spt,:]-locs[sim,0,:]
aaa = np.arctan2(vvv[1],vvv[0])*180/np.pi
print(ddd,aaa)
fig = plt.figure()
for ndx in range(3):
    ax = fig.add_subplot(1,3,ndx+1)
    X1 = feed_dict[ph['X'][ndx][0]]
    plt.imshow(X1[sim,:,:,0],cmap='gray')


# In[ ]:

pt = 3
im = 3

y = feed_dict[ph['y']]
print(y[im,:].reshape([3,6]))

plt.figure()
plt.imshow(xs[im,0,...],cmap='gray')
plt.scatter(locs[im,:,0]+5,locs[im,:,1]+5,hold=True)
plt.scatter(nlocs[im,:,0],nlocs[im,:,1],hold=True,c='r')

Z = feed_dict[ph['X'][0][pt]]
plt.figure()
plt.imshow(Z[im,...,0],cmap='gray')
Z = feed_dict[ph['X'][1][pt]]
plt.figure()
plt.imshow(Z[im,...,0],cmap='gray')
Z = feed_dict[ph['X'][2][pt]]
plt.figure()
plt.imshow(Z[im,...,0],cmap='gray')
print(feed_dict[ph['S'][0][pt]][im,...].reshape(18,2))


# In[ ]:

import poseEval2
reload(poseEval2)
from poseEval2 import *
import tensorflow as tf

from romainLegConfig import bottomconf as conf

tf.reset_default_graph()

restore = True

ph,feed_dict,out,queue,_ = poseEvalNetInit(conf)
feed_dict[ph['phase_train']] = False
evalSaver = createEvalSaver(conf) 

loss = tf.nn.l2_loss(out-ph['y'])
correct_pred = tf.cast(tf.equal(out>0,ph['y']>0),tf.float32)
accuracy = tf.reduce_mean(correct_pred)

opt = tf.train.AdamOptimizer(learning_rate=                   ph['learning_rate']).minimize(loss)

sess = tf.InteractiveSession()
data,coord,threads = createCursors(sess,queue,conf)

evalstartat = restoreEval(sess,evalSaver,restore,conf,feed_dict)


# In[ ]:

import copy

# patch and locs both distorted
nlocs,locs,xs = updateFeedDict(conf,'train',distort=True,sess=sess,data=data,feed_dict=feed_dict,ph=ph)
curout1 = sess.run(out,feed_dict=feed_dict)
y = copy.deepcopy(feed_dict[ph['y']])

# only patch distorted
alllocs = nlocs
dd = alllocs-locs # distance of neg points to actual locations
ddist = np.sqrt(np.sum(dd**2,axis=2))
ind_labels = ddist/conf.eval_minlen/2
ind_labels = 1-2*ind_labels
ind_labels = ind_labels.clip(min=-1,max=1)
psz = conf.poseEval2_psz
x0,x1,x2 = PoseTools.multi_scale_images(xs.transpose([0, 2, 3, 1]),
                                        conf.rescale, conf.scale, conf.l1_cropsz, conf)
for ndx in range(conf.n_classes):
    feed_dict[ph['X'][0][ndx]] = extract_patches(x0,alllocs[:,ndx,:],psz)
    feed_dict[ph['X'][1][ndx]] = extract_patches(x1,old_div((alllocs[:,ndx,:]),conf.scale),psz)
    feed_dict[ph['X'][2][ndx]] = extract_patches(x2,old_div((alllocs[:,ndx,:]),(conf.scale**2)),psz)
    feed_dict[ph['S'][0][ndx]] = np.reshape(locs-locs[:,ndx:ndx+1,:],[conf.batch_size,2*conf.n_classes])
feed_dict[ph['y']] = ind_labels
curout2 = sess.run(out,feed_dict=feed_dict)

# only loc distorted
alllocs = locs
dd = alllocs-locs # distance of neg points to actual locations
ddist = np.sqrt(np.sum(dd**2,axis=2))
ind_labels = ddist/conf.eval_minlen/2
ind_labels = 1-2*ind_labels
ind_labels = ind_labels.clip(min=-1,max=1)
psz = conf.poseEval2_psz
x0,x1,x2 = PoseTools.multi_scale_images(xs.transpose([0, 2, 3, 1]),
                                        conf.rescale, conf.scale, conf.l1_cropsz, conf)
for ndx in range(conf.n_classes):
    feed_dict[ph['X'][0][ndx]] = extract_patches(x0,alllocs[:,ndx,:],psz)
    feed_dict[ph['X'][1][ndx]] = extract_patches(x1,old_div((alllocs[:,ndx,:]),conf.scale),psz)
    feed_dict[ph['X'][2][ndx]] = extract_patches(x2,old_div((alllocs[:,ndx,:]),(conf.scale**2)),psz)
    feed_dict[ph['S'][0][ndx]] = np.reshape(nlocs-nlocs[:,ndx:ndx+1,:],[conf.batch_size,2*conf.n_classes])
feed_dict[ph['y']] = ind_labels
curout3 = sess.run(out,feed_dict=feed_dict)

# none distorted
alllocs = locs
dd = alllocs-locs # distance of neg points to actual locations
ddist = np.sqrt(np.sum(dd**2,axis=2))
ind_labels = ddist/conf.eval_minlen/2
ind_labels = 1-2*ind_labels
ind_labels = ind_labels.clip(min=-1,max=1)
psz = conf.poseEval2_psz
x0,x1,x2 = PoseTools.multi_scale_images(xs.transpose([0, 2, 3, 1]),
                                        conf.rescale, conf.scale, conf.l1_cropsz, conf)
for ndx in range(conf.n_classes):
    feed_dict[ph['X'][0][ndx]] = extract_patches(x0,alllocs[:,ndx,:],psz)
    feed_dict[ph['X'][1][ndx]] = extract_patches(x1,old_div((alllocs[:,ndx,:]),conf.scale),psz)
    feed_dict[ph['X'][2][ndx]] = extract_patches(x2,old_div((alllocs[:,ndx,:]),(conf.scale**2)),psz)
    feed_dict[ph['S'][0][ndx]] = np.reshape(locs-locs[:,ndx:ndx+1,:],[conf.batch_size,2*conf.n_classes])
feed_dict[ph['y']] = ind_labels

curout4 = sess.run(out,feed_dict=feed_dict)


# In[ ]:

print(alllocs.shape)
print(y[3,:])
print(ind_labels[3,:])


# In[ ]:

print(np.random.randint(3))


# In[ ]:

print(y[3,:])


# In[ ]:

for im in range(8):

    print(y[im,:].reshape([3,6]))
    print() 
    print(curout1[im,:].reshape([3,6]))
    print() 
    print(curout2[im,:].reshape([3,6]))
    print() 
    print(curout3[im,:].reshape([3,6]))
    print() 
    print(curout4[im,:].reshape([3,6]))

    plt.figure()
    plt.imshow(xs[im,0,...],cmap='gray')
    plt.scatter(locs[im,:,0]+5,locs[im,:,1]+5,hold=True)
    plt.scatter(nlocs[im,:,0],nlocs[im,:,1],hold=True,c='r')


# In[ ]:

import h5py
L = h5py.File(conf.labelfile,'r')

if 'pts' in L:
    pts = np.array(L['pts'])
    v = conf.view
else:
    pp = np.array(L['labeledpos'])
    nmovie = pp.shape[1]
    pts = np.zeros([0,conf.n_classes,2])
    v = 0
    for ndx in range(nmovie):
        curpts = np.array(L[pp[0,ndx]])
        frames = np.where(np.invert( np.any(np.isnan(curpts),axis=(1,2))))[0]
        nptsPerView = np.array(L['cfg']['NumLabelPoints'])[0,0]
        pts_st = int(conf.view*nptsPerView)
        selpts = pts_st + conf.selpts
        curlocs = curpts[:,:,selpts]
        curlocs = curlocs[frames,:,:]
        curlocs = curlocs.transpose([0,2,1])
        pts = np.append(pts,curlocs[:,:,:],axis=0)



# In[18]:

allxs = []
alllocs = []
for ndx in range(100):
    xs,locs = readImages(conf,'train',False,sess,data)
    allxs.append(xs)
    alllocs.append(locs)
allxs = np.array(allxs)
alllocs = np.array(alllocs)

allxs = allxs.reshape([8*100,592,256,1])
alllocs = alllocs.reshape([8*100,18,2])


# In[44]:

selpt1 = 6
selpt2 = selpt1-6
shp = shape_from_locs(alllocs)
ptchs = extract_patches(allxs,alllocs[:,selpt1,:],128) 
print(ptchs.shape)


# In[32]:

kk = np.zeros(12)
for ndx1 in range(12):
    ndx2 = ndx1+6
    lbls = shp[:,ndx1,ndx2,:]
    mm = np.mean(lbls,axis=0)
    kk[ndx1] = np.count_nonzero(mm>0.1)
print(kk)    
print(mm.shape)


# In[46]:

print(lbls.mean(axis=0))


# In[45]:

lbls = shp[:,selpt1,selpt2,:]

plt.figure()
pp = lbls[:,0,0]==1
plt.imshow(np.mean(ptchs[pp,:,:,0],axis=0),cmap='gray',vmax=255)
plt.figure()
pp = lbls[:,1,0]==1
plt.imshow(np.mean(ptchs[pp,:,:,0],axis=0),cmap='gray',vmax=255)
plt.figure()
pp = lbls[:,2,0]==1
plt.imshow(np.mean(ptchs[pp,:,:,0],axis=0),cmap='gray',vmax=255)

