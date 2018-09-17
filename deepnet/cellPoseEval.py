from __future__ import division
from __future__ import print_function

# coding: utf-8

# In[4]:

# Interactive plots from
# http://matplotlib.1069221.n5.nabble.com/how-to-create-interactive-plots-in-jupyter-python3-notebook-td46804.html
from builtins import zip
from builtins import range
from past.utils import old_div
get_ipython().magic(u'pylab notebook')
import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='.*/ipykernel/.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='.*/widgets/.*')


# In[ ]:

import poseEval
reload(poseEval)
import romainLegConfig
reload(romainLegConfig)
from romainLegConfig import bottomconf as conf
# from romainLegConfig import side1conf as conf
import tensorflow as tf
tf.reset_default_graph()
poseEval.poseEvalTrain(conf,restore=False)


# In[1]:

# Gradient analysis
import poseEval
reload(poseEval)
from poseEval import *
import tensorflow as tf
from romainLegConfig import bottomconf as conf

useNet = True
restore = True

tf.reset_default_graph()
ph,feed_dict,out,queue,out_dict = poseEvalNetInit(conf)
feed_dict[ph['phase_train']] = True
feed_dict[ph['keep_prob']] = 1.
evalSaver = createEvalSaver(conf) 
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(out,ph['y'])
correct_pred = tf.equal(tf.argmax(out,1),tf.argmax(ph['y'],1))

loss = tf.reduce_mean(cross_entropy)
tf.summary.scalar('cross_entropy',loss)

opt = tf.train.AdamOptimizer(learning_rate=                   ph['learning_rate']).minimize(loss)

merged = tf.summary.merge_all()


sess =  tf.InteractiveSession()
data,coord,threads = createCursors(sess,queue,conf)
updateFeedDict(conf,'train',distort=True,sess=sess,data=data,feed_dict=feed_dict,ph=ph)
if useNet:
    evalstartat = restoreEval(sess,evalSaver,restore,conf,feed_dict)
    initializeRemainingVars(sess,feed_dict)

            #* conf.gamma**math.floor(excount/conf.step_size)
feed_dict[ph['learning_rate']] = 0
feed_dict[ph['keep_prob']] = 1.
feed_dict[ph['phase_train']] = False



# In[2]:

# Interactive plots from
# http://matplotlib.1069221.n5.nabble.com/how-to-create-interactive-plots-in-jupyter-python3-notebook-td46804.html
get_ipython().magic(u'pylab notebook')
import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='.*/ipykernel/.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='.*/widgets/.*')

for ndx in range(np.random.randint(50)):
    alllocs = updateFeedDict(conf,'val',distort=True,sess=sess,data=data,feed_dict=feed_dict,ph=ph)
if useNet:
    oo,cc = sess.run([out,cross_entropy],feed_dict=feed_dict)
else:
    oo = feed_dict[ph['y']]
    cc = np.zeros([8,])

ims = feed_dict[ph['X'][0]]

tt = oo.reshape(8,18)
tt = tt.clip(min=0,max=1)
vv = feed_dict[ph['y']].reshape(8,18)
pp = np.abs(tt-vv)
nc = 2; nr = 4
fig = plt.figure(figsize=[12,25])
mrk = ['o','*','+']
ss = [30,100,50]
for idx in range(ims.shape[0]):
    ax = fig.add_subplot(nr,nc,idx+1)
    ax.imshow(ims[idx,:,:,0],cmap='gray')
    tstr = []
    for jj in range(3):
        sz = ss[jj]*(3*pp[idx,jj*6:(jj+1)*6]+0.5)
        ax.scatter(alllocs[idx,jj*6:(jj+1)*6,0],alllocs[idx,jj*6:(jj+1)*6,1],
                   c=np.linspace(0,1,6),cmap=cm.jet,marker=mrk[jj],s=sz)
        tstr.append(','.join('{:.2f}'.format(a) for a in tt[idx,jj*6:(jj+1)*6]))
        tstr.append(','.join('{:.2f}'.format(a) for a in vv[idx,jj*6:(jj+1)*6]))
    ax.set_title('\n'.join(tstr))


# In[3]:

# measure different accuracies

pp = np.zeros([800,18])
ii = np.zeros([800,18])
count = 100
for ndx in range(count):
    alllocs = updateFeedDict(conf,'val',distort=True,sess=sess,data=data,feed_dict=feed_dict,ph=ph)
    oo,cc = sess.run([out,cross_entropy],feed_dict=feed_dict)
    oo = oo.reshape(8,18)
    vv = feed_dict[ph['y']].reshape(8,18)
    pp[ndx*8:(ndx+1)*8,:] = oo-vv
    ii[ndx*8:(ndx+1)*8,:] = oo
 


# In[10]:

qq = np.zeros([800,18])
ff = np.round(qq).astype('int')
ff[1:5,1:5]


# In[4]:

kk = np.abs(pp).mean(axis=0)
print(kk)


# In[5]:

plt.figure()
plt.hist(pp.flatten())
plt.figure()
plt.hist(ii.flatten())


# In[ ]:

vv = tf.global_variables()



import re
aa = [v for v in vv if not re.search('Adam|batch_norm|beta',v.name)]
gg = sess.run(tf.gradients(loss,aa),feed_dict=feed_dict)

kk = sess.run(aa,feed_dict=feed_dict)

ss = [np.sum(np.abs(g.flatten())) for g in gg]
ww = [np.sum(np.abs(g.flatten())) for g in kk]

rr = [old_div(s,w) for s,w in zip(ss,ww)]


# In[ ]:

#for new
bb = [[r,n.name] for r,n in zip(rr,aa)]
for b in bb:
    print(b)
bbnew = bb


# In[ ]:

#for reloaded
bb = [[r,n.name] for r,n in zip(rr,aa)]
for b in bb:
    print(b)


# In[ ]:

for v in vv:
    if re.search('beta|gamma',v.name):
        print(v.eval())


# In[ ]:

import re
aa = [v for v in vv if not re.search('Adam|batch_norm|beta',v.name)]
gg = sess.run(tf.gradients(loss,aa),feed_dict=feed_dict)

kk = sess.run(aa,feed_dict=feed_dict)


# In[ ]:

ss = [np.sum(np.abs(g.flatten())) for g in gg]
ww = [np.sum(np.abs(g.flatten())) for g in kk]


# In[ ]:

rr = [old_div(s,w) for s,w in zip(ss,ww)]

bb = [[r,n.name] for r,n in zip(rr,aa)]
for b in bb:
    print(b)


# In[ ]:

plt.hist(rr)


# In[83]:

from romainLegConfig import bottomconf as conf
import h5py

L = h5py.File(conf.labelfile,'r')
pts = np.array(L['labeledpos'])
curpts = np.array(L[pts[0,0]])
frames = np.where(np.invert( np.any(np.isnan(curpts[:,:,:]),axis=(1,2))))[0]
loc = curpts[frames[0],:,:]

print(loc.shape)    


# In[84]:

jj = curpts[frames,:,:]
print(jj.shape)


# In[101]:

kk = np.logspace(4,10,3,base=2)
kk = np.concatenate([[0,],kk,[np.inf,]])
# kk = np.insert(kk,0,0)
# kk = np.insert(kk,np.inf,-1)
print(kk)
jj = np.random.random(8)*1200
hh = np.digitize(jj,kk)
print(hh)
print(jj)


# In[104]:

pp = np.random.random([5,12])
pp = np.reshape(pp,[6,-1])
pp.shape

