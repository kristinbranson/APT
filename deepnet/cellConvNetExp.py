from __future__ import print_function

# coding: utf-8

# In[1]:

from builtins import range
import tensorflow as tf
from numpy import linalg as LA

nclus = 2
indim = [50]
niter = 500
nsample = 500
lr = 0.01
scale = 0.2
c1 = np.random.randn(indim[0])
c2 = np.random.randn(indim[0])
print((LA.norm(c1.flatten()),LA.norm(c2.flatten())))
print(np.dot(c1.flatten(),c2.flatten()))
sess = tf.InteractiveSession()


# In[4]:

def snet(X,indim,nhidden=2):
    kk = np.random.randn(indim[0],2)
    kk = kk.astype('single')
    weights = tf.Variable(kk)
    biases = tf.Variable(tf.zeros([2]))
    mmult = tf.matmul(X,weights)
    return tf.nn.relu(mmult+biases)

def genSamples(c1,indim,scale,num):
    nn = np.random.randn(num,indim[0])
    nn = nn*scale + c1
    return nn

Xph = tf.placeholder(tf.float32, [None, indim[0]])
label_ph = tf.placeholder(tf.float32,[None,2])
Xout = snet(Xph,indim,nclus)
loss = tf.nn.l2_loss(Xout-label_ph)

opt = tf.train.AdamOptimizer(learning_rate= lr).minimize(loss)

init = tf.initialize_all_variables()
sess.run(init)
for ndx in range(niter):
    s1 = genSamples(c1,indim,scale,nsample)
    s2 = genSamples(c2,indim,scale,nsample)
    s = np.concatenate([s1,s2],0)
    l1 = np.hstack([np.ones([1,nsample]),np.zeros([1,nsample])])
    l2 = np.hstack([np.zeros([1,nsample]),np.ones([1,nsample])])
    l = np.vstack([l1,l2])
    l = np.transpose(l)
    feed_dict={Xph:s, label_ph:l}
    sess.run(opt,feed_dict=feed_dict)
    s1 = genSamples(c1,indim,0.2,nsample)
    s2 = genSamples(c2,indim,0.2,nsample)
    s = np.concatenate([s1,s2],0)
    feed_dict={Xph:s, label_ph:l}
    curloss = sess.run(loss,feed_dict=feed_dict)
    if ndx%50 == 0:
        print((ndx,curloss))
    
    


# In[1]:

import tensorflow as tf
from numpy import linalg as LA

nclus = 2
indim = [50]
niter = 500
nsample = 500
lr = 0.01
scale = 0.2
c1 = np.random.randn(indim[0])
c2 = np.random.randn(indim[0])
rdir = np.random.randn(indim[0])
tres = np.dot(rdir,c2-c1)
print((LA.norm(c1.flatten()),LA.norm(c2.flatten())))
print(np.dot(c1.flatten(),c2.flatten()))
print(np.dot(rdir.flatten(),c2.flatten()))
print(np.dot(rdir.flatten(),c1.flatten()))


def snet(X,indim,nhidden=2):
    weights = tf.Variable(tf.random_normal(indim+[2],stddev=0.2)),
    biases = tf.Variable(tf.zeros([2]))
    mmult = tf.matmul(X,weights)
    return tf.nn.relu(mmult+biases)


def genSamples(c1,indim,scale,num):
    nn = np.random.randn(num,indim[0])
    nn = nn*scale + c1
    return nn

Xph = tf.placeholder(tf.float32, [None, indim[0]])
label_ph = tf.placeholder(tf.float32,[None,2])
Xout = snet(Xph,indim,nclus)
loss = tf.nn.l2_loss(Xout-label_ph)

opt = tf.train.AdamOptimizer(learning_rate= lr).minimize(loss)

sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
sess.run(init)
for ndx in range(niter):
    s = genSamples(np.zeros(indim[0]),indim,scale,nsample)
#     s2 = genSamples(c2,indim,scale,nsample)
#     s = np.concatenate([s1,s2],0)
#     l1 = np.hstack([np.ones([1,nsample]),np.zeros([1,nsample])])
#     l2 = np.hstack([np.zeros([1,nsample]),np.ones([1,nsample])])
#     l = np.vstack([l1,l2])
#     l = np.transpose(l)
    l = 1.0*(s[:,0:1]>0)
    l = np.concatenate([l,l],1)
    feed_dict={Xph:s, label_ph:l}
    sess.run(opt,feed_dict=feed_dict)
    s = genSamples(np.zeros(indim[0]),indim,scale,nsample)
    l = 1.0*(s[:,0:1]>0)
    l = np.concatenate([l,l],1)
    feed_dict={Xph:s, label_ph:l}
    curloss = sess.run(loss,feed_dict=feed_dict)
    if ndx%50 == 0:
        print((ndx,curloss))
    
    


# In[4]:

aa = np.zeros(3)
bb = np.hstack([aa,aa])
print(bb)

gg = np.dot(s,rdir[:,np.newaxis])
print(gg.shape)

