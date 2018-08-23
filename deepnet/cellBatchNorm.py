from __future__ import print_function

# coding: utf-8

# In[1]:

import tensorflow as tf

beta = tf.Variable(tf.constant(0.0, shape=[3]),
    name='beta', trainable=True)
gamma = tf.Variable(tf.constant(1.0, shape=[3]),
    name='gamma', trainable=True)
ema = tf.train.ExponentialMovingAverage(decay=0.9)
beta1 = tf.Variable(tf.constant(0.0, shape=[3]),
    name='beta1', trainable=True)
gamma1 = tf.Variable(tf.constant(1.0, shape=[3]),
    name='gamma1', trainable=True)

ema_apply_op = ema.apply([beta, gamma])
ema_mean, ema_var = ema.average(beta), ema.average(gamma)

vv = ema.variables_to_restore()
for v in vv:
    print(v)
    
# ema1 = tf.train.ExponentialMovingAverage(decay=0.999)
# ema_apply_op = ema1.apply([beta1, gamma1])
# ema_mean1, ema_var1 = ema.average(beta1), ema.average(gamma1)


for v in vv:
    print(v)

