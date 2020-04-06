from __future__ import print_function

# coding: utf-8

# In[1]:

from builtins import range
import tensorflow as tf

efile = '/home/mayank/work/poseEstimation/cache/romainLegBottom/eval_test_summary/events.out.tfevents.1484832405.mayankWS'


# In[2]:


vals = []
names = []
first = True
for aa in tf.train.summary_iterator(efile):
    if not len(aa.summary.value):
        continue
    if first:
        for ndx,bb in enumerate(aa.summary.value):
            vals.append([])
            names.append(bb.tag)
        first = False
    for ndx,bb in enumerate(aa.summary.value):
        vals[ndx].append(bb.simple_value)


# In[3]:

for n in names:
    print(n)


# In[4]:

for ndx in range(50,len(names)):
    plt.figure()
    plt.plot(vals[ndx])
    plt.title(names[ndx])


# In[8]:

ndx = 51
plt.figure()
plt.plot(vals[ndx][2:])
plt.title(names[ndx])

