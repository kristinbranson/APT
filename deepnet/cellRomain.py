
# coding: utf-8

# In[11]:

from builtins import range
import romainLegConfig
reload(romainLegConfig)
from romainLegConfig import bottomconf as conf
import multiResData
reload(multiResData)

multiResData.create_tf_record_from_lbl(conf, split=True)

import romainLegConfig
reload(romainLegConfig)
from romainLegConfig import side1conf as conf
import multiResData
reload(multiResData)

multiResData.create_tf_record_from_lbl(conf, split=True)


import romainLegConfig
reload(romainLegConfig)
from romainLegConfig import side2conf as conf
import multiResData
reload(multiResData)

multiResData.create_tf_record_from_lbl(conf, split=True)


##

import PoseTrain
reload(PoseTrain)
from romainLegConfig import bottomconf as conf
import tensorflow as tf

tf.reset_default_graph()
pobj = PoseTrain.PoseTrain(conf)
pobj.baseTrain(restore=True,trainType=0)


##

import PoseTrain
reload(PoseTrain)
from romainLegConfig import side2conf as conf
import tensorflow as tf

tf.reset_default_graph()
pobj = PoseTrain.PoseTrain(conf)
pobj.baseTrain(restore=False,trainType=0)

##

import PoseTrain
reload(PoseTrain)
from romainLegConfig import bottomconf as conf
import tensorflow as tf

tf.reset_default_graph()
pobj = PoseTrain.PoseTrain(conf)
pobj.mrfTrain(restore=True,trainType=0)

##

import PoseTrain
reload(PoseTrain)
from romainLegConfig import side1conf as conf
import tensorflow as tf

tf.reset_default_graph()
pobj = PoseTrain.PoseTrain(conf)
pobj.mrfTrain(restore=False,trainType=0)


## In[4]:

import PoseTrain
reload(PoseTrain)
from PoseTrain import *
import tensorflow as tf

from romainLegConfig import bottomconf as conf

tf.reset_default_graph()
restore = True

self = PoseTrain(conf)

self.createPH()
self.createFeedDict()
self.feed_dict[self.ph['phase_train_base']] = False
self.feed_dict[self.ph['keep_prob']] = 0.5
self.trainType = 0
doBatchNorm = self.conf.doBatchNorm

with tf.variable_scope('base'):
    self.createBaseNetwork(doBatchNorm)
self.cost = tf.nn.l2_loss(self.basePred - self.ph['y'])
self.openDBs()
self.createOptimizer()
self.createBaseSaver()

sess = tf.InteractiveSession()
self.createCursors(sess)
self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
self.restoreBase(sess, restore)

##
for ndx in range(np.random.randint(15)):
    self.updateFeedDict(self.DBType.Val, sess=sess, distort=True)

pred = sess.run(self.basePred,feed_dict=self.feed_dict)
plocs = PoseTools.get_base_pred_locs(pred, conf)

selim = np.random.randint(conf.batch_size)
plt.imshow(self.xs[selim,0,:,:],cmap='gray')
plt.scatter(self.locs[selim,:,0],self.locs[selim,:,1])
plt.scatter(plocs[selim,:,0],plocs[selim,:,1],c='r')