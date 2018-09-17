
import sys
import os
from easydict import EasyDict as edict

sys.path.append('pose-tensorflow')
sys.path.append('../poseTF')
import train
from poseConfig import config as conf
import config
import  logging
import tensorflow as tf
import predict
from dataset.factory import create as create_dataset
import tensorflow as tf
from dataset.pose_dataset import Batch
from matplotlib import pyplot as plt
import  numpy as np

reload(config)
conf.n_classes = 5
conf.cachedir = '/groups/branson/bransonlab/mayank/PoseTF/cache/apt_interface/DoubPend_view0'
conf.dlc_rescale = 1
kk = edict(conf.__dict__)
cc = config.convert_to_deepcut(conf)
cc.dataset = '/groups/branson/home/kabram/PycharmProjects/deepcut/Generating_a_Training_Set/UnaugmentedDataSet_stephen-trainJan30/stephen-train_stephen100shuffle1.mat'
cc.cachedir = '/groups/branson/bransonlab/mayank/PoseTF/cache/apt_interface/dlc'
cc.init_weights = '/groups/branson/home/kabram/PycharmProjects/deepcut/pose-tensorflow/models/pretrained/resnet_v1_50.ckpt'
cc.dlc_steps = 20000
cc.display_steps = 50
cc.save_step = 2000
cc.expname = 'test'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cc.imgDim = 1
doTrain = True
doTest = False
##
logging.getLogger().setLevel(logging.INFO)
if doTrain:
    tf.reset_default_graph()
    reload(train)
    train.train(cc)


##
if doTest:
    cfg = cc

    name = 'deeplabcut'
    os.path.join( cfg.cachedir, cfg.expname + '_' + name)

    ckpt_file = os.path.join(cfg.cachedir,'checkpoint')
    latest_ckpt = tf.train.get_checkpoint_state( cfg.cachedir, ckpt_file)

    cfg['init_weights'] = latest_ckpt.model_checkpoint_path

    dataset = create_dataset(cfg)

    tf.reset_default_graph()
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)

    batch_np = dataset.next_batch()

    cur_out = sess.run(outputs, feed_dict={inputs: batch_np[Batch.inputs]})

    scmap, locref = predict.extract_cnn_output(cur_out, cfg)

    # Extract maximum scoring location from the heatmap, assume 1 person
    loc_pred = predict.argmax_pose_predict(scmap, locref, cfg.stride)
    loc_in = batch_np[Batch.locs]
    dd = np.sqrt(np.sum(np.square(loc_pred[:, :, :2] - loc_in), axis=-1))

    ims = batch_np[Batch.inputs]
    locs = batch_np[Batch.locs]
    in_hmap = batch_np[Batch.part_score_targets]

    f, ax = plt.subplots(1,3)
    ix = np.random.randint(cfg.batch_size)
    dx = np.random.randint(cfg.n_classes)
    ax = ax.flatten()
    mm = np.zeros(scmap.shape[1:3]+(3,))
    mm[:,:,0] = scmap[ix,:,:,dx]
    mm[:,:,1] = in_hmap[ix,:,:,dx]/5
    ax[0].imshow(ims[ix,:,:,0],'gray')
    ax[0].scatter(locs[ix,:,0],locs[ix,:,1])
    ax[1].imshow(mm)
    ax[2].imshow(in_hmap[ix,:,:,dx])
