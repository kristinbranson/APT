"""
Modified by Mayank Kabra

Adapted from DeepLabCut2.0 Toolbox (deeplabcut.org)
copyright A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow

"""
import logging, os
import threading
import argparse
from pathlib import Path
import tensorflow as tf
#vers = (tf.__version__).split('.')
TF=tf.compat.v1
import tf_slim as slim
#from tf_slim.nets import resnet_v1

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import Batch
from deeplabcut.pose_estimation_tensorflow.dataset.factory import create as create_dataset
from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net
from deeplabcut.pose_estimation_tensorflow.util.logging import setup_logging
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.utils.auxfun_videos import imread, imresize
import numpy as np
import json
import pickle
import PoseTools
import random
import time


class LearningRate(object):
    def __init__(self, cfg):
        self.steps = cfg.multi_step
        self.current_step = 0

    def get_lr(self, iteration):
        lr = self.steps[self.current_step][0]
        if iteration == self.steps[self.current_step][1]:
            self.current_step += 1

        return lr

def get_batch_spec(cfg):
    num_joints = cfg.num_joints
    batch_size = cfg.batch_size
    return {
        Batch.inputs: [batch_size, None, None, 3],
        Batch.part_score_targets: [batch_size, None, None, num_joints],
        Batch.part_score_weights: [batch_size, None, None, num_joints],
        Batch.locref_targets: [batch_size, None, None, num_joints * 2],
        Batch.locref_mask: [batch_size, None, None, num_joints * 2],
        Batch.locs: [batch_size, 1, None, 2]
    }

def setup_preloading(batch_spec):
    placeholders = {name: TF.placeholder(tf.float32, shape=spec) for (name, spec) in batch_spec.items()}
    names = placeholders.keys()
    placeholders_list = list(placeholders.values())

    QUEUE_SIZE = 20
    q = tf.queue.FIFOQueue(QUEUE_SIZE, [tf.float32]*len(batch_spec))
    enqueue_op = q.enqueue(placeholders_list)
    batch_list = q.dequeue()

    batch = {}
    for idx, name in enumerate(names):
        batch[name] = batch_list[idx]
        batch[name].set_shape(batch_spec[name])
    return batch, enqueue_op, placeholders, q


def load_and_enqueue(sess, enqueue_op, coord, dataset, placeholders):
    while not coord.should_stop():
        batch_np = dataset.next_batch()
        food = {pl: batch_np[name] for (name, pl) in placeholders.items()}
        try:
            sess.run(enqueue_op, feed_dict=food)
        except tf.errors.CancelledError:
            # Just ignore this error
            logging.debug("Ignoring tf.errors.CancelledError in load_and_enqueue()")
            pass


def start_preloading(sess, enqueue_op, dataset, placeholders):
    coord = TF.train.Coordinator()

    t = threading.Thread(target=load_and_enqueue,
                         args=(sess, enqueue_op, coord, dataset, placeholders))
    t.start()

    return coord, t

def get_optimizer(loss_op, cfg):
    learning_rate = TF.placeholder(tf.float32, shape=[])

    if cfg.optimizer == "sgd":
        optimizer = TF.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif cfg.optimizer == "adam":
        optimizer = TF.train.AdamOptimizer(learning_rate)
    else:
        raise ValueError('unknown optimizer {}'.format(cfg.optimizer))
    train_op = slim.learning.create_train_op(loss_op, optimizer)

    return learning_rate, train_op


def save_td(cfg, train_info):
    cachedir = str(Path(cfg.snapshot_prefix).parent)
    name = Path(cfg.snapshot_prefix).stem
    if name == 'deepnet':
        train_data_file = os.path.join(cachedir, 'traindata')
    else:
        train_data_file = os.path.join(cachedir, cfg.expname + '_' + name + '_traindata')

    # train_data_file = os.path.join( cfg.cachedir, 'traindata')
    json_data = {}
    for x in train_info.keys():
        json_data[x] = np.array(train_info[x]).astype(np.float64).tolist()
    with open(train_data_file + '.json', 'w') as json_file:
        json.dump(json_data, json_file)
    with open(train_data_file, 'wb') as train_data_file:
        pickle.dump([train_info, cfg], train_data_file, protocol=2)


def get_read_fn(cfg_dict):
    # Adapted from pose_defaultdataset.next_batch
    cfg = create_cfg(cfg_dict)
    cfg.batch_size = 1
    cfg.shuffle = False

    dataset = create_dataset(cfg)

    n = dataset.num_images
    def read_fn():
        imidx, mirror = dataset.next_training_sample()
        data_item = dataset.get_training_sample(imidx)
        im_file = data_item.im_path
        if not os.path.isabs(im_file):
            ims = imread(os.path.join(cfg.project_path,im_file), mode='RGB')
        else:
            ims = imread(im_file,mode='RGB')
        joints = np.copy(data_item.joints)
        loc_in = joints[0,:,1:]
        ims = ims[np.newaxis,...]

        # scale = cfg.global_scale
        # batch_np = dataset.make_batch(data_item, scale, False)

        if cfg.img_dim == 1:
            ims = ims[:,:,:,0:1]
        info = [0, 0, 0]
        return ims, loc_in, info

    return read_fn, n



def train(cfg_dict,displayiters,saveiters,maxiters,max_to_keep=5,keepdeconvweights=True,allow_growth=False):

    random.seed(3)
    start_path=os.getcwd()
    # os.chdir(str(Path(config_yaml).parents[0])) #switch to folder of config_yaml (for logging)
    setup_logging()
    cfg = create_cfg(cfg_dict)

    if cfg.dataset_type=='default' or cfg.dataset_type=='tensorpack' or cfg.dataset_type=='deterministic':
        print("Switching batchsize to 1, as default/tensorpack/deterministic loaders do not support batches >1. Use imgaug loader.")
        cfg['batch_size']=1 #in case this was edited for analysis.-

    dataset = create_dataset(cfg)
    start = time.time()
    for n in range(100):
        kk = dataset.next_batch() # for debugging
    logging.info('Time for inputting {}'.format( (time.time()-start)/1000))
    batch_spec = get_batch_spec(cfg)
    batch, enqueue_op, placeholders, q = setup_preloading(batch_spec)
    net = pose_net(cfg)
    losses = net.train(batch)
    total_loss = losses['total_loss']
    outputs = net.heads
    train_info = {'train_dist':[],'train_loss':[],'val_dist':[],'val_loss':[],'step':[]}

    for k, t in losses.items():
        TF.summary.scalar(k, t)
    merged_summaries = TF.summary.merge_all()


    if 'snapshot' in Path(cfg.init_weights).stem and keepdeconvweights:
        print("Loading already trained DLC with backbone:", cfg.net_type)
        variables_to_restore = slim.get_variables_to_restore()
    else:
        print("Loading ImageNet-pretrained", cfg.net_type)
        #loading backbone from ResNet, MobileNet etc.
        if 'resnet' in cfg.net_type:
            variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])
        elif 'mobilenet' in cfg.net_type:
            variables_to_restore = slim.get_variables_to_restore(include=["MobilenetV2"])
        else:
            print("Wait for DLC 2.3.")

    restorer = TF.train.Saver(variables_to_restore)
    saver = TF.train.Saver(max_to_keep=max_to_keep) # selects how many snapshots are stored, see https://github.com/AlexEMG/DeepLabCut/issues/8#issuecomment-387404835

    if allow_growth==True:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = TF.Session(config=config)
    else:
        sess = TF.Session()

    coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)
    train_writer = TF.summary.FileWriter(cfg.log_dir, sess.graph)
    learning_rate, train_op = get_optimizer(total_loss, cfg)

    sess.run(TF.global_variables_initializer())
    sess.run(TF.local_variables_initializer())

    # Restore variables from disk.
    try:
        print(f'Initializing weights from {cfg.init_weights}')
        wtfile = cfg.init_weights
        if wtfile.endswith('.index'):
            wtfile = ''.join(wtfile.rsplit('.index',1))
        restorer.restore(sess, wtfile)
    except Exception as e:
        print(f'Could not load weights from {cfg.init_weights}')
        print(e)

    if maxiters==None:
        max_iter = int(cfg.multi_step[-1][1])
    else:
        max_iter = min(int(cfg.multi_step[-1][1]),int(maxiters))
        #display_iters = max(1,int(displayiters))
        print("Max_iters overwritten as",max_iter)

    if displayiters==None:
        display_iters = max(1,int(cfg.display_iters))
    else:
        display_iters = max(1,int(displayiters))
        print("Display_iters overwritten as",display_iters)

    if saveiters==None:
        save_iters=max(1,int(cfg.save_iters))

    else:
        save_iters=max(1,int(saveiters))
        print("Save_iters overwritten as",save_iters)

    cum_loss = 0.0
    lr_gen = LearningRate(cfg)

    stats_path = os.path.join(cfg.project_path,'learning_stats.csv')
    lrf = open(str(stats_path), 'w')

    print("Training parameter:")
    print(cfg)
    print("Starting training....")
    for it in range(max_iter+1):
        current_lr = lr_gen.get_lr(it)
        [_, loss_val, summary] = sess.run([train_op, total_loss, merged_summaries],
                                          feed_dict={learning_rate: current_lr})
        cum_loss += loss_val
        train_writer.add_summary(summary, it)

        if it % display_iters == 0:

            cur_out, batch_out = sess.run([outputs, batch], feed_dict={learning_rate: current_lr})
            pred = [cur_out['part_pred'],cur_out['locref']]
            scmap, locref = predict.extract_cnn_output(pred, cfg)

            # Extract maximum scoring location from the heatmap, assume 1 person
            loc_pred = predict.argmax_pose_predict(scmap, locref, cfg.stride)
            if loc_pred.ndim == 2:
                loc_pred = loc_pred[np.newaxis,np.newaxis,...]
            loc_in = batch_out[Batch.locs]
            if loc_pred.shape[2] == loc_in.shape[2]:
                dd = np.sqrt(np.sum(np.square(loc_pred[:,:,:,:2]-loc_in),axis=-1))
            else:
                dd = np.array([0])
            dd = dd/cfg.global_scale
            average_loss = cum_loss / display_iters
            cum_loss = 0.0
            logging.info("iteration: {} dist: {:.2f} loss: {} lr: {}"
                         .format(it, dd.mean(), "{0:.4f}".format(average_loss), current_lr))
            # lrf.write("{}, {:.2f}, {:.5f}, {}\n".format(it, dd.mean(), average_loss, current_lr))
            # lrf.flush()

            train_info['step'].append(it)
            train_info['train_loss'].append(loss_val)
            train_info['val_loss'].append(loss_val)
            train_info['val_dist'].append(dd.mean())
            train_info['train_dist'].append(dd.mean())

            save_td(cfg, train_info)


        # Save snapshot
        if (it % save_iters == 0 and it != 0) or it == max_iter:
            model_name = cfg.snapshot_prefix
            saver.save(sess, model_name, global_step=it)

    lrf.close()
    coord.request_stop()
    sess.run(q.close(cancel_pending_enqueues=True))
    coord.join([thread],stop_grace_period_secs=60,ignore_live_threads=True)
    sess.close()

    #return to original path.
    os.chdir(str(start_path))

def create_cfg(cfg_dict):
    curd = os.path.realpath(__file__)
    bdir = os.path.split(os.path.split(curd)[0])[0]
    config_yaml = os.path.join(bdir,'pose_cfg.yaml')
    cfg = load_config(config_yaml)
    for k in cfg_dict.keys():
        cfg[k] = cfg_dict[k]
    return  cfg

def get_pred_fn(cfg_dict, model_file=None):

    cfg = create_cfg(cfg_dict)
    name = Path(cfg.snapshot_prefix).stem

    if model_file is None:
        ckpt_file = os.path.join(cfg.project_path, name + '_ckpt')
        latest_ckpt = tf.train.get_checkpoint_state(cfg.project_path, ckpt_file)
        model_file = latest_ckpt.model_checkpoint_path
        init_weights = model_file
    else:
        init_weights = model_file

    TF.reset_default_graph()
    TF.disable_eager_execution()

    cfg.init_weights = init_weights
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)

    def pred_fn(all_f):
        if cfg.img_dim == 1:
            cur_im = np.tile(all_f,[1,1,1,3])
        else:
            cur_im = all_f

        if cfg.dlc_use_apt_preprocess:
            cur_im, _ = PoseTools.preprocess_ims(cur_im, in_locs=np.zeros([cur_im.shape[0], cfg.num_joints, 2]), conf=cfg, distort=False, scale=1/cfg.global_scale)
        else:
            scale = cfg.global_scale
            nims = []
            for ndx  in range(all_f.shape[0]):
                image = cur_im[ndx,...]
                nims.append(imresize(image, scale) if scale != 1 else image)
            cur_im = np.array(nims)

        cur_out = sess.run(outputs, feed_dict={inputs: cur_im})
        scmap, locref = predict.extract_cnn_output(cur_out, cfg)
        pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
        pose = pose[np.newaxis,:,:2]/cfg.global_scale
        ret_dict = {}
        ret_dict['locs'] = pose
        ret_dict['hmaps'] = scmap[np.newaxis,...]
        ret_dict['conf'] = np.max(scmap[np.newaxis,...], axis=(1,2))
        return ret_dict

    def close_fn():
        sess.close()

    return pred_fn, close_fn, model_file

def model_files(cfg_dict, name='deepnet'):
    cfg = create_cfg(cfg_dict)
    ckpt_file = os.path.join(cfg.project_path, name + '_ckpt')
    if not os.path.exists(ckpt_file):
        return []
    latest_ckpt = tf.train.get_checkpoint_state(cfg.project_path, ckpt_file)
    latest_model_file = latest_ckpt.model_checkpoint_path
    import glob
    all_model_files = glob.glob(latest_model_file + '.*')
    train_data_file = os.path.join( cfg.project_path, 'traindata')
    all_model_files.extend([ckpt_file, train_data_file])

    return all_model_files



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to yaml configuration file.')
    cli_args = parser.parse_args()

    train(Path(cli_args.config).resolve())

##

