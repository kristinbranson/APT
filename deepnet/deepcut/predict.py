import numpy as np

import tensorflow as tf

from nnet.net_factory import pose_net


def setup_pose_prediction(cfg, init_weights):
    inputs = tf.placeholder(tf.float32, shape=[cfg.batch_size   , None, None, 3])

    net_heads = pose_net(cfg).test(inputs)
    outputs = [net_heads['part_prob']]
    if cfg.location_refinement:
        outputs.append(net_heads['locref'])

    restorer = tf.train.Saver()

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, init_weights)

    return sess, inputs, outputs


def extract_cnn_output(outputs_np, cfg):
    scmap = outputs_np[0]
    locref = None
    if cfg.location_refinement:
        # locref = np.squeeze(outputs_np[1])
        #  MK: edit on July 9 2018.
        # The squeeze fails if batch size is 1.
        # it anyway seems redundant.
        locref = outputs_np[1]
        shape = locref.shape
        locref = np.reshape(locref, (shape[0], shape[1], shape[2],-1, 2))
        locref *= cfg.locref_stdev
    return scmap, locref


def argmax_pose_predict(scmap, offmat, stride):
    """Combine scoremat and offsets to the final pose."""
    num_joints = scmap.shape[3]
    pose = []
    for ndx in range(scmap.shape[0]):
        pose.append([])
        for joint_idx in range(num_joints):
            maxloc = np.unravel_index(np.argmax(scmap[ndx, :, :, joint_idx]),
                                      scmap[ndx,:, :, joint_idx].shape)
            offset = np.array(offmat[ndx][maxloc][joint_idx])[::-1]
            pos_f8 = (np.array(maxloc).astype('float') * stride + 0.5 * stride +
                      offset)
            pose[ndx].append(np.hstack((pos_f8[::-1],
                                   [scmap[ndx][maxloc][joint_idx]])))
    return np.array(pose)
