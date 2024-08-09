from PoseBase import PoseBase
import PoseTools
import os
import tarfile
import tensorflow
vv = [int(v) for v in tensorflow.__version__.split('.')]
if (vv[0]==1 and vv[1]>12) or vv[0]==2:
    tf = tensorflow.compat.v1
else:
    tf = tensorflow
# from batch_norm import batch_norm_mine_old as batch_norm
if vv[0]==1:
    from tensorflow.contrib.layers import batch_norm
    import tensorflow.contrib.slim as slim
    from tensorflow.contrib.slim.nets import resnet_v1

    from tensorflow.contrib.layers import xavier_initializer

else:
    from tensorflow.compat.v1.layers import batch_normalization as batch_norm_temp
    def batch_norm(inp,decay,is_training,renorm=False,data_format=None):
        return batch_norm_temp(inp,momentum=decay,training=is_training)
    import tf_slim as slim
    from tf_slim.nets import resnet_v1
    from tensorflow.keras.initializers import GlorotUniform as  xavier_initializer
import urllib
import resnet_official
import convNetBase as CNB
from PoseCommon_dataset import conv_relu3, conv_relu
import numpy as np
from upsamp import upsample_init_value


class Pose_resnet_unet(PoseBase):

    def __init__(self, conf,name='deepnet'):
        PoseBase.__init__(self, conf,name=name,hmaps_downsample=1)

        self.conf.use_pretrained_weights = True

        self.resnet_source = self.conf.get('mdn_resnet_source','official_tf')
        if self.resnet_source == 'official_tf':
            url = 'http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC.tar.gz'
            script_dir = os.path.dirname(os.path.realpath(__file__))
            wt_dir = os.path.join(script_dir,'pretrained')
            wt_file = os.path.join(wt_dir,'resnet_v2_fp32_savedmodel_NHWC','1538687283','variables','variables.index')
            if not os.path.exists(wt_file):
                print('Downloading pretrained weights..')
                if not os.path.exists(wt_dir):
                    os.makedirs(wt_dir)
                sname, header = urllib.urlretrieve(url)
                tar = tarfile.open(sname, "r:gz")
                print('Extracting pretrained weights..')
                tar.extractall(path=wt_dir)
            self.pretrained_weights = os.path.join(wt_dir,'resnet_v2_fp32_savedmodel_NHWC','1538687283','variables','variables')
        else:
            url = 'http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz'
            script_dir = os.path.dirname(os.path.realpath(__file__))
            wt_dir = os.path.join(script_dir,'pretrained')
            wt_file = os.path.join(wt_dir,'resnet_v1_50.ckpt')
            if not os.path.exists(wt_file):
                print('Downloading pretrained weights..')
                if not os.path.exists(wt_dir):
                    os.makedirs(wt_dir)
                sname, header = urllib.urlretrieve(url)
                tar = tarfile.open(sname, "r:gz")
                print('Extracting pretrained weights..')
                tar.extractall(path=wt_dir)
            self.pretrained_weights = os.path.join(wt_dir,'resnet_v1_50.ckpt')


    def create_network(self):

        conv = lambda a, b: conv_relu3( a,b,self.ph['phase_train'])

        im, locs, info, hmap = self.inputs

        if self.conf.get('pretrain_freeze_bnorm', True):
            pretrain_update_bnorm = False
        else:
            pretrain_update_bnorm = self.ph['phase_train']

        if self.resnet_source == 'slim':
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = resnet_v1.resnet_v1_50(im,
                                          global_pool=False, is_training=pretrain_update_bnorm)
                l_names = ['conv1', 'block1/unit_2/bottleneck_v1', 'block2/unit_3/bottleneck_v1',
                           'block3/unit_5/bottleneck_v1', 'block4']
                down_layers = [end_points['resnet_v1_50/' + x] for x in l_names]

                ex_down_layers = conv(self.inputs[0], 64)
                down_layers.insert(0, ex_down_layers)
                n_filts = [32, 64, 64, 128, 256, 512]

        elif self.resnet_source == 'official_tf':
            mm = resnet_official.Model( resnet_size=50, bottleneck=True, num_classes=self.conf.n_classes, num_filters=64, kernel_size=7, conv_stride=2, first_pool_size=3, first_pool_stride=2, block_sizes=[3, 4, 6, 3], block_strides=[1, 2, 2, 2], final_size=2048, resnet_version=2, data_format='channels_last',dtype=tf.float32)
            resnet_out = mm(im, pretrain_update_bnorm)
            down_layers = mm.layers
            down_layers.pop(1) # remove one of the layers of size imsz/4, imsz/4 at index 1
            ex_down_layers = conv(self.inputs[0], 64)
            down_layers.insert(0, ex_down_layers)
            n_filts = [32, 64, 64, 128, 256, 512, 1024]


        prev_in = None
        for ndx in reversed(range(len(down_layers))):

            if prev_in is None:
                X = down_layers[ndx]
            else:
                X = tf.concat([prev_in, down_layers[ndx]],axis=-1)

            sc_name = 'layerup_{}_0'.format(ndx)
            with tf.variable_scope(sc_name):
                X = conv(X, n_filts[ndx])

            if ndx is not 0:
                sc_name = 'layerup_{}_1'.format(ndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filts[ndx])

                layers_sz = down_layers[ndx-1].get_shape().as_list()[1:3]
                with tf.variable_scope('u_{}'.format(ndx)):
                    X_sh = X.get_shape().as_list()
                    # w_mat = np.zeros([4, 4, X_sh[-1], X_sh[-1]])
                    # for wndx in range(X_sh[-1]):
                    #     w_mat[:, :, wndx, wndx] = 1.
                    w_sh = [4, 4, X_sh[-1], X_sh[-1]]
                    w_mat = upsample_init_value(w_sh, alg='bl', dtype=np.float32)

                    w = tf.get_variable('w', [4, 4, X_sh[-1], X_sh[-1]], initializer=tf.constant_initializer(w_mat))
                    out_shape = [X_sh[0], layers_sz[0], layers_sz[1], X_sh[-1]]
                    X = tf.nn.conv2d_transpose(X, w, output_shape=out_shape, strides=[1, 2, 2, 1], padding="SAME")
                    biases = tf.get_variable('biases', [out_shape[-1]], initializer=tf.constant_initializer(0))
                    conv_b = X + biases

                    bn = batch_norm(conv_b, is_training=self.ph['phase_train'], decay=0.99)
                    X = tf.nn.relu(bn)

                    # X = CNB.upscale('u_{}'.format(ndx), X, layers_sz)

            prev_in = X

        n_filt = X.get_shape().as_list()[-1]
        n_out = self.conf.n_classes
        weights = tf.get_variable("out_weights", [3,3,n_filt,n_out],
                                  initializer=xavier_initializer())
        biases = tf.get_variable("out_biases", n_out,
                                 initializer=tf.constant_initializer(0.))
        conv = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.add(conv, biases, name = 'unet_pred')
        X = 2*tf.sigmoid(X)-1
        return X
