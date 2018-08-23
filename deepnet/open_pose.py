from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal,constant
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.callbacks import Callback
from keras.applications.vgg19 import VGG19
from scipy import stats
from keras.optimizers import Optimizer
from keras import backend as K
from keras.legacy import interfaces

import sys
import os
import re
import math
import PoseTools
import os
import  numpy as np
import json
import tensorflow as tf
import keras.backend as K


name = 'open_pose'


# ---------------------
# ----- Optimizer -----
#----------------------
class MultiSGD(Optimizer):
    """
    Modified SGD with added support for learning multiplier for kernels and biases
    as suggested in: https://github.com/fchollet/keras/issues/5920

    Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, lr_mult=None, **kwargs):
        super(MultiSGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.lr_mult = lr_mult

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):

            if p.name in self.lr_mult:
                multiplied_lr = lr * self.lr_mult[p.name]
            else:
                multiplied_lr = lr

            v = self.momentum * m - multiplied_lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - multiplied_lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(MultiSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# ---------------------
# ----- Model ---------
#----------------------

def relu(x): return Activation('relu')(x)

def conv(x, nf, ks, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    return x

def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x

def vgg_block(x, weight_decay):
    # Block 1
    x = conv(x, 64, 3, "conv1_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")

    # Block 2
    x = conv(x, 128, 3, "conv2_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1")

    # Block 3
    x = conv(x, 256, 3, "conv3_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_2", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_3", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_4", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool3_1")

    # Block 4
    x = conv(x, 512, 3, "conv4_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "conv4_2", (weight_decay, 0))
    x = relu(x)

    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPM", (weight_decay, 0))
    x = relu(x)

    return x

def stage1_block(x, num_p, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))

    return x

def stageT_block(x, num_p, stage, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))

    return x

def apply_mask(x, mask, stage, branch):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    w = Multiply(name=w_name)([x, mask]) # vec_weight
    return w

def get_training_model(weight_decay, br1=38,br2=19):

    stages = 6
    np_branch1 = br1
    np_branch2 = br2

    img_input_shape = (None, None, 3)
    vec_input_shape = (None, None, br1)
    heat_input_shape = (None, None, br2)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    vec_weight_input = Input(shape=vec_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    inputs.append(vec_weight_input)
    inputs.append(heat_weight_input)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, weight_decay)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)
    w1 = apply_mask(stage1_branch1_out, vec_weight_input, 1, 1)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
    w2 = apply_mask(stage1_branch2_out, heat_weight_input, 1, 2)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    outputs.append(w1)
    outputs.append(w2)

    # stage sn >= 2
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
        w1 = apply_mask(stageT_branch1_out, vec_weight_input, sn, 1)

        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
        w2 = apply_mask(stageT_branch2_out, heat_weight_input, sn, 2)

        outputs.append(w1)
        outputs.append(w2)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=inputs, outputs=outputs)

    return model

def get_testing_model(br1=38,br2=19):
    stages = 6
    np_branch1 = br1
    np_branch2 = br2

    img_input_shape = (None, None, 3)

    img_input = Input(shape=img_input_shape)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, None)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, None)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, None)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    # stage t >= 2
    stageT_branch1_out = None
    stageT_branch2_out = None
    for sn in range(2, stages + 1):
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, None)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=[img_input], outputs=[stageT_branch1_out, stageT_branch2_out])

    return model


# ---------------------
# -- Data Generator ---
#----------------------


def create_affinity_labels(locs, imsz, graph):
    n_out = len(graph)
    n_ex = locs.shape[0]
    out = np.zeros([n_ex,imsz[0],imsz[1],n_out*2])
    for cur in range(n_ex):
        for ndx, e in enumerate(graph):
            start_x, start_y = locs[cur, e[0]-1, :]
            end_x, end_y = locs[cur,e[1]-1,:]
            ll = np.sqrt( (start_x-end_x)**2 + (start_y-end_y)**2)

            dx = (end_x - start_x)/ll/2
            dy = (end_y - start_y)/ll/2
            xx = np.round(np.linspace(start_x,end_x,6000))
            yy = np.round(np.linspace(start_y,end_y,6000))
            zz = np.stack([xx,yy])
            xx = np.round(np.linspace(start_x+dy,end_x+dy,6000))
            yy = np.round(np.linspace(start_y-dx,end_y-dx,6000))
            zz = np.concatenate([zz,np.stack([xx,yy])],axis=1)
            xx = np.round(np.linspace(start_x-dy,end_x-dy,6000))
            yy = np.round(np.linspace(start_y+dx,end_y+dx,6000))
            zz = np.concatenate([zz,np.stack([xx,yy])],axis=1)
            # zz now has all the pixels that are along the line.
            zz = np.unique(zz,axis=1)
            # zz now has all the unique pixels that are along the line with thickness 1.
            for x,y in zz.T:
                if x >= out.shape[2] or y>= out.shape[1]:
                    continue
                out[cur,int(y),int(x),ndx*2] = (end_x-start_x)/ll
                out[cur,int(y),int(x),ndx*2+1] = (end_y-start_y)/ll

    return  out

def create_label_images(locs, imsz):
    n_out = locs.shape[1]
    n_ex = locs.shape[0]
    out = np.zeros([n_ex,imsz[0],imsz[1],n_out])
    for cur in range(n_ex):
        for ndx in range(n_out):
            x,y = np.meshgrid(range(imsz[1]),range(imsz[0]))
            x = x-locs[cur,ndx,0]
            y = y - locs[cur,ndx,1]
            dd = np.sqrt(x**2+y**2)
            out[cur,:,:,ndx] = stats.norm.pdf(dd)/stats.norm.pdf(0)
    out[out<0.05] = 0.
    return  out

class DataIteratorTF(object):


    def __init__(self, conf, db_type, distort, shuffle):
        self.conf = conf
        if db_type == 'val':
            filename = os.path.join(self.conf.cachedir, self.conf.valfilename) + '.tfrecords'
        elif db_type == 'train':
            filename = os.path.join(self.conf.cachedir, self.conf.trainfilename) + '.tfrecords'
        else:
            raise IOError, 'Unspecified DB Type'
        self.file = filename
        self.iterator  = None
        self.distort = distort
        self.shuffle = shuffle
        self.batch_size = self.conf.batch_size
        self.vec_num = len(conf.op_affinity_graph)
        self.heat_num = self.conf.n_classes
        self.N = PoseTools.count_records(filename)


    def reset(self):
        if self.iterator:
            self.iterator.close()
        self.iterator = tf.python_io.tf_record_iterator(self.file)
        print('========= Resetting ==========')


    def read_next(self):
        if not self.iterator:
            self.iterator = tf.python_io.tf_record_iterator(self.file)
        try:
            record = self.iterator.next()
        except StopIteration:
            self.reset()
            record = self.iterator.next()

        return  record

    def next(self):

        all_ims = []
        all_locs = []
        for b_ndx in range(self.batch_size):
            n_skip = np.random.randint(30) if self.shuffle else 0
            for _ in range(n_skip+1):
                record = self.read_next()

            example = tf.train.Example()
            example.ParseFromString(record)
            height = int(example.features.feature['height'].int64_list.value[0])
            width = int(example.features.feature['width'].int64_list.value[0])
            depth = int(example.features.feature['depth'].int64_list.value[0])
            expid = int(example.features.feature['expndx'].float_list.value[0]),
            t = int(example.features.feature['ts'].float_list.value[0]),
            img_string = example.features.feature['image_raw'].bytes_list.value[0]
            img_1d = np.fromstring(img_string, dtype=np.uint8)
            reconstructed_img = img_1d.reshape((height, width, depth))
            locs = np.array(example.features.feature['locs'].float_list.value)
            locs = locs.reshape([self.conf.n_classes, 2])
            if 'trx_ndx' in example.features.feature.keys():
                trx_ndx = int(example.features.feature['trx_ndx'].int64_list.value[0])
            else:
                trx_ndx = 0
            all_ims.append(reconstructed_img)
            all_locs.append(locs)

        ims = np.stack(all_ims)
        locs = np.stack(all_locs)

        if self.conf.imgDim == 1:
            ims = np.tile(ims, 3)

        mask_sz = [int(x/self.conf.op_label_scale) for x in self.conf.imsz]
        mask_sz1 = [self.batch_size,] + mask_sz + [2*self.vec_num]
        mask_sz2 = [self.batch_size,] + mask_sz + [self.heat_num]
        mask_im1 = np.ones(mask_sz1)
        mask_im2 = np.ones(mask_sz2)

        ims, locs = PoseTools.preprocess_ims(ims, locs, self.conf,
                                            self.distort, self.conf.op_rescale)

        label_ims = create_label_images(locs/self.conf.op_label_scale, mask_sz)
#        label_ims = PoseTools.create_label_images(locs/self.conf.op_label_scale, mask_sz,1,2)
        label_ims = np.clip(label_ims,0,1)

        affinity_ims = create_affinity_labels(locs/self.conf.op_label_scale,
                                              mask_sz, self.conf.op_affinity_graph)


        return [ims, mask_im1, mask_im2], \
                [affinity_ims, label_ims,
                 affinity_ims, label_ims,
                 affinity_ims, label_ims,
                 affinity_ims, label_ims,
                 affinity_ims, label_ims,
                 affinity_ims, label_ims ]


    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


# ---------------------
# -- Training ---------
#----------------------

def set_openpose_defaults(conf):
    conf.label_blur_rad = 5
    conf.rrange = 5
    conf.display_steps = 50 # this is same as batches per epoch
    conf.dl_steps = 600000
    conf.batch_size = 10
    conf.n_steps = 4.41
    conf.gamma = 0.333


def training(conf):

    base_lr = 4e-5  # 2e-5
    momentum = 0.9
    weight_decay = 5e-4
    lr_policy = "step"
    batch_size = conf.batch_size
    gamma = conf.gamma
    stepsize = int(float(conf.dl_steps)/conf.n_steps)
    # stepsize = 68053  # 136106 #   // after each stepsize iterations update learning rate: lr=lr*gamma
    iterations_per_epoch = conf.display_step
    max_iter = conf.dl_steps/iterations_per_epoch
    restart = True
    last_epoch = 0

    assert conf.dl_steps % iterations_per_epoch == 0, 'For open-pose dl steps must be a multiple of display steps'
    assert conf.save_step % iterations_per_epoch == 0, 'For open-pose save steps must be a multiple of display steps'

    model_file = os.path.join(conf.cachedir, conf.expname + '_' + name + '-{epoch:d}')
    model = get_training_model(weight_decay, br1=len(conf.op_affinity_graph) * 2, br2=conf.n_classes)

    # load previous weights or vgg19 if this is the first run
    from_vgg = dict()
    for blk in range(1,5):
        for lvl in range(1,3):
            from_vgg['conv{}_{}'.format(blk,lvl)] = 'block{}_conv{}'.format(blk,lvl)
    print("Loading vgg19 weights...")
    vgg_model = VGG19(include_top=False, weights='imagenet')
    for layer in model.layers:
        if layer.name in from_vgg:
            vgg_layer_name = from_vgg[layer.name]
            layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
            print("Loaded VGG19 layer: " + vgg_layer_name)

    # prepare generators
    train_di = DataIteratorTF(conf, 'train', True, True)
    train_di2 = DataIteratorTF(conf, 'train', True, True)
    val_di = DataIteratorTF(conf, 'train', False, False)

    # setup lr multipliers for conv layers
    lr_mult = dict()
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            # stage = 1
            if re.match("Mconv\d_stage1.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2

            # stage > 1
            elif re.match("Mconv\d_stage.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 4
                lr_mult[bias_name] = 8

            # vgg
            else:
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2

    # configure loss functions
    def eucl_loss(x, y):
        return K.sum(K.square(x - y)) / batch_size / 2
    losses = {}
    for stage in range(1,7):
        for lvl in range(1,3):
            losses['weight_stage{}_L{}'.format(stage,lvl)] = eucl_loss

    # lr decay.
    def step_decay(epoch):
        initial_lrate = base_lr
        steps = epoch * iterations_per_epoch
        lrate = initial_lrate * math.pow(gamma, math.floor(steps / stepsize))
        return lrate

    # Callback to do writing pring stuff.
    class OutputObserver(Callback):
        def __init__(self, conf, dis):
            self.train_di, self.val_di = dis
            self.train_info = {}
            self.train_info['step'] = []
            self.train_info['train_dist'] = []
            self.train_info['train_loss'] = []
            self.train_info['val_dist'] = []
            self.train_info['val_loss'] = []
            self.config = conf
            self.force = False

        def on_epoch_end(self, epoch, logs={}):
            step = (epoch+1) * conf.display_step
            val_x, val_y = self.val_di.next()
            val_out = self.model.predict(val_x)
            val_loss = self.model.evaluate(val_x, val_y, verbose=0)
            train_x, train_y = self.train_di.next()
            train_out = self.model.predict(train_x)
            train_loss = self.model.evaluate(train_x, train_y, verbose=0)

            # dist only for last layer
            tt1 = PoseTools.get_pred_locs(val_out[-1]) - \
                  PoseTools.get_pred_locs(val_y[-1])
            tt1 = np.sqrt(np.sum(tt1 ** 2, 2))
            val_dist = np.nanmean(tt1)*self.config.op_label_scale
            tt1 = PoseTools.get_pred_locs(train_out[-1]) - \
                  PoseTools.get_pred_locs(train_y[-1])
            tt1 = np.sqrt(np.sum(tt1 ** 2, 2))
            train_dist = np.nanmean(tt1)*self.config.op_label_scale
            self.train_info['val_dist'].append(val_dist)
            self.train_info['val_loss'].append(val_loss[0])
            self.train_info['train_dist'].append(train_dist)
            self.train_info['train_loss'].append(train_loss[0])
            self.train_info['step'].append(int(step))

            p_str = ''
            for k in self.train_info.keys():
                p_str += '{:s}:{:.2f} '.format(k, self.train_info[k][-1])
            print(p_str)

            train_data_file = os.path.join( self.config.cachedir, self.config.expname + '_' + name + '_traindata')

            json_data = {}
            for x in self.train_info.keys():
                json_data[x] = np.array(self.train_info[x]).astype(np.float64).tolist()
            with open(train_data_file + '.json', 'w') as json_file:
                json.dump(json_data, json_file)

            if step % conf.save_step == 0:
                model.save(os.path.join(conf.cachedir, conf.expname + '_' + name + '-{}'.format(step)))


    # configure callbacks
    lrate = LearningRateScheduler(step_decay)
    checkpoint = ModelCheckpoint(
        model_file, monitor='loss', verbose=0, save_best_only=False,
        save_weights_only=True, mode='min', period=conf.save_step)
    obs = OutputObserver(conf, [train_di2, val_di])
    callbacks_list = [lrate, obs] #checkpoint,

    # sgd optimizer with lr multipliers
    multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)

    # start training
    model.compile(loss=losses, optimizer=multisgd)

    # training
    model.fit_generator(train_di,
                        steps_per_epoch=conf.display_step,
                        epochs=max_iter,
                        callbacks=callbacks_list,
                        verbose=0,
                        # validation_data=val_di,
                        # validation_steps=val_samples // batch_size,
#                        use_multiprocessing=True,
#                        workers=4,
                        initial_epoch=last_epoch
                        )

    # force saving in case the max iter doesn't match the save step.
    model.save(os.path.join(conf.cachedir, conf.expname + '_' + name + '-{}'.format(max_iter)))
    obs.on_epoch_end(max_iter)


def get_pred_fn(conf, model_file=None):
    model = get_testing_model(br1=len(conf.op_affinity_graph) * 2, br2=conf.n_classes)
    if model_file is None:
        latest_model_file = PoseTools.get_latest_model_file_keras(conf, name)
    else:
        latest_model_file = model_file
    print("Loading the weights from {}.. ".format(latest_model_file))
    model.load_weights(latest_model_file)

    def pred_fn(all_f):
        if all_f.shape[3] == 1:
            all_f = np.tile(all_f,[1,1,1,3])
        xs, _ = PoseTools.preprocess_ims(
            all_f, in_locs=np.zeros([conf.batch_size, conf.n_classes, 2]), conf=conf,
            distort=False, scale=conf.op_rescale)
        pred = model.predict(xs)[-1]
        base_locs = PoseTools.get_pred_locs(pred)
        base_locs = base_locs * conf.op_rescale * conf.op_label_scale
        return base_locs, pred

    close_fn = K.clear_session

    return pred_fn, close_fn, latest_model_file
