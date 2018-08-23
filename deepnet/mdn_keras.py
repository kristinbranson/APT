import os
import PoseTools
import tensorflow as tf
import numpy as np
from scipy import stats
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Multiply
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.initializers import random_normal,constant
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from keras.callbacks import Callback
import keras.backend as K
import keras
import math
import json

name = 'keras_unet'


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


def create_label_images(locs, imsz):
    n_out = locs.shape[1]
    n_ex = locs.shape[0]
    out = np.zeros([n_ex,imsz[0],imsz[1],n_out])
    for cur in range(n_ex):
        for ndx in range(n_out):
            x,y = np.meshgrid(range(imsz[1]),range(imsz[0]))
            x = x-locs[cur,ndx,0]
            y = y - locs[cur,ndx,1]
            dd = np.sqrt(x**2+y**2)/3
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
        self.heat_num = self.conf.n_classes
        self.N = PoseTools.count_records(filename)


    def reset(self):
        if self.iterator:
            self.iterator.close()
        self.iterator = tf.python_io.tf_record_iterator(self.file)
        #print('========= Resetting ==========')


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

        ims, locs = PoseTools.preprocess_ims(ims, locs, self.conf,
                                            self.distort, self.conf.unet_rescale)

        label_sz = [x/self.conf.unet_rescale for x in self.conf.imsz]
        label_ims = create_label_images(locs/self.conf.unet_rescale, label_sz)
        label_ims = np.clip(label_ims,0,1)

        return [ims,locs], label_ims


    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

def relu(x): return Activation('relu')(x)

def conv(x, nf, ks, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
    #           kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)

    x = BatchNormalization()(x)
    return x

def pooling(x, ks, st, name):
    # x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    x = AveragePooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x


def get_training_model(conf, weight_decay):

    inputs = []
    outputs = []

    imsz = [x/conf.unet_rescale for x in conf.imsz]
    img_input_shape = imsz + [conf.imgDim,]

    img_input = Input(img_input_shape)
    locs_input = Input([conf.n_classes,2])
    inputs.append(img_input)
    inputs.append(locs_input)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]
    n_stages = 6
    x = img_normalized
    base_filt = 32
    down_layers = [x,]
    for ndx in range(n_stages):
        x = conv(x, 32*(2**ndx), 3, 'conv_{}_0'.format(ndx),weight_decay)
        x = relu(x)
        x = conv(x, 32*(2**ndx), 3, 'conv_{}_1'.format(ndx),weight_decay)
        x = relu(x)
        x = pooling(x,2,2,'pool_{}'.format(ndx))
        down_layers.append(x)

    x = conv(x, 32 * (2 ** n_stages), 3, 'top_0', weight_decay)
    x = relu(x)
    x = conv(x, 32 * (2 ** n_stages), 3, 'top_1', weight_decay)
    x = relu(x)

    up_layers = [x,]
    for ndx in reversed(range(n_stages)):
        x = Concatenate()([x,down_layers[ndx+1]])
        x = conv(x, 32*(2**ndx), 3, 'up_conv_{}_0'.format(ndx),weight_decay)
        x = relu(x)
        x = conv(x, 32*(2**ndx), 3, 'up_conv_{}_1'.format(ndx),weight_decay)
        x = relu(x)
        x = keras.layers.UpSampling2D(size=(2,2))(x)
        x_shape = x.shape.as_list()
        d_shape = down_layers[ndx].shape.as_list()
        x_crop = d_shape[2] - x_shape[2]
        y_crop = d_shape[1] - x_shape[1]
        x = keras.layers.ZeroPadding2D(padding=((0,y_crop),(0,x_crop)))(x)
        up_layers.append(x)

#    x = conv(x, conf.n_classes, 3, 'out_conv',weight_decay)

    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None
    x = Conv2D(conf.n_classes, (3, 3), padding='same', name='out_conv',
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               bias_initializer=constant(0.0))(x)

    outputs.append(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def training(conf):

    train_di = DataIteratorTF(conf, 'train', True, True)
    train_di2 = DataIteratorTF(conf, 'train', True, True)
    val_di = DataIteratorTF(conf, 'train', False, False)

    weight_decay = 5e-4
    base_lr = 4e-5
    momentum = 0.9
    gamma = 0.1
    batch_size = conf.batch_size
    stepsize = 20000/3
    iterations_per_epoch = conf.display_step
    max_iter = 20000/iterations_per_epoch
    last_epoch = 0

    model_file = os.path.join(conf.cachedir, conf.expname + '_' + name + '-{epoch:d}')
    model = get_training_model(conf, (weight_decay,0))

    def step_decay(epoch):
        initial_lrate = base_lr
        steps = epoch * iterations_per_epoch
        # lrate = initial_lrate * math.pow(gamma, math.floor(steps / stepsize))
        lrate = initial_lrate * math.pow(gamma, float(steps) / stepsize)
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
            tt1 = PoseTools.get_pred_locs(val_out) - \
                  PoseTools.get_pred_locs(val_y)
            tt1 = np.sqrt(np.sum(tt1 ** 2, 2))
            val_dist = np.nanmean(tt1)*self.config.unet_rescale
            tt1 = PoseTools.get_pred_locs(train_out) - \
                  PoseTools.get_pred_locs(train_y)
            tt1 = np.sqrt(np.sum(tt1 ** 2, 2))
            train_dist = np.nanmean(tt1)*self.config.unet_rescale
            self.train_info['val_dist'].append(val_dist)
            self.train_info['val_loss'].append(val_loss)
            self.train_info['train_dist'].append(train_dist)
            self.train_info['train_loss'].append(train_loss)
            self.train_info['step'].append(int(step))

            p_str = ''
            for k in self.train_info.keys():
                p_str += '{:s}:{:.2f} '.format(k, self.train_info[k][-1])
            print(p_str)
#            print('Learning Rate {}'.format(K.eval(self.model.optimizer.lr)))
            train_data_file = os.path.join( self.config.cachedir, self.config.expname + '_' + name + '_traindata')

            json_data = {}
            for x in self.train_info.keys():
                json_data[x] = np.array(self.train_info[x]).astype(np.float64).tolist()
            with open(train_data_file + '.json', 'w') as json_file:
                json.dump(json_data, json_file)

            if step % conf.save_step == 0:
                model.save(os.path.join(conf.cachedir, conf.expname + '_' + name + '-{}'.format(step)))

    lr_mult = dict()
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            kernel_name = layer.weights[0].name
            bias_name = layer.weights[1].name
            lr_mult[kernel_name] = 1
            lr_mult[bias_name] = 2


    # configure callbacks
    lrate = LearningRateScheduler(step_decay)
    checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=0, save_best_only=False,
        save_weights_only=True, mode='min', period=conf.save_step)
    obs = OutputObserver(conf, [train_di2, val_di])
    callbacks_list = [lrate, obs] #checkpoint,

    # sgd optimizer with lr multipliers
    # multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)

    def eucl_loss(x, y):
        return K.sum(K.square(x - y)) / batch_size / 2

    losses = [eucl_loss,]
    # start training
    model.compile(loss=losses, optimizer=keras.optimizers.Adam(lr=base_lr,clipnorm=1.))

    # training
    model.fit_generator(train_di,
                        steps_per_epoch=conf.display_step,
                        epochs=max_iter,
                        callbacks=callbacks_list,
                        verbose=0,
                        # validation_data=val_di,
                        # validation_steps=val_samples // batch_size,
                       use_multiprocessing=True,
                       workers=8,
                        initial_epoch=last_epoch
                        )

    # force saving in case the max iter doesn't match the save step.
    model.save(os.path.join(conf.cachedir, conf.expname + '_' + name + '-{}'.format(max_iter)))
    obs.on_epoch_end(max_iter)
