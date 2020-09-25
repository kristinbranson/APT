
import os
import copy
import numpy as np

import logging

import multiResData as mrd
import PoseTools
import tfdatagen as opdata
import apt_dpk

__all__ = ["TrainingGeneratorTFRecord"]

logr = logging.getLogger('APT')


# Adapter class for use with DeepPoseKit. Some code adapted or taken from DeepPoseKit by Jake Graving et al
# https://github.com/jgraving/deepposekit

class TrainingGeneratorTFRecord:
    """
    TrainingGenerator mock class for use with APT. Reads from tfrecords
    rather than using a dpk DataGenerator; otherwise acts like a
    TrainingGenerator.


    Parameters (from TrainingGenerator; now set from APT conf)
    ----------
    downsample_factor : int, default = 0
        The factor for determining the output shape of the confidence
        maps for estimating keypoints. This is determined as
        shape // 2**downsample_factor. The default is 0, which
        produces confidence maps that are the same shape
        as the input images.
    use_graph : bool, default = True
        Whether to generate confidence maps for the parent graph
        as lines drawn between connected keypoints. This can help reduce
        keypoint estimation error when training the network.
    augmenter : class or list, default = None
        A imgaug.Augmenter, or list of imgaug.Augmenter
        for applying augmentations to images and keypoints.
        Default is None, which applies no augmentations.
    shuffle : bool, default = True
        Whether to randomly shuffle the data.
    sigma : float, default = 3
        The standard deviation of the Gaussian confidence peaks. HI/INPUT RES
        This is scaled to sigma // 2**downsample_factor.
    validation_split : float, default = 0.0
        Float between 0 and 1. Fraction of the training data to be used
        as validation data. The generator will set apart this fraction
        of the training data, will not generate this data unless
        the `validation` flag is set to True when the class is called.
    graph_scale : float, default = 1.0
        Float between 0 and 1. A factor to scale the edge
        confidence map values to y * edge_scale.
        The default is 1.0 which does not scale the confidence
        values. This is useful for preventing the edge channels
        from dominating the error when training a smaller network.
        This arg is not used when `use_graph` is set to False.
    random_seed : int, default = None
        set random seed for selecting validation data
    """

    def __init__(
        self,
        conf0,
        random_seed=None,
    ):

        self.random_seed = random_seed
        if self.random_seed:
            np.random.seed(self.random_seed)

        conf = copy.deepcopy(conf0)

        self.conf = conf

        downsample_factor = conf.dpk_downsample_factor
        assert isinstance(downsample_factor, int) and downsample_factor >= 0
        self.downsample_factor = downsample_factor
        conf.dpk_output_sigma = conf.dpk_input_sigma / 2.0 ** downsample_factor

        if conf.dpk_output_sigma < 0.5:
            logr.warning("Small output sigma: dpk_output_sigma={}".format(conf.dpk_output_sigma))

        valtfr = os.path.join(conf.cachedir, conf.valfilename) + '.tfrecords'
        trntfr = os.path.join(conf.cachedir, conf.trainfilename) + '.tfrecords'
        assert os.path.exists(trntfr), "path {} not found".format(trntfr)
        if not os.path.exists(valtfr):
            logr.info("Cannot find val db; using train db {}".format(trntfr))
            valtfr = trntfr

        self.trntfr = trntfr
        self.valtfr = valtfr
        self.n_train = PoseTools.count_records(trntfr)
        self.n_validation = PoseTools.count_records(valtfr)
        trnmddict = mrd.read_tfrecord_metadata(trntfr)
        valmddict = mrd.read_tfrecord_metadata(valtfr)
        for k in ['height', 'width', 'depth']:
            assert trnmddict[k] == valmddict[k]

        logr.info("TGTFR. Using trn={}, ntrn={}".format(trntfr, self.n_train))
        logr.info("TGTFR. Using val={}, nval={}".format(valtfr, self.n_validation))

        # as far as TGTFR and all DPK-related, the ims have post-pad, post-rescale sz
        self.height = conf.dpk_imsz_net[0]
        self.width = conf.dpk_imsz_net[1]
        self.n_channels = trnmddict['depth']
        self.image_shape = (self.height, self.width, self.n_channels)
        conf.dpk_output_shape = (
            self.height // 2 ** downsample_factor,
            self.width // 2 ** downsample_factor,
        )
        #self.n_keypoints = conf.n_classes

        # Initialize skeleton attributes
        # self.graph = self.generator.graph
        # self.swap_index = self.generator.swap_index

        if conf.dpk_use_augmenter:
            augtype = conf.dpk_augmenter_type['type']
            conf.dpk_augmenter = apt_dpk.make_imgaug_augmenter(
                augtype, conf.dpk_swap_index
            )
            logr.info("TGTFR. created dpk_augmenter, type {}, swapidx {}.".format(
                augtype, conf.dpk_swap_index))

        # self.on_epoch_end()
        # we get a row here just to figure out shapes/sizes
        # batch_size not important
        g = self.get_generator(batch_size=self.conf.batch_size,
                               validation=False,
                               confidence=True,
                               debug=True,
                               silent=True)
        ims0, tgts0, locs0, info0 = next(g)
        if isinstance(tgts0, list):
            tgts0 = tgts0[0]
        # tgts0 can be a list if conf.dpk_n_outputs > 1.
        # dpk_n_outputs can mutate later but n_output_channels should not change
        self.n_output_channels = tgts0.shape[-1]
        self.keypoints_shape = locs0.shape[1:]
        assert self.keypoints_shape == (conf.n_classes, 2,)
        logr.info("TGTFR. n_output_chans={}".format(self.n_output_channels))

        self.use_tfdata = getattr(conf, 'dpk_use_tfdata', True) # set to True to use tfdatas instead of generators

        logr.info("TGTFR. use_tfdata: {}".format(self.use_tfdata))

    def get_tfdataset(self, batch_size, validation, confidence, n_outputs,
                      shuffle=None, infinite=None, **kwargs):

        distort = not validation

        if not confidence:
            assert n_outputs == 1

        if shuffle is None:
            shuffle = not validation

        if infinite is None:
            infinite = True

        #assert not (shuffle and not infinite)  # shuffling can skip a lot of records

        for k in kwargs:
            logr.info("Ignoring kwarg: {}".format(k))

        ds = opdata.create_tf_datasets(self.conf,
                                       batch_size,
                                       n_outputs,
                                       is_val=validation,
                                       distort=distort,
                                       shuffle=shuffle,
                                       infinite=infinite,
                                       drawconf=confidence,
                                       )
        return ds

    def get_generator(self, batch_size, validation, confidence,
                      shuffle=None, infinite=None, **kwargs):
        '''


        :param validation:
        :param confidence:
        :param shuffle: opt, if not provided is based on validation
        :param infinite:
            validation_data+infinite notes:
            * for fit_generator, validation_data, we want infinite as the validation_steps is specified and K will just
            call the generator repeatedly.
            * for APTKerasCbk, again we want infinite
            * the only time we don't want infinite is with val data and dpk/engine/evaluate, which currently is written
            as an eval over the entire val generator.

        :param kwargs:
        :return:
        '''

        tfrfilename = self.valtfr if validation else self.trntfr
        distort = not validation

        if confidence:
            ppfcn = 'ims_locs_preprocess_dpk'
        else:
            ppfcn = 'ims_locs_preprocess_dpk_noconf_nodistort'
            assert self.conf.dpk_n_outputs == 1

        if shuffle is None:
            shuffle = not validation

        if infinite is None:
            infinite = True

        assert not (shuffle and not infinite)  # shuffling can skip a lot of records

        g = opdata.make_data_generator(
            tfrfilename,
            self.conf,
            distort,
            shuffle,
            ppfcn,
            infinite=infinite,
            batch_size=batch_size,
            **kwargs,
        )
        return g

    '''
    def __len__(self):
        """The number of batches per epoch"""
        if self.validation:
            return self.n_validation // self.batch_size
        else:
            return self.n_train // self.batch_size
    '''

    def __call__(self, n_outputs=1, batch_size=32, validation=False, confidence=True, **kwargs):
        """
        Return a generator

        Parameters
        ----------
        n_outputs : int, default = 1
            The number of outputs to generate.
            This is needed for applying intermediate supervision
            to a network with multiple output layers.
        batch_size : int, default = 32
            Number of samples in each batch
        validation: bool, default False
            If set to True, will generate the validation set.
            Otherwise, generates the training set.
        confidence: bool, default True
            If set to True, will generate confidence maps.
            Otherwise, generates keypoints.

        """

        if batch_size != self.conf.batch_size:
            logr.warning('batch specification ({}) differs from conf.batch_size ({})!'.format(
                batch_size, self.conf.batch_size))

        if self.use_tfdata:
            return self.get_tfdataset(batch_size, validation, confidence, n_outputs, **kwargs)
        else:
            # This is dumb, this is only set here to pass into tfdatagen.data_generator;
            # it shouldn't persist. however self.conf is a deepcopied/separate obj now

            self.conf.dpk_n_outputs = n_outputs
            return self.get_generator(batch_size,
                                      validation,
                                      confidence,
                                      **kwargs)

    def get_config(self):
        #if self.augmenter:
        #    augmenter = True
        #else:
        #    augmenter = False
        config = {
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            # "validation_split": self.validation_split,
            "downsample_factor": self.downsample_factor,
            "output_shape": self.conf.dpk_output_shape,
            "n_output_channels": self.n_output_channels,
            # "shuffle": self.shuffle,
            #"sigma": self.sigma,
            "output_sigma": self.conf.dpk_output_sigma,
            "use_graph": self.conf.dpk_use_graph,
            "graph_scale": self.conf.dpk_graph_scale,
            "random_seed": self.random_seed,
            "use_augmenter": self.conf.dpk_use_augmenter,
            "augmenter_type": self.conf.dpk_augmenter_type,
            "augmenter": repr(self.conf.dpk_augmenter),
            "image_shape": self.image_shape,
            "keypoints_shape": self.keypoints_shape,
            "use_tfdata": self.use_tfdata,
        }
        return config  # xxxAL image_height
        #base_config = self.generator.get_config()
        #return dict(list(config.items()) + list(base_config.items()))

    '''
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        idx0 = index * self.batch_size
        idx1 = (index + 1) * self.batch_size
        if self.validation:
            indexes = self.val_range[idx0:idx1]
        else:
            indexes = self.train_range[idx0:idx1]

        # Generate data
        X, y = self.generate_batch(indexes)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.train_range = np.arange(self.n_train)
        self.val_range = np.arange(self.n_validation)
        if self.shuffle:
            np.random.shuffle(self.train_range)
            np.random.shuffle(self.val_range)

    def load_batch(self, indexes):
        if self.validation:
            batch_index = self.val_index[indexes]
        else:
            batch_index = self.train_index[indexes]
        return self.generator[batch_index]

    def augment(self, images, keypoints):
        images_aug = []
        keypoints_aug = []
        for idx in range(images.shape[0]):
            images_idx = images[idx, None]
            keypoints_idx = keypoints[idx, None]
            augmented_idx = self.augmenter(images=images_idx, keypoints=keypoints_idx)
            images_aug_idx, keypoints_aug_idx = augmented_idx
            images_aug.append(images_aug_idx)
            keypoints_aug.append(keypoints_aug_idx)

        images_aug = np.concatenate(images_aug)
        keypoints_aug = np.concatenate(keypoints_aug)
        return images_aug, keypoints_aug

    def generate_batch(self, indexes):
        """Generates data containing batch_size samples"""
        X, y = self.load_batch(indexes)
        if self.augmenter and not self.validation:
            X, y = self.augment(X, y)
        if self.confidence:
            y = draw_confidence_maps(
                images=X,
                keypoints=y,
                graph=self.graph,
                output_shape=self.output_shape,
                use_graph=self.use_graph,
                sigma=self.output_sigma,
            )
            y *= 255
            if self.use_graph:
                y[..., self.n_keypoints:] *= self.graph_scale  # scale grps, limbs, globals
        if self.n_outputs > 1:
            y = [y for idx in range(self.n_outputs)]

        return X, y
    '''

