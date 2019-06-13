from PoseBaseGeneral import PoseBaseGeneral
import tensorflow as tf

from hourglass.hourglass_tiny import HourglassModel

#from tensorflow.contrib.layers import batch_norm
import logging
import sys
import numpy as np


class Pose_hg(PoseBaseGeneral):
    '''
        Inherit this class to use your own network with APT.
        If the name of your networks is <netname>, then the new class should be created in
        python file Pose_<netname>.py and the class name should also be Pose_<netname>. The new class name should also be the same.
        The function that need to overridden are:
        * convert_locs_to_target
        * train
        * get_pred_fn

        We use the tensorflow dataset pipeline to read and process the input images which makes the training fast. For this reason, image preprocessing and target creation (e.g. generating heatmaps) functions have to be defined separately so that they can be injected in the dataset pipeline.
        Override preprocess_ims if you want to define your own image pre-processing function. By default, it'll down sample the input image (if required), augment the images using the augmentation, and adjust contrast (if required).
        In convert_locs_to_target, create target outputs (e.g. heatmaps) that you want the network the predict.
        In train, create the network, train and save trained models.
        In get_pred_fn, restore the network and create a function that will predict the landmark locations on input images.

    '''


    def __init__(self, conf):
        '''
        Initialize the pose object.

        :param conf: Configuration object has all the parameters defined in params_netname.yaml. Some important parameters are:
            imsz: 2 element tuple specifying the size of the input image.
            img_dim: Number of channels in input image
            batch_size
            rescale: How much to downsample the input image before feeding into the network.
            dl_steps: Number of steps to run training for.
            In addition, any configuration setting defined in APT_basedir/trackers/dt/params_<netname>.yaml will be available to objects Pose_<netname> in file Pose_<netname> are created.

        '''

        PoseBaseGeneral.__init__(self, conf)
        self.hgmodel = None  # scalar HourglassModel

    def convert_locs_to_targets(self, locs):
        '''
        Override this function to to convert labels into targets (e.g. heatmaps).
        You can use PoseTools.create_label_images to generate the target heatmaps.
        You can use PoseTools.create_affinity_labels to generate the target part affinity field heatmaps.
        Return the results as a list. This list will be available as tensors self.inputs[3], self.inputs[4] and so on for computing the loss.
        '''

        assert False, 'This function must be overridden'

    def train(self):
        '''
        :return:

        Implement network creation and and its training in this function. The input and output tensors are in self.inputs.
        self.inputs[0] has the preprocessed and augmented images as b x h x w x c
        self.inputs[1] has the landmark positions as b x n x 2
        self.inputs[2] has information about the movie number, frame number and trx number as b x 3
        self.inputs[3] onwards has the outputs that are produced by convert_locs_to_targets
         that provice the
        The train function should save models to self.conf.cachedir every self.conf.save_step. Also save a final model at the end of the training with step number self.conf.dl_steps. APT expects the model files to be named 'deepnet-<step_no>' (i.e., follow the format used by tf.train.Saver for saving models e.g. deepnet-10000.index).
        Before each training step call self.fd_train() which will setup the data generator to generate augmented input images in self.inputs from training database during the next call to sess.run. This also sets the self.ph['phase_train'] to True which can be used by batch norm. Use self.fd_val() will generate non-augmented inputs from training DB and to set the self.ph['phase_train'] to false for batch norm.
        To view updated training metrics in APT training update window, call self.append_td(step,train_loss,train_dist) every self.conf.display_step. train_loss is the current training loss, while train_dist is the mean pixel distance between the predictions and labels.
        '''


        hgm = HourglassModel(
            nStack = 1,
            nFeat = 256,
            nLow = 4,
            outputDim = XXXXX CONF,
        self.batchSize = batch_size
        self.dropout_rate = drop_rate
        self.learning_rate = lear_rate
        self.decay = decay
        self.decay_step = decay_step
        self.logdir_train = logdir_train
        self.logdir_test = logdir_test
        self.training = training)


        hgm.generate_model_al(imph,gtm)


        assert False, 'This function should be overridden'



