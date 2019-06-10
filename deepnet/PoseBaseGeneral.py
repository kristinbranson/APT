from PoseCommon_dataset import PoseCommon, initialize_remaining_vars
import PoseTools
import tensorflow as tf
import logging
import sys
import numpy as np
import os
import math
from collections import OrderedDict

class PoseBaseGeneral(PoseCommon):
    '''
    Inherit this class to use your own network with APT.
    If the name of your networks is <netname>, then the new class should be created in python file Pose_<netname>.py and the class name should also be Pose_<netname>. The new class name should also be the same.
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

        PoseCommon.__init__(self, conf,name='deepnet')

    def get_var_list(self):
        return tf.global_variables()


    def preproc_func(self, ims, locs, info, distort):
        '''
        This function is added into tensorflow dataset pipeline using tf.py_func. The outputs returned by this function are available tf tensors in self.inputs array.
        :param ims: Input image as B x H x W x C
        :param locs: Labeled part locations as B x N x 2
        :param info: Information about the input as B x 3. (:,0) is the movie number, (:,1) is the frame number and (:,2) is the animal number (if the project has trx).
        :param distort: Whether to augment the data or not.
        :return: augmented images, augmented labeled locations, input information, heatmaps.
        '''

        conf = self.conf
        # Scale and augment the training image and labels
        ims, locs = self.preprocess_ims(ims,locs,conf,distort,conf.rescale)
        out = self.convert_locs_to_targets(locs)
        # Return the results as float32.
        out_32 = [o.astype('float32') for o in out]
        return [ims.astype('float32'), locs.astype('float32'), info.astype('float32')] + out_32


    def preprocess_ims(self,ims,locs,conf,distort,rescale):
        '''
        Override this function to change how images are preprocessed. Ensure that the return objects are float32.

        :param ims: Batch of input images b x h x w x c
        :param locs: Landmark locations b x n x 2
        :param conf: Configuration object
        :param distort: Whether to distort for augmentation or not.
        :param rescale: Downsample the image by this much.
        :return: [ims,locs] Returns preprocessed and augmented images and landmark locations

        '''
        ims, locs = PoseTools.preprocess_ims(ims, locs, conf, distort, rescale)
        return ims, locs

    def convert_locs_to_targets(self,locs):
        '''
        Override this function to to convert labels into targets (e.g. heatmaps).
        You can use PoseTools.create_label_images to generate the target heatmaps.
        You can use PoseTools.create_affinity_labels to generate the target part affinity field heatmaps.
        Return the results as a list. This list will be available as tensors self.inputs[3], self.inputs[4] and so on for computing the loss.
        '''

        assert False, 'This function must be overridden'


    def find_input_sizes(self):
        # Find the number of outputs from preproc function for the tf.py_func using dummpy inputs, which inserts the preprocessing code into the tensorflow dataset pipeline. The size of output of preproc function is also used to set the size of the tf tensors in self.inputs which will be used during create_network.
        conf = self.conf
        b_sz = conf.batch_size
        imsz = conf.imsz
        img_dim = conf.img_dim
        n_classes = conf.n_classes

        dummy_ims = np.random.rand(b_sz,imsz[0],imsz[1] ,img_dim)
        dummy_locs = np.ones([b_sz,n_classes,2]) * min(imsz)/2
        dummy_info = np.ones([b_sz,3])
        pp_out = self.preproc_func(dummy_ims,dummy_locs,dummy_info,True)
        self.input_dtypes = [tf.float32,]*len(pp_out)
        input_sizes = []
        for i in pp_out:
            input_sizes.append(i.shape)
        self.input_sizes = input_sizes


    def set_input_sizes(self):
        # Set the size for the input tensors
        for ndx, i in enumerate(self.inputs):
            i.set_shape(self.input_sizes[ndx])


    def train_wrapper(self, restore=False):
        '''
        Sets up the inputs pipeline, the network and the loss function
        '''

        self.find_input_sizes()

        def train_pp(ims,locs,info):
            return self.preproc_func(ims,locs,info, True)
        def val_pp(ims,locs,info):
            return self.preproc_func(ims,locs,info, False)

        self.train_py_map = lambda ims, locs, info: tuple(tf.py_func( train_pp, [ims, locs, info], self.input_dtypes))
        self.val_py_map = lambda ims, locs, info: tuple(tf.py_func( val_pp, [ims, locs, info], self.input_dtypes ))

        self.setup_train()
        self.set_input_sizes()
        self.init_td()

        # train
        self.train()

        # reset in case we want to use tensorflow for other stuff.
        tf.reset_default_graph()


    def append_td(self,step,train_loss,train_dist=0.,learning_rate=0.):
        cur_dict = OrderedDict()
        cur_dict['val_dist'] = train_dist
        cur_dict['train_dist'] = train_dist
        cur_dict['train_loss'] = train_loss
        cur_dict['val_loss'] = train_loss
        cur_dict['step'] = step
        cur_dict['l_rate'] = learning_rate
        self.update_td(cur_dict)
        train_data_file = os.path.join( self.conf.cachedir,'traindata')
        self.save_td(train_data_file)


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

        assert False, 'This function should be overridden'


    def setup_pred(self):
        self.find_input_sizes()
        self.create_ph_fd()
        self.create_input_ph()
        self.fd_val()


    def preproc_pred(self,ims):
        '''
        Preprocess the images for predictions
        :param ims: batch of input images b x h x w x c
        :return: Preprocessed images
        '''
        locs = np.zeros([self.conf.batch_size,self.conf.n_classes,2])
        info = np.zeros([self.conf.batch_size,3])
        ims, _ = self.preprocess_ims(ims,locs,self.conf,False,self.conf.rescale)
        return ims


    def get_pred_fn(self, model_file=None):
        '''
        :param model_file: Model_file to use. If not specified the latest trained should be used.
        :return: pred_fn: Function that predicts the 2D locations given a batch of input images.
        :return close_fn: Function to call to close the predictions (eg. the function should close the tensorflow session)
        :return model_file_used: Returns the model file that is used for prediction.

        Creates a prediction function that returns the pose prediction as a python array of size [batch_size,n_pts,2].
        This function should creates the network, start a tensorflow session and load the latest model.

        At the start of the function call:
            self.setup_pred()
        to setup the feed dicts

        At the start of pred_fn call:
            self.preproc_pred(ims)

        Example implementation of functions are shown.

        '''


        assert False, 'This function should be overridden'

        # setup the feed dicts
        self.setup_pred()
        # Ater setup_pred, self.inputs[0] is now  a placeholder in which input images should be fed. It can be used in the same manner as self.inputs are used during training as inputs to network.

        # create the network
        # self.create_network()
        sess = tf.Session()

        # restore the weights
        # self.restore()

        def pred_fn(ims):
            '''
            :param ims:
            :return:
            This is the function that is used for predicting the location on a batch of images.
            The input is a numpy array B x H x W x C of images.
            The predicted locations should be B x N x 2
            The predicted locations should be in the original image scale.
            The predicted locations should be returned in a dict with key 'locs'
            '''

            ims = self.preproc_pred(ims)
            self.fd[self.inputs[0]] = ims
            pred = sess.run(self.pred, self.fd)
            base_locs = self.convert_preds_to_locs(pred)
            base_locs = base_locs * conf.rescale
            ret_dict = {}
            ret_dict['locs'] = base_locs
            return ret_dict

        def close_fn():
            sess.close()

        return pred_fn, close_fn, latest_model_file


    def restore_net_common(self, model_file=None):
        '''
        :param model_file: Model file to restore the network form. If None, use the latest model.
        This function creates the network, creates the session and load the saved model.
        '''
        create_network_fn = self.create_network
        logging.info('--- Loading the model by reconstructing the graph ---')
        self.find_input_sizes()
        self.setup_pred()
        self.pred = create_network_fn()
        self.create_saver()
        sess = tf.Session()
        latest_model_file = self.restore(sess, model_file)
        initialize_remaining_vars(sess)

        try:
            self.restore_td()
        except (AttributeError,IOError):  # If the conf file has been modified
            logging.warning("Couldn't load the training data")
            self.init_td()

        for i in self.inputs:
            self.fd[i] = np.zeros(i.get_shape().as_list())

        return sess, latest_model_file


    def create_input_ph(self):
        # when we want to manually feed in data
        self.inputs = []
        for ndx, cur_d in enumerate(self.input_dtypes):
            self.inputs.append(tf.placeholder(cur_d,self.input_sizes[ndx]))