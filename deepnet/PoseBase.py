'''
    This class provides the easiest way to add your own networks to APT. To add your networks, if your network name is _mynet_, you'll have to create a new file Pose_mynet.py in which you need to define a new class of the same name (Pose_mynet) which should inherit from PoseBase.
    ```
    from PoseBase import PoseBase
    class Pose_mynet(PoseBase):
        def train(self):
        ...
    ```

    If the network generates heatmaps for each landmark (i.e., body part), then you need to override

    * `PoseBase.PoseBase.create_network`:
    * Supply the appropriate hmaps_downsample to the __init__ function when you inherit. Eg, self.hmaps_downsample = 8

    If your networks output is different than only heatmaps, you'll have to override

    * `PoseBase.PoseBase.preproc_func`: (to generate the target outputs other than heatmaps).
    * `PoseBase.PoseBase.create_network`:
    * `PoseBase.PoseBase.loss`:
    * `PoseBase.PoseBase.convert_preds_to_locs`:

    In addition, if you want to change the training procedure, you'll have to override

    * `PoseBase.PoseBase.train` function:
    * `PoseBase.PoseBase.get_pred_fn` function:

    To use pretrained weights, put the location of the pretrained weights in self.pretrained_weights. Eg
    ``` def __init__(self, conf):
            ...
            script_dir = os.path.dirname(os.path.realpath(__file__))
            wt_dir = os.path.join(script_dir, 'pretrained')
            self.pretrained_weights =  os.path.join(wt_dir,'resnet_v1_50.ckpt')
    ```

    The Pose_mynet object once created, will have access to configuration settings that were set by the user in APT GUI in the self.conf object. Some of the useful configuration settings are:

    * imsz: 2 element tuple specifying the size of the input image.
    * img_dim: Number of channels in input image.
    * batch_size: Batch size defined by user.
    * rescale: How much to downsample the input image before feeding into the network.
    * dl_steps: Number of steps to run training for.
    Details of other settings can be found APT/tracker/dt/params_deeptrack.json.

'''

from PoseCommon_dataset import PoseCommon, initialize_remaining_vars
import PoseTools
import tensorflow
vv = [int(v) for v in tensorflow.__version__.split('.')]
if (vv[0]==1 and vv[1]>12) or vv[0]==2:
    tf = tensorflow.compat.v1
else:
    tf = tensorflow

import logging
import sys
import numpy as np
import os
import math

class PoseBase(PoseCommon):
    '''
    '''


    def __init__(self, conf, name='deepnet',hmaps_downsample=1):
        ''' Initialize the pose object.

        Args:
            conf: Configuration object has all the parameters defined in params_netname.yaml.
            hmaps_downsample: The amount that the networks heatmaps are downsampled as compared to input image. Ignored if preproc_func is overridden.

        '''

        PoseCommon.__init__(self, conf,name=name)
        self.hmaps_downsample = hmaps_downsample
        self.conf.use_pretrained_weights = True
        self.pretrained_weights = None

    def get_var_list(self):
        return tf.global_variables()


    def preproc_func(self,*args, distort=True):
        '''
        Override this function to change how images are preprocessed. Ensure that the return objects are float32.
        This function is added into tensorflow dataset pipeline using tf.py_func. The outputs returned by this function are available tf tensors in self.inputs array.
        Args:
            ims: Input image as B x H x W x C
            locs: Labeled part locations as B x N x 2
            info: Information about the input as B x 3. (:,0) is the movie number, (:,1) is the frame number and (:,2) is the animal number (if the project has trx).
            distort: Whether to augment the data or not.
        Returns:
            List as [augmented images, augmented labeled locations, input information, heatmaps]
        '''
        ims,locs,info = args[:3]
        if len(args)>3:
            occ = args[3]
        else:
            occ = np.zeros(locs.shape[:-1])

        conf = self.conf
        # Scale and augment the training image and labels
        ims, locs = PoseTools.preprocess_ims(ims, locs, conf, distort, conf.rescale)
        out = self.convert_locs_to_targets(locs,occ)
        # Return the results as float32.
        out_32 = [o.astype('float32') for o in out]
        return [ims.astype('float32'), locs.astype('float32'), info.astype('float32')] + out_32


    def setup_network(self):
        '''
        Setups the network prediction and loss.
        :return:
        '''
        pred = self.create_network()
        self.pred = pred
        self.cost = self.loss(self.inputs[3:],pred)


    def create_network(self):
        '''
        Use self.inputs to create a network.
        By default (if preproc_function is not overridden), the training data is supplied as a list in self.inputs. For batch size B, image size H x W x C and downsample factor s and N landmarks:
            self.inputs[0] tensor has the images [B,H//s,W//s,C]
            self.inputs[1] tensor has the locations in an array of [B,N,2]
            self.inputs[2] tensor has the [movie_id, frame_id, animal_id] information. Mostly useful for debugging.
            self.inputs[3] tensor has the heatmaps, which should ideally not be used for creating the network. Use it only for computing the loss.
        Information about whether it is training phase or test phase for batch norm is available in self.ph['phase_train']
        If preproc function is overridden, then self.inputs will have the outputs of preproc function.
        This function must return the network's output such as the predicted heatmaps.
        '''
        assert False, 'This function must be overridden'
        return None


    def convert_locs_to_targets(self,locs,occ):
        '''
        Override this function to change how labels are converted into target heatmaps.
        You can use PoseTools.create_label_images to generate the target heatmaps.
        You can use PoseTools.create_affinity_labels to generate the target part affinity field heatmaps.
        Return the results as a list. This list will available as the first input to loss function.
        '''

        conf = self.conf
        hmaps_rescale = self.hmaps_downsample
        hsz = [ math.ceil( (i // conf.rescale)/hmaps_rescale) for i in conf.imsz]
        # Creates target heatmaps by placing gaussians with sigma label_blur_rad at location locs.
        hmaps = PoseTools.create_label_images(locs/hmaps_rescale, hsz, 1, conf.label_blur_rad)
        return [hmaps]



    def convert_preds_to_locs(self, pred):
        '''
        Converts the networks output to 2D predictions.
        Override this function to write your function to convert the networks output (as numpy array) to locations. Note the locations should be in input images scale i.e., not downsampled by self.rescale
        :param pred: Output of network as python/numpy arrays
        :return: 2D locations as batch_size x num_pts x 2
        '''
        return PoseTools.get_pred_locs(pred)*self.hmaps_downsample


    def get_conf_from_preds(self, pred):
        '''
        Computes the prediction confidence.
        :param pred: Output of network as python/numpy arrays
        :return: Confidence for predictions. batch_size x num_pts
        '''
        return np.max(pred, axis=(1, 2))


    def compute_dist(self, pred, locs):
        '''
        Function is used by APT to display the mean pixel error during training
        :param preds: predictions from define_network
        :param locs: label locations as python array
        :return: Mean pixel error between network's prediction and labeled location.
        '''
        tt1 = self.convert_preds_to_locs(pred) - locs
        tt1 = np.sqrt(np.sum(tt1 ** 2, 2))
        return np.nanmean(tt1)


    def loss(self,targets, pred):
        '''
        :param targets: Has the targets (e.g hmaps/pafs) created in convert_locs_to_targets
        :param pred: Has the output of network created in define_network
                    It'll be a list if networks output was a list.
        :return: The loss function to be optimized.
        Override this to define your own loss function.
        '''
        hmap_loss = tf.sqrt(tf.nn.l2_loss(targets[0] - self.pred)) / self.conf.label_blur_rad / self.conf.n_classes
        if self.conf.get('normalize_loss_batch',False):
            hmap_loss = hmap_loss/self.conf.batch_size

        return hmap_loss


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


    def train_wrapper(self, restore=False,model_file=None):
        '''
        Sets up the inputs pipeline, the network and the loss function
        '''

        self.find_input_sizes()

        def train_pp(*args):
            return self.preproc_func(*args, distort=True)
        def val_pp(*args):
            return self.preproc_func(*args, distort = False)

        self.train_py_map = lambda *args: tuple(tf.py_func( train_pp,args, self.input_dtypes))
        self.val_py_map = lambda *args: tuple(tf.py_func( val_pp, args, self.input_dtypes ))

        self.setup_train()
        self.set_input_sizes()

        # create the network
        self.setup_network()

        # train
        self.train(restore=restore,model_file=model_file)

        # reset in case we want to use tensorflow for other stuff.
        tf.reset_default_graph()


    def train(self, restore=False,model_file=None):
        '''
        :param restore: Whether to start training from previously saved model or start from scratch.
        :return:

        This function trains the network my minimizing the loss function using Adam optimizer along with gradient norm clipping.
        The learning rate schedule is exponential decay: lr = conf.learning_rate*(conf.gamma**(float(cur_step)/conf.decay_steps))
        Override this function to implement a different training function. If you override this, do override the get_pred_fn as well.
        For saving the models, call self.create_saver() after creating the network but before creating a session to setup the saver. Call self.save(sess, step_no)  every self.conf.save_step to save intermediate models. At the end of training, again call self.save(sess, self.conf.dl_steps) to save the final model. During prediction (get_pred_fn), you can restore the latest model by calling self.create_saver() after creating the network but before creating a session, and then calling self.restore(sess). In most cases, if you use self.create_saver(), then you don't need to override get_pred_fn().
        If you want to use your own saver, the train function should save models to self.conf.cachedir every self.conf.save_step. Also save a final model at the end of the training with step number self.conf.dl_steps. APT expects the model files to be named 'deepnet-<step_no>' (i.e., follow the format used by tf.train.Saver for saving models e.g. deepnet-10000.index). If you write your own saver, then you'll have to override get_pred_fn() for restoring the model.
        Before each training step call self.fd_train() which will setup the data generator to generate augmented input images in self.inputs from training database during the next call to sess.run. This also sets the self.ph['phase_train'] to True for batch norm. Use self.fd_val() will generate non-augmented inputs from training DB and to set the self.ph['phase_train'] to false for batch norm.
        To view updated training status in APT, call self.update_and_save_td(step,sess) after each training step. Note update_and_save_td uses the output of loss function to find the loss and convert_preds_to_locs function to find the distance between prediction and labeled locations.
        '''

#        base_lr = self.conf.learning_rate
        learning_rate = self.conf.get('learning_rate_multiplier',1.)*self.conf.get('base_lr',0.0001)

        PoseCommon.train_quick(self, learning_rate=learning_rate,restore=restore,model_file=model_file)



    def get_pred_fn(self, model_file=None):
        '''
        :param model_file: Model_file to use. If not specified the latest trained should be used.
        :return: pred_fn: Function that predicts the 2D locations given a batch of input images.
        :return close_fn: Function to call to close the predictions (eg. the function should close the tensorflow session)
        :return model_file_used: Returns the model file that is used for prediction.

        Creates a prediction function that returns the pose prediction as a python array of size [B, N, 2].
        This function should create the network, start a tensorflow session and load the latest model.
        If you used self.create_saver() for saving the models, you can restore the latest model by calling self.create_saver() after creating the network but before creating a session, and then calling self.restore(sess)
        '''

        try:
            sess, latest_model_file = self.restore_net_common(model_file=model_file)
        except tf.errors.InternalError:
            logging.exception(
                'Could not create a tf session. Probably because the CUDA_VISIBLE_DEVICES is not set properly')
            sys.exit(1)

        conf = self.conf
        def pred_fn(all_f):
            '''
            :param all_f:
            :return:
            This is the function that is used for predicting the location on a batch of images.
            The input is a numpy array B x H x W x C of images, and
            output an numpy array of predicted locations.
            predicted locations should be B x N x 2
            PoseTools.get_pred_locs can be used to convert heatmaps into locations.
            The predicted locations should be in the original image scale.
            The predicted locations, confidence and hmaps(if used) should be returned in a dict with keys 'locs', 'conf' and 'hmaps'.
            If overriding this function, ensure that you call self.fd_val() which will set the self.ph[phase_train] to false for batch norm.
            '''

            bsize = conf.batch_size
            xs, locs_in = PoseTools.preprocess_ims(all_f, in_locs=np.zeros([bsize, self.conf.n_classes, 2]), conf=self.conf, distort=False, scale=self.conf.rescale)

            self.fd[self.inputs[0]] = xs
            self.fd_val()
            try:
                pred = sess.run(self.pred, self.fd)
            except tf.errors.ResourceExhaustedError:
                logging.exception('Out of GPU Memory. Either reduce the batch size or scale down the images')
                exit(1)
            base_locs = self.convert_preds_to_locs(pred)
            base_locs = base_locs * conf.rescale
            ret_dict = {}
            ret_dict['locs'] = base_locs
            ret_dict['hmaps'] = pred
            ret_dict['conf'] = self.get_conf_from_preds(pred)
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
