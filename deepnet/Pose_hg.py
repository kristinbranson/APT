from PoseBaseGeneral import PoseBaseGeneral
import tensorflow as tf
import time
import datetime
import os

from hourglass.hourglass_tiny import HourglassModel

#from tensorflow.contrib.layers import batch_norm
import logging
import sys
import numpy as np


def make_gaussian_al(height, width, sigma=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    sigma is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]

    # AL: don't know why this normalization
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

def generate_hm_al(height, width, joints, s):
    """ Generate a full Heap Map for every joint in an array
    Args:
        height			: Wanted Height for the Heat Map
        width			: Wanted Width for the Heat Map
        joints			: Array of Joints n x 2
        s  				: sigma for Gaussian

    Returns:
        hm 				: height x width x num_joints heatmaps
    """

    num_joints = joints.shape[0]
    hm = np.zeros((height, width, num_joints), dtype=np.float32)
    for i in range(num_joints):
        if True:  # not(np.array_equal(joints[i], [-1,-1])) and weight[i] == 1:
            hm[:, :, i] = make_gaussian_al(height, width, sigma=s, center=(joints[i, 0], joints[i, 1]))
    # else:
    # hm[:,:,i] = np.zeros((height,width))
    return hm


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

        (imnr, imnc) = self.conf.imsz
        assert imnr % 4 == 0, "Image size must be divisible by 4"
        assert imnc % 4 == 0, "Image size must be divisible by 4"

        self.gtsz = (imnr/4, imnc/4)
        self.hgmodel = None  # scalar HourglassModel


    def convert_locs_to_targets(self, locs):
        '''
        Override this function to to convert labels into targets (e.g. heatmaps).
        You can use PoseTools.create_label_images to generate the target heatmaps.
        You can use PoseTools.create_affinity_labels to generate the target part affinity field heatmaps.
        Return the results as a list. This list will be available as tensors self.inputs[3], self.inputs[4] and so on for computing the loss.

        locs: Landmark locations b x n x 2. 3rd dimension is x/y. upper-left pixel is (0,0).

        Returns:
             Single-element list [gtmaps] where gtmaps is
                b x self.hgmodel.nStack x self.gtsz[0] x self.gtsz[1] x n
        '''

        bsize = self.conf.batch_size
        npts = locs.shape[1]
        nstack = 1
        assert locs.shape == (bsize, npts, 2), "Dimension mismatch"
        sigma = self.conf.label_blur_rad
        (gtnr, gtnc) = self.gtsz

        locsgt = locs.copy() / 4.0
        gtmapsz = (bsize, nstack, gtnr, gtnc, npts)
        gtmaps = np.zeros(gtmapsz, np.float32)
        for i in range(bsize):
            locsgt_this = locsgt[i, :, :]
            hm = generate_hm_al(gtnr, gtnc, locsgt_this, sigma)
            hm = np.expand_dims(hm, axis=0)
            hm = np.repeat(hm, nstack, axis=0)
            gtmaps[i] = hm

        return [gtmaps]


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

        (imgs, locs, _, gtmaps) = self.inputs[0:4]
        (nBatch, npts) = locs.shape.as_list()[0:2]

        hgm = HourglassModel(
            nStack=1,
            nFeat=256,
            nLow=4,
            outputDim=npts,
            batch_size=nBatch,
            drop_rate=0.2,
            lear_rate=2.5e-4,
            decay=0.96,
            decay_step=2000,
            logdir_train=self.conf.cachedir,
            logdir_test=self.conf.cachedir,
            training=True)

        self.hgmodel = hgm

        hgm.generate_model_al(imgs,gtmaps)


        nEpochs = 1
        epochSize = self.conf.dl_steps
        savestep = self.conf.save_step
        dispstep = self.conf.save_td_step

        with tf.name_scope('Session'):
            with tf.device(hgm.gpu):
                hgm._init_weight()
                hgm._define_saver_summary(summary=False)
                #self._train(nEpochs, epochSize, saveStep, validIter=10)
                with tf.name_scope('Train'):
                    startTime = time.time()
                    #self.resume = {}
                    #self.resume['accur'] = []
                    #self.resume['loss'] = []
                    #self.resume['err'] = []
                    for epoch in range(nEpochs):
                        epochstartTime = time.time()
                        avg_cost = 0.
                        cost = 0.
                        print('Epoch :' + str(epoch) + '/' + str(nEpochs) + '\n')
                        # Training Set
                        for i in range(epochSize):
                            #percent = ((i + 1) / epochSize) * 100
                            #num = np.int(20 * percent / 100)
                            #tToEpoch = int((time.time() - epochstartTime) * (100 - percent) / (percent))
                            #sys.stdout.write(
                            #    '\r Train: {0}>'.format("=" * num) + "{0}>".format(" " * (20 - num)) + '||' + str(
                            #        percent)[:4] + '%' + ' -cost: ' + str(cost)[:6] + ' -avg_loss: ' +
                            #        str(avg_cost)[:5] + ' -timeToEnd: ' + str(tToEpoch) + ' sec.')
                            #sys.stdout.flush()

                            #img_train, gt_train, weight_train = next(self.generator)
                            self.fd_train()
                            if i % savestep == 0:
                                _, c, summary = hgm.Session.run([hgm.train_rmsprop, hgm.loss, hgm.train_op])
                                # Save summary (Loss + Accuracy)
                                #self.train_summary.add_summary(summary, epoch * epochSize + i)
                                #self.train_summary.flush()
                                with tf.name_scope('save'):
                                    save_path = os.path.join(self.conf.cachedir, 'deepnet')
                                    hgm.saver.save(hgm.Session,
                                                   save_path,
                                                   global_step=i,
                                                   write_meta_graph=False)
                                    logging.info('Saved state to %s-%d' % (save_path, i))
                            else:
                                _, c, = hgm.Session.run([hgm.train_rmsprop, hgm.loss])
                            if i % dispstep == 0:
                                logging.info('Loss is {:8.4f}'.format(c))
                            #cost += c
                            #avg_cost += c / epochSize
                        epochfinishTime = time.time()
                        # Save Weight (axis = epoch)
                        #weight_summary = self.Session.run(self.weight_op,
                        #                                  {self.img: img_train, self.gtMaps: gt_train})
                        #self.train_summary.add_summary(weight_summary, epoch)
                        #self.train_summary.flush()
                        # self.weight_summary.add_summary(weight_summary, epoch)
                        # self.weight_summary.flush()
                        #print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(
                        #    int(epochfinishTime - epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(
                        #    ((epochfinishTime - epochstartTime) / epochSize))[:4] + ' sec.')
                        # self.resume['loss'].append(cost)
                        # Validation Set
                        # accuracy_array = np.array([0.0] * len(self.joint_accur))
                        # for i in range(validIter):
                        #    img_valid, gt_valid, w_valid = next(
                        #        self.generator)  # XXXAL looks like a bug want self.valid_gen
                        #    accuracy_pred = self.Session.run(self.joint_accur,
                        #                                     feed_dict={self.img: img_valid, self.gtMaps: gt_valid})
                        #    accuracy_array += np.array(accuracy_pred, dtype=np.float32) / validIter
                        #print('--Avg. Accuracy =', str((np.sum(accuracy_array) / len(accuracy_array)) * 100)[:6], '%')
                        #self.resume['accur'].append(accuracy_pred)
                        #self.resume['err'].append(np.sum(accuracy_array) / len(accuracy_array))
                        #valid_summary = self.Session.run(self.test_op,
                        #                                 feed_dict={self.img: img_valid, self.gtMaps: gt_valid})
                        #self.test_summary.add_summary(valid_summary, epoch)
                        #self.test_summary.flush()
                    print('Training Done')
                    #print('Resume:' + '\n' + '  Epochs: ' + str(nEpochs) + '\n' + '  n. Images: ' + str(
                    #    nEpochs * epochSize * self.batchSize))
                    #print('  Final Loss: ' + str(cost) + '\n' + '  Relative Loss: ' + str(
                    #    100 * self.resume['loss'][-1] / (self.resume['loss'][0] + 0.1)) + '%')
                    #print('  Relative Improvement: ' + str(
                    #    (self.resume['err'][-1] - self.resume['err'][0]) * 100) + '%')
                    print('  Training Time: ' + str(datetime.timedelta(seconds=time.time() - startTime)))




