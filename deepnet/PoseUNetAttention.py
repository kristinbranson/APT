import PoseUNet
import PoseCommon
from multiResData import *
import tensorflow.contrib.graph_editor as ge


class PoseUNetAttention(PoseUNet.PoseUNet):
    def __init__(self, conf, name='pose_u_att',
                 unet_name='pose_unet'):
        PoseCommon.PoseCommon.__init__(self, conf, name)
        self.dep_nets = PoseUNet.PoseUNet(conf, unet_name)
        self.net_name = 'pose_u_att'
        self.dep_nets.keep_prob = 1.
        self.db_name = '_att'

        self.min_score = -1
        self.att_layers = conf.att_layers
        self.att_hist = conf.att_hist  # amount of history for attention
        self.joint = True
        self.debug_layers = None
        self.sel_layers = None
        self.sel_layers_ndx = None

    def create_ph_fd(self):
        PoseCommon.PoseCommon.create_ph_fd(self)
        self.ph['x'] = self.dep_nets.ph['x']
        self.ph['y'] = self.dep_nets.ph['y']
        self.ph['phase_train'] = self.dep_nets.ph['phase_train']
        self.ph['keep_prob'] = self.dep_nets.ph['keep_prob']

        # This should ideally be in create_network().
        # But in this case the placeholders depend on the size of dependent UNet.
        # So we have to create it in here to figure out the sizes.
        self.dep_nets.pred = self.dep_nets.create_network()

        unet = self.dep_nets
        assert self.dep_nets.n_conv == 2, 'this logic only works n_conv =2'
        middle_layer = len(unet.all_layers) / 2
        sel_layers = [(middle_layer + self.dep_nets.n_conv * x - 1) for x in self.att_layers]
        for ndx, s in enumerate(sel_layers):
            sz = unet.all_layers[s].get_shape().as_list()
            self.ph['prev_in_{}'.format(ndx)] = tf.placeholder(tf.float32, [None, self.att_hist, ] + sz[1:])

        self.sel_layers_ndx = sel_layers
        self.ph['scores'] = tf.placeholder(tf.float32, [None, self.att_hist, self.conf.n_classes])
        self.read_and_decode = lambda a, b: read_and_decode(a, b, [self.att_hist, self.sel_layers])

    def create_fd(self):
        self.dep_nets.create_fd()
        self.fd = self.dep_nets.fd
        for ndx, s in enumerate(self.att_layers):
            sz = self.ph['prev_in_{}'.format(ndx)].get_shape().as_list()
            sz[0] = self.conf.batch_size
            self.fd[self.ph['prev_in_{}'.format(ndx)]] = np.zeros(sz)
        self.fd[self.ph['scores']] = np.zeros([self.conf.batch_size, self.att_hist, self.conf.n_classes])

    def fd_train(self):
        self.fd[self.ph['phase_train']] = True
        self.fd[self.ph['keep_prob']] = self.dep_nets.keep_prob

    def fd_val(self):
        self.fd[self.ph['phase_train']] = False
        self.fd[self.ph['keep_prob']] = 1.

    def update_fd(self, db_type, sess, distort):

        self.read_images(db_type, distort, sess, distort, self.conf.unet_rescale)
        self.fd[self.ph['x']] = self.xs
        self.fd[self.ph['scores']] = np.array([d[0] for d in self.extra_data])
        for ndx in range(len(self.sel_layers)):
            cur_name = 'prev_in_{}'.format(ndx)
            self.fd[self.ph[cur_name]] = np.array([d[1][ndx] for d in self.extra_data])

        rescale = self.conf.unet_rescale
        im_sz = [self.conf.imsz[0] / rescale, self.conf.imsz[1] / rescale, ]
        label_ims = PoseTools.create_label_images(
            self.locs / rescale, im_sz, 1, self.conf.label_blur_rad)
        self.fd[self.ph['y']] = label_ims

    def create_network(self):

        unet = self.dep_nets
        sel_layers = self.sel_layers_ndx
        self.sel_layers = []
        for ndx, s in enumerate(sel_layers):
            self.sel_layers.append(unet.all_layers[s])

        debug_layers = []
        with tf.variable_scope('pose_u_att'):
            for ndx, ll in enumerate(self.sel_layers):
                with tf.variable_scope('att_layer_{}'.format(ndx)):
                    n_channels = ll.get_shape().as_list()[-1]

                    # no batch norm for this. 2 fully connected layers to generate the attention weights from scores.
                    int1 = tf.contrib.layers.fully_connected(self.ph['scores'], n_channels, activation_fn=tf.nn.relu)
                    wts = tf.sigmoid(tf.contrib.layers.fully_connected(int1, n_channels, activation_fn=None))
                    # with sigmoid the weights are > 0 and bounded

                    debug_layers.append(wts)

                    # multiply by linear to weight closer inputs more as compared to far away
                    t_wts = tf.range(self.att_hist, 0, -1, dtype=tf.float32) / self.att_hist
                    debug_layers.append(t_wts)
                    t_wts = tf.expand_dims(t_wts, 1)
                    t_wts = tf.expand_dims(t_wts, 0)
                    wts = tf.multiply(t_wts, wts)
                    debug_layers.append(wts)

                    # wts is no Batch x T X C
                    # ensure weights sum to 1 for each channel
                    wts_sum = tf.reduce_sum(wts, axis=1, keep_dims=True)
                    wts = tf.div(wts, wts_sum)
                    debug_layers.append(wts)

                    # exapnd dims to match the activations in prev_in_{}
                    wts = tf.expand_dims(wts, 2)
                    wts = tf.expand_dims(wts, 2)

                    # multiply wts with previous activations
                    cur_activations = self.ph['prev_in_{}'.format(ndx)]
                    att_prod = tf.multiply(wts, cur_activations)
                    debug_layers.append(att_prod)

                    # sum along the dimension 1 to get attention context.
                    # After att_context should have the same size as layer in.
                    att_context = tf.reduce_sum(att_prod, axis=1)
                    debug_layers.append(att_context)

                    # concat with current examples layer.
                    ll_concat = tf.concat([ll, att_context], axis=-1)
                    att_out = PoseCommon.conv_relu3(ll_concat, n_channels, self.ph['phase_train'], self.ph['keep_prob'])

                    layer_2_remove = unet.all_layers[sel_layers[ndx] + 1]
                    ge.swap_outputs(att_out, layer_2_remove)

        self.debug_layers = debug_layers
        return self.dep_nets.pred

    def init_net(self, train_type=0, restore=True):
        self.init_train(train_type=train_type)
        self.create_network()
        self.joint = True

        sess = tf.InteractiveSession()
        self.init_and_restore(sess, restore, ['loss', 'dist'])
        return sess

    def create_tfrecord_trx(self, split=True, distort=True):

        orig_bs = self.conf.batch_size
        if (self.att_hist % self.conf.batch_size) != 0:
            self.conf.batch_size = 1
            nbatches = self.att_hist
            bsz = 1
        else:
            nbatches = self.att_hist / self.conf.batch_size
            bsz = self.conf.batch_size

        if split:
            sess = self.dep_nets.init_net(0, True)
        else:
            sess = self.dep_nets.init_net(1, True)
        unet = self.dep_nets

        # select layers to save
        assert self.dep_nets.n_conv == 2, 'this logic only works n_conv =2'
        middle_layer = len(unet.all_layers) / 2
        sel_layers = [(middle_layer + self.dep_nets.n_conv * x - 1) for x in self.att_layers]
        for ndx, s in enumerate(sel_layers):
            sz = unet.all_layers[s].get_shape().as_list()[1:]
            print('{} -- Selected layer {} with size {}'.format(ndx, -s, sz))
            sz = unet.all_layers[s + 1].get_shape().as_list()[1:]
            print('{} -- Next layer {} with size {}'.format(ndx, -s, sz))
            sz = unet.all_layers[s - 1].get_shape().as_list()[1:]
            print('{} -- Previous layer {} with size {}'.format(ndx, -s, sz))

        preds = [unet.all_layers[s] for s in sel_layers]
        # preds.append(unet.pred)

        # read the detail of previous db to keep the same splits.
        train_info = []
        val_info = []
        if split:
            n_train = PoseTools.count_records(os.path.join(self.conf.cachedir, self.conf.trainfilename + '.tfrecords'))
            n_val = PoseTools.count_records(os.path.join(self.conf.cachedir, self.conf.valfilename + '.tfrecords'))
            for ndx in range(n_train):
                unet.setup_train(sess, distort=False)
                train_info.extend(unet.info)
            for ndx in range(n_val):
                unet.setup_val(sess)
                val_info.extend(unet.info)

        # ##### creating the db. Most stuff is copied from multiResData.createTFRecordFromLblWithTrx

        conf = self.conf
        is_val, local_dirs, _ = load_val_data(conf)

        lbl = h5py.File(conf.labelfile, 'r')
        npts_per_view = np.array(lbl['cfg']['NumLabelPoints'])[0, 0]
        trx_files = get_trx_files(lbl, local_dirs)

        env, valenv = create_envs(conf, split, db_type='att')
        view = conf.view
        count = 0
        valcount = 0
        selpts = int(view * npts_per_view) + conf.selpts

        for ndx, dirname in enumerate(local_dirs):

            trx = sio.loadmat(trx_files[ndx])['trx'][0]
            curpts = trx_pts(lbl, ndx)
            cap = movies.Movie(local_dirs[ndx])

            for trx_ndx in range(len(trx)):
                frames = np.where(np.invert(np.all(np.isnan(curpts[trx_ndx, :, :, :]), axis=(1, 2))))[0]
                # cap = cv2.VideoCapture(localdirs[ndx])
                cur_trx = trx[trx_ndx]

                for fndx, fnum in enumerate(frames):

                    if not check_fnum(fnum, cap, dirname, ndx):
                        continue

                    if split:
                        # TODO handle multiple trx_ndx?? Right now all matches go into validation
                        if val_info.count([ndx, fnum, trx_ndx]) > 0:
                            curenv = valenv
                        else:
                            curenv = env
                    else:
                        curenv = env

                    # initialize stuff
                    ims = np.zeros([self.att_hist, self.conf.imsz[0], self.conf.imsz[1], self.conf.imgDim])
                    cur_preds = []
                    for p in preds:
                        cur_preds.append(np.zeros([self.att_hist, ] + p.get_shape().as_list()[1:]))
                    max_scores = self.min_score * np.ones([self.att_hist, self.conf.n_classes])

                    in_shape = unet.ph['x'].get_shape().as_list()
                    for cur_b in range(nbatches):
                        cur_s = cur_b * bsz
                        cur_e = (cur_b + 1) * bsz
                        xs = np.zeros([bsz, ] + in_shape[1:])

                        for indx, hist in enumerate(range(cur_s, cur_e)):
                            cur_fnum = fnum - hist - 1

                            if not check_fnum(cur_fnum, cap, dirname, ndx):
                                continue

                            framein, curloc = get_patch_trx(cap, cur_trx, cur_fnum, conf.imsz[0],
                                                            curpts[trx_ndx, cur_fnum, :, selpts])
                            framein = framein[:, :, 0:conf.imgDim]
                            ims[hist, ...] = framein

                            cur_xs, _ = PoseTools.preprocess_ims(framein[np.newaxis, ...], curloc[np.newaxis, ...],
                                                                 self.conf, distort=distort,
                                                                 scale=self.conf.unet_rescale)
                            xs[indx, ...] = cur_xs[0, ...]

                        unet.fd[unet.ph['x']] = xs
                        unet.fd[unet.ph['phase_train']] = False
                        unet.fd[unet.ph['keep_prob']] = unet.keep_prob if distort else 1

                        cur_layers, cur_out = sess.run([preds, unet.pred], unet.fd)

                        for p_ndx in range(len(preds)):
                            cur_preds[p_ndx][cur_s:cur_e, ...] = cur_layers[p_ndx]
                        max_scores[cur_s:cur_e, ...] = np.max(cur_out, axis=(1, 2))

                    framein, curloc = get_patch_trx(cap, cur_trx, fnum, conf.imsz[0], curpts[trx_ndx, fnum, :, selpts])
                    framein = framein[:, :, 0:conf.imgDim]

                    rows, cols = framein.shape[0:2]
                    depth = conf.imgDim

                    image_raw = framein.tostring()
                    feature = {
                        'height': int64_feature(rows),
                        'width': int64_feature(cols),
                        'depth': int64_feature(depth),
                        'locs': float_feature(curloc.flatten()),
                        'expndx': float_feature(ndx),
                        'ts': float_feature(fnum),
                        'trx_ndx': int64_feature(trx_ndx),
                        'image_raw': bytes_feature(image_raw),
                        'context_pred_scores': float_feature(max_scores.flatten())}
                    for p_ndx, p in enumerate(preds):
                        feature['context_raw_{}'.format(p_ndx)] = float_feature(cur_preds[p_ndx].flatten())
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    curenv.write(example.SerializeToString())

                    if curenv is valenv:
                        valcount += 1
                    else:
                        count += 1

                    if (count + valcount) % 20 == 19:
                        sys.stdout.write('.')
                    if (count + valcount) % 400 == 399:
                        sys.stdout.write('\n')

            cap.close()  # close the movie handles
            print('Done %d of %d movies, count:%d val:%d' % (ndx + 1, len(local_dirs), count, valcount))
        env.close()
        valenv.close() if split else None
        print('%d,%d number of pos examples added to the db and valdb' % (count, valcount))
        lbl.close()

        sess.close()
        tf.reset_default_graph()
        self.conf.batch_size = orig_bs


def read_and_decode(filename_queue, conf, att_params):
    att_hist = att_params[0]
    sel_layers = att_params[1]
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features_dict = {'height': tf.FixedLenFeature([], dtype=tf.int64),
                     'width': tf.FixedLenFeature([], dtype=tf.int64),
                     'depth': tf.FixedLenFeature([], dtype=tf.int64),
                     'trx_ndx': tf.FixedLenFeature([], dtype=tf.int64),
                     'locs': tf.FixedLenFeature(shape=[conf.n_classes, 2], dtype=tf.float32),
                     'expndx': tf.FixedLenFeature([], dtype=tf.float32),
                     'ts': tf.FixedLenFeature([], dtype=tf.float32),
                     'image_raw': tf.FixedLenFeature([], dtype=tf.string),
                     'context_pred_scores': tf.FixedLenFeature(shape=[att_hist, conf.n_classes], dtype=tf.float32)
                     }

    names = []
    for ndx, layer in enumerate(sel_layers):
        cur_sz = [att_hist, ] + layer.get_shape().as_list()[1:]
        cur_name = 'context_raw_{}'.format(ndx)
        features_dict[cur_name] = tf.FixedLenFeature(shape=cur_sz, dtype=tf.float32)
        names.append(cur_name)

    features = tf.parse_single_example(
        serialized_example,
        features=features_dict)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, conf.imsz + (conf.imgDim,))

    locs = tf.cast(features['locs'], tf.float64)
    exp_ndx = tf.cast(features['expndx'], tf.float64)
    trx_ndx = tf.cast(features['trx_ndx'], tf.int64)

    ts = tf.cast(features['ts'], tf.float64)  # tf.constant([0]); #
    scores = tf.cast(features['context_pred_scores'], tf.float32)
    att_layers = []
    for cur_name in names:
        cur_l = tf.cast(features[cur_name], tf.float32)
        att_layers.append(cur_l)

    return image, locs, [exp_ndx, ts, trx_ndx], scores, att_layers


def read_and_decode_without_session(filename, conf, npred, count=0):
    # code that shows how to read the tf record file raw.
    # npred is the number of context_raw predictions.

    xx = tf.python_io.tf_record_iterator(filename)
    record = xx.next()
    for _ in range(count):
        record = xx.next()

    example = tf.train.Example()
    example.ParseFromString(record)
    height = int(example.features.feature['height'].int64_list.value[0])
    width = int(example.features.feature['width'].int64_list.value[0])
    depth = int(example.features.feature['depth'].int64_list.value[0])
    expid = int(example.features.feature['expndx'].float_list.value[0]),
    t = int(example.features.feature['ts'].float_list.value[0]),
    trx_ndx = int(example.features.feature['trx_ndx'].int64_list.value[0]),
    img_string = example.features.feature['image_raw'].bytes_list.value[0]
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((height, width, depth))
    locs = np.array(example.features.feature['locs'].float_list.value)
    locs = locs.reshape([conf.n_classes, 2])
    max_scores = np.array(example.features.feature['context_pred_scores'].float_list.value)

    layer_out = []
    for ndx in range(npred):
        cur_name = 'context_raw_{}'.format(ndx)
        cur_layer = np.array(example.features.feature[cur_name].float_list.value)
        layer_out.append(cur_layer)

    return reconstructed_img, locs, layer_out, expid, t, trx_ndx, max_scores
