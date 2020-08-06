from __future__ import print_function
from __future__ import division

import numpy as np
import logging

def upsample_filt(alg='nn', dtype=None):
    if alg == 'nn':
        x = np.array([[0., 0., 0., 0.],
                      [0., 1., 1., 0.],
                      [0., 1., 1., 0.],
                      [0., 0., 0., 0.]], dtype=dtype)
    elif alg == 'bl':
        x = np.array(
            [[0.0625, 0.1875, 0.1875, 0.0625],
             [0.1875, 0.5625, 0.5625, 0.1875],
             [0.1875, 0.5625, 0.5625, 0.1875],
             [0.0625, 0.1875, 0.1875, 0.0625]], dtype=dtype)
    else:
        assert False
    return x

def upsample_init_value(shape, alg='nn', dtype=None):
    # Return numpy array for initialization value

    print("upsample initializer desired shape: {}".format(shape))
    f = upsample_filt(alg, dtype)

    filtnr, filtnc, kout, kin = shape
    assert kout == kin  # for now require equality
    if kin > kout:
        wstr = "upsample filter has more inputs ({}) than outputs ({}). Using truncated identity".format(kin, kout)
        logging.warning(wstr)

    xinit = np.zeros(shape)
    for i in range(kout):
        xinit[:, :, i, i] = f

    return xinit