#from __future__ import division
#from __future__ import print_function

import time
import collections.abc
import json
import numpy as np
import copy
from easydict import EasyDict as edict

from keras import backend as K

import h5py


def dictsubsetpfix(d, pfix):
    return dict((k, d[k]) for k in d if k.startswith(pfix))


def dictdiff(d1, d2, print_fcn=print):
    if not isinstance(d1, collections.abc.Mapping):
        print_fcn("using vars(d1)")
        d1 = vars(d1)
    if not isinstance(d2, collections.abc.Mapping):
        print_fcn("using vars(d2)")
        d2 = vars(d2)

    k1 = set(d1.keys())
    k2 = set(d2.keys())

    k1not2 = k1-k2
    k2not1 = k2-k1
    k = k1.intersection(k2)

    print_fcn("{} keys in d1 not in d2: {}".format(len(k1not2), k1not2))
    print_fcn("{} keys in d2 not in d1: {}".format(len(k2not1), k2not1))

    for kk in k:
        v1 = d1[kk]
        v2 = d2[kk]
        try:
            tf = v1 == v2
            if isinstance(tf, np.ndarray):
                tf = np.all(tf)
            if not tf:
                print_fcn("{}: values differ, {} vs {}".format(kk, v1, v2))
        except:
            print_fcn("Error caught comparing key {}".format(kk))

    print_fcn("{} total common keys checked".format(len(k)))

def h5diffattrs(x1, x2, path):
    mismatch = []
    a1 = x1.attrs
    a2 = x2.attrs
    lista1 = list(a1)
    assert lista1 == list(a2)
    print("{}: checking {} attrs: {}".format(path, len(lista1), lista1))
    for a in a1:
        v1 = a1.get(a)
        v2 = a2.get(a)
        assert type(v1) == type(v2)
        if isinstance(v1, bytes):
            try:
                v1 = json.loads(v1.decode('utf-8'))
                v2 = json.loads(v2.decode('utf-8'))
                tfmatch = (v1 == v2)
            except json.decoder.JSONDecodeError:
                tfmatch = (v1 == v2)
        elif isinstance(v1, np.ndarray):
            tfmatch = np.array_equal(v1, v2)
        else:
            assert False

        if not tfmatch:
            print('!!! attr mismatch: {}'.format(a))
            mismatch += [path + '#' + a]

    return mismatch


def h5diffgrps(g1, g2, path):
    mismatch = []

    mismatch += h5diffattrs(g1, g2, path)
    for k in g1:
        assert k in g2
        v1 = g1[k]
        v2 = g2[k]
        path2 = path + "." + k
        isg1 = isinstance(v1, h5py.Group)
        isg2 = isinstance(v2, h5py.Group)
        assert isg1 == isg2
        if isg1:
            print("{}: group, entering".format(path2))
            mismatch += h5diffgrps(v1, v2, path2)
        else:
            assert v1.shape == v2.shape
            assert v1.dtype == v2.dtype
            mismatch += h5diffattrs(v1, v2, path2)
            if np.allclose(np.array(v1),np.array(v2)):
                print("{}: CLOSE!! val. shape={}, dtype={}.".format(path2, v1.shape, v1.dtype))
            else:
                print("{}: val. shape={}, dtype={}.".format(path2, v1.shape, v1.dtype))
    return mismatch

def h5diff(h1, h2):
    '''
    uni-directional compare. run it the other way too
    :param h1:
    :param h2:
    :return:
    '''
    with h5py.File(h1, 'r') as f1, h5py.File(h2, 'r') as f2:
        mismatch = h5diffgrps(f1, f2, '')
    return mismatch

def dict_copy_with_edict_convert(d):
    '''
    Deep-copy a dict, while:
      * converting all easydicts to dicts.
      * replacing all empty-dict values with '<emptydict>'

    Only easydicts that are direct dict-values are converted.

    Needed for saving to MATfiles, where easydicts and empty
    dict vals cause corruption

    :param d: dict or easydict
    :return: deep-copy of d with all easydicts converted
    '''

    x = d

    if isinstance(x, dict) or isinstance(x, edict):
        x2 = dict(x)  # shallow copy
        if len(x2) == 0:
            x2 = '<empty dict>'
        else:
            for k, v in x2.items():
                x2[k] = dict_copy_with_edict_convert(v)
    else:
        x2 = copy.deepcopy(x)

    return x2


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self.name = None
        self._start_time = None
        self._recorded_times = []

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        self._recorded_times.append(elapsed_time)

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop()


def get_model_memory_usage(batch_size, model):
    #import numpy as np
    #from keras import backend as K
    shapes_mem_count = 0
    for l in model.layers:
        if l in model._input_layers:
            print("Skipping input layer with shape {}".format(l.output_shape))
            continue
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem
    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0
    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

