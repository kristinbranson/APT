from __future__ import division
from __future__ import print_function

import time
import collections.abc

import numpy as np

from keras import backend as K

def dictsubsetpfix(d, pfix):
    return dict((k, d[k]) for k in d if k.startswith(pfix))


def dictdiff(d1, d2, print_fcn=print):
    if not isinstance(d1, collections.abc.Mapping):
        print("using vars(d1)")
        d1 = vars(d1)
    if not isinstance(d2, collections.abc.Mapping):
        print("using vars(d2)")
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
        tf = v1 == v2
        if isinstance(tf, np.ndarray):
            tf = np.all(tf)
        if not tf:
            print("{}: values differ, {} vs {}".format(kk, v1, v2))

    print_fcn("{} total common keys checked".format(len(k)))


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

