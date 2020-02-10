from __future__ import division
from __future__ import print_function

import time

import numpy as np

def dictsubsetpfix(d, pfix):
    return dict((k, d[k]) for k in d if k.startswith(pfix))


def dictdiff(d1, d2, print_fcn=print):
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