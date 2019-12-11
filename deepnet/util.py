from __future__ import division
from __future__ import print_function

import numpy as np

def dictdiff(d1, d2):
    k1 = set(d1.keys())
    k2 = set(d2.keys())

    k1not2 = k1-k2
    k2not1 = k2-k1
    k = k1.intersection(k2)

    print("{} keys in d1 not in d2: {}".format(len(k1not2), k1not2))
    print("{} keys in d2 not in d1: {}".format(len(k2not1), k2not1))

    for kk in k:
        v1 = d1[kk]
        v2 = d2[kk]
        tf = v1 == v2
        if isinstance(tf, np.ndarray):
            tf = np.all(tf)
        if not tf:
            print("{}: values differ, {} vs {}".format(kk, v1, v2))

    print("{} total common keys checked".format(len(k)))