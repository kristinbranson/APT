import subprocess
# import pprint
import re
import platform
# import distutils.spawn
import os
import shutil

import sys
ISPY3 = sys.version_info >= (3, 0)

PrettyOutput = False

def get_gpu_memory_free():

    nvidiasmi = 'nvidia-smi'
    memtotal = []
    memfree = []
    # check if nvidia-smi exists
    # if distutils.spawn.find_executable(nvidiasmi) is None:  # distutils.spawn.find_executable() gives deprecation warning
    if shutil.which(nvidiasmi) is None:
        if platform.system() == "Windows":
            nvpath = None
            if 'NVTOOLSEXT_PATH' in os.environ:
                try:
                    nvpath = os.environ['NVTOOLSEXT_PATH']
                except:
                    pass
            if nvpath is None:
                #print('Could not find path to nvidia-smi. OS = Windows, environmental variable NVTOOLSEXT_PATH is not set')
                return memfree, memtotal
            nvpath,tail = os.path.split(nvpath)
            if tail == '':
                nvpath = os.path.dirname(nvpath)
            nvidiasmi = os.path.join(nvpath,'NVSMI','nvidia-smi.exe')
            if not os.path.isfile(nvidiasmi):
                print('Could not find path to nvidia-smi. OS = Windows, looked at %s'%nvidiasmi)

    sp = subprocess.Popen([nvidiasmi, '-q' ,'-d','MEMORY'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out_str = sp.communicate()

    if ISPY3:
        out_list = out_str[0].decode().split('\n')
    else:
        out_list = out_str[0].split('\n')

    gpuidx = -1
    memtype = 'unknown'

    gpubusinfo_re = re.compile('GPU \d+:')
    memtypecurr_re = re.compile('([a-zA-Z0-9]+) Memory Usage')
    memval_re = re.compile('(\d+) MiB')

    for i in range(len(out_list)):
        item = out_list[i]
        gpubusinfo = gpubusinfo_re.search(item)
        memtypecurr = memtypecurr_re.search(item)
        if gpubusinfo is not None:
            gpuidx += 1
        elif memtypecurr is not None:
            memtype = memtypecurr.groups()
            memtype = memtype[0]
        else:
            m = item.split(':')
            if len(m) < 2:
                continue
            key = m[0].strip()
            val = m[1].strip()
            if key == 'Attached GPUs':
                ngpus = int(val)
                gpuidx = -1
                memfree = [0]*ngpus
                memtotal = [0]*ngpus
            elif key == 'Free' and memtype == 'FB':
                m = memval_re.match(val).groups()
                memfreecurr = float(m[0])
                memfree[gpuidx] += memfreecurr
            elif key == 'Total' and memtype == 'FB':
                m = memval_re.match(val).groups()
                memtotalcurr = float(m[0])
                memtotal[gpuidx] += memtotalcurr

    return memfree,memtotal
                
if __name__ == "__main__":

    memfree,memtotal = get_gpu_memory_free()
    ngpus = len(memfree)
    if PrettyOutput:
        for i in range(ngpus):
            print("GPU %d: %f MiB free / %f MiB"%(i,memfree[i],memtotal[i]))
    else:
        print('GPU,free,total')
        for i in range(ngpus):
            print("%d,%f,%f"%(i,memfree[i],memtotal[i]))

