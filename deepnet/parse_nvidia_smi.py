import subprocess
import pprint
import re

PrettyOutput = False

def get_gpu_memory_free():

    sp = subprocess.Popen(['nvidia-smi', '-q' ,'-d','MEMORY'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out_str = sp.communicate()

    out_list = out_str[0].split('\n')

    gpuidx = -1
    memtype = 'unknown'

    gpubusinfo_re = re.compile('GPU \d+:')
    memtypecurr_re = re.compile('([a-zA-Z]+) Memory Usage')
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

