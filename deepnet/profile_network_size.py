
## Multi-animal

# you might have to set cuda_visible in terminal before launching python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 0

import multiprocessing
import numpy as np

# multiprocessing.set_start_method('spawn')
import nvidia_smi # Install using pip install nvidia-ml-py3
import time

lbl_file = '/groups/branson/home/kabram/temp/ma_expts/alice/trn_packdir_23022022/grone/conf_crop.json'
all_types = ['multi_mdn_joint_torch','multi_openpose','detect_mmdetect']
json_trn_file = '/groups/branson/home/kabram/temp/ma_expts/alice/trn_packdir_23022022/grone/loc_neg.json'

scale_range = [2,1.5,1,0.75,0.5]
bszs = range(2,12,4)
crop_sz = 352
import PoseTools as pt

out_file = 'data/network_size_ma.mat'

nvidia_smi.nvmlInit()

all_mem_use = {}
##
for cur_type in all_types:

    # Create the dbs and skip db creation later for faster stuff
    # parent_conn, child_conn = multiprocessing.Pipe()
    # p = multiprocessing.Process(target=find_mem_ma, args=(lbl_file, json_trn_file, cur_type, bszs[0], scale_range[0], child_conn))
    # p.start()
    # while p.is_alive():
    #     time.sleep(2)
    # success = parent_conn.recv()
    # p.join()

    xx = []
    for scale in scale_range:
        cc = []
        for bsz in bszs:
            cmd = f'{lbl_file} -name sz_test -no_except -json_trn_file {json_trn_file} -conf_params rescale {scale} batch_size {bsz} multi_loss_mask True dl_steps 100 op_hires_ndeconv 0 -cache /groups/branson/bransonlab/mayank/apt_cache_2  -type {cur_type} train -use_cache'

            parent_conn, child_conn = multiprocessing.Pipe()
            p = multiprocessing.Process(target=pt.find_mem,args=(cmd,child_conn,True))
            p.start()
            max_mem = 0
            while p.is_alive():
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                mem = info.used
                max_mem = max(mem,max_mem)
                time.sleep(2)

            success = parent_conn.recv()
            p.join()
            cc.append([max_mem,success])
            time.sleep(5)

        xx.append(cc)
    all_mem_use[cur_type] = np.array(xx)


nvidia_smi.nvmlShutdown()
from scipy import io
io.savemat(out_file,{'mem_use':all_mem_use,
    'batch_size':np.array(bszs),'im_sz':np.array(crop_sz),'scales':np.array(scale_range)})


## single animal

# you might have to set cuda_visible in terminal before launching python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 0
create_db = False

import multiprocessing

# multiprocessing.set_start_method('spawn')
import nvidia_smi # Install using pip install nvidia-ml-py3
import time
import numpy as np


lbl_file = '/groups/branson/home/kabram/temp/ma_expts/alice/trn_packdir_23022022/2stageHT/conf_nocrop.json'
all_types = ['mdn_joint_fpn','mmpose','openpose','deeplabcut']
json_trn_file = '/groups/branson/home/kabram/temp/ma_expts/alice/trn_packdir_23022022/2stageHT/loc_neg.json'

scale_range = [2,1.5,1,0.75,0.5]
bszs = range(2,12,4)
crop_sz = 160
import PoseTools as pt

out_file = 'data/network_size.mat'


nvidia_smi.nvmlInit()

all_mem_use = {}

##
for cur_type in all_types:

    if create_db:
    # Create the dbs and skip db creation later for faster stuff
        parent_conn, child_conn = multiprocessing.Pipe()
        cmd = f'{lbl_file} -name sz_test -no_except -json_trn_file {json_trn_file} -conf_params rescale {scale_range[0]} batch_size {bszs[0]}  dl_steps 100 op_hires_ndeconv 0 -cache /groups/branson/bransonlab/mayank/apt_cache_2 -stage second  -type {cur_type} train -use_cache'
        p = multiprocessing.Process(target=pt.find_mem, args=(cmd, child_conn,False))
        p.start()
        max_mem = 0
        for count in range(300):
            if not p.is_alive():
                break
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            mem = info.used
            max_mem = max(mem, max_mem)
            time.sleep(2)
        success = parent_conn.recv()
        p.join()

    xx = []
    for scale in scale_range:
        cc = []
        for bsz in bszs:
            cmd = f'{lbl_file} -name sz_test -no_except -json_trn_file {json_trn_file} -conf_params rescale {scale} batch_size {bsz}  dl_steps 100 op_hires_ndeconv 0 -cache /groups/branson/bransonlab/mayank/apt_cache_2 -stage second  -type {cur_type} train -use_cache'

            parent_conn, child_conn = multiprocessing.Pipe()
            p = multiprocessing.Process(target=pt.find_mem,args=(cmd,child_conn,True))
            p.start()
            max_mem = 0
            done = False
            for count in range(300):
                if not p.is_alive():
                    done = True
                    break
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                mem = info.used
                max_mem = max(mem,max_mem)
                time.sleep(2)

            if not done:
                success = True
                p.terminate()
            else:
                success = parent_conn.recv()

            p.join()
            cc.append([max_mem,success])
            time.sleep(5)

        xx.append(cc)
    all_mem_use[cur_type] = np.array(xx)


nvidia_smi.nvmlShutdown()
from scipy import io
io.savemat(out_file,{'mem_use':all_mem_use,
    'batch_size':np.array(bszs),'im_sz':np.array(crop_sz),'scales':np.array(scale_range)})


## Single animal - old .. do not use, see above for new code

all_types = ['openpose','mdn','unet','resnet_unet','deeplabcut']
import APT_interface as apt
import tensorflow as tf
import os
import multiprocessing
import tempfile
import time
import numpy as np
import urllib.request

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

files = [['https://www.dropbox.com/s/mo2jfdmzzsz6btc/alice_sz90_stripped.lbl?dl=1',181],
         ['https://www.dropbox.com/s/88m1l9u8oi056io/alice_sz150_stripped.lbl?dl=1',300],
         ['https://www.dropbox.com/s/uvyxjpxepdajoy7/alice_sz200_stripped.lbl?dl=1',400]
         ]


bszs = range(2,12,4)

def find_mem(lbl_file,bsz,net_type,conn,tdir):
    ss = os.path.splitext(os.path.split(lbl_file)[1])[0]
    cmd = '-cache {} -name {} -conf_params batch_size {} dl_steps 10 display_step 10 op_affinity_graph (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16) -type {} {} train -use_cache '.format(tdir,ss,bsz,net_type, lbl_file)

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('batch size: {} label file:{}'.format(bsz,lbl_file))
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    try:
        apt.main(cmd.split())
    except:
        pass
    sess = tf.get_default_session()
    if sess is None:
        sess = tf.Session()
    mem_use = sess.run(tf.contrib.memory_stats.MaxBytesInUse())/1024/1024
    tf.reset_default_graph()
    conn.send(mem_use)
    print(mem_use)

tdir = tempfile.mkdtemp()
local_files = []
for cur_f in files:
    out_file = os.path.join(tdir,'file_{}.lbl'.format(cur_f[1]))
    urllib.request.urlretrieve(cur_f[0],out_file)
    local_files.append(out_file)

all_mem_use = {}
for cur_type in all_types:
    xx = []
    for sndx, lbl_file in enumerate(local_files):
        cc = []
        for bsz in bszs:
            parent_conn, child_conn = multiprocessing.Pipe()
            p = multiprocessing.Process(target=find_mem,args=(lbl_file,bsz, cur_type,child_conn,tdir))
            p.start()
            mm = parent_conn.recv()
            p.join(2)
            time.sleep(5)

            if p.is_alive():
                p.terminate()
            cc.append([mm])
            time.sleep(15)
        xx.append(cc)
    all_mem_use[cur_type] = np.array(xx)


from scipy import io
imsz = np.array([f[1] for f in files])
io.savemat('data/network_size.mat',{'mem_use':all_mem_use,
    'batch_size':np.array(bszs),'im_sz':np.array(imsz)})

