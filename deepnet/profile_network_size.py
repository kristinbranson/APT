
all_types = ['openpose','mdn','unet','resnet_unet','deeplabcut','leap']
all_types = ['openpose','mdn','unet','resnet_unet','deeplabcut']
import APT_interface as apt
import tensorflow as tf
import os
import multiprocessing
import tempfile
import time
import numpy as np

files = [['https://www.dropbox.com/s/mo2jfdmzzsz6btc/alice_sz90_stripped.lbl?dl=0',181],
         ['https://www.dropbox.com/s/88m1l9u8oi056io/alice_sz150_stripped.lbl?dl=0',300],
         ['https://www.dropbox.com/s/uvyxjpxepdajoy7/alice_sz200_stripped.lbl?dl=0',400]
         ]


bszs = range(2,12,3)

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

    cmd = 'wget -q -O {} {}'.format(out_file,cur_f[0])
    os.system(cmd)
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
io.savemat('data/network_size.mat',all_mem_use)

##

