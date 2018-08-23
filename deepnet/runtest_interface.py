# script to test APT_interface on 3 different projects

import APT_interface
import tensorflow as tf
import os
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
dstr = datetime.datetime.now().strftime('%Y%m%d')

## Double pendulum
lbl_file = '/groups/branson/home/kabram/bransonlab/PoseTF/data/apt_interface/pend/Doub_pend_stripped_modified.lbl'
mov_file = '/groups/branson/home/leea30/apt/deeptrackIntegrate20180427/doubpend.mp4'
out_file = os.path.splitext(mov_file)[0] + '_interface_test_{}.trk'.format(dstr)
name = 'pend_test_apt'
cmd = '{} -name {} train'.format(lbl_file,name)
# APT_interface.main(cmd.split())
cmd = '{} -name {} track -mov {} -out {} -end_frame 1000'.format(lbl_file, name, mov_file, out_file)
# APT_interface.main(cmd.split())

## stephen for multi view
lbl_file = '/groups/branson/bransonlab/mayank/PoseTF/data/apt_interface/stephen/shdl_stripped_homogeneousims_modified.lbl'
mov_files = ['/groups/branson/bransonlab/mayank/PoseTF/data/apt_interface/stephen/fly516/C001H001S0001/C001H001S0001_c.avi',
            '/groups/branson/bransonlab/mayank/PoseTF/data/apt_interface/stephen/fly516/C002H001S0001/C002H001S0001_c.avi']
out_files = []
for mov_file in mov_files:
    out_files.append(os.path.splitext(mov_file)[0] + '_interface_test_{}.trk'.format(dstr))

name = 'stephen_test_apt'
cmd = '{} -name {} train'.format(lbl_file,name)
#APT_interface.main(cmd.split())
cmd = '{} -name {} track -mov {} {} -out {} {}'.format(
    lbl_file, name, mov_files[0], mov_files[1], out_files[0], out_files[1])
#APT_interface.main(cmd.split())



## Alice: for projects with trx files.
lbl_file = '/groups/branson/bransonlab/mayank/PoseTF/data/alice/multitarget_bubble_expandedbehavior_20180425_modified.lbl'
mov_file = '/groups/branson/bransonlab/mayank/PoseTF/data/apt_interface/alice/cx_GMR_SS00168_CsChr_RigD_20150909T111218/movie.ufmf'
trx_file = '/groups/branson/bransonlab/mayank/PoseTF/data/apt_interface/alice/cx_GMR_SS00168_CsChr_RigD_20150909T111218/registered_trx.mat'
out_file = os.path.splitext(mov_file)[0] + '_interface_test_{}.trk'.format(dstr)
name = 'alice_test_apt'
cmd = '{} -name {} train'.format(lbl_file,name)
#APT_interface.main(cmd.split())
cmd = '{} -name {} track -mov {} -trx {} -out {} -end_frame 1000'.format(lbl_file, name, mov_file, trx_file, out_file)
APT_interface.main(cmd.split())

