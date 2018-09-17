import os
import subprocess
import argparse
import datetime
import time
import sys
import fnmatch
import re

#!!!!NEED TO RUN: source /groups/branson/bransonlab/mayank/venv/bin/activate   AT COMMAND PROMPT BEFORE RUNNING THIS SCRIPT!!!!!

#base_dir ='/groups/branson/bransonlab/mayank/stephen_copy'
cut_off =  time.mktime(datetime.datetime(2018,3,20).timetuple()) # trk files newer than this won't be retracked
bsize = 20 # size of each batch
max_jobs = 1000 #16 # max number of jobs that should be running on cluster to start a new batch of jobs
coresPerMatJob = 3 #Number of cores to use for each Matlab (non-GPU) job.  Smaller=cheaper, larger=fewer complaints from SciComp. SciComp recommmends = 3
dltfilename="/groups/huston/hustonlab/flp-chrimson_experiments/fly2DLT_lookupTableStephen.csv"
redo = True
n_batches = 1000

def main(argv):
    base_dir = '/groups/huston/hustonlab/flp-chrimson_experiments/'
    temp_dir = os.path.join('/groups/huston/hustonlab/flp-chrimson_experiments', 'tempTrackingOutput')
    name = argv[0]
    if len(argv)>1:
        base_dir = argv[1]
    if len(argv)>2:
        temp_dir = argv[2]
    if len(argv)>3:
        net_name = argv[3]
        use_net_name = True
    else:
        use_net_name = False

    with open(dltfilename,'r') as f:
        x = f.readlines()
    valid_flies = [z.split(',')[0] for z in x]

    donelist_s = os.path.join(temp_dir,'done_' + name + '_slist.txt')
    donelist_f = os.path.join(temp_dir,'done_' + name + '_flist.txt')
    missinglist_s = os.path.join(temp_dir,'missingAll_' + name + '_slist.txt')
    missinglist_f = os.path.join(temp_dir,'missingAll_' + name + '_flist.txt')
    if redo:
        count = 0
        cur_batch = 0
        curflist = os.path.join(temp_dir,'missing_' + name + '{:05d}_flist.txt'.format(cur_batch))
        curslist = os.path.join(temp_dir,'missing_' + name + '{:05d}_slist.txt'.format(cur_batch))
        absent = os.path.join(temp_dir,'absent_' + name + '_flist.txt')
        fr = open(curflist,'w')
        si = open(curslist,'w')
        done_f = open(donelist_f,'w')
        done_s = open(donelist_s,'w')
        miss_f = open(missinglist_f,'w')
        miss_s = open(missinglist_s,'w')
        absent_f = open(absent,'w')
    
        for root, dirnames, filenames in os.walk(base_dir):
            dirnames.sort()
            if cur_batch >= n_batches: 
                break
            for fnames in fnmatch.filter(filenames, 'C001H0*S*.avi'):
    
                cur_mov = os.path.join(root,fnames)
                front_mov = cur_mov.replace('C001','C002')
                if not os.path.exists(front_mov):
                    absent_f.write('{}\n'.format(front_mov))
                    continue
                cur_trk = os.path.join(root,fnames.replace('avi','trk'))
                front_trk = cur_trk.replace('C001','C002')
                side_exist = os.path.exists(cur_trk)
                side_new = os.path.getmtime(cur_trk)>cut_off if side_exist else False
                front_exist = os.path.exists(front_trk)
                front_new = os.path.getmtime(front_trk)>cut_off if front_exist else False
    
                fly_reg = re.search('fly(\d+)',cur_mov,re.IGNORECASE)
                if fly_reg is not None:
                    fly_num = fly_reg.groups()[0]
                else:
                    print('Couldnt find fly num for ')
                    print(cur_mov)
                valid =  valid_flies.count(fly_num)>0
                if not valid:
                    continue
            
                skip = side_exist and side_new and front_exist and front_new
                if not skip:
                    fr.write("{}\n".format(front_mov))
                    si.write("{}\n".format(cur_mov))
                    miss_s.write("{}\n".format(cur_mov))
                    miss_f.write("{}\n".format(front_mov))
                    count += 1
                else:
                    done_s.write("{}\n".format(cur_mov))
                    done_f.write("{}\n".format(front_mov))
    
                if count > bsize:
                    count = 0
                    fr.close()
                    si.close()
                    cur_batch += 1
                    if cur_batch >= n_batches: break
                    curflist = os.path.join(temp_dir,'missing_' + name + '{:05d}_flist.txt'.format(cur_batch))
                    curslist = os.path.join(temp_dir,'missing_' + name + '{:05d}_slist.txt'.format(cur_batch))
                    fr = open(curflist,'w')
                    si = open(curslist,'w')
                           
    
        done_f.close()
        done_s.close()
        miss_f.close()
        miss_s.close()
        fr.close()
        si.close()    
        absent_f.close()
    else:
        tfiles = os.listdir(temp_dir)
        flist = fnmatch.filter(tfiles,'missing_' + name + '*_flist.txt')
        cur_batch = len(flist)-1
    
    tot_batch = cur_batch + 1
    cur_batch = 0
    while cur_batch < tot_batch:

        print('Cleaning up')
        for b_num in range(cur_batch):
            curflist = os.path.join(temp_dir,'missing_' + name + '{:05d}_flist.txt'.format(b_num))
            curslist = os.path.join(temp_dir,'missing_' + name + '{:05d}_slist.txt'.format(b_num))
            cmd = 'python cleanupStephen.py -o {} -s {} -f {}'.format(temp_dir,curslist, curflist)
            subprocess.call(cmd,shell=True)
        cmd = 'python cleanupStephen.py -o {} -s {} -f {}'.format(temp_dir,donelist_s,donelist_f)
        subprocess.call(cmd,shell=True)
        
        bjobs_status = subprocess.check_output('ssh 10.36.11.34 ". /misc/lsf/conf/profile.lsf; bjobs"',shell=True)
        njobs = bjobs_status.count('gpu')
        if njobs > (max_jobs)/2:
            print('{} jobs running on GPU queue. Sleeping for 5 minutes'.format(njobs))
            time.sleep(300) # sleep for a minute
            continue

        for b in range(max_jobs): #submit max_jobs/2 new jobs
            if cur_batch >= tot_batch:
                continue
            curflist = os.path.join(temp_dir,'missing_' + name + '{:05d}_flist.txt'.format(cur_batch))
            curslist = os.path.join(temp_dir,'missing_' + name + '{:05d}_slist.txt'.format(cur_batch))
            sing_script = os.path.join(temp_dir,'missing_'+ name + '_singularity_{}.sh'.format(cur_batch))
            sing_err = os.path.join(temp_dir,'missing_'+ name + '_singularity_{}.err'.format(cur_batch))
            sing_log = os.path.join(temp_dir,'missing_'+ name + '_singularity_{}.log'.format(cur_batch))
            with open(sing_script,'w') as f:
                f.write('#!/bin/bash\n')
                f.write('env | grep CUDA\n')
                f.write('echo "JOBID $LSB_JOBID"\n')
                f.write('bjobs -uall -m `hostname -s`\n')
                f.write('nvidia-smi\n')
                f.write('. /opt/venv/bin/activate\n')
                f.write('cd /groups/branson/home/kabram/bransonlab/poseTF_bkps/poseTF_noqueue\n')
                f.write("if nvidia-smi | grep -q 'No devices were found'; then \n")
                f.write('{ echo "No GPU devices were found. quitting"; exit 1; }\n')
                f.write('fi\n')
                f.write('numCores2use={} \n'.format(coresPerMatJob))
                f.write('python trackStephenHead_KB.py -s {} -f {} -d {} -o {} -ncores $numCores2use -rt '.format(curslist,curflist, dltfilename, temp_dir))
                if use_net_name:
                    f.write('-net {}'.format(net_name))
                f.write('\n')

            os.chmod(sing_script,0755)

            cmd = '''ssh 10.36.11.34 '. /misc/lsf/conf/profile.lsf; bsub -oo {} -eo {} -n2 -gpu "num=1" -q gpu_any "singularity exec --nv /misc/local/singularity/branson_v2.simg {}"' '''.format(sing_log, sing_err, sing_script) #-n2 because SciComp says we need 2 slots for the RAM
            subprocess.call(cmd,shell=True)
            print('Submitted jobs for batch {}'.format(cur_batch))

            cur_batch += 1

 

if __name__ == "__main__":
   main(sys.argv[1:])

