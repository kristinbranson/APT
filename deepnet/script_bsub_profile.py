import bsub
import os

nconfigs = 19
outdir = '/groups/branson/home/bransonk/tracking/code/APT_develop/deepnet/bsubout'
cwd = os.path.dirname(os.path.abspath(__file__))
for cfgi in range(4,5):
    jobname = f'mem{cfgi:02d}'
    logfile = os.path.join(outdir,f'{jobname}.log')
    errfile = os.path.join(outdir,f'{jobname}.err')
    basecmd = f'python profile_network_size_kb.py --cfgi0 {cfgi} --cfgi1 {cfgi+1}'
    bsub.launch_job(basecmd, sshserver='login1', queue='gpu_a100', ncores=8, condaenv='APT', logfile=logfile, errfile=errfile, name=jobname, dryrun=False, verbose=1, rundir=cwd)