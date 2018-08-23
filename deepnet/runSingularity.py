import sys
import argparse
import os
import subprocess

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd",help="cmd to run",nargs='*')
    parser.add_argument('-name',dest='name',help='name for the singularity files',default='default')
    parser.add_argument('-dir',dest='dir',help='directory to create the singularity files',default=os.getcwd()+'/singularity_stuff')

    args = parser.parse_args(argv)
    name = args.name
    dir = args.dir
    cmd = ' '.join(args.cmd)
    this_dir = os.path.dirname(os.path.realpath(__file__))
    sing_script = os.path.join(dir, name + '_singularity.sh')
    sing_log = os.path.join(dir, name + '_singularity.log')
    sing_err = os.path.join(dir, name + '_singularity.err')
    with open(sing_script, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('. /opt/venv/bin/activate\n')
        f.write('cd {}\n'.format(this_dir))
        f.write("if nvidia-smi | grep -q 'No devices were found'; then \n")
        f.write('{ echo "No GPU devices were found. quitting"; exit 1; }\n')
        f.write('fi\n')
        f.write('numCores2use={} \n'.format(1))
        f.write('python {}'.format(cmd))

    os.chmod(sing_script, 0755)

    cmd = '''ssh 10.36.11.34 '. /misc/lsf/conf/profile.lsf; bsub -oo {}  -n2 -gpu "num=1" -q gpu_any "singularity exec --nv /misc/local/singularity/branson_v2.simg {}"' '''.format(
        sing_log, sing_script)  # -n2 because SciComp says we need 2 slots for the RAM
    subprocess.call(cmd, shell=True)
    print('Submitted job: {}'.format(cmd))


if __name__ == "__main__":
    main(sys.argv[1:])
