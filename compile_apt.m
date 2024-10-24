script_file = 'APT_deployed.m';
out_file = 'APT_deployed';
out_dir = 'compiled';
if ~exist(out_dir,'dir')
  mkdir(out_dir)
end
apt_dir = pwd;
tic;
mcc_cmd = 'mcc -m ';
mcc_cmd = [mcc_cmd ' -a ' 'unittest' ...
  ' -a ' 'matlab' ...
  ' -a ' 'external' ...
  ' -a ' 'java' ...
  ' -a ' 'gfx' ...
  ' -a ' 'deepnet' ...
  ' -a ' 'docs' ...  
  ' -I ' 'APT.m'...
  ' -R ' '''-logfile,APT_deployed.log''' ...
  ' -o ' out_file ...
  ' -d ' out_dir ...
  ];

mcc_cmd = [mcc_cmd ' ' script_file];
eval(mcc_cmd);
tcompile = toc;
fprintf(sprintf('-- Time to compile %.2f --\n',tcompile));
% cmd = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/groups/branson/home/kabram/bransonlab/MCR/2018a/v94/runtime/glnxa64:/groups/branson/home/kabram/bransonlab/MCR/2018a/v94/bin/glnxa64:/groups/branson/home/kabram/bransonlab/MCR/2018a/v94/sys/os/glnxa64:/groups/branson/home/kabram/bransonlab/MCR/2018a/v94/extern/bin/glnxa64;';
cmd = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/groups/branson/home/kabram/bransonlab/MCR/v911/runtime/glnxa64:/groups/branson/home/kabram/bransonlab/MCR/v911/bin/glnxa64:/groups/branson/home/kabram/bransonlab/MCR/v911/sys/os/glnxa64:/groups/branson/home/kabram/bransonlab/MCR/v911/extern/bin/glnxa64;';
cmd = [cmd  ' ./' out_dir '/' out_file ' 1 roianma'];
system(cmd);
