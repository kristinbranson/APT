NCORESPERJOB = 1;
curdir = pwd;
TMP_ROOT_DIR = '/scratch/bransonk';
MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');
MCR = '/groups/branson/bransonlab/share/MCR/v717';
SCRIPT = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/test/distrib/run_test.sh';

saverootdir = '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults20150623';
if ~exist(saverootdir,'dir'),
  mkdir(saverootdir);
end

[~,n] = fileparts(expdir);
savedircurr = saverootdir;

jobid = sprintf('track_%s_%s',n,datestr(now,'yyyymmddTHHMMSSFFF'));
testresfile = fullfile(savedircurr,[jobid,'.mat']);
scriptfile = fullfile(savedircurr,[jobid,'.sh']);
outfile = fullfile(savedircurr,[jobid,'.log']);

fid = fopen(scriptfile,'w');
fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
fprintf(fid,'fi\n');
fprintf(fid,'%s %s %s %s %s moviefilestr %s\n',...
  SCRIPT,MCR,expdir,trainresfile,testresfile,ld.moviefilestr);
fclose(fid);
unix(sprintf('chmod u+x %s',scriptfile));

cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
  curdir,NCORESPERJOB,jobid,outfile,scriptfile);

unix(cmd);
