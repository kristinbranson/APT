curdir = '/groups/branson/home/rodriguezgonzalezj/Documents/pose_tracking';
SCRIPT = [curdir,'/rcpr_v1_stable/scripts/ScriptTrack_video/for_redistribution_files_only/run_ScriptTrack_video.sh'];
tmpdatafile = [curdir,'/mouse/epxlists/params_M130.txt'];
TMP_ROOT_DIR = '/scratch/rodriguezgonzalezj';
MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');
MCR = '/groups/branson/home/rodriguezgonzalezj/MATLAB/R2013b/';  
tmpdir = [curdir,'/mouse/temp'];

[file,folder]=uigetfile('*.txt');
expfile=fullfile(folder,file);

fid = fopen(expfile,'r');
%[expdirs_all,moviefiles_all,labeledpos]=read_exp_list_labeled(fid,dotest);
[~,moviefiles_all]=read_exp_list_NONlabeled(fid);

nbatches=numel(moviefiles_all);
%%
jobids = [];
outfiles = cell(nbatches,1);
cmdchmod = cell(nbatches,1);
cmd = cell(nbatches,1);

for batchi=1:nbatches
    moviefile = moviefiles_all{batchi};
    timestamp = datestr(now,TimestampFormat);
    jobid = sprintf('pv%s_%03d',timestamp,batchi);
    outfiles{batchi} = fullfile(tmpdir,['log_',jobid,'.txt']);
    scriptfile = fullfile(tmpdir,[jobid,'.sh']);
    fid = fopen(scriptfile,'w');
    fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
    fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
    fprintf(fid,'fi\n');
    fprintf(fid,'%s %s %05d %s %s\n',...
      SCRIPT,MCR,batchi,moviefile,tmpdatafile);
    fclose(fid);

    cmdchmod{batchi} = (sprintf('chmod u+x %s',scriptfile));
        cmd{batchi} = sprintf('qsub -N %s -j y -b y -l short=true -o %s -cwd %s',...
      jobid,outfiles{batchi},scriptfile);

%     cmd{batchi} = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -N %s -j y -b y -l short=true -o %s -cwd %s',...
%       curdir,jobid,outfiles{batchi},scriptfile);
%     [tmp1,tmp2] = unix(cmd);
%     if tmp1 ~= 0,
%       error('Error submitting job %d:\n%s ->\n%s\n',batchi,cmd,tmp2);
%     end
%     m = regexp(tmp2,'job (\d+) ','once','tokens');
%     jobids = [jobids,str2double(m)]; %#ok<AGROW>
end