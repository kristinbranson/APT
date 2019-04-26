function MakeMATLABClusterScript(shfile,jobid,script,argstr,varargin)

[MCR,TMP_ROOT_DIR] = myparse(varargin,'MCR','','TMP_ROOT_DIR','');

if isempty(TMP_ROOT_DIR),
  [~,me] = unix('whoami');
  me = strtrim(me);
  TMP_ROOT_DIR = fullfile('/scratch',me);
end

MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');

if isempty(MCR),
  tmp = ver('MATLAB');
  v = ['v',strrep(tmp.Version,'.','')];
  MCR = fullfile('/groups/branson/bransonlab/share/MCR',v);
  assert(exist(MCR,'dir'));
end

fid = fopen(shfile,'w');
if fid < 0,
  error('Could not open file %s for writing',shfile);
end


fprintf(fid,'#!/bin/bash\n\n');
fprintf(fid,'source ~/.bashrc\n');
fprintf(fid,'unset DISPLAY\n');
fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
fprintf(fid,'fi\n');
fprintf(fid,'%s %s %s\n',script,MCR,argstr);
fclose(fid);
unix(sprintf('chmod u+x %s',shfile));
