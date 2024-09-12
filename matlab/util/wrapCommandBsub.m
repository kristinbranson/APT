function cmdout = wrapCommandBsub(cmdin,varargin)

[nslots,gpuqueue,logfile,jobname,additionalArgs] = ...
  myparse(varargin,...
          'nslots',DeepTracker.default_jrcnslots_train,...
          'gpuqueue',DeepTracker.default_jrcgpuqueue,...
          'logfile','/dev/null',...
          'jobname','', ...
          'additionalArgs','');
esccmd = escape_string_for_bash(cmdin) ;
if isempty(jobname),
  jobnamestr = '';
else
  jobnamestr = sprintf('-J %s', jobname) ;
end
% NB: Line below sends *both* stdout and stderr to the file named by logfile
quotedlogfile = escape_string_for_bash(logfile) ;
cmdout = sprintf('bsub -n %d -gpu "num=1" -q %s -o %s -R"affinity[core(1)]" %s %s %s',...
                 nslots,gpuqueue,quotedlogfile,jobnamestr,additionalArgs,esccmd);