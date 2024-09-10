function codestr = codeGenCondaGeneral(basecmd,varargin)

% Take a base command and run it in a sing img
[condaEnv,gpuid] = myparse(varargin,...
                           'condaEnv','APT',...
                           'gpuid',0);
codestr = synthesize_conda_command(['activate ',condaEnv]);
if ~isnan(gpuid),
  if ispc,
    envcmd = sprintf('set CUDA_DEVICE_ORDER=PCI_BUS_ID&& set CUDA_VISIBLE_DEVICES=%d',gpuid);
  else
    envcmd = sprintf('export CUDA_DEVICE_ORDER=PCI_BUS_ID && export CUDA_VISIBLE_DEVICES=%d',gpuid);
  end
  codestr = [codestr,' && ',envcmd];
end
codestr = [codestr,' && ',basecmd];
  
end
