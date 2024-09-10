function codestr = trackCodeGenDocker(backend,fileinfo,frm0,frm1,varargin)

% varargin: see trackCodeGenBase, except for 'cache' and 'view'
assert(backend.type==DLBackEnd.Docker) ;
[baseargs,dockerargs,mntPaths,containerName] = myparse(varargin,...
  'baseargs',{},'dockerargs',{},'mntPaths',{},'containerName','');

baseargs = [{'cache' fileinfo.cache} baseargs];
basecmd = APTInterf.trackCodeGenBase(fileinfo,frm0,frm1,baseargs{:});

if isempty(containerName),
  if iscell(fileinfo.outtrk),
    [~,containerName] = fileparts(fileinfo.outtrk{1});
  else
    [~,containerName] = fileparts(fileinfo.outtrk);
  end
end

codestr = backend.wrapBaseCommand(basecmd,containerName,...
                                  'bindpath',mntPaths,dockerargs{:});

end
