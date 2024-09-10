function codestr = trackCodeGenListFileSSHBsubSing(backend, trksysinfo,...
                                                   trnID,nettype,netmode,view,varargin)

[baseargs,singargs,bsubargs,sshargs] = ...
  myparse(varargin,...
          'baseargs',{},...
          'singargs',{},...
          'bsubargs',{},...
          'sshargs',{});      

cache = trksysinfo.dmcRootDir;
dlconfigfile = trksysinfo.trainConfigLnx;
errfile = trksysinfo.errfile;
outfile = trksysinfo.outfile;
listfile = trksysinfo.listfile;

codebase = DeepTracker.trackCodeGenBaseListFile(trnID,cache,dlconfigfile,...
                                                outfile,errfile,nettype,view,listfile,baseargs{:});
singimg = pick_singularity_image(backend, netmode) ;
singargs2 = add_pair_to_key_value_list(singargs, 'singimg', singimg) ;      
codesing = DeepTracker.codeGenSingGeneral(codebase,singargs2{:});
codebsub = codeGenBsubGeneral(codesing,bsubargs{:});
codestr = DeepTracker.codeGenSSHGeneral(codebsub,sshargs{:});      

end        
