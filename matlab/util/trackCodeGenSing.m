function codestr = trackCodeGenSing(backend,fileinfo,frm0,frm1,varargin)

[baseargs,singargs] = myparse(varargin,...
  'baseargs',{},...
  'singargs',{});
baseargs = [baseargs {'confparamsfilequote','\\\"','ignore_local',1}];
basecmd = APTInterf.trackCodeGenBase(fileinfo,frm0,frm1,baseargs{:});
singimg = pick_singularity_image(backend, fileinfo.netMode) ;
singargs2 = add_pair_to_key_value_list(singargs, 'singimg', singimg) ;
codestr = DeepTracker.codeGenSingGeneral(basecmd, singargs2{:});

end  % function

