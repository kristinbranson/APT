function codestr = trackCodeGenBsubSing(backend, fileinfo,frm0,frm1,varargin)

[baseargs,singargs,bsubargs] = myparse(varargin,...
  'baseargs',{},...
  'singargs',{},...
  'bsubargs',{});
basecmd = trackCodeGenSing(backend, fileinfo,frm0,frm1,'baseargs',baseargs,'singargs',singargs);
codestr = codeGenBsubGeneral(basecmd,bsubargs{:});

end  % function
  
