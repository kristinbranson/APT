function codestr = trackCodeGenConda(backend,fileinfo,frm0,frm1,varargin)  %#ok<INUSD> 

[baseargs,condaargs,outfile] = ...
  myparse(varargin,'baseargs',{},'condaargs',{},'outfile','');

addnlbaseargs = {'cache' fileinfo.cache 'filequote' '"' 'updateWinPaths2LnxContainer' false};
baseargs = [addnlbaseargs baseargs];
  
basecmd = APTInterf.trackCodeGenBase(fileinfo,frm0,frm1,baseargs{:});
if ~isempty(outfile),
  basecmd = sprintf('%s > %s 2>&1',basecmd,outfile);
end
codestr = codeGenCondaGeneral(basecmd,condaargs{:});

end
