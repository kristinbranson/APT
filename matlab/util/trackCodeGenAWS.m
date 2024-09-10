function codestr = trackCodeGenAWS(backend,fileinfo,frm0,frm1,baseargs)  %#ok<INUSD> 
% movRemoteFull: can be cellstr when tracking all views
% trkRemoteFull: "
% 
% baseargs: PV cell vector that goes to .trackCodeGenBase

deepnetroot = '/home/ubuntu/APT/deepnet';
baseargs = [baseargs {'cache' fileinfo.cache}];
codestrbase = APTInterf.trackCodeGenBase(fileinfo,frm0,frm1,...
  'deepnetroot',deepnetroot,baseargs{:});

codestr = {
   'cd /home/ubuntu/APT/deepnet;';
   'export LD_LIBRARY_PATH=/home/ubuntu/src/cntk/bindings/python/cntk/libs:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/mpi/lib;';
   codestrbase
  };
codestr = cat(2,codestr{:});

end  % function

