function  readframe = klb_get_readframe_fcn(filename,varargin)

[dim,numThreads] = myparse(varargin,'dim',3,'numThreads',[]);
readframe = @(f) klbreadframe(f);

  function im = klbreadframe(f)
    im = readKLBslice(filename,f,dim,numThreads);
  end
end
