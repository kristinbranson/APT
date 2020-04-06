function  readframe = h5_get_readframe_fcn(filename,headerinfo,varargin)

readframe = @(fs) h5readframe(fs);
start = ones(1,headerinfo.ndims);
count = headerinfo.sz;
count(headerinfo.dims) = 1;
start(headerinfo.fixdims) = headerinfo.fixidx;
count(headerinfo.fixdims) = 1;

  function im = h5readframe(f)
    start(headerinfo.dims) = f;
    im = h5read(filename,headerinfo.datasetname,start,count);
  end
end
