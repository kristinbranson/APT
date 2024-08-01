function res = parforOverVideo(readframe,frms,fun,varargin)

[args,bufsize,verbose] = myparse(varargin,'args',{},...
  'bufsize',128,'verbose',false);

ncurrmov = numel(frms);
useparfor = true;
if useparfor,
  
  imbuf = cell(bufsize,1);
  niters = ceil(ncurrmov/bufsize);
  res = cell(1,ncurrmov);
  for iter = 1:niters,
    istart = (iter-1)*bufsize+1;
    iend = min(ncurrmov,iter*bufsize);
    ncurriter = iend-istart+1;
    for i = 1:ncurriter,
      imbuf{i} = readframe(frms(istart+i-1));
    end
    rescurr = cell(1,ncurriter);
    parfor i = 1:ncurriter,
      frm = frms(istart+i-1);
      rescurr{i} = fun(imbuf{i},frm,istart+i-1,args{:});
    end
    res(istart:iend) = rescurr;
    if verbose,
      fprintf('%d / %d frames processed.\n',iend,ncurrmov);
    end
  end
  
else
  res = cell(1,ncurrmov);
  for i = 1:ncurrmov,
    im = readframe(frms(i));
    res{i} = fun(im,frms(i),i,args{:});
  end
end

end