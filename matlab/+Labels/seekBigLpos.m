function [tffound,f] = seekBigLpos(lpos,f0,df,iTgt)
  % lpos: [npts x d x nfrm x ntgt]
  % f0: starting frame
  % df: frame increment
  % iTgt: target of interest
  % 
  % tffound: logical
  % f: first frame encountered with (non-nan) label, applicable if
  %   tffound==true
  
  if isempty(lpos),
    tffound = false;
    f = f0;
    return;
  end
  
  [npts,d,nfrm,ntgt] = size(lpos); %#ok<ASGLU>
  assert(d==2);
  
  f = f0+df;
  while 0<f && f<=nfrm
    for ipt = 1:npts
      %for j = 1:2
      if ~isnan(lpos(ipt,1,f,iTgt))
        tffound = true;
        return;
      end
      %end
    end
    f = f+df;
  end
  tffound = false;
  f = nan;
end  % function
