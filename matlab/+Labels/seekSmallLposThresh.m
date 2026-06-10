function [tffound,f] = seekSmallLposThresh(lpos,f0,df,th,cmp)
  % lpos: [npts x nfrm]
  % f0: starting frame
  % df: frame increment
  % th: threshold
  % cmp: comparitor
  % 
  % tffound: logical
  % f: first frame encountered with (non-nan) label that satisfies 
  % comparison with threshold, applicable if tffound==true
  
  switch cmp
    case '<',  cmp = @lt;
    case '<=', cmp = @le;
    case '>',  cmp = @gt;
    case '>=', cmp = @ge;
  end
      
  [npts,nfrm] = size(lpos);
  
  f = f0+df;
  while 0<f && f<=nfrm
    for ipt=1:npts
      if cmp(lpos(ipt,f),th)
        tffound = true;
        return;
      end
    end
    f = f+df;
  end
  tffound = false;
  f = nan;
end  % function
