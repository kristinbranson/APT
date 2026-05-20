function [tffound,f] = seekSmallLpos(lpos,f0,df)
  % lpos: [npts x nfrm]
  % f0: starting frame
  % df: frame increment
  % 
  % tffound: logical
  % f: first frame encountered with (non-nan) label, applicable if
  %   tffound==true
  
  [npts,nfrm] = size(lpos);
  
  f = f0+df;
  while 0<f && f<=nfrm
    for ipt=1:npts
      if ~isnan(lpos(ipt,f))
        tffound = true;
        return;
      end
    end
    f = f+df;
  end
  tffound = false;
  f = nan;
end  % function
