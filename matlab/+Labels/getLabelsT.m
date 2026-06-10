function [tfhasdata,p,occ,t0,t1] = getLabelsT(s,itgt)
  % get labels/occ for given target.
  %
  % p: [2npts x nf]. nf=t1-t0+1
  % occ: [npts x nf] logical
  % t0/t1: start/end frames (inclusive) labeling 2nd dims of p, occ.

  tf = s.tgt==itgt;
  frms = s.frm(tf);
  tfhasdata = ~isempty(frms);
  if tfhasdata
    t0 = min(frms);
    t1 = max(frms);
    nf = t1-t0+1;
  else
    t0 = nan;
    t1 = nan;
    nf = 0;
  end
  p = nan(2*s.npts,nf);
  occ = false(s.npts,nf);

  if tfhasdata
    idx = frms-t0+1;
    p(:,idx) = s.p(:,tf);
    occ(:,idx) = s.occ(:,tf);
  end
end  % function
