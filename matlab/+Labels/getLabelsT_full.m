function [p,occ] = getLabelsT_full(s,itgt,nf)
  % get labels/occ for given target.
  % nf: total number of frames for target/mov
  %
  % p: [2npts x nf]
  % occ: [npts x nf] logical
  
  p = nan(2*s.npts,nf);
  occ = false(s.npts,nf);
  tf = s.tgt==itgt;
  frms = s.frm(tf);
  p(:,frms) = s.p(:,tf);
  occ(:,frms) = s.occ(:,tf);
end  % function
