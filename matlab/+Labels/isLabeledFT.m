function [tf,p,occ,ts] = isLabeledFT(s,frm,itgt)
  % Could get "getLabelsFT"
  %
  % p, occ, ts have appropriate size/vals even if tf==false
  
  i = find(s.frm==frm & s.tgt==itgt,1);
  tf = ~isempty(i);
  if tf
    p = s.p(:,i);
    occ = s.occ(:,i);
    ts = s.ts(:,i);
  else
    p = nan(2*s.npts,1);
    occ = zeros(s.npts,1,Labels.CLS_OCC());
    ts = -inf(s.npts,1);
  end
end  % function
