function [tf,p,occ,ts] = isLabeledPerPtFT(s,frm,itgt)
  % [tf,p,occ,ts] = isLabeledPerPtFT(s,frm,itgt)
  % Added by KB 20220202, similar to isLabeledFT
  % tf(i) indicates whether landmark i is labeled. 
  % p, occ, ts returned are the landmarks locations, occluded labels,
  % and timestamps
  i = find(s.frm==frm & s.tgt==itgt,1);
  tf = ~isempty(i);
  if tf
    p = s.p(:,i);
    tf = any(~isnan(reshape(p,[numel(p)/2 2])),2);
    occ = s.occ(:,i);
    ts = s.ts(:,i);
  else
    tf = false(s.npts,1);
    p = nan(2*s.npts,1);
    occ = zeros(s.npts,1,Labels.CLS_OCC());
    ts = -inf(s.npts,1);
  end
end  % function
