function [s,tfchanged] = rmFTP(s,frm,itgt,pts)
  % remove labels for given frm/itgt      
  i = find(s.frm==frm & s.tgt==itgt);
  tfchanged = ~isempty(i); % found our (frm,tgt)
  if tfchanged
    ptidx = false(size(s.occ,1),1);
    ptidx(pts) = true;
    s.p(repmat(ptidx,[2,1]),i) = Labels.getUnlabeledValue();
    s.ts(ptidx,i) = Labels.getUnlabeledValue();
    s.occ(ptidx,i) = false;
  end
end  % function
