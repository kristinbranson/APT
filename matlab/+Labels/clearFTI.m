function [s,tfchanged] = clearFTI(s,frm,itgt,ipt)
  i = find(s.frm==frm & s.tgt==itgt);
  tfchanged = ~isempty(i); % found our (frm,tgt)
  if tfchanged
    s.p([ipt ipt+s.npts],i) = nan;
    s.ts(ipt,i) = now;
    s.occ(ipt,i) = 0;
    % s.frm(ipt,1) and s.tgt(ipt,1) unchanged
  end
end  % function
