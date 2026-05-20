function [s,tfchanged] = rmFT(s,frm,itgt)
  % remove labels for given frm/itgt      
  i = find(s.frm==frm & s.tgt==itgt);
  tfchanged = ~isempty(i); % found our (frm,tgt)
  if tfchanged
    s.p(:,i) = [];
    s.ts(:,i) = [];
    s.occ(:,i) = [];
    s.frm(i,:) = [];
    s.tgt(i,:) = [];
  end
end  % function
