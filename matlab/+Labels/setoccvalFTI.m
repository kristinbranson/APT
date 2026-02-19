function s = setoccvalFTI(s,frm,itgt,ipt,val)
  % creates a new label if nec
  % ipt: can be vector
  % val: can be scalar for scalar expansion or vec same size as ipt
  i = find(s.frm==frm & s.tgt==itgt);
  if isempty(i)
    % new label
    s.p(:,end+1) = nan;
    s.ts(:,end+1) = now;
    s.occ(:,end+1) = 0;
    s.frm(end+1,1) = frm;
    s.tgt(end+1,1) = itgt;
    i = size(s.p,2);
  end
  s.occ(ipt,i) = val;
  s.ts(:,i) = now;
end  % function
