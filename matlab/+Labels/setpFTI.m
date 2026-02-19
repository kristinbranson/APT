function s = setpFTI(s,frm,itgt,ipt,xy)
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
  s.p([ipt ipt+s.npts],i) = xy(:);
  s.ts(ipt,i) = now;
end  % function
