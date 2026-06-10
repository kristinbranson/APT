function s = setpFT(s,frm,itgt,xy)
  i = find(s.frm==frm & s.tgt==itgt);
  if isempty(i)
    % new label
    s.p(:,end+1) = xy(:);
    s.ts(:,end+1) = now;
    s.occ(:,end+1) = 0;
    s.frm(end+1,1) = frm;
    s.tgt(end+1,1) = itgt;
  else
    % updating existing label
    s.p(:,i) = xy(:);
    s.ts(:,i) = now;
    % s.occ(:,i) unchanged
    % s.frm(i) "
    % s.tgt(i) "
  end
end  % function
