function s2 = indexPts(s,ipts)
  % "subsidx" given pts
  
  s2 = struct();
  ip = [ipts(:); ipts(:)+s.npts]; % xs and ys for these pts
  s2.p = s.p(ip,:);
  s2.ts = s.ts(ipts,:);
  s2.occ = s.occ(ipts,:);
  s2.frm = s.frm;
  s2.tgt = s.tgt;
end  % function
