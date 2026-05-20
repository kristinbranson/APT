function s = fromtableMA(t)
  sz = size(t.p);
  npts = sz(end)/2;
  p = t.p;
  ts = t.pTS;
  occ = t.tfocc;
  nlbls = nnz(~all(isnan(p),3));
  s = Labels.new(npts,nlbls);      
  count = 1;
  for fndx = 1:size(p,1)
    curt = 1;
    for tndx = 1:size(p,2)
      if all(isnan(p(fndx,tndx,:))), continue; end
      s.p(:,count) = p(fndx,tndx,:);
      s.ts(:,count) = ts(fndx,tndx,:);
      s.occ(:,count) = occ(fndx,tndx,:);
      s.frm(count) = t.frm(fndx);
      s.tgt(count) = uint32(curt);
      curt = curt+1;
      count = count+1;
    end
  end

end  % function
