function s = fromtable(t)
  if any(strcmp(t.Properties.VariableNames,'mov'))
    assert(all(t.mov==t.mov(1)));
    warningNoTrace('.mov column will be ignored.');        
  end

  if ndims(t.p)==3
    s = Labels.fromtableMA(t);
    return;
  end
  
  n = height(t);
  sz = size(t.p);
  npts = sz(end)/2;
  s = Labels.new(npts,n);      
  p = t.p.';
  ts = t.pTS.';
  occ = t.tfocc.';
  s.p(:) = p(:);
  s.ts(:) = ts(:);
  s.occ(:) = occ(:);
  s.frm(:) = t.frm;
  s.tgt(:) = t.iTgt;
  % if max(t.iTgt)==0
  %   % when exporting for MA, all the iTgt can get set to 0
  %   tgt = zeros(size(s.frm));
  %   for i=1:numel(s.frm)
  %     tgt(i) = sum(s.frm(1:i)==s.frm(i));
  %   end
  %   s.tgt(:) = uint32(tgt);
  % end
end  % function
