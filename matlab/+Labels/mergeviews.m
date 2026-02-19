function s = mergeviews(sarr)
  % sarr: array of Label structures

  if isscalar(sarr)
    s = sarr;
    return;
  end
  
  assert(isequal(sarr.npts),'npts must be equal across views.');
  assert(isequal(sarr.frm),'frames must be equal across views.');
  assert(isequal(sarr.tgt),'targets must be equal across views.');
  
  npts = sarr(1).npts;
  nview = numel(sarr);
  for i=1:nview
    sarr(i).p = reshape(sarr(i).p,npts,2,[]);
  end
  s = sarr(1);
  s.npts = npts*nview;
  s.p = cat(1,sarr.p);
  s.p = reshape(s.p,2*s.npts,[]);
  s.ts = cat(1,sarr.ts);
  s.occ = cat(1,sarr.occ);
  % .frm, .tgt unchanged
end  % function
