function s1 = remapLandmarks(s,new2oldpts)      
  oldnpts = s.npts;
  n = size(s.p,2);
  newnpts = numel(new2oldpts);
  isold = new2oldpts > 0;
  s1 = s;
  
  p = reshape(s.p,[oldnpts,2,n]);
  p1 = nan([newnpts,2,n],class(s.p));
  p1(isold,:,:) = p(new2oldpts(isold),:,:);
  s1.p = reshape(p1,[newnpts*2,n]);
  s1.ts = nan([newnpts,n],class(s.ts));
  s1.ts(isold,:) = s.ts(new2oldpts(isold),:);
  s1.occ = zeros([newnpts,n],class(s.occ));
  s1.occ(isold,:) = s.occ(new2oldpts(isold),:);
  s1.npts = newnpts;      
end  % function
