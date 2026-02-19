function s = new(npts,n)
  if nargin<2
    n = 0;
  end
  s = struct();
  s.npts = npts;
  s.p = nan(npts*2,n); % pos
  s.ts = nan(npts,n); % ts -- time a bout was added
  s.occ = zeros(npts,n,Labels.CLS_OCC()); % "tag"
  s.frm = zeros(n,1,Labels.CLS_MD());
  s.tgt = zeros(n,1,Labels.CLS_MD());
end  % function
