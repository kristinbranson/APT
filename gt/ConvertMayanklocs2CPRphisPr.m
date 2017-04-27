function phisPr = ConvertMayanklocs2CPRphisPr(locs)

if iscell(locs),
  phisPr = cell(size(locs));
  for v = 1:numel(locs),
    phisPr{v} = ConvertMayankR2CPRphisPr(locs{v});
  end
  return
end

[F,nfids,d,nreps] = size(locs);
phisPr = reshape(locs,[F,nfids*d,nreps]);
