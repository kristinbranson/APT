Q = load('../RomainLeg/RomainCombined_fixed.lbl','-mat');
for ndx = 1:numel(Q.movieFilesAll)
  Q.movieFilesAll{ndx} = ['/localhome/kabram/' Q.movieFilesAll{ndx}(20:end)];
end

for ndx = 1:numel(Q.movieFilesAll)
  if ~exist(Q.movieFilesAll{ndx},'file'),
    fprintf('%d %s doesnt exist\n',ndx,Q.movieFilesAll{ndx});
  end
end