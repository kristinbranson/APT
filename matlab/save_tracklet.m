function save_tracklet(trx,matfile)

% if nargin < 3,
%   d = 2;
% end

td = struct;
td.startframes = cat(1,trx.firstframe);
td.endframes = cat(1,trx.endframe);
ntargets = numel(trx);
td.pTrk = cell(ntargets,1);
if isfield(trx,'pocc'),
  td.pTrkTag = cell(ntargets,1);
end
if isfield(trx,'TS'),
  td.pTrkTS = cell(ntargets,1);
end
td.pTrkiTgt = 1:ntargets;
for i = 1:ntargets,
  if isfield(trx,'id'),
    td.pTrkiTgt(i) = trx(i).id;
  end
%   npts = size(trx(i).p,1)/d;
%   td.pTrk{i} = reshape(trx(i).p,[npts,d,size(trx(i).p,2)]);
  td.pTrk{i} = trx(i).p;
  if isfield(trx,'pocc'),
    td.pTrkTag{i} = trx(i).pocc;
  end
  if isfield(trx,'TS'),
    td.pTrkTS{i} = trx(i).TS;
  end
end
if isfield(trx,'movfile'),
  td.movfile = trx(1).movfile;
end

save(matfile,'-struct','td');