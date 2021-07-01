function tlt = save_tracklet(trx,matfile)
% tlt = save_tracklet(trx,matfile)
% matfile: pass [] to just return tracklet data without saving

tfnosave = isequal(matfile,[]);

tlt = struct;
tlt.startframes = cat(1,trx.firstframe);
tlt.endframes = cat(1,trx.endframe);
ntargets = numel(trx);
tlt.pTrk = cell(ntargets,1);
if isfield(trx,'pocc'),
  tlt.pTrkTag = cell(ntargets,1);
end
if isfield(trx,'TS'),
  tlt.pTrkTS = cell(ntargets,1);
end
tlt.pTrkiTgt = 1:ntargets;
for i = 1:ntargets,
  if isfield(trx,'id'),
    tlt.pTrkiTgt(i) = trx(i).id;
  end
%   npts = size(trx(i).p,1)/d;
%   td.pTrk{i} = reshape(trx(i).p,[npts,d,size(trx(i).p,2)]);
  tlt.pTrk{i} = trx(i).p;
  if isfield(trx,'pocc'),
    tlt.pTrkTag{i} = trx(i).pocc;
  end
  if isfield(trx,'TS'),
    tlt.pTrkTS{i} = trx(i).TS;
  end
end
if isfield(trx,'movfile'),
  tlt.movfile = trx(1).movfile;
end

if ~tfnosave
  save(matfile,'-struct','tlt');
end