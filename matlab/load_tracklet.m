function trx = load_tracklet(matfile,frompy)
% load_tracklet(matfile,frompy)
% load_tracklet(matcontents,frompy)
%


if nargin < 2,
  frompy = false;
end

if ischar(matfile)
  td = load(matfile,'-mat');
else
  td = matfile;
end

ntargets = numel(td.pTrk);
for i = 1:ntargets,
  trxcurr = struct;
  trxcurr.id = td.pTrkiTgt(i);
  trxcurr.p = td.pTrk{i};
  if isfield(td,'pTrkTag'),
    trxcurr.pocc = td.pTrkTag{i};
  end
  if isfield(td,'pTrkTS'),
    trxcurr.TS = td.pTrkTS{i};
  end
  trxcurr.firstframe = td.startframes(i);
  trxcurr.endframe = td.endframes(i);
  trxcurr.nframes = trxcurr.endframe - trxcurr.firstframe + 1;
  trxcurr.off = 1-trxcurr.firstframe;
  if isfield(td,'movfile'),
    trxcurr.movfile = td.movfile;
  end    
  if frompy,
    trxcurr.id = trxcurr.id + 1;
    trxcurr.p = trxcurr.p + 1;
    if isfield(td,'pTrkTS'),
      trxcurr.TS = trxcurr.TS + 1;
    end
    trxcurr.firstframe = trxcurr.firstframe + 1;
    trxcurr.endframe = trxcurr.endframe + 1;
  end    
  if i == 1,
    trx = trxcurr;
  else
    trx(i) = trxcurr;
  end
end
if isfield(td,'pTrkTS'),
  ts = td.pTrkTS;
end
if frompy,
  ts = ts + 1;
end
