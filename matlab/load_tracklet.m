function trx = load_tracklet(matfile,frompy)
% trx = load_tracklet(matfile,frompy)
% trx = load_tracklet(matcontents,frompy)


if nargin < 2,
  frompy = false;
end

if ischar(matfile)
  td = load(matfile,'-mat');
else
  td = matfile;
end

% % aux fields like confidences, occlusion-scores etc
% fldsaux = fieldnames(td);
% fldsaux = fldsaux(startsWith(fldsaux,'pTrk'));
% fldsaux = setdiff(fldsaux,{'pTrk' 'pTrkFrm' 'pTrkTS' 'pTrkTag' 'pTrkiTgt'});
% fldsaux = fldsaux(:)';

ntargets = numel(td.pTrk);

if ntargets==0
  % fields made here match those added below in case of ntargets>=1
  trx = TrxUtil.newptrx(0,td.npts);
end

for i = 1:ntargets,
  trxcurr = struct;
  trxcurr.id = td.pTrkiTgt(i);
  trxcurr.p = td.pTrk{i};
  if isfield(td,'pTrkTag') || isprop(td,'pTrkTag') 
    if ~ischar(td.pTrkTag) %|| ~(strcmp(td.pTrkTag,'__UNSET__'))
      trxcurr.pocc = td.pTrkTag{i};
    end
  end
  if isfield(td,'pTrkTS') || isprop(td,'pTrkTS')
    trxcurr.TS = td.pTrkTS{i};
  end
%   for f=fldsaux,f=f{1}; %#ok<FXSET>
%     trxcurr.(f) = td.(f){i};
%   end
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

% if isfield(td,'pTrkTS'),
%   ts = td.pTrkTS;
% end
% if frompy,
%   ts = ts + 1;
% end
