function tblout = findNearestNborTrx(tbl,movifo,tfaf,trxCache)
% Find the nearest neighboring trx/target for each row in a tbl
%
% tbl: [n x <width>] MFT table
% tfaf: cellstr. tbl.iTgt are indices into trxFilesAllFull 
% trxCache: trxCache 
%
% nbrDist: [nx1] L2 nearest nbor distance
% nbrTgt: [nx1] nearest nbor target index

fcn = @(iMov,frm,iTgt)lcl(iMov,frm,iTgt,movifo,tfaf,trxCache);
tblout = rowfun(fcn,tbl,'InputVariables',{'mov' 'frm' 'iTgt'},...
  'NumOutputs',2,'OutputVariableNames',{'nbrDist' 'nbrTgt'});

function [nbrDist,nbrTgt] = lcl(iMov,frm,iTgt,movifo,tfaf,trxCache)

ifo = movifo{iMov};
trxfile = tfaf{iMov};
[trx,frm2trx] = Labeler.getTrxCacheStc(trxCache,trxfile,ifo.nframes);
ntgt = numel(trx);
assert(ntgt==size(frm2trx,2));

[xi,yi] = readtrx(trx,frm,iTgt);

distj = inf(ntgt,1);
for jTgt=1:ntgt
  if jTgt~=iTgt && frm2trx(frm,jTgt)
    [xj,yj] = readtrx(trx,frm,jTgt);
    distj(jTgt) = sqrt((xi-xj)^2+(yi-yj)^2);
  end
end

[nbrDist,nbrTgt] = nanmin(distj);