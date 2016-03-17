function [pRetestT,pRetestTRed] = retestAL(td,iRetest,tr,res,resITst,varargin)
% Take previous tracked results and re-start
% 
% td: dataset containing frame to be retested
% iRetest: iTrl into td for frame to retest
% tr: trained classifier
% res: results structure (classifier applied to subset of td containing iRetest)
% resITst: iTrl that labels results res

[rootdir,ignoreChan,forceChan] = myparse(varargin,...
    'rootdir',[],...
    'ignoreChan',false,...
    'forceChan',true); % if true, compute channels and use them (ignoreChan ignored)
  
assert(isscalar(iRetest));

%% channels
if forceChan
  if isempty(td.Ipp) || isempty(td.Ipp{iRetest})
    fprintf(1,'Computing Ipp!\n');
    td.computeIpp('iTrl',iRetest);
  end
  tfChan = true;
else
  tfChan = ~isempty(td.Ipp) && ~ignoreChan;
end
if tfChan
  assert(~isempty(td.IppInfo));
  nChan = numel(td.IppInfo);  
  fprintf(1,'Using %d additional channels.\n',nChan);
  
  im = td.I{iRetest};
  impp = td.Ipp{iRetest};
  assert(size(impp,3)==nChan);  
  Is = {cat(3,im,impp)};
else
  Is = td.I(iRetest,:);
end
bb = td.bboxes(iRetest,:);

%% pIni
iRes = find(iRetest==resITst);
assert(isscalar(iRes));
pIni = res.pTstT(iRes,:,:,end);
nRep = size(pIni,2);
assert(size(pIni,3)==td.D);
pIni = permute(pIni,[1 3 2]);
assert(isequal(size(pIni),[1 td.D nRep]));

%% Test on test set
mdl = tr.regModel.model;
[~,~,~,~,pRetestT] = test_rcpr([],bb,Is,tr.regModel,tr.regPrm,tr.prunePrm,pIni);
pRetestT = reshape(pRetestT,[1 nRep mdl.D tr.regModel.T+1]);

%% Select best preds for each time
[N,R,D,Tp1] = size(pRetestT);
pRetestTRed = nan(N,D,Tp1);
prunePrm = tr.prunePrm;
prunePrm.prune = 1;
for t = 1:Tp1
  fprintf('Pruning t=%d\n',t);
  pTmp = permute(pRetestT(:,:,:,t),[1 3 2]); % [NxDxR]
  pRetestTRed(:,:,t) = rcprTestSelectOutput(pTmp,tr.regPrm,prunePrm);
end
