function testAL(tpfile,tdfile,trfile,varargin)
% Take a TrainData, TrainRes, and produce test results

[rootdir,tdIfile,tdIfileVar,iTst,testres,ignoreChan,forceChan,skipLoss,...
  iterMovie,datatype,td] = myparse(varargin,...
    'rootdir','/groups/flyprojects/home/leea30/cpr/jan',... % place to look for files
    'tdIfile','',... % traindata Index file for testing; if not specified, use td.iTst. SPECIAL CASE: 'every5'
    'tdIfileVar','',...
    'iTst',[],... % direct specification of iTst (indices into td). If specified, tfIfile/tdIfileVar ignored.
    'testres',[],...% previously computed/saved test results
    'ignoreChan',false,...
    'forceChan',true,... % if true, compute channels and use them (ignoreChan ignored)
    'skipLoss',false,...% if true, don't compute loss/stats comparing tracked results to GT
    'iterMovie',false,... % if true, make a movie-over-iterations     
    'datatype','REQ',... % for computeIpp
    'td',[]); % loaded td;
  
if ischar(skipLoss)
  skipLoss = str2double(skipLoss);
end

tfTestRes = ~isempty(testres);

tdfilefull = fullfile(rootdir,tdfile);
if isempty(td)
  td = load(tdfilefull);
  td = td.td;
  fprintf(1,'Loaded TD: %s\n',tdfilefull);
end

if ~isempty(iTst)
  fprintf(1,'iTst specified: %s\n',mat2str(iTst));
  td.iTst = iTst;
elseif ~isempty(tdIfile)
  if strcmp(tdIfile,'every5')
    td.iTst = 1:5:numel(td.I);  
    fprintf(1,'tdIfile: every 5. from 1 to nframes=%d\n',numel(td.I));
  else  
    tdIfilefull = fullfile(rootdir,tdIfile);
    tdI = load(tdIfilefull);
    tdI = tdI.(tdIfileVar);
    td.iTst = tdI;
    
    fprintf(1,'tdIfile supplied: %s, var %s.\n',tdIfilefull,tdIfileVar);
  end
else  
    fprintf(1,'No tdIfile, using .iTst.\n');
    %td.iTst = find(td.isFullyLabeled);
end
fprintf(1,'td.NTst=%d\n',td.NTst);
if td.NTst<=20
   disp(td.MDTst);
end

try
  td.summarize(td.iTst);
catch ME
  fprintf(2,'Error summarizing training data: %s\n\n',ME.getReport());
end

if forceChan
 % assert(isempty(td.Ipp),'TEMPORARY');
  fprintf(1,'Computing Ipp!\n');
  pause(3);
  td.computeIpp([],[],[],datatype,true,'iTrl',td.iTst);
  tfChan = true;
else
  tfChan = ~isempty(td.Ipp) && ~ignoreChan;
end

%% channels
if tfChan
  assert(~isempty(td.IppInfo));
  nChan = numel(td.IppInfo);  
  fprintf(1,'Using %d additional channels.\n',nChan);
  
  Is = cell(td.NTst,1);
  for i = 1:td.NTst
    iTrl = td.iTst(i);
    
    im = td.I{iTrl};
    impp = td.Ipp{iTrl};
    assert(size(impp,3)==nChan);
    
    Is{i} = cat(3,im,impp);
  end
else
  Is = td.I(td.iTst,:);
end

tpfilefull = fullfile(rootdir,tpfile);
fprintf(1,'Using params file: %s\n',tpfilefull);
sPrm = ReadYaml(tpfilefull);
prmTestInit = sPrm.TestInit;

%%
trfilefull = fullfile(rootdir,trfile);
tr = load(trfilefull,'-mat');
fprintf(1,'Train results file: %s\n',trfilefull);

%% Test on test set
if tfTestRes
  pTstT = testres.pTstT;
else
  mdl = tr.regModel.model;
  pGTTrnNMu = nanmean(tr.regModel.pGtN,1);
  pIni = shapeGt('initTest',[],td.bboxesTst,mdl,[],...
    repmat(pGTTrnNMu,td.NTst,1),prmTestInit.Nrep,prmTestInit.augrotate);
  VERBOSE = 0;
  [~,p_t] = rcprTest1(Is,tr.regModel,pIni,tr.regPrm,tr.ftrPrm,td.bboxesTst,VERBOSE,tr.prunePrm);  
  pTstT = reshape(p_t,[td.NTst prmTestInit.Nrep mdl.D tr.regModel.T+1]);
end

%% Select best preds for each time
if tfTestRes
  pTstTRed = testres.pTstTRed;
else
  [N,R,D,Tp1] = size(pTstT);
  pTstTRed = nan(N,D,Tp1);
  prunePrm = tr.prunePrm;
  prunePrm.prune = 1;
  for t = 1:Tp1
    fprintf('Pruning t=%d\n',t);
    pTmp = permute(pTstT(:,:,:,t),[1 3 2]); % [NxDxR]
    pTstTRed(:,:,t) = rcprTestSelectOutput(pTmp,tr.regModel.model,prunePrm);
  end
end

if ~skipLoss
  
  %%
  hFig = Shape.vizLossOverTime(td.pGTTst,pTstTRed,'md',td.MDTst);

  %%
  hFig(end+1) = figure('WindowStyle','docked');
  iTst = td.iTst;
  tfTstLbled = ismember(iTst,find(td.isFullyLabeled));
  Shape.vizDiff(td.ITst(tfTstLbled),td.pGTTst(tfTstLbled,:),...
    pTstTRed(tfTstLbled,:,end),tr.regModel.model,...
    'fig',gcf,'nr',4,'nc',4,'md',td.MDTst(tfTstLbled,:));
end

%% Save results
if true %~tfTestRes
  resultsfolder = FS.formTestResultsFolderName(trfile,tdfile,tdIfile,tdIfileVar);
  assert(exist(resultsfolder','dir')==0);
  mkdir(rootdir,resultsfolder);
  resdir = fullfile(rootdir,resultsfolder);
  fprintf('Training and saving results to: %s\n',resdir);

  %%
  if ~skipLoss
    savefig(hFig,fullfile(resdir,'results.fig'));
  end
  save(fullfile(resdir,'res.mat'),'-v7.3','pIni','pTstT','pTstTRed');

  if iterMovie
    % Movie
    NTRIALS = 1;
    trls = randsample(td.NTst,NTRIALS);
    for iTrl = trls(:)'
      movname = fullfile(resdir,sprintf('vizROTD_iTrl%04d',iTrl));
      hFig(end+1) = figure;
      Shape.vizRepsOverTimeDensity(td.ITst,pTstT,iTrl,tr.regModel.model,...
        'fig',gcf,'smoothsig',20,'movie',true,'moviename',movname);
    end
  end
end

if isdeployed
  close all force;
end
  

% %% Train on training set
% %cd(RCPR);
% cmd = sprintf('git --git-dir=%s/.git rev-parse HEAD',RCPR);
% [~,cmdout] = system(cmd);
% trainNote = cmdout(1:5);
% 
% PAT = '^td_(?<base>[a-zA-Z0-9_]+)__[0-9]{8,8}+\.mat';
% names = regexp(tdfile,PAT,'names');
% 
% trname = sprintf('%s@%s@%s@%s@%s@%s',names.base,tdIfile,tdIfileVar,tp.Name,...
%     trainNote,datestr(now,'yyyymmddTHHMM'));
% trfilefull = fullfile(rootdir,trname);
% fprintf('Training and saving results to: %s\n',trfilefull);
% 
% diary([trfilefull '.dry']);
% [regModel,regPrm,prunePrm,phisPr,err] = train(...
%     td.pGTTrn,td.bboxesTrn,td.ITrn,'savefile',trfilefull,tpargs{:});
% diary off;


