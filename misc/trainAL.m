function trainAL(datafile,prmfile,varargin)
% Take a CPRData, TrainDataI, and TrainParams and produce a TrainRes

[rootdir,tdIfile,tdIfileVar,ignoreChan,forceChan,datatype,td] = myparse(varargin,...
  'rootdir','/groups/flyprojects/home/leea30/cpr/jan',... % place to look for files
  'tdIfile','',... % traindata Index file; if not specified, use td.iTrn
  'tdIfileVar','',...
  'ignoreChan',false,...  % if true, then ignore channel data if present
  'forceChan',true,...    % if true, compute channels and use them (ignoreChan ignored)
  'datatype','REQ',...    % for computeIpp
  'td',[]...             % if supplied, don't load from MAT.
  );                      

if isunix
  if isdeployed
    sha = 'unkDep';
  else
    cmd = sprintf('git --git-dir=%s/.git rev-parse HEAD',CPR.Root);
    [~,cmdout] = system(cmd);
    sha = cmdout(1:5);
  end
else
  sha = 'unkSHA';
end

trname = FS.formTrainedClassifierName(datafile,tdIfile,tdIfileVar,prmfile,sha);
trfilefull = fullfile(rootdir,trname);

diary([trfilefull '.dry']);

datafilefull = fullfile(rootdir,datafile);
if isempty(td)
  td = load(datafilefull);
  flds = fieldnames(td);
  assert(isscalar(flds));
  td = td.(flds{1});
  fprintf(1,'Loaded TD: %s, varname %s\n',datafilefull,flds{1});
end
if ~isempty(tdIfile)
  tdIfilefull = fullfile(rootdir,tdIfile);
  tdI = load(tdIfilefull);
  tdI = tdI.(tdIfileVar);
  td.iTrn = tdI;
  
  fprintf(1,'tdIfile supplied: %s, var %s.\n',tdIfilefull,tdIfileVar);
else
  fprintf(1,'No tdIfile, using indices supplied with td.\n');
end
fprintf(1,'td.NTrn=%d\n',td.NTrn);

try
  td.summarize(td.iTrn);
catch ME
  fprintf(2,'Error summarizing training data: %s\n\n',ME.getReport());
end

if forceChan
  %assert(isempty(td.Ipp),'TEMPORARY');
  fprintf(1,'Computing Ipp!\n');
  pause(3);
  td.computeIpp([],[],[],datatype,true,'iTrl',td.iTrn);
  tfChan = true;
else
  tfChan = ~isempty(td.Ipp) && ~ignoreChan;
end
if tfChan
  Is = td.getCombinedIs(td.iTrn);
  nChanTot = numel(td.IppInfo)+1;
  assert(size(Is{1},3)==nChanTot);
else
  Is = td.I(td.iTrn,:);
end

tpfilefull = fullfile(rootdir,prmfile);
% tp = load(tpfilefull);
% tp = tp.tp;
fprintf(1,'Using params file: %s\n',tpfilefull);
% tpargs = tp.getPVs();
sPrm = ReadYaml(tpfilefull);

if tfChan
  %tpargs(end+1:end+2,1) = {'nChn'; nChan}; % original image counts as channel
  %tpargs(end+1:end+2,1) = {'nChn'; nChan+1}; % original image counts as channel
  sPrm.Ftr.nChn = nChanTot;
end

iPt = sPrm.TrainInit.iPt;
d = sPrm.Model.d;
assert(d==2);
nfidsInTD = size(td.pGT,2)/d; 
if isempty(iPt)
  assert(nfidsInTD==sPrm.Model.nfids);
  iPt = 1:nfidsInTD;
end
iPGT = [iPt iPt+nfidsInTD];
fprintf(1,'iPGT: %s\n',mat2str(iPGT));
  
%% Train on training set
fprintf('Training and saving results to: %s\n',trfilefull);
train(td.pGTTrn(:,iPGT),td.bboxesTrn,Is,...
  'savefile',trfilefull,...
  'modelPrms',sPrm.Model,...
  'regPrm',sPrm.Reg,...
  'ftrPrm',sPrm.Ftr,...
  'initPrm',sPrm.TrainInit,...
  'prunePrm',sPrm.Prune,...
  'docomperr',false);
diary off;


