function trainAL(datafile,prmfile,varargin)
% Take a CPRData, TrainDataI, and TrainParams and produce a TrainRes

[rootdir,tdIfile,tdIfileVar] = myparse(varargin,...
    'rootdir','/groups/flyprojects/home/leea30/cpr/jan',... % place to look for files
    'tdIfile','',... % traindata Index file; if not specified, use td.iTrn
    'tdIfileVar','');

datafilefull = fullfile(rootdir,datafile);
td = load(datafilefull);
td = td.td;
fprintf(1,'Loaded TD: %s\n',datafilefull);
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

tfChan = ~isempty(td.Ipp);
if tfChan
  assert(~isempty(td.IppInfo));
  nChan = numel(td.IppInfo);  
  fprintf(1,'Using %d additional channels.\n',nChan);
  
  Is = cell(td.NTrn,1);
  for i = 1:td.NTrn
    iTrl = td.iTrn(i);
    
    im = td.I{iTrl};
    impp = td.Ipp{iTrl};
    assert(size(impp,3)==nChan);
    
    Is{i} = cat(3,im,impp);
  end
else
  Is = td.I(td.iTrn,:);
end

tpfilefull = fullfile(rootdir,prmfile);
tp = load(tpfilefull);
tp = tp.tp;
fprintf(1,'Using params file: %s\n',tpfilefull);
tpargs = tp.getPVs();

if tfChan
  tpargs(end+1:end+2,1) = {'nChn'; nChan+1}; % original image counts as channel
end

%% Train on training set
%cd(RCPR);
if isunix
  cmd = sprintf('git --git-dir=%s/.git rev-parse HEAD',RCPR);
  [~,cmdout] = system(cmd);
  sha = cmdout(1:5);
else
  sha = 'unkSHA';
end

trname = FS.formTrainedClassifierName(datafile,tdIfile,tdIfileVar,prmfile,sha);
trfilefull = fullfile(rootdir,trname);
fprintf('Training and saving results to: %s\n',trfilefull);

diary([trfilefull '.dry']);
[regModel,regPrm,prunePrm,phisPr,err] = train(...
    td.pGTTrn,td.bboxesTrn,Is,'savefile',trfilefull,tpargs{:});
diary off;


