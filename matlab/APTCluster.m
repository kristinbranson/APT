function APTCluster(varargin)
% Compiled APT operations
%
% % Do a full retrain of a project's tracker
% APTCluster(lblFile,'retrain')
%
% % Track a single movie
% APTCluster(lblFile,'track',moviefullpath)
%
% APTCluster(lblFile,'track',moviefullpath,trxfullpath)
%
% % Options passed to Labeler.trackAndExport, 'trackArgs'
% APTCluster(lblFile,'track',moviefullpath,trxfullpath,varargin)
% 
% % Track a set of movies
% APTCluster(lblFile,'trackbatch',moviesetfile,varargin)
%
% % CrossValidation
% APTCluster(lblFile,'xv','tableFile',tableFile,...
%     'tableSplitFile',tableSplitFile,'paramFile',paramFile,...
%     'paramPatchFile',paramPatchFile,'outdir',outdir);
% APTCluster(lblFile,'trntrk','tableFileTrn',tblFileTrn,...
%     'tableFileTrk',tblFileTrk,'paramFile',paramFile,...
%     'paramPatchFile',paramPatchFile,'outdir',outdir);

startTime = tic;

% smoke/diagnostic test for when deployed
% APTCluster hello
% APTCluster hello diag
if strcmp(varargin{1},'hello')
  disp('### APTCluster hello! ###');
  if numel(varargin)>1 && strcmp(varargin{2},'diag')
    croot = ctfroot;
    aptroot = APT.getRoot;
    fprintf(1,'ctfroot is %s, aptroot is %s\n',croot,aptroot);
    cmd = sprintf('find %s',croot);
    system(cmd);
  end
  
  return
end

lblFile = varargin{1};
action = varargin{2};

if exist(lblFile,'file')==0
  error('APTCluster:file','Cannot find project file: ''%s''.',lblFile);
end

lObj = Labeler('isgui',false);
set(lObj.hFig,'Visible','off');
fprintf('APTCluster: ''%s'' on project ''%s''.\n',action,lblFile);
%fprintf('Time to start labeler: %f\n',toc(startTime));
startTime = tic;
switch action
  case 'retrain'
    varargin = varargin(3:end);
    [tableFile,paramFile,paramPatchFile] = ...
      myparse(varargin,...
      'tableFile','',... % (opt) mat-filename containing an MFTtable of training rows
      'paramFile','',... % (opt) mat-filename containing a single param struct that will be fed to lObj.trackSetParams
      'paramPatchFile',''... % (opt) textfile containing parameter 'patch'
      );
    
    lObj.projLoad(lblFile);
    if lObj.movieReadPreLoadMovies
      fprintf(' ... preload is on.\n');
    end
    if lObj.gtIsGTMode
      lObj.gtSetGTMode(false,'warnChange',true);
    end
    
    [tCPR,iTrk] = lObj.trackGetTracker('cpr');
    if tCPR~=lObj.tracker
      fprintf(1,'Setting tracker to CPR.\n');
      lObj.trackSetCurrentTracker(iTrk);
    end
        
    retrainArgs = cell(1,0);
    [lblP,lblF,lblE] = fileparts(lblFile);
    outfileBase = lblF;
    tfTable = ~isempty(tableFile);
    if tfTable
      tblMFT = MFTable.loadTableFromMatfile(tableFile);
      fprintf(1,'Loaded training table (%d rows) from %s.\n',height(tblMFT),tableFile);
      retrainArgs = [retrainArgs {'tblMFTtrn' tblMFT}];
      [~,tableFileS,~] = fileparts(tableFile);
      outfileBase = [outfileBase '_' tableFileS];
    end
    
    outfileBase = lclSetParamsApplyPatches(lObj,paramFile,paramPatchFile,outfileBase);
    
    outfileBase = [outfileBase '_retrain' datestr(now,'yyyymmddTHHMMSS')];
    outfile = fullfile(lblP,[outfileBase lblE]);
    lObj.trackRetrain(retrainArgs{:});
    fprintf('APTCluster: saving retrained project: %s\n',outfile);
    lObj.projSaveRaw(outfile);
  case 'track'
    lObj.projLoad(lblFile,'nomovie',true);
    mov = varargin{3};
    if nargin>3
      trxfile = varargin{4};
      trackArgs = varargin(5:end);
    else
      trxfile = '';
      trackArgs = {};
    end      
    lclTrackAndExportSingleMov(lObj,mov,trxfile,trackArgs); 
  case 'track2'
    % we went astray
    lObj.projLoad(lblFile,'nomovie',true);
    movfiles = varargin{3};
    trxfiles = varargin{4};
    trkfiles = varargin{5};
    trackArgs = varargin(6:end);
    for imov=1:numel(movfiles)
      args = [{'rawtrkname' trkfiles{imov}} trackArgs(:)'];
      lclTrackAndExportSingleMov(lObj,movfiles{imov},trxfiles{imov},args);
    end
  case 'trackbatch'
    lObj.projLoad(lblFile,'nomovie',true);
%    fprintf('Time to load project: %f\n',toc(startTime)); startTime = tic;
    movfile = varargin{3};
    if exist(movfile,'file')==0
      error('APTCluster:file','Cannot find batch movie file ''%s''.',movfile);
    end
    movs = importdata(movfile);
    if ~iscellstr(movs) 
      error('APTCluster:movfile','Error reading batch movie file ''%s''.',movfile);
    end
    nmov = numel(movs);
%    fprintf('Time to read movie info: %f\n',toc(startTime)); startTime = tic;
    for iMov = 1:nmov
      lclTrackAndExportSingleMov(lObj,movs{iMov},'',{});
    end
  case 'xv'
    varargin = varargin(3:end);
    [auxRoot,tableFile,tableSplitFile,paramFile,paramPatchFile,outDir] = ...
      myparse(varargin,...
      'auxRoot','',... % (opt) if supplied, root dir for all following args
      'tableFile','',... % (opt) mat-filename containing an MFTtable for rows to consider in XV
      'tableSplitFile','',... % (opt) mat-filename containing split variables. If specified, tableFile must be specced %      'tableSplitFileVar','',... % (opt) variable name in tableSplitFile. <tableSplitFile>.(tableSplitFielVar) should be a [height(<tableFile>) x nfold] logical where true indicates train and false indicates test
      'paramFile','',... % (opt) mat-filename containing a single param struct that will be fed to lObj.trackSetParams
      'paramPatchFile','',... % (opt) textfile containing parameter 'patch'
      'outDir',''... (opt) location to place xv output. Defaults to lblfile path
      );
    
    lObj.projLoad(lblFile);
    if lObj.movieReadPreLoadMovies
      fprintf(' ... preload is on.\n');
    end
    if lObj.gtIsGTMode
      lObj.gtSetGTMode(false,'warnChange',true);
    end

    tfAuxRoot = ~isempty(auxRoot);
    tfTable = ~isempty(tableFile);
    tfSplit = ~isempty(tableSplitFile);
%     xvArgs = cell(1,0);
    xvArgs = {'wbObj',WaitBarWithCancelCmdline('APTCluster xv')};
    [lblP,lblF,lblE] = fileparts(lblFile);
    outfileBase = ['xv_' lblF];
    if tfTable
      if tfAuxRoot
        tableFile = fullfile(auxRoot,tableFile);
      end
      [~,tableFileS,~] = fileparts(tableFile);
      tblMFT = MFTable.loadTableFromMatfile(tableFile);
      fprintf(1,'Loaded table (%d rows) from %s.\n',height(tblMFT),tableFile);
      xvArgs = [xvArgs {'tblMFgt' tblMFT}];
      outfileBase = [outfileBase '_' tableFileS];
    end
    if tfSplit
      assert(tfTable);
      if tfAuxRoot
        tableSplitFile = fullfile(auxRoot,tableSplitFile);
      end
      [~,tableSplitFileS,~] = fileparts(tableSplitFile);
      split = loadSingleVariableMatfile(tableSplitFile);      
      if ~(islogical(split) && ismatrix(split) && size(split,1)==height(tblMFT))
        error('Expected split definition to be a logical matrix with %d rows.\n',...
          height(tblMFT));
      end
      kfold = size(split,2);
      fprintf(1,'Loaded split (%d fold) from %s.\n',kfold,tableSplitFile);
      xvArgs = [xvArgs {'kfold' kfold 'partTst' split}];
      outfileBase = [outfileBase '_' tableSplitFileS];
    end
    
    if tfAuxRoot && ~isempty(paramFile)
      paramFile = fullfile(auxRoot,paramFile);
    end
    if tfAuxRoot && ~isempty(paramPatchFile)
      paramPatchFile = fullfile(auxRoot,paramPatchFile);
    end    
    outfileBase = lclSetParamsApplyPatches(lObj,paramFile,paramPatchFile,outfileBase);
    
    outfileBase = [outfileBase '_' datestr(now,'yyyymmddTHHMMSS')];
    
    lObj.trackCrossValidate(xvArgs{:});
    
    savestuff = struct();
    %assert(false,'TODO: react');
    savestuff.sPrm = lObj.trackGetParams();
    savestuff.xvArgs = xvArgs;
    savestuff.xvRes = lObj.xvResults;
    savestuff.xvResTS = lObj.xvResultsTS; %#ok<STRNU>
    if isempty(outDir)
      outDir = lblP;
    else
      if tfAuxRoot
        outDir = fullfile(auxRoot,outDir);
      end
    end
    outfile = fullfile(outDir,[outfileBase '.mat']);    
    fprintf('APTCluster: saving xv results: %s\n',outfile);
    save(outfile,'-mat','-struct','savestuff');
    
  case 'trntrk'
    varargin = varargin(3:end);
    [tblFileTrn,tblFileTrk,paramFile,paramPatchFile,outDir] = ...
      myparse(varargin,...
      'tblFileTrn','',... % (opt) mat-filename containing an MFTtable with training rows
      'tblFileTrk','',... % (opt) mat-filename containing an MFTtable with tracking rows
      'paramFile','',... % (opt) mat-filename containing a single param struct that will be fed to lObj.trackSetParams
      'paramPatchFile','',... % (opt) textfile containing parameter 'patch'
      'outDir',''... (opt) location to place output. Defaults to lblfile path
      );
    
    lObj.projLoad(lblFile);
    if lObj.movieReadPreLoadMovies
      fprintf(' ... preload is on.\n');
    end
    if lObj.gtIsGTMode
      lObj.gtSetGTMode(false,'warnChange',true);
    end

    assert(~isempty(tblFileTrn));
    assert(~isempty(tblFileTrk));
    [lblP,lblF,lblE] = fileparts(lblFile);
    outfileBase = ['trntrk_' lblF];
    
    tblTrn = MFTable.loadTableFromMatfile(tblFileTrn);
    fprintf(1,'Loaded train table (%d rows) from %s.\n',height(tblTrn),tblFileTrn);
    [~,tblFileTrnS,~] = fileparts(tblFileTrn);
    outfileBase = [outfileBase '_' tblFileTrnS];

    tblTrk = MFTable.loadTableFromMatfile(tblFileTrk);
    fprintf(1,'Loaded track table (%d rows) from %s.\n',height(tblTrk),tblFileTrk);
    [~,tblFileTrkS,~] = fileparts(tblFileTrk);
    outfileBase = [outfileBase '_' tblFileTrkS];
    
    outfileBase = lclSetParamsApplyPatches(lObj,paramFile,paramPatchFile,outfileBase);
    
    outfileBase = [outfileBase '_' datestr(now,'yyyymmddTHHMMSS')];
    
    args = {'wbObj',WaitBarWithCancelCmdline('APTCluster trntrk')};
    tblRes = lObj.trackTrainTrackEval(tblTrn,tblTrk,args{:});
    
    savestuff = struct();
    assert(false,'TODO: react');
    savestuff.sPrm = lObj.trackGetParams();
    savestuff.tblRes = tblRes;
    savestuff.tblResTS = now;
    if isempty(outDir)
      outDir = lblP;
    end
    outfile = fullfile(outDir,[outfileBase '.mat']);    
    fprintf('APTCluster: saving results: %s\n',outfile);
    save(outfile,'-mat','-struct','savestuff');
    
  case 'gtcompute'
    varargin = varargin(3:end);
    [outDir] = ...
      myparse(varargin,...
      'outDir',''... (opt) location to place xv output. Defaults to lblfile path
      );
    
    lObj.projLoad(lblFile);
    tGT = lObj.gtComputeGTPerformance(); %#ok<NASGU>
    
    [lblFileP,lblFileS,~] = fileparts(lblFile);
    outfile = [lblFileS '_gtcomputed_' datestr(now,'yyyymmddTHHMMSS') '.mat'];
    
    if isempty(outDir)
      outDir = lblFileP;
    end
    outfile = fullfile(outDir,outfile);
    
    fprintf('APTCluster: saving GT results: %s\n',outfile);
    save(outfile,'-mat','tGT');
    
  otherwise
    error('APTCluster:action','Unrecognized action ''%s''.',action);
end

fprintf('Real processing done, total time: %f\n',toc(startTime)); startTime = tic;

delete(lObj);

%fprintf('Time to close APT: %f\n',toc(startTime)); startTime = tic;

close all force;
%fprintf('Time to close everything else: %f\n',toc(startTime)); startTime = tic;
fprintf('APTCluster finished.\n');

function outfileBase = lclSetParamsApplyPatches(lObj,...
  paramFile,paramPatchFile,outfileBase)

tfParam = ~isempty(paramFile);
tfPPatch = ~isempty(paramPatchFile);
if tfParam
  sPrm = loadSingleVariableMatfile(paramFile);
  fprintf(1,'Loaded parameters from %s.\n',paramFile);
  %assert(false,'TODO: react');
  lObj.trackSetParams(sPrm);
  [~,paramFileS,~] = fileparts(paramFile);
  outfileBase = [outfileBase '_' paramFileS];
end
if tfPPatch
  assert(false,'TODO: react');
  sPrm = lObj.trackGetParams();
  sPrm = HPOptim.readApplyPatch(sPrm,paramPatchFile);
  assert(false,'TODO: react');
  lObj.trackSetParams(sPrm);
  [~,paramPatchFileS,~] = fileparts(paramPatchFile);
  outfileBase = [outfileBase '_' paramPatchFileS];
end

function lclTrackAndExportSingleMov(lObj,mov,trx,trackArgs)
% Trx: optional, specify '' for no-trx

startTime = tic;
isClean = false;

if strcmp(lObj.tracker.algorithmName,'cpr'),
  lObj.cleanUpProjTempDir();
  isClean = true;
else
  % set to cpr
  lObj.trackSetCurrentTracker(1);
  assert(strcmp(lObj.tracker.algorithmName,'cpr'));
end

if lObj.gtIsGTMode
  error('APTCluster:gt','Unsupported for GT mode.');
end
if lObj.isMultiView
  error('APTCluster:multiview','Unsupported for multiview projects.');
end
if exist(mov,'file')==0
  error('APTCluster:file','Cannot find movie file ''%s''.',mov);
end
tfTrxIn = ~isempty(trx);
if tfTrxIn && exist(trx,'file')==0
  error('APTCluster:file','Cannot find trx file ''%s''.',trx);
end

mov = FSPath.fullyLocalizeStandardizeChar(mov,struct());
[tfMovInProj,iMov] = ismember(mov,lObj.movieFilesAllFull);
if tfTrxIn
  trx = FSPath.fullyLocalizeStandardizeChar(trx,struct());
  [tfTrxInProj,iTrx] = ismember(trx,lObj.trxFilesAllFull);
end

if tfMovInProj
  if tfTrxIn
    if tfTrxInProj && iTrx==iMov
      % (mov,trx) is already in proj
      
      % no action; iMov is set        
    else
      warningNoTrace('Movie ''%s'' exists in project, but not with trxfile ''%s''.',...
        mov,trx);
      % Attempt to add new (mov,trx) pair
      lObj.movieAdd(mov,trx,'offerMacroization',false);
      iMov = numel(lObj.movieFilesAllFull);        
    end
  else
    % no action; iMov is set
  end
else
  % mov is not in proj
  lObj.movieAdd(mov,trx,'offerMacroization',false);
  iMov = numel(lObj.movieFilesAllFull);
end
lObj.movieSet(iMov);
assert(strcmp(lObj.movieFilesAllFull{lObj.currMovie},mov));

% filter/massage trackArgs
trackArgs = trackArgs(:);

i = find(strcmpi(trackArgs,'rawtrkname'));
assert(isempty(i) || isscalar(i));
trkFilenameArgs = trackArgs(i:i+1);
trackArgs(i:i+1,:) = [];

i = find(strcmpi(trackArgs,'startFrame'));
assert(isempty(i) || isscalar(i));
startArgs = trackArgs(i:i+1);
trackArgs(i:i+1,:) = [];
if numel(startArgs)==2 && ischar(startArgs{2})
  startArgs{2} = str2double(startArgs{2});
end  
i = find(strcmpi(trackArgs,'endFrame'));
assert(isempty(i) || isscalar(i));
endArgs = trackArgs(i:i+1);
trackArgs(i:i+1,:) = [];
if numel(endArgs)==2 && ischar(endArgs{2})
  endArgs{2} = str2double(endArgs{2});
end
i = find(strcmp(trackArgs,'storeFullTracking'));
assert(isempty(i) || isscalar(i));
storeFullTrackingArgs = trackArgs(i:i+1);
trackArgs(i:i+1,:) = [];
if ~isempty(storeFullTrackingArgs),
  switch lower(storeFullTrackingArgs{2}),
    case 'none',
      lObj.tracker.storeFullTracking = StoreFullTrackingType.NONE;
    case 'finaliter',
      lObj.tracker.storeFullTracking = StoreFullTrackingType.FINALITER;
    case 'alliters',
      lObj.tracker.storeFullTracking = StoreFullTrackingType.ALLITERS;
    otherwise
      warning('Unknown storeFullTracking type %s, using default %s',storeFullTrackingArgs{2},lObj.tracker.storeFullTracking);
  end
else
  fprintf('Using default storeFullTracking type %s.\n',lObj.tracker.storeFullTracking);
end
i = find(strcmp(trackArgs,'nReps'));
assert(isempty(i) || isscalar(i));
if ~isempty(i),
  forceNReps = trackArgs{i+1};
  trackArgs(i:i+1,:) = [];
  lObj.tracker.setNTestReps(forceNReps);
end
i = find(strcmp(trackArgs,'nIters'));
assert(isempty(i) || isscalar(i));
if ~isempty(i),
  forceNIters = trackArgs{i+1};
  trackArgs(i:i+1,:) = [];
  lObj.tracker.setNIters(forceNIters);
end
tfStartEnd = numel(startArgs)==2 && numel(endArgs)==2;
if tfStartEnd
  frms = startArgs{2}:endArgs{2};
  tm = MFTSet(MovieIndexSetVariable.CurrMov,FrameSetFixed(frms),...
    FrameDecimationFixed.EveryFrame,TargetSetVariable.AllTgts);
else
  tm = MFTSetEnum.CurrMovAllTgtsEveryFrame;
end
fprintf('Tracking preprocessing time: %f\n',toc(startTime)); startTime = tic;
lObj.trackAndExport(tm,'trackArgs',trackArgs,trkFilenameArgs{:});
if ~isClean,
  lObj.cleanUpProjTempDir();
end
fprintf('Time to track, total: %f\n',toc(startTime)); startTime = tic;

