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
%     'tableSplitFile',tableSplitFile,'paramFile',paramFile);

startTime = tic;

lblFile = varargin{1};
action = varargin{2};
  
if exist(lblFile,'file')==0
  error('APTCluster:file','Cannot find project file: ''%s''.',lblFile);
end

lObj = Labeler();
set(lObj.hFig,'Visible','off');
fprintf('APTCluster: ''%s'' on project ''%s''.\n',action,lblFile);
fprintf('Time to start labeler: %f\n',toc(startTime));
startTime = tic;
switch action
  case 'retrain'
    lObj.projLoad(lblFile);
    lObj.trackRetrain();
    [lblP,lblF,lblE] = fileparts(lblFile);
    outfile = fullfile(lblP,[lblF '_retrain' datestr(now,'yyyymmddTHHMMSS') lblE]);
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
  case 'trackbatch'
    lObj.projLoad(lblFile,'nomovie',true);
    fprintf('Time to load project: %f\n',toc(startTime)); startTime = tic;
    movfile = varargin{3};
    if exist(movfile,'file')==0
      error('APTCluster:file','Cannot find batch movie file ''%s''.',movfile);
    end
    movs = importdata(movfile);
    if ~iscellstr(movs) 
      error('APTCluster:movfile','Error reading batch movie file ''%s''.',movfile);
    end
    nmov = numel(movs);
    fprintf('Time to read movie info: %f\n',toc(startTime)); startTime = tic;
    for iMov = 1:nmov
      lclTrackAndExportSingleMov(lObj,movs{iMov},'',{});
    end
  case 'xv'
    varargin = varargin(3:end);
    [tableFile,tableSplitFile,paramFile] = ...
      myparse(varargin,...
      'tableFile','',... % (opt) mat-filename containing an MFTtable for rows to consider in XV
      'tableSplitFile','',... % (opt) mat-filename containing split variables. If specified, tableFile must be specced %      'tableSplitFileVar','',... % (opt) variable name in tableSplitFile. <tableSplitFile>.(tableSplitFielVar) should be a [height(<tableFile>) x nfold] logical where true indicates train and false indicates test
      'paramFile',''... % (opt) mat-filename containing a single param struct that will be fed to lObj.trackSetParams
      );
    
    lObj.projLoad(lblFile);

    tfTable = ~isempty(tableFile);
    tfSplit = ~isempty(tableSplitFile);
    tfParam = ~isempty(paramFile);
    xvArgs = cell(1,0);
    [lblP,lblF,lblE] = fileparts(lblFile);
    outfileBase = 'xv';
    if tfTable
      [~,tableFileS,~] = fileparts(tableFile);
      tblMFT = MFTable.loadTableFromMatfile(tableFile);
      fprintf(1,'Loaded table (%d rows) from %s.\n',height(tblMFT),tableFile);
      xvArgs = [xvArgs {'tblMFgt' tblMFT}];
      outfileBase = [outfileBase '_' tableFileS];
    end
    if tfSplit
      assert(tfTable);
      [~,tableSplitFileS,~] = fileparts(tableSplitFile);
      split = loadSingleVariableMatfile(tableSplitFile);      
      if ~(islogical(split) && ismatrix(split) && size(split,1)==height(tblMFT))
        error('Expected split definition to be a logical matrix with %d rows.\n',...
          height(tblMFT));
      end
      kfold = size(split,2);
      fprintf(1,'Loaded split (%d fold) from %s.\n',kfold,tableSplitFile);
      xvArgs = [xvArgs {'kfold' kfold 'partTrn' split}];
      outfileBase = [outfileBase '_' tableSplitFileS];
    end
    if tfParam
      [~,paramFileS,~] = fileparts(paramFile);
      sPrm = loadSingleVariableMatfile(paramFile);
      fprintf(1,'Loaded parameters from %s.\n',paramFile);
      lObj.trackSetParams(sPrm);
      outfileBase = [outfileBase '_' paramFileS];
    end
    outfileBase = [outfileBase '_' datestr(now,'yyyymmddTHHMMSS')];
    
    lObj.trackCrossValidate(xvArgs{:});
    
    savestuff = struct();
    savestuff.sPrm = lObj.trackGetParams();
    savestuff.xvArgs = xvArgs;
    savestuff.xvRes = lObj.xvResults;
    savestuff.xvResTS = lObj.xvResultsTS; %#ok<STRNU>
    outfile = fullfile(lblP,[outfileBase '.mat']);    
    fprintf('APTCluster: saving xv results: %s\n',outfile);
    save(outfile,'-mat','-struct','savestuff');
  otherwise
    error('APTCluster:action','Unrecognized action ''%s''.',action);
end

fprintf('Real processing done, total time: %f\n',toc(startTime)); startTime = tic;

delete(lObj);

fprintf('Time to close APT: %f\n',toc(startTime)); startTime = tic;

close all force;
fprintf('Time to close everything else: %f\n',toc(startTime)); startTime = tic;
fprintf('APTCluster finished.\n');

function lclTrackAndExportSingleMov(lObj,mov,trx,trackArgs)
% Trx: optional, specify '' for no-trx

startTime = tic;

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
      lObj.movieAdd(mov,trx);
      iMov = numel(lObj.movieFilesAllFull);        
    end
  else
    % no action; iMov is set
  end
else
  % mov is not in proj
  lObj.movieAdd(mov,trx);
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
fprintf('Time to track, total: %f\n',toc(startTime)); startTime = tic;

