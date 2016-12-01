function APTCluster(varargin)
% APTCluster(varargin)
%
% % Do a full retrain of a project's tracker
% APTCluster(lblFile,'retrain')
%
% % Track a single movie
% APTCluster(lblFile,'track',moviefullpath)
% % Options passed to Labeler.trackAndExport, 'trackArgs'
% APTCluster(lblFile,'track',moviefullpath,varargin)
%
% % Prune legs of a trkfile, saving results alongside trkfile
% APTCluster(0,'prunejan',trkfullpath,sigd,leg)

lblFile = varargin{1};
action = varargin{2};

if isequal(lblFile,'0') || isequal(lblFile,0)
  % NON-LBLFILE ACTIONS
  
  switch action
    case 'prunejan'
      trkfile = varargin{3};
      if exist(trkfile,'file')==0
        error('APTCluster:file','Cannot find file: ''%s''.',trkfile);
      end
      sigd = varargin{4};
      if ~isnumeric(sigd)
        sigd = str2double(sigd);
      end
      ipt = varargin{5};
      if ~isnumeric(ipt)
        ipt = str2double(ipt);
      end
      
      assert(~verLessThan('matlab','R2016a'),'''prunejan'' requires MATLAB R2016a or later.');
      
      trk = load(trkfile,'-mat');
      trkPFull = trk.pTrkFull;
      [npttrk,d,nRep,nfrm] = size(trkPFull);
      trkD = npttrk*d;
      trkPFull = reshape(trkPFull,[trkD nRep nfrm]);
      trkPFull = permute(trkPFull,[3 2 1]);
      trkPiPt = trk.pTrkiPt;
      assert(numel(trkPiPt)==npttrk);

      IMNR = 256;
      IMNC = 256;
      %LEGS = 4:7;
      %NLEG = numel(LEGS);
      NLEG = 1;
      pLegsPruned = nan(NLEG,2,nfrm);
      pLegsPrunedAbs = nan(NLEG,2,nfrm);
      %for iLeg = 1:NLEG
        %ipt = LEGS(iLeg);
        pObj = CPRPrune(IMNR,IMNC,sigd);
        pObj.init(trkPFull,trkPiPt,ipt,1,nfrm);
        pObj.run();
        pLegsPruned(1,:,:) = pObj.prnTrk';
        pLegsPrunedAbs(1,:,:) = pObj.prnTrkAbs';
      %end
      trkPruned = TrkFile(pLegsPruned,'pTrkiPt',ipt);
      trkPrunedAbs = TrkFile(pLegsPrunedAbs,'pTrkiPt',ipt);
      
      [trkfileP,trkfileS] = fileparts(trkfile);
      filebase = sprintf('_prune%d.trk',ipt);
      filebaseAbs = sprintf('_pruneAbs%d.trk',ipt);
      trkfilePruned = fullfile(trkfileP,[trkfileS filebase]);
      trkfilePrunedAbs = fullfile(trkfileP,[trkfileS filebaseAbs]);
      
      fprintf(1,'Saving: %s...\n',trkfilePruned);
      trkPruned.save(trkfilePruned);
      fprintf(1,'Saving: %s...\n',trkfilePrunedAbs);
      trkPrunedAbs.save(trkfilePrunedAbs);            
    otherwise
      assert(false,'Unrecognized action.');
  end
  
  return;
end
 
if exist(lblFile,'file')==0
  error('APTCluster:file','Cannot find project file: ''%s''.',lblFile);
end

lObj = Labeler();
lObj.projLoad(lblFile);

switch action
  case 'retrain'
    fprintf('APTCluster: ''%s'' on project ''%s''.\n',action,lblFile);

    lObj.trackRetrain();
    [p,f,e] = fileparts(lblFile);
    outfile = fullfile(p,[f '_retrain' datestr(now,'yyyymmddTHHMMSS') e]);
    fprintf('APTCluster: saving retrained project: %s\n',outfile);
    lObj.projSaveRaw(outfile);
  case 'track'
    mov = varargin{3};
    trackArgs = varargin(4:end);
    lclTrackAndExportSingleMov(lObj,mov,trackArgs);    
  case 'trackbatch'
    movfile = varargin{3};
    if exist(movfile,'file')==0
      error('APTCluster:file','Cannot find batch movie file ''%s''.',movfile);
    end
    movs = importdata(movfile);
    if ~iscellstr(movs)
      error('APTCluster:movfile','Error reading batch movie file ''%s''.',movfile);
    end
    nmov = numel(movs);
    for iMov = 1:nmov
      lclTrackAndExportSingleMov(lObj,movs{iMov},{});
    end
  otherwise
    error('APTCluster:action','Unrecognized action ''%s''.',action);
end

delete(lObj);
close all force;


function lclTrackAndExportSingleMov(lObj,mov,trackArgs)
if exist(mov,'file')==0
  error('APTCluster:file','Cannot find movie file ''%s''.',mov);
end
[tf,iMov] = ismember(mov,lObj.movieFilesAllFull);
if tf
  % none
else
  lObj.movieAdd(mov,'');
  iMov = numel(lObj.movieFilesAllFull);
end
lObj.movieSet(iMov);
assert(strcmp(lObj.movieFilesAllFull{lObj.currMovie},mov));

% filter/massage trackArgs
trackArgs = trackArgs(:);

i = find(strcmpi(trackArgs,'trkFilename'));
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

i = find(strcmpi(trackArgs,'stripTrkPFull'));
assert(isempty(i) || isscalar(i));
if isscalar(i) && ischar(trackArgs{i+1})
  trackArgs{i+1} = str2double(trackArgs{i+1});
end

tfStartEnd = numel(startArgs)==2 && numel(endArgs)==2;
if tfStartEnd
  tm = TrackMode.CurrMovCustomFrames;
  tm.info = startArgs{2}:endArgs{2};
else
  tm = TrackMode.CurrMovEveryFrame;
end
lObj.trackAndExport(tm,'trackArgs',trackArgs,trkFilenameArgs{:});