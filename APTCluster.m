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
% APTCluster(0,'prunejan',trkfullpath,sigd)

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
      
      assert(~verLessThan('matlab','R2016a'),'''prunejan'' requires MATLAB R2016b or later.');
      
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
      LEGS = 4:7;      
      NLEG = numel(LEGS);
      pLegsPruned = nan(NLEG,2,nfrm);
      pLegsPrunedAbs = nan(NLEG,2,nfrm);
      for iLeg = 1:NLEG
        ipt = LEGS(iLeg);
        pObj = CPRPrune(IMNR,IMNC,sigd);
        pObj.init(trkPFull,trkPiPt,ipt,1,nfrm);
        pObj.run();
        pLegsPruned(iLeg,:,:) = pObj.prnTrk';
        pLegsPrunedAbs(iLeg,:,:) = pObj.prnTrkAbs';
      end
      trkPruned = TrkFile(pLegsPruned,'pTrkiPt',LEGS);
      trkPrunedAbs = TrkFile(pLegsPrunedAbs,'pTrkiPt',LEGS);
      
      [trkfileP,trkfileS] = fileparts(trkfile);
      trkfilePruned = fullfile(trkfileP,[trkfileS '_legsPruned.trk']);
      trkfilePrunedAbs = fullfile(trkfileP,[trkfileS '_legsPrunedAbs.trk']);
      
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

% filter trackArgs
i = find(strcmpi(trackArgs,'trkFilename'));
assert(isempty(i) || isscalar(i));
trackArgs = trackArgs(:);
trkFilenameArgs = trackArgs(i:i+1);
trackArgs(i:i+1,:) = [];

tm = TrackMode.CurrMovEveryFrame;
lObj.trackAndExport(tm,'trackArgs',trackArgs,trkFilenameArgs{:});