function lObj = createProject(baseProj,movFiles,varargin)
% Create/Augment APT project from base project and list of movies
%
% Note, if your project will be run on multiple platforms, it may be worth
% running VideoTest on movFiles.
%
% baseProj: a "base" project with the correct number of views, points, etc.
%   This could be an empty project with no movies, or a project with existing
%   movies/labels.
% movFiles: [nmov x nview] cellstr of movies to add to baseProj.
%
% 

[tblLbls,calibFiles,cropRois,projname,outfile,diaryfile] = myparse(varargin,...
  'tblLbls',[],... (opt) MFTable with fields MFTable.FLDSCORE, ie .mov, .frm, .iTgt, .tfocc, .p. 
               ...   % * .mov are positive ints, row indices into movFiles
               ...   % * Currently .iTgt must always be 1
               ...   % * tblLbls.tfocc should be logical of size [nmovxnLabelPoints]
               ...   % * tblLbls.p should have size [nmovxnLabelPoints*2].
               ...   %   The raster order is (fastest first): 
               ...   %     {physical pt,view,coordinate (x vs y)} 
  'calibFiles',[],... % (opt) [nx1] cellstr of calibration files for each movie
  'cropRois',[],... (opt) [nx4xnview] crop rois for movies
  'projname','',... % (opt) char, projectname
  'outfile','',...   % (opt) output file where new proj will be saved
  'diaryfile',''... % (opt) diary file
  );

lObj = Labeler();
lObj.projLoad(baseProj);
lObj.projname = projname;

assert(iscellstr(movFiles));
[nmov,nview] = size(movFiles);
if lObj.nview~=nview
  error('Number of views in base project (%d) does not match specified ''movFiles''.',...
    lObj.nview);
end

tfTblLbls = ~isempty(tblLbls);
tfCalibFiles = ~isempty(calibFiles);
tfCropRois = ~isempty(cropRois);
tfDiaryFile = ~isempty(diaryfile);
if tfTblLbls
  tblfldsassert(tblLbls,MFTable.FLDSCORE);
end
if tfCalibFiles
  if ~(iscellstr(calibFiles) && numel(calibFiles)==nmov)
    error('''calibFiles'' must be a cellstr with one element for each movieset.');
  end
end
if tfCropRois
  szassert(cropRois,[nmov 4 nview]);  
end
if tfDiaryFile
  diary(diaryfile);
  fprintf('Diary started at ''%s''.\n',diaryfile);
  oc = onCleanup(@()diary('off'));
end

% nphyspts = lObj.nPhysPoints;
% npts = lObj.nLabelPoints;

for imov=1:nmov  
  if nview==1
    lObj.movieAdd(movFiles{imov},[]);
  else
    lObj.movieSetAdd(movFiles(imov,:));
  end
  lObj.movieSet(lObj.nmovies);
  pause(1); % prob unnec, give UI a little time
  assert(imov==lObj.currMovie);

  nfrm = lObj.nframes;
  fprintf(1,'mov %d. %d frms.\n',imov,nfrm);
  
  if tfTblLbls
    tfMov = tblLbls.mov==imov;
    tblLblsThis = tblLbls(tfMov,:);
    tblLblsThis(:,{'mov'}) = [];
    lObj.labelPosBulkImportTbl(tblLblsThis);
    fprintf(1,' ... imported %d lbled rows.\n',height(tblLblsThis));
  end  
  if tfCalibFiles
    crFile = calibFiles{imov};
    crFile = strtrim(crFile);
    
    if isempty(crFile)
      fprintf(1,' ... no calibfile.\n');
    elseif exist(crFile,'file')==0
      fprintf(1,' ... calibfile DNE: %s.\n',crFile);
    else
      warnst = warning('off','MATLAB:load:variableNotFound'); % sh specific
      crObj = CalRig.loadCreateCalRigObjFromFile(crFile);
      warning(warnst);
      lObj.viewCalSetCurrMovie(crObj);
      fprintf(1,' ... set crObj class ''%s'' from calibfile %s.\n',...
        class(crObj),crFile);
    end
  end
  if tfCropRois
    for ivw=1:nview
      lObj.cropSetNewRoiCurrMov(ivw,cropRois(imov,:,ivw));
    end
    fprintf(1,' ... set crops for %d views.\n',nview);
  end
end

if ~isempty(outfile)
  lObj.projSaveRaw(outfile);
  fprintf(1,'Project ''%s'' saved.\n',outfile);
else
  lObj.projSaveAs();
end
