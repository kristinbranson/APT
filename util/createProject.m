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

[gt,tblLbls,calibFiles,calibObjs,cropRois,projname,outfile,diaryfile] = ...
  myparse(varargin,...
  'gt',false,... % if true, augment GT movies with movFiles (and remainder of opt args)
  'tblLbls',[],... (opt) MFTable with fields MFTable.FLDSCORE, ie .mov, .frm, .iTgt, .tfocc, .p. 
               ...   % * .mov are positive ints, row indices into movFiles
               ...   % * Currently .iTgt must always be 1
               ...   % * tblLbls.tfocc should be logical of size [nmovxnLabelPoints]
               ...   % * tblLbls.p should have size [nmovxnLabelPoints*2].
               ...   %   The raster order is (fastest first): 
               ...   %     {physical pt,view,coordinate (x vs y)} 
  'calibFiles',[],... % (opt) [nx1] cellstr of calibration files for each movie
  'calibObjs',[],... % (opt) [nx1] cell array of CalRig objects. Specify at most one of 'calibFiles' or 'calibObjs'
  'cropRois',[],... (opt) [nx4xnview] crop rois for movies
  'projname','',... % (opt) char, projectname
  'outfile','',...   % (opt) output file where new proj will be saved
  'diaryfile',''... % (opt) diary file
  );

lObj = Labeler();
lObj.projLoad(baseProj);
lObj.projname = projname;  

assert(iscellstr(movFiles));
[nmovadd,nview] = size(movFiles);
if lObj.nview~=nview
  error('Number of views in base project (%d) does not match specified ''movFiles''.',...
    lObj.nview);
end

tfTblLbls = ~isempty(tblLbls);
tfCalibFiles = ~isempty(calibFiles);
tfCalibObjs = ~isempty(calibObjs);
tfCropRois = ~isempty(cropRois);
tfDiaryFile = ~isempty(diaryfile);
if tfTblLbls
  tblfldsassert(tblLbls,MFTable.FLDSCORE);
end
assert(~(tfCalibFiles && tfCalibObjs));
if tfCalibFiles
  if ~(iscellstr(calibFiles) && numel(calibFiles)==nmovadd)
    error('''calibFiles'' must be a cellstr with one element for each movieset.');
  end
end
if tfCalibObjs
  if ~(iscell(calibObjs) && numel(calibObjs)==nmovadd)
    error('''calibObjs'' must be a cell array of CalRig objects with one element for each movieset.');
  end
end
if tfCropRois
  szassert(cropRois,[nmovadd 4 nview]);  
end
if tfDiaryFile
  diary(diaryfile);
  fprintf('Diary started at ''%s''.\n',diaryfile);
  oc = onCleanup(@()diary('off'));
end

if gt~=lObj.gtIsGTMode
  fprintf(1,'Entering gtmode=%d.\n',gt);
  lObj.gtSetGTMode(gt);
end

% nphyspts = lObj.nPhysPoints;
% npts = lObj.nLabelPoints;

for imovadd=1:nmovadd
  if nview==1
    lObj.movieAdd(movFiles{imovadd},[]);
  else
    lObj.movieSetAdd(movFiles(imovadd,:));
  end
  lObj.movieSet(lObj.nmoviesGTaware);
  pause(1); % prob unnec, give UI a little time
  %assert(imov==lObj.currMovie);

  nfrm = lObj.nframes;
  fprintf(1,'movadd %d. %d frms.\n',imovadd,nfrm);
  
  if tfTblLbls
    tfMov = tblLbls.mov==imovadd;
    tblLblsThis = tblLbls(tfMov,:);
    tblLblsThis(:,{'mov'}) = [];
    lObj.labelPosBulkImportTbl(tblLblsThis);
    fprintf(1,' ... imported %d lbled rows.\n',height(tblLblsThis));
  end  
  if tfCalibFiles
    crFile = calibFiles{imovadd};
    crFile = strtrim(crFile);
    
    if isempty(crFile)
      fprintf(1,' ... no calibfile.\n');
    elseif exist(crFile,'file')==0
      fprintf(1,' ... calibfile not found: %s.\n',crFile);
    else
      %warnst = warning('off','MATLAB:load:variableNotFound'); % sh specific
      crObj = CalRig.loadCreateCalRigObjFromFile(crFile);
      %warning(warnst);
      lObj.viewCalSetCurrMovie(crObj);
      fprintf(1,' ... set calibration object class ''%s'' from calibfile %s.\n',...
        class(crObj),crFile);
    end
  end
  if tfCalibObjs
    crObj = calibObjs{imovadd};
    assert(isa(crObj,'CalRig'),'''calibObjs'' must contain CalRig objects.');
    lObj.viewCalSetCurrMovie(crObj);
    fprintf(1,' ... set calibration object class ''%s''.\n',class(crObj));
  end
  if tfCropRois
    for ivw=1:nview
      lObj.cropSetNewRoiCurrMov(ivw,cropRois(imovadd,:,ivw));
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
