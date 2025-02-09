function dispatchMainFigureCallback(callbackName, varargin)
feval(callbackName, varargin{:}) ;


function pumTrack_Callback(src,evt,handles)
labeler = handles.labeler;
labeler.trackModeIdx = src.Value;



function slider_frame_Callback(hObject,evt,varargin)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

debugtiming = false;
if debugtiming,
  starttime = tic() ;
end

handles = guidata(hObject);
labeler = handles.labeler;

if ~labeler.hasProject
  set(hObject,'Value',0);  
  return;
end
if ~labeler.hasMovie
  set(hObject,'Value',0);  
  msgbox('There is no movie open.');
  return;
end

v = get(hObject,'Value');
f = round(1 + v * (labeler.nframes - 1));

cmod = handles.figure.CurrentModifier;
if ~isempty(cmod) && any(strcmp(cmod{1},{'control' 'shift'}))
  if f>labeler.currFrame
    tfSetOccurred = labeler.frameUp(true);
  else
    tfSetOccurred = labeler.frameDown(true);
  end
else
  tfSetOccurred = labeler.setFrameProtected(f);
end
  
if ~tfSetOccurred
  sldval = (labeler.currFrame-1)/(labeler.nframes-1);
  if isnan(sldval)
    sldval = 0;
  end
  set(hObject,'Value',sldval);
end

if debugtiming,
  fprintf('Slider callback setting to frame %d took %f seconds\n',f,toc(starttime));
end


function edit_frame_Callback(hObject,~,handles)
if ~handles.labeler.doProjectAndMovieExist()
  return;
end

labeler = handles.labeler;

f = str2double(get(hObject,'String'));
if isnan(f)
  set(hObject,'String',num2str(labeler.currFrame));
  return;
end
f = min(max(1,round(f)),labeler.nframes);
if ~labeler.trxCheckFramesLive(f)
  set(hObject,'String',num2str(labeler.currFrame));
  warnstr = sprintf('Frame %d is out-of-range for current target.',f);
  warndlg(warnstr,'Out of range');
  return;
end
set(hObject,'String',num2str(f));
if f ~= labeler.currFrame
  labeler.setFrame(f)
end 
  


function pbClear_Callback(hObject, eventdata, handles)
if ~handles.labeler.doProjectAndMovieExist()
  return;
end
handles.labeler.lblCore.clearLabels();
handles.labeler.CheckPrevAxesTemplate();


function tbAccept_Callback(hObject, eventdata, handles)
% debugtiming = false;
% if debugtiming,
%   starttime = tic;
% end

if ~handles.labeler.doProjectAndMovieExist()
  return;
end
lc = handles.labeler.lblCore;
switch lc.state
  case LabelState.ADJUST
    lc.acceptLabels();
    %handles.labeler.InitializePrevAxesTemplate();
  case LabelState.ACCEPTED
    lc.unAcceptLabels();    
    %handles.labeler.CheckPrevAxesTemplate();
  otherwise
    assert(false);
end

% if debugtiming,
%   fprintf('toggleAccept took %f seconds\n',toc(starttime));
% end

function tblTrx_cell_selected_(src,evt) %#ok<*DEFNU>
% Current/last row selection is maintained in hObject.UserData

handles = guidata(src.Parent);
labeler = handles.labeler;
if ~(labeler.hasTrx || labeler.maIsMA)
  return;
end

rows = evt.Indices;
rows = rows(:,1); % AL20210514: rows is nx2; columns are {rowidxs,colidxs} at least in 2020b
%rowsprev = src.UserData;
src.UserData = rows;
dat = get(src,'Data');

if isscalar(rows)
  idx = dat{rows(1),1};
  labeler.setTarget(idx);
  %labeler.labelsOtherTargetHideAll();
else
  % 20210514 Skipping this for now; possible performance hit
  
  % addon to existing selection
  %rowsnew = setdiff(rows,rowsprev);  
  %idxsnew = cell2mat(dat(rowsnew,1));
  %labeler.labelsOtherTargetShowIdxs(idxsnew);
end

hlpRemoveFocus(src,handles);



% 20170428
% Notes -- Zooms Views Angles et al
% 
% Zoom.
% In APT we refer to the "zoom" as effective magnification determined by 
% the axis limits, ie how many pixels are shown along x and y. Currently
% the pixels and axis are always square.
% 
% The zoom level can be adjusted in a variety of ways: via the zoom slider,
% the Unzoom button, the manual zoom tools in the toolbar, or 
% View > Zoom out. 
%
% Camroll.
% When Trx are available, the movie can be rotated so that the Trx are
% always at a given orientation (currently, "up"). This is achieved by
% "camrolling" the axes, ie setting axes.CameraUpVector. Currently
% manually camrolling is not available.
%
% CamViewAngle.
% The CameraViewAngle is the AOV of the 'camera' viewing the axes. When
% "camroll" is off (either there are no Trx, or rotateSoTargetIsUp is
% off), axis.CameraViewAngleMode is set to 'auto' and MATLAB selects a
% CameraViewAngle so that the axis fills its outerposition. When camroll is
% on, MATLAB by default chooses a CameraViewAngle that is relatively wide, 
% so that the square axes is very visible as it rotates around. This is a
% bit distracting so currently we choose a smaller CamViewAngle (very 
% arbitrarily). There may be a better way to handle this.

function sldZoom_Callback(hObject, eventdata, ~)
% log(zoomrad) = logzoomradmax + sldval*(logzoomradmin-logzoomradmax)
handles = guidata(hObject);

if ~handles.labeler.doProjectAndMovieExist()
  return;
end

labeler = handles.labeler;
v = hObject.Value;
userdata = hObject.UserData;
logzoomrad = userdata(2)+v*(userdata(1)-userdata(2));
zoomRad = exp(logzoomrad);
handles.controller.videoZoom(zoomRad);
hlpRemoveFocus(hObject,handles);

function pbResetZoom_Callback(hObject, eventdata, handles)
hAxs = handles.axes_all;
hIms = handles.images_all;
assert(numel(hAxs)==numel(hIms));
arrayfun(@zoomOutFullView,hAxs,hIms,false(1,numel(hIms)));

function pbSetZoom_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
labeler.targetZoomRadiusDefault = diff(handles.axes_curr.XLim)/2;

function pbRecallZoom_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
% TODO this is broken!!
handles.controller.videoCenterOnCurrTarget();
handles.controller.videoZoom(labeler.targetZoomRadiusDefault);

% function tblSusp_CellSelectionCallback(hObject, eventdata, handles)
% labeler = handles.labeler;
% if verLessThan('matlab','R2015b')
%   jt = labeler.gdata.tblSusp.UserData.jtable;
%   row = jt.getSelectedRow; % 0 based
%   frm = jt.getValueAt(row,0);
%   iTgt = jt.getValueAt(row,1);
%   if ~isempty(frm)
%     frm = frm.longValueReal;
%     iTgt = iTgt.longValueReal;
%     labeler.setFrameAndTarget(frm,iTgt);
%     hlpRemoveFocus(hObject,handles);
%   end
% else
%   row = eventdata.Indices(1);
%   dat = hObject.Data;
%   frm = dat(row,1);
%   iTgt = dat(row,2);
%   labeler.setFrameAndTarget(frm,iTgt);
%   hlpRemoveFocus(hObject,handles);
% end

function tbTLSelectMode_Callback(hObject, eventdata, handles)
if ~handles.labeler.doProjectAndMovieExist()
  return;
end
tl = handles.labelTLInfo;
tl.selectOn = hObject.Value;

function pbClearSelection_Callback(hObject, eventdata, handles)
if ~handles.labeler.doProjectAndMovieExist()
  return;
end
tl = handles.labelTLInfo;
tl.selectClearSelection();

% function cbkFreezePrevAxesToMainWindow(src,evt)
% handles = guidata(src);
% handles.labeler.setPrevAxesMode(PrevAxesMode.FROZEN);

% function cbkUnfreezePrevAxes(src,evt)
% handles = guidata(src);
% handles.labeler.setPrevAxesMode(PrevAxesMode.LASTSEEN);

%% menu
function menu_file_save_Callback(hObject, eventdata, handles)
labeler = handles.labeler ;
labeler.setStatus('Saving project...');
handles.labeler.projSaveSmart();
handles.labeler.projAssignProjNameFromProjFileIfAppropriate();
handles.labeler.clearStatus()

function menu_file_saveas_Callback(hObject, eventdata, handles)
handles.labeler.setStatus('Saving project...');
handles.labeler.projSaveAs();
handles.labeler.projAssignProjNameFromProjFileIfAppropriate();
handles.labeler.clearStatus()

function menu_file_load_Callback(hObject, eventdata, handles)

labeler = handles.labeler;
controller = handles.controller ;
labeler.setStatus('Loading Project...') ;
if controller.raiseUnsavedChangesDialogIfNeeded() ,
  currMovInfo = labeler.projLoad();
  if ~isempty(currMovInfo)
    controller.movieManagerController_.setVisible(true);
    wstr = ...
      sprintf(strcatg('Could not find file for movie(set) %d: %s.\n\nProject opened with no movie selected. ', ...
                      'Double-click a row in the MovieManager or use the ''Switch to Movie'' button to start working on a movie.'), ...
              currMovInfo.iMov, ...
              currMovInfo.badfile);
    warndlg(wstr,'Movie not found','modal');
  end
end
labeler.clearStatus()

% function tfcontinue = hlpSave(labelerObj)
% tfcontinue = true;
% 
% if ~verLessThan('matlab','9.6') && batchStartupOptionUsed
%   return;
% end
% 
% OPTION_SAVE = 'Save first';
% OPTION_PROC = 'Proceed without saving';
% OPTION_CANC = 'Cancel';
% if labelerObj.doesNeedSave ,
%   res = questdlg('You have unsaved changes to your project. If you proceed without saving, your changes will be lost.',...
%     'Unsaved changes',OPTION_SAVE,OPTION_PROC,OPTION_CANC,OPTION_SAVE);
%   switch res
%     case OPTION_SAVE
%       labelerObj.projSaveSmart();
%       labelerObj.projAssignProjNameFromProjFileIfAppropriate();
%     case OPTION_CANC
%       tfcontinue = false;
%     case OPTION_PROC
%       % none
%   end
% end



function menu_file_managemovies_Callback(src, evt, handles)  %#ok<INUSD>
if ~isempty(handles.controller.movieManagerController_) && isvalid(handles.controller.movieManagerController_) ,
  handles.controller.movieManagerController_.setVisible(true);
else
  handles.labeler.lerror('LabelerGUI:movieManagerController','Please create or load a project.');
end



function menu_file_import_labels_trk_curr_mov_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
if ~labeler.hasMovie
  handles.labeler.lerror('LabelerGUI:noMovie','No movie is loaded.');
end
labeler.gtThrowErrIfInGTMode();
iMov = labeler.currMovie;
haslbls1 = labeler.labelPosMovieHasLabels(iMov); % TODO: method should be unnec
haslbls2 = labeler.movieFilesAllHaveLbls(iMov)>0;
assert(haslbls1==haslbls2);
if haslbls1
  resp = questdlg('Current movie has labels that will be overwritten. OK?',...
    'Import Labels','OK, Proceed','Cancel','Cancel');
  if isempty(resp)
    resp = 'Cancel';
  end
  switch resp
    case 'OK, Proceed'
      % none
    case 'Cancel'
      return;
    otherwise
      assert(false); 
  end
end
handles.labeler.labelImportTrkPromptGenericSimple(iMov,...
  'labelImportTrk','gtok',false);

function menu_file_import_labels2_trk_curr_mov_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
if ~labeler.hasMovie
  handles.labeler.lerror('LabelerGUI:noMovie','No movie is loaded.');
end
iMov = labeler.currMovie; % gt-aware
handles.labeler.setStatus('Importing tracking results...');
labeler.labelImportTrkPromptGenericSimple(iMov,'labels2ImportTrk','gtok',true);
handles.labeler.clearStatus();

function menu_file_export_labels_trks_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
[tfok,rawtrkname] = handles.controller.getExportTrkRawNameUI('labels',true);
if ~tfok
  return;
end
handles.labeler.setStatus('Exporting tracking results...');
labeler.labelExportTrk(1:labeler.nmoviesGTaware,'rawtrkname',rawtrkname);
handles.labeler.clearStatus();

function menu_file_export_labels_table_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
fname = labeler.getDefaultFilenameExportLabelTable();
[f,p] = uiputfile(fname,'Export File');
if isequal(f,0)
  return;
end
fname = fullfile(p,f);  
VARNAME = 'tblLbls';
s = struct();
s.(VARNAME) = labeler.labelGetMFTableLabeled('useMovNames',true); 
save(fname,'-mat','-struct','s');
fprintf('Saved table ''%s'' to file ''%s''.\n',VARNAME,fname);

function menu_file_import_labels_table_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
lastFile = RC.getprop('lastLabelMatfile');
if isempty(lastFile)
  lastFile = pwd;
end
[fname,pth] = uigetfile('*.mat','Load Labels',lastFile);
if isequal(fname,0)
  return;
end
fname = fullfile(pth,fname);
t = loadSingleVariableMatfile(fname);
labeler.labelPosBulkImportTbl(t);
fprintf('Loaded %d labeled frames from file ''%s''.\n',height(t),fname);

function menu_file_export_stripped_lbl_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
fname = labeler.getDefaultFilenameExportStrippedLbl();
[f,p] = uiputfile(fname,'Export File');
if isequal(f,0)
  return
end
fname = fullfile(p,f);
handles.labeler.setStatus(sprintf('Exporting training data to %s',fname));
labeler.projExportTrainData(fname)
fprintf('Saved training data to file ''%s''.\n',fname);
handles.labeler.clearStatus();

function menu_file_crop_mode_Callback(hObject,evtdata,handles)

labeler = handles.labeler;

if ~isempty(labeler.tracker) && ~labeler.gtIsGTMode && labeler.labelPosMovieHasLabels(labeler.currMovie),
  res = questdlg('Frames of the current movie are labeled. Editing the crop region for this movie will cause trackers to be reset. Continue?');
  if ~strcmpi(res,'Yes'),
    return;
  end
end

handles.labeler.setStatus('Switching crop mode...');
labeler.cropSetCropMode(~labeler.cropIsCropMode);
handles.labeler.clearStatus();

function menu_file_clean_tempdir_Callback(hObject,evtdata,handles)

handles.labeler.setStatus('Deleting temp directories...');
handles.labeler.projRemoveOtherTempDirs();
handles.labeler.clearStatus();

function menu_file_bundle_tempdir_Callback(hObject,evtdata,handles)
handles.labeler.setStatus('Bundling the temp directory...');
handles.labeler.projBundleTempDir();
handles.labeler.clearStatus();


function menu_help_Callback(hObject, eventdata, handles)

function menu_help_labeling_actions_Callback(hObject, eventdata, handles)
lblCore = handles.labeler.lblCore;
if isempty(lblCore)
  h = 'Please open a movie first.';
else
  h = lblCore.getLabelingHelp();
end
msgbox(h,'Labeling Actions','help',struct('Interpreter','tex','WindowStyle','replace'));

function menu_help_about_Callback(hObject, eventdata, handles)
about(handles.labeler);

function menu_setup_sequential_mode_Callback(hObject,eventdata,handles)
handles.controller.menuSetupLabelModeCbkGeneric(hObject);

function menu_setup_sequential_add_mode_Callback(hObject,eventdata,handles)
handles.controller.menuSetupLabelModeCbkGeneric(hObject);

function menu_setup_template_mode_Callback(hObject,eventdata,handles)
handles.controller.menuSetupLabelModeCbkGeneric(hObject);

function menu_setup_highthroughput_mode_Callback(hObject,eventdata,handles)
handles.controller.menuSetupLabelModeCbkGeneric(hObject);

function menu_setup_multiview_calibrated_mode_2_Callback(hObject,eventdata,handles)
handles.controller.menuSetupLabelModeCbkGeneric(hObject);

function menu_setup_multianimal_mode_Callback(hObject,eventdata,handles)
handles.controller.menuSetupLabelModeCbkGeneric(hObject);

function menu_setup_label_overlay_montage_Callback(hObject,evtdata,handles)
handles.labeler.setStatus('Plotting all labels on one axes to visualize label distribution...');
labeler = handles.labeler;
if labeler.hasTrx
  labeler.labelOverlayMontage();
  labeler.labelOverlayMontage('ctrMeth','trx');
  labeler.labelOverlayMontage('ctrMeth','trx','rotAlignMeth','trxtheta');
  % could also use headtail for centering/alignment but skip for now.  
else % labeler.maIsMA, or SA-non-trx
  labeler.labelOverlayMontage();
  if ~labeler.isMultiView
    labeler.labelOverlayMontage('ctrMeth','centroid');
    tfHTdefined = ~isempty(labeler.skelHead) && ~isempty(labeler.skelTail);
    if tfHTdefined  
      labeler.labelOverlayMontage('ctrMeth','centroid','rotAlignMeth','headtail');
    else
      warningNoTrace('For aligned overlays, define head/tail points in Track>Landmark Paraneters.');
    end
  end
end
handles.labeler.clearStatus();

% function menu_setup_label_overlay_montage_trx_centered_Callback(hObject,evtdata,handles)
% 
% handles.labeler.setStatus('Plotting all labels on one axes to visualize label distribution...');
% labeler = handles.labeler;
% hFig(1) = labeler.labelOverlayMontage('ctrMeth','trx','rotAlignMeth','none'); 
% try
%   hFig(2) = labeler.labelOverlayMontage('ctrMeth','trx',...
%     'rotAlignMeth','headtail','hFig0',hFig(1)); 
% catch ME
%   warningNoTrace('Could not create head-tail aligned montage: %s',ME.message);
%   hFig(2) = figurecascaded(hFig(1));
% end
% hFig(3) = labeler.labelOverlayMontage('ctrMeth','trx',...
%   'rotAlignMeth','trxtheta','hFig0',hFig(2)); %#ok<NASGU>
% handles.labeler.clearStatus();

function menu_setup_label_outliers_Callback(hObject,evtdata,handles)
handles.labeler.setStatus('Finding outliers in labels...');
labeler = handles.labeler;
label_outlier_gui(labeler);
handles.labeler.clearStatus();

function menu_setup_set_nframe_skip_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
lc = labeler.lblCore;
assert(isa(lc,'LabelCoreHT'));
nfs = lc.nFrameSkip;
ret = inputdlg('Select labeling frame increment','Set increment',1,{num2str(nfs)});
if isempty(ret)
  return;
end
val = str2double(ret{1});
lc.nFrameSkip = val;
labeler.labelPointsPlotInfo.HighThroughputMode.NFrameSkip = val;
% This state is duped between labelCore and lppi b/c the lifetimes are
% different. LabelCore exists only between movies etc, and is initted from
% lppi. Hmm

function menu_setup_streamlined_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
lc = labeler.lblCore;
assert(isa(lc,'LabelCoreMultiViewCalibrated2'));
lc.streamlined = ~lc.streamlined;

function menu_setup_ma_twoclick_align_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
lc = labeler.lblCore;
tftc = ~lc.tcOn;
labeler.isTwoClickAlign = tftc; % store the state
lc.setTwoClickOn(tftc);
hObject.Checked = onIff(tftc); % skip listener business for now

function menu_setup_set_labeling_point_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
ipt = labeler.lblCore.iPoint;
ret = inputdlg('Select labeling point','Point number',1,{num2str(ipt)});
if isempty(ret)
  return;
end
ret = str2double(ret{1});
labeler.lblCore.setIPoint(ret);


function menu_setup_use_calibration_Callback(hObject, eventdata, handles)

labeler = handles.labeler;
lc = labeler.lblCore;
if lc.supportsCalibration,
  lc.toggleShowCalibration();
  hObject.Checked = onIff(lc.showCalibration);
else
  hObject.Checked = 'off';
end

function menu_setup_load_calibration_file_Callback(hObject, eventdata, handles)

lastCalFile = RC.getprop('lastCalibrationFile');
if isempty(lastCalFile)
  lastCalFile = pwd;
end
[fname,pth] = uigetfile('*.mat','Load Calibration File',lastCalFile);
if isequal(fname,0)
  return;
end
fname = fullfile(pth,fname);

crObj = CalRig.loadCreateCalRigObjFromFile(fname);

labeler = handles.labeler;
vcdPW = labeler.viewCalProjWide;
if isempty(vcdPW)
  resp = questdlg('Should calibration apply to i) all movies in project or ii) current movie only?',...
    'Calibration load',...
    'All movies in project',...
    'Current movie only',...
    'Cancel',...
    'All movies in project');
  if isempty(resp)
    resp = 'Cancel';
  end
  switch resp
    case 'All movies in project'
      tfProjWide = true;      
    case 'Current movie only'
      tfProjWide = false;      
    otherwise
      return;
  end
else
  tfProjWide = vcdPW;
end

% Currently there is no UI for altering labeler.viewCalProjWide once it is set

if tfProjWide
  labeler.viewCalSetProjWide(crObj);%,'tfSetViewSizes',tfSetViewSizes);
else
  labeler.viewCalSetCurrMovie(crObj);%,'tfSetViewSizes',tfSetViewSizes);
end


%set_use_calibration(handles,true);
lc = labeler.lblCore;
if lc.supportsCalibration,
  lc.setShowCalibration(true);
end
handles.menu_setup_use_calibration.Checked = onIff(lc.showCalibration);

RC.saveprop('lastCalibrationFile',fname);

% function menu_setup_unlock_all_frames_Callback(hObject, eventdata, handles)
% handles.labeler.labelPosSetAllMarked(false);
% function menu_setup_lock_all_frames_Callback(hObject, eventdata, handles)
% handles.labeler.labelPosSetAllMarked(true);

function menu_view_show_bgsubbed_frames_Callback(hObject,evtdata,handles)
tf = ~strcmp(hObject.Checked,'on');
labeler = handles.labeler;
labeler.movieViewBGsubbed = tf;

function menu_view_adjustbrightness_Callback(hObject, eventdata, handles)
[tfproceed,iAxRead,iAxApply] = hlpAxesAdjustPrompt(handles.controller);
if tfproceed
  try
  	hConstrast = imcontrast_kb(handles.axes_all(iAxRead));
  catch ME
    switch ME.identifier
      case 'images:imcontrast:unsupportedImageType'
        error(ME.identifier,'%s %s',ME.message,'Try View > Convert to grayscale.');
      otherwise
        ME.rethrow();
    end
  end
	addlistener(hConstrast,'ObjectBeingDestroyed',...
		@(s,e) closeImContrast(handles.controller,iAxRead,iAxApply));
end
  
function menu_view_converttograyscale_Callback(hObject, eventdata, handles)
tf = ~strcmp(hObject.Checked,'on');
labeler = handles.labeler;
labeler.movieForceGrayscale = tf;
if labeler.hasMovie
  % Pure convenience: update image for user rather than wait for next 
  % frame-switch. Could also put this in Labeler.set.movieForceGrayscale.
  labeler.setFrame(labeler.currFrame,'tfforcereadmovie',true);
end
function menu_view_gammacorrect_Callback(hObject, eventdata, handles)
[tfok,~,iAxApply] = hlpAxesAdjustPrompt(handles.controller);
if ~tfok
	return;
end
val = inputdlg('Gamma value:','Gamma correction');
if isempty(val)
  return;
end
gamma = str2double(val{1});
ViewConfig.applyGammaCorrection(handles.images_all,handles.axes_all,...
  handles.axes_prev,iAxApply,gamma);
		
function menu_file_quit_Callback(hObject, eventdata, handles)
controller = handles.controller ;
controller.quitRequested() ;
%CloseGUI(handles);

% function cbkShowPredTxtLblChanged(src,evt)
% labeler = evt.AffectedObject;
% handles = labeler.gdata;
% onOff = onIff(~labeler.showPredTxtLbl);
% handles.menu_view_showhide_advanced_hidepredtxtlbls.Checked = onOff;

function menu_view_hide_trajectories_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
labeler.setShowTrx(~labeler.showTrx);

function menu_view_plot_trajectories_current_target_only_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
labeler.setShowTrxCurrTargetOnly(~labeler.showTrxCurrTargetOnly);

function menu_view_trajectories_centervideoontarget_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
labeler.movieCenterOnTarget = ~labeler.movieCenterOnTarget;
function menu_view_rotate_video_target_up_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
labeler.movieRotateTargetUp = ~labeler.movieRotateTargetUp;
function menu_view_flip_flipud_movie_only_Callback(hObject, eventdata, handles)
[tfproceed,~,iAxApply] = hlpAxesAdjustPrompt(handles.controller);
if tfproceed
  labeler = handles.labeler;
  labeler.movieInvert(iAxApply) = ~labeler.movieInvert(iAxApply);
  if labeler.hasMovie
    labeler.setFrame(labeler.currFrame,'tfforcereadmovie',true);
  end
end
function menu_view_flip_flipud_Callback(hObject, eventdata, handles)
[tfproceed,~,iAxApply] = hlpAxesAdjustPrompt(handles.controller);
if tfproceed
  for iAx = iAxApply(:)'
    ax = handles.axes_all(iAx);
    ax.YDir = toggleAxisDir(ax.YDir);
  end
  handles.labeler.UpdatePrevAxesDirections();
end
function menu_view_flip_fliplr_Callback(hObject, eventdata, handles)
[tfproceed,~,iAxApply] = hlpAxesAdjustPrompt(handles.controller);
if tfproceed
  for iAx = iAxApply(:)'
    ax = handles.axes_all(iAx);
    ax.XDir = toggleAxisDir(ax.XDir);
%     if ax==handles.axes_curr
%       ax2 = handles.axes_prev;
%       ax2.XDir = toggleAxisDir(ax2.XDir);
%     end
    handles.labeler.UpdatePrevAxesDirections();
  end
end
function menu_view_show_axes_toolbar_Callback(hObject, eventdata, handles)
ax = handles.axes_curr;
if strcmp(hObject.Checked,'on')
  onoff = 'off';
else
  onoff = 'on';
end
ax.Toolbar.Visible = onoff;
hObject.Checked = onoff;
% For now not listening to ax.Toolbar.Visible for cmdline changes


function menu_view_fit_entire_image_Callback(hObject, eventdata, handles)
hAxs = handles.axes_all;
hIms = handles.images_all;
assert(numel(hAxs)==numel(hIms));
arrayfun(@zoomOutFullView,hAxs,hIms,true(1,numel(hAxs)));
handles.labeler.movieCenterOnTarget = false;


function menu_view_hide_labels_Callback(hObject, eventdata, handles)
lblCore = handles.labeler.lblCore;
if ~isempty(lblCore)
  lblCore.labelsHideToggle();
end

function menu_view_hide_predictions_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
tracker = labeler.tracker;
if ~isempty(tracker)
  tracker.hideVizToggle();
end

function menu_view_show_preds_curr_target_only_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
tracker = labeler.tracker;
if ~isempty(tracker)
  tracker.showPredsCurrTargetOnlyToggle();
end

function menu_view_hide_imported_predictions_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
labeler.labels2VizToggle();



function menu_view_show_imported_preds_curr_target_only_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
labeler.labels2VizSetShowCurrTargetOnly(~labeler.labels2ShowCurrTargetOnly);



% function menu_track_cpr_show_replicates_Callback(hObject, eventdata, handles)
% tObj = handles.labeler.tracker;
% vsr = tObj.showVizReplicates;
% vsrnew = ~vsr;
% sft = tObj.storeFullTracking;
% if vsrnew && sft==StoreFullTrackingType.NONE
%   warningNoTrace('Tracker will store replicates for final CPR iterations.');
%   tObj.storeFullTracking = StoreFullTrackingType.FINALITER;
% end
% tObj.showVizReplicates = vsrnew;

% % when trackerInfo is updated, update the tracker info text in the main APT window
% function cbkTrackerInfoChanged(src,evt)
% 
% tObj = evt.AffectedObject;
% tObj.labeler.gdata.text_trackerinfo.String = tObj.getTrackerInfoString();



function menu_view_show_tick_labels_Callback(hObject, eventdata, handles)
% just use checked state of menu for now, no other state
toggleOnOff(hObject,'Checked');
hlpTickGridBang(handles.axes_all, handles.menu_view_show_tick_labels, handles.menu_view_show_grid) ;



function menu_view_show_grid_Callback(hObject, eventdata, handles)
% just use checked state of menu for now, no other state
toggleOnOff(hObject,'Checked');
hlpTickGridBang(handles.axes_all, handles.menu_view_show_tick_labels, handles.menu_view_show_grid) ;



function menu_track_setparametersfile_Callback(hObject, eventdata, handles)
% Really, "configure parameters"

labeler = handles.labeler;
if any(labeler.bgTrnIsRunningFromTrackerIndex()),
  warndlg('Cannot change training parameters while trackers are training.','Training in progress','modal');
  return;
end
handles.labeler.setStatus('Setting training parameters...');

[tPrm,do_update] = labeler.trackSetAutoParams();

sPrmNew = ParameterSetup(handles.figure,tPrm,'labelerObj',labeler); % modal

if isempty(sPrmNew)
  if do_update
    RC.saveprop('lastCPRAPTParams',sPrmNew);
    %cbkSaveNeeded(labeler,true,'Parameters changed');
    labeler.setDoesNeedSave(true,'Parameters changed') ;
  end
  % user canceled; none
else
  labeler.trackSetParams(sPrmNew);
  RC.saveprop('lastCPRAPTParams',sPrmNew);
  %cbkSaveNeeded(labeler,true,'Parameters changed');
  labeler.setDoesNeedSave(true,'Parameters changed') ;
end

handles.labeler.clearStatus();


function menu_track_settrackparams_Callback(hObject, eventdata, handles)

labeler = handles.labeler;
handles.labeler.setStatus('Setting tracking parameters...');

[tPrm] = labeler.trackGetTrackParams();

sPrmTrack = ParameterSetup(handles.figure,tPrm,'labelerObj',labeler); % modal

if ~isempty(sPrmTrack),
  sPrmNew = labeler.trackSetTrackParams(sPrmTrack);
  RC.saveprop('lastCPRAPTParams',sPrmNew);
  %cbkSaveNeeded(labeler,true,'Parameters changed');
  labeler.setDoesNeedSave(true, 'Parameters changed') ;
end

handles.labeler.clearStatus();


function menu_track_auto_params_update_Callback(hObject,eventdata,handles)

checked = get(hObject,'Checked');
set(hObject,'Checked',~checked);
handles.labeler.trackAutoSetParams = ~checked;
labeler = handles.labeler;
labeler.setDoesNeedSave(true, 'Auto compute training parameters changed') ;


function menu_track_use_all_labels_to_train_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
tObj = labeler.tracker;
if isempty(tObj)
  handles.labeler.lerror('LabelerGUI:tracker','No tracker for this project.');
end
if tObj.hasTrained && tObj.trnDataDownSamp
  resp = questdlg('A tracker has already been trained with downsampled training data. Proceeding will clear all previous trained/tracked results. OK?',...
    'Clear Existing Tracker','Yes, clear previous tracker','Cancel','Cancel');
  if isempty(resp)
    resp = 'Cancel';
  end
  switch resp
    case 'Yes, clear previous tracker'
      % none
    case 'Cancel'
      return;
  end
end
tObj.trnDataUseAll();

% function menu_track_select_training_data_Callback(hObject, eventdata, handles)
% tObj = handles.labeler.tracker;
% if tObj.hasTrained
%   resp = questdlg('A tracker has already been trained. Downsampling training data will clear all previous trained/tracked results. Proceed?',...
%     'Clear Existing Tracker','Yes, clear previous tracker','Cancel','Cancel');
%   if isempty(resp)
%     resp = 'Cancel';
%   end
%   switch resp
%     case 'Yes, clear previous tracker'
%       % none
%     case 'Cancel'
%       return;
%   end
% end
% tObj.trnDataSelect();

% function menu_track_set_landmark_matches_Callback(hObject,eventdata,handles)
% handles.labeler.setStatus('Defining landmark matches...');
% labeler = handles.labeler;
% instructions = ['These part matches are used for data augmentation when training using deep learning. ' ...
%                 'To create more training data, we can flip the original images. This requires knowing ' ...
%                 'which parts on the left of the animal correspond to the same parts on the right side. ' ...
%                 'Use this GUI to select these pairings of parts. Each part can only belong to at most ' ...
%                 'one pair. Some parts will not be part of any pair, e.g. parts that go down the ' ...
%                 'mid-line of the animal should not have a mate.'];
% matches = defineLandmarkMatches(labeler,'edges',labeler.flipLandmarkMatches,'instructions',instructions);
% labeler.setFlipLandmarkMatches(matches);
% handles.labeler.clearStatus();

% function menu_track_training_data_montage_Callback(hObject,eventdata,handles)
% handles.labeler.setStatus('Plotting training examples...');
% labeler = handles.labeler;
% labeler.tracker.trainingDataMontage();
% handles.labeler.clearStatus();

function menu_track_trainincremental_Callback(hObject, eventdata, handles)
handles.labeler.trainIncremental();

function menu_go_targets_summary_Callback(hObject, eventdata, handles)
if handles.labeler.maIsMA
  TrkInfoUI(handles.labeler);
else
  handles.controller.raiseTargetsTableFigure();
end

function menu_go_nav_prefs_Callback(hObject, eventdata, handles)
handles.labeler.navPrefsUI();

function menu_go_gt_frames_Callback(hObject, eventdata, handles)
handles.labeler.gtShowGTManager();

function menu_evaluate_crossvalidate_Callback(hObject, eventdata, handles)

labeler = handles.labeler;

tbl = labeler.labelGetMFTableLabeled;  
if labeler.maIsMA
  tbl = tbl(:,1:2);
  tbl = unique(tbl);
  str = 'frames';
else
  tbl = tbl(:,1:3);
  str = 'targets';
end
n = height(tbl);

inputstr = sprintf('This project has %d labeled %s.\nNumber of folds for k-fold cross validation:',...
  n,str);
resp = inputdlg(inputstr,'Cross Validation',1,{'3'});
if isempty(resp)
  return;
end
nfold = str2double(resp{1});
if round(nfold)~=nfold || nfold<=1
  handles.labeler.lerror('LabelerGUI:xvalid','Number of folds must be a positive integer greater than 1.');
end

tbl.split = ceil(nfold*rand(n,1));

t = labeler.tracker;
t.trainsplit(tbl);


function menu_track_clear_tracking_results_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
% legacy behavior not sure why; maybe b/c the user is prob wanting to increase avail mem
%labeler.preProcInitData(); 
res = questdlg('Are you sure you want to clear tracking results?');
if ~strcmpi(res,'yes'),
  return;
end
handles.labeler.setStatus('Clearing tracking results...');
tObj = labeler.tracker;
tObj.clearTrackingResults();
handles.labeler.clearStatus();
%msgbox('Tracking results cleared.','Done');


% function menu_track_cpr_storefull_dont_store_Callback(hObject, eventdata, handles)
% tObj = handles.labeler.tracker;
% svr = tObj.showVizReplicates;
% if svr
%   qstr = 'Replicates will no longer by shown. OK?';
%   resp = questdlg(qstr,'Tracking Storage','OK, continue','No, cancel','OK, continue');
%   if isempty(resp)
%     resp = 'No, cancel';
%   end
%   if strcmp(resp,'No, cancel')
%     return;
%   end
%   tObj.showVizReplicates = false;
% end
% tObj.storeFullTracking = StoreFullTrackingType.NONE;
% 
% 
% function menu_track_cpr_storefull_store_final_iteration_Callback(...
%   hObject, eventdata, handles)
% tObj = handles.labeler.tracker;
% tObj.storeFullTracking = StoreFullTrackingType.FINALITER;
% 
% 
% function menu_track_cpr_storefull_store_all_iterations_Callback(...
%   hObject, eventdata, handles)
% tObj = handles.labeler.tracker;
% tObj.storeFullTracking = StoreFullTrackingType.ALLITERS;
% 
% 
% function menu_track_cpr_view_diagnostics_Callback(...
%   hObject, eventdata, handles)
% labeler = handles.labeler;
% 
% % Look for existing/open CPRVizTrackDiagsGUI
% h = handles.controller.findSatelliteByTag_('figCPRVizTrackDiagsGUI') ;
% if ~isempty(h) && isvalid(h) ,
%   figure(h) ;
%   return
% end
% 
% lc = labeler.lblCore;
% if ~isempty(lc) && ~lc.hideLabels
%   warningNoTrace('LabelerGUI:hideLabels','Hiding labels.');
%   lc.labelsHide();
% end
% hVizGUI = CPRVizTrackDiagsGUI(handles.labeler);
% handles.controller.addSatellite(hVizGUI) ;
% guidata(handles.figure,handles);


% function menu_track_track_and_export_Callback(hObject, eventdata, handles)
% labeler = handles.labeler;
% [tfok,rawtrkname] = handles.controller.getExportTrkRawNameUI();
% if ~tfok
%   return
% end
% labeler.trackAndExport([],'rawtrkname',rawtrkname);


function menu_track_batch_track_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
tbobj = TrackBatchGUI(labeler);
tbobj.run();


function menu_track_all_movies_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
mIdx = labeler.allMovIdx();
toTrackIn = labeler.mIdx2TrackList(mIdx);
tbobj = TrackBatchGUI(labeler,'toTrack',toTrackIn);
% [toTrackOut] = tbobj.run();
tbobj.run();
% todo: import predictions


function menu_track_current_movie_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
mIdx = labeler.currMovIdx;
toTrackIn = labeler.mIdx2TrackList(mIdx);
mdobj = SpecifyMovieToTrackGUI(labeler,labeler.gdata.figure,toTrackIn);
[toTrackOut,dostore] = mdobj.run();
if ~dostore,
  return;
end
trackBatch('labeler',labeler,'toTrack',toTrackOut);


function menu_track_id_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
labeler.track_id = ~labeler.track_id;
set(handles.menu_track_id,'checked',labeler.track_id);


function menu_file_clear_imported_Callback(hObject,evtdata,handles)
labeler = handles.labeler;
labeler.labels2Clear();

function menu_file_export_all_movies_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
nMov = labeler.nmoviesGTaware;
if nMov==0
  handles.labeler.lerror('LabelerGUI:noMov','No movies in project.');
end
iMov = 1:nMov;
[tfok,rawtrkname] = handles.controller.getExportTrkRawNameUI();
if ~tfok
  return;
end
labeler.trackExportResults(iMov,'rawtrkname',rawtrkname);

function menu_track_set_labels_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
tObj = labeler.tracker;
if labeler.gtIsGTMode
  handles.labeler.lerror('LabelerGUI:gt','Unsupported in GT mode.');
end

if ~isempty(tObj) && tObj.hasBeenTrained() && (~labeler.maIsMA)
  % single animal. Use prediction if available else use imported below
  [tfhaspred,xy,tfocc] = tObj.getTrackingResultsCurrFrm(); %#ok<ASGLU>
  itgt = labeler.currTarget;

  if ~tfhaspred(itgt)
    if (labeler.nTrx>1)
      msgbox('No predictions for current frame.');
      return;    
    else % for single animal use imported predictions if available
      iMov = labeler.currMovie;
      frm = labeler.currFrame;  
      [tfhaspred,xy] = labeler.labels2{iMov}.getPTrkFrame(frm);
      if ~tfhaspred
        msgbox('No predictions for current frame.');
        return;    
      end
    end
  else
    xy = xy(:,:,itgt); % "targets" treatment differs from below
  end

  disp(xy);
  
  % AL20161219: possibly dangerous, assignLabelCoords prob was intended
  % only as a util method for subclasses rather than public API. This may
  % not do the right thing for some concrete LabelCores.
%   labeler.lblCore.assignLabelCoords(xy);

  lpos2xy = reshape(xy,labeler.nLabelPoints,2);
  %assert(size(lpos2,4)==1); % "targets" treatment differs from above
  %lpos2xy = lpos2(:,:,frm);
  labeler.labelPosSet(lpos2xy);

  labeler.lblCore.newFrame(frm,frm,1);

else
  iMov = labeler.currMovie;
  frm = labeler.currFrame;
  if iMov==0
    handles.labeler.lerror('LabelerGUI:setLabels','No movie open.');
  end
  
  if labeler.maIsMA
    % We need to be smart about which to use. 
    % If only one of imported or prediction exist for the current frame then use whichever exists
    % If both exist for current frame, then don't do anything and error.
    
    useImported = true;
    usePred = true;
    % getting imported info old sytle. Doesn't work anymore
    
%     s = labeler.labels2{iMov};
%     itgtsImported = Labels.isLabeledF(s,frm);
%     ntgtsImported = numel(itgtsImported);
 
    % check if we can use imported
    imp_trk = labeler.labeledpos2trkViz;
    if isempty(imp_trk)
      useImported=false;
    elseif isnan(imp_trk.currTrklet)
      useImported=false;
    else
      s = labeler.labels2{iMov};
      iTgtImp = imp_trk.currTrklet;
      if isnan(iTgtImp)
        useImported = false;
      else
        [tfhaspred,~,~] = s.getPTrkFrame(frm,'collapse',true);
        if ~tfhaspred(iTgtImp)
          useImported = false;
        end
      end
    end
      
    % check if we can use pred
    if isempty(tObj)
      usePred = false;
    elseif isempty(tObj.trkVizer)
      usePred = false;
    else
      [tfhaspred,xy,tfocc] = tObj.getTrackingResultsCurrFrm(); %#ok<ASGLU>
      iTgtPred = tObj.trkVizer.currTrklet;
      if isnan(iTgtPred)
        usePred = false;
      elseif ~tfhaspred(iTgtPred)
        usePred = false;      
      end
    end
    
    if usePred && useImported
      msgbox('Both imported and prediction exist for current frame. Cannot decide which to use. Skipping');
      return
    end
    
    if (~usePred) && (~useImported)
      msgbox('No predictions for current frame or no valid tracklet selected. Nothing to use as a label');
      return
    end

    if useImported
      s = labeler.labels2{iMov};
      iTgt = imp_trk.currTrklet;
      [~,xy,tfocc] = s.getPTrkFrame(frm,'collapse',true);      
    else
      iTgt = tObj.trkVizer.currTrklet;
      [~,xy,tfocc] = tObj.getTrackingResultsCurrFrm();
    end
    xy = xy(:,:,iTgt); % "targets" treatment differs from below
    occ = tfocc(:,iTgt);
    ntgts = labeler.labelNumLabeledTgts();
    labeler.setTargetMA(ntgts+1);
    labeler.labelPosSet(xy,occ);
    labeler.updateTrxTable();
    iTgt = labeler.currTarget;
    labeler.lblCore.tv.updateTrackResI(xy,occ,iTgt);

  else
    if labeler.nTrx>1
      handles.labeler.lerror('LabelerGUI:setLabels','Unsupported for multiple targets.');
    end
    %lpos2 = labeler.labeledpos2{iMov};
    %MK 20230728, labels2 now should always be TrkFile, but keeping other
    %logic around just in case. Needs work for multiview though.
    if isa(labeler.labels2{iMov} ,'TrkFile')      
      [~,p] = labeler.labels2{iMov}.getPTrkFrame(frm);
    else
      p = Labels.getLabelsF(labeler.labels2{iMov},frm,1);
    end
    lpos2xy = reshape(p,labeler.nLabelPoints,2);
    %assert(size(lpos2,4)==1); % "targets" treatment differs from above
    %lpos2xy = lpos2(:,:,frm);
    labeler.labelPosSet(lpos2xy);
    
    labeler.lblCore.newFrame(frm,frm,1);
  end
end

function menu_track_background_predict_start_Callback(hObject,eventdata,handles)
tObj = handles.labeler.tracker;
if tObj.asyncIsPrepared
  tObj.asyncStartBgRunner();
else
  if ~tObj.hasTrained
    errordlg('A tracker has not been trained.','Background Tracking');
    return;
  end
  tObj.asyncPrepare();
  tObj.asyncStartBgRunner();
end
  
function menu_track_background_predict_end_Callback(hObject,eventdata,handles)
tObj = handles.labeler.tracker;
if tObj.asyncIsPrepared
  tObj.asyncStopBgRunner();
else
  warndlg('Background worker is not running.','Background tracking');
end

function menu_track_background_predict_stats_Callback(hObject,eventdata,handles)
tObj = handles.labeler.tracker;
if tObj.asyncIsPrepared
  tObj.asyncComputeStats();
else
  warningNoTrace('LabelerGUI:bgTrack',...
    'No background tracking information available.','Background tracking');
end

function menu_evaluate_gtmode_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
controller = handles.controller ;

handles.labeler.setStatus('Switching between Labeling and Ground Truth Mode...');

gt = labeler.gtIsGTMode;
gtNew = ~gt;
labeler.gtSetGTMode(gtNew);
% hGTMgr = labeler.gdata.GTMgr;
if gtNew
  mmc = controller.movieManagerController_ ;
  mmc.setVisible(true);
  figure(mmc.hFig);
end
handles.labeler.clearStatus();

function menu_evaluate_gtloadsuggestions_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
LabelerGT.loadSuggestionsUI(labeler);

function menu_evaluate_gtsetsuggestions_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
LabelerGT.setSuggestionsToLabeledUI(labeler);

function menu_evaluate_gtcomputeperf_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
assert(labeler.gtIsGTMode);
labeler.gtComputeGTPerformance();

function menu_evaluate_gtcomputeperfimported_Callback(hObject,eventdata,handles)
labeler = handles.labeler;
assert(labeler.gtIsGTMode);
labeler.gtComputeGTPerformance('useLabels2',true);

function menu_evaluate_gtexportresults_Callback(hObject,eventdata,handles)
labeler = handles.labeler;

tblRes = labeler.gtTblRes;
if isempty(tblRes)
  errordlg('No GT results are currently available.','Export GT Results');
  return;
end

%assert(labeler.gtIsGTMode);
fname = labeler.getDefaultFilenameExportGTResults();
[f,p] = uiputfile(fname,'Export File');
if isequal(f,0)
  return;
end
fname = fullfile(p,f);  
VARNAME = 'tblGT';
s = struct();
s.(VARNAME) = tblRes;
save(fname,'-mat','-struct','s');
fprintf('Saved table ''%s'' to file ''%s''.\n',VARNAME,fname);
  
function pumInfo_Callback(hObject, eventdata, handles)
cprop = get(hObject,'Value');
handles.labelTLInfo.setCurProp(cprop);
cpropNew = handles.labelTLInfo.getCurProp();
if cpropNew ~= cprop,
  set(hObject,'Value',cpropNew);
end
hlpRemoveFocus(hObject,handles);

function pbPlaySeg_Callback(hObject, eventdata, handles)
if ~handles.labeler.doProjectAndMovieExist()
  return
end
handles.controller.play('playsegment', 'videoPlaySegFwdEnding') ;

function pbPlaySegRev_Callback(hObject, eventdata, handles)
if ~handles.labeler.doProjectAndMovieExist()
  return
end
handles.controller.play('playsegmentrev', 'videoPlaySegRevEnding') ;

function pbPlay_Callback(hObject, eventdata, handles)
if ~handles.labeler.doProjectAndMovieExist()
  return
end
handles.controller.play('play', 'videoPlay') ;

%% Cropping
% function handles = cropInitImRects(handles)
% deleteValidGraphicsHandles(handles.cropHRect);
% handles.cropHRect = ...
%   arrayfun(@(x)imrect(x,[nan nan nan nan]),handles.axes_all,'uni',0); %#ok<IMRECT> 
% handles.cropHRect = cat(1,handles.cropHRect{:}); % ML 2016a ish can't concat imrects in arrayfun output
% arrayfun(@(x)set(x,'Visible','off','PickableParts','none','UserData',true),...
%   handles.cropHRect); % userdata: see cropImRectSetPosnNoPosnCallback
% for ivw=1:numel(handles.axes_all)
%   posnCallback = @(zpos)cbkCropPosn(zpos,ivw,handles.figure);
%   handles.cropHRect(ivw).addNewPositionCallback(posnCallback);
% end

% function cropReactNewCropMode(handles,tf)
% 
% % CROPCONTROLS = {
% %   'pushbutton_exitcropmode'
% %   'tbAdjustCropSize'
% %   'pbClearAllCrops'
% %   'txCropMode'
% %   };
% REGCONTROLS = {
%   'pbClear'
%   'tbAccept'
%   'pbTrain'
%   'pbTrack'
%   'pumTrack'};
% 
% onIfTrue = onIff(tf);
% offIfTrue = onIff(~tf);
% 
% %cellfun(@(x)set(handles.(x),'Visible',onIfTrue),CROPCONTROLS);
% set(handles.uipanel_cropcontrols,'Visible',onIfTrue);
% set(handles.text_trackerinfo,'Visible',offIfTrue);
% 
% cellfun(@(x)set(handles.(x),'Visible',offIfTrue),REGCONTROLS);
% handles.menu_file_crop_mode.Checked = onIfTrue;
% 
% cropUpdateCropHRects(handles);
% cropUpdateCropAdjustingCropSize(handles,false);


% function cropUpdateCropHRects(handles)
% % Update handles.cropHRect from labeler.cropIsCropMode, labeler.currMovie and
% % labeler.movieFilesAll*cropInfo
% %
% % rect props set:
% % - position
% % - visibility, pickableparts
% %
% % rect props NOT set:
% % - resizeability. 
% 
% labeler = handles.labeler;
% tfCropMode = labeler.cropIsCropMode;
% [tfHasCrop,roi] = labeler.cropGetCropCurrMovie();
% if tfCropMode && tfHasCrop
%   nview = labeler.nview;
%   imnc = labeler.movierawnc;
%   imnr = labeler.movierawnr;
%   szassert(roi,[nview 4]);
%   szassert(imnc,[nview 1]);
%   szassert(imnr,[nview 1]);
%   for ivw=1:nview
%     h = handles.cropHRect(ivw);
%     cropImRectSetPosnNoPosnCallback(h,CropInfo.roi2RectPos(roi(ivw,:)));
%     set(h,'Visible','on','PickableParts','all');
%     fcn = makeConstrainToRectFcn('imrect',[1 imnc(ivw)],[1 imnr(ivw)]);
%     h.setPositionConstraintFcn(fcn);
%   end
% else
%   arrayfun(@(x)cropImRectSetPosnNoPosnCallback(x,[nan nan nan nan]),...
%     handles.cropHRect);
%   arrayfun(@(x)set(x,'Visible','off','PickableParts','none'),handles.cropHRect);
% end


% function cropImRectSetPosnNoPosnCallback(hRect,pos)
% % Set the hRect's graphics position without triggering its
% % PositionCallback. Works in concert with cbkCropPosn
% tfSetPosnLabeler0 = get(hRect,'UserData');
% set(hRect,'UserData',false);
% hRect.setPosition(pos);
% set(hRect,'UserData',tfSetPosnLabeler0);


function tbAdjustCropSize_Callback(hObject, eventdata, handles)
cropUpdateCropAdjustingCropSize(handles);
tb = handles.tbAdjustCropSize;
if tb.Value==tb.Min
  % user clicked "Done Adjusting"
  warningNoTrace('All movies in a given view must share the same crop size. The sizes of all crops have been updated as necessary.'); 
elseif tb.Value==tb.Max
  % user clicked "Adjust Crop Size"
  labeler = handles.labeler;
  if ~labeler.cropProjHasCrops
    labeler.cropInitCropsAllMovies;
    fprintf(1,'Default crop initialized for all movies.\n');
    handles.controller.cropUpdateCropHRects_();
  end
end
function pbClearAllCrops_Callback(hObject, eventdata, handles)
handles.labeler.cropClearAllCrops();


% --------------------------------------------------------------------
function menu_file_export_labels2_trk_curr_mov_Callback(hObject, eventdata, handles)
% hObject    handle to menu_file_export_labels2_trk_curr_mov (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

labeler = handles.labeler;
iMov = labeler.currMovie;
if iMov==0
  handles.labeler.lerror('LabelerGUI:noMov','No movie currently set.');
end
[tfok,rawtrkname] = handles.controller.getExportTrkRawNameUI();
if ~tfok
  return;
end
labeler.trackExportResults(iMov,'rawtrkname',rawtrkname);


% --------------------------------------------------------------------
function menu_file_import_export_advanced_Callback(hObject, eventdata, handles)
% hObject    handle to menu_file_import_export_advanced (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function menu_track_tracking_algorithm_Callback(hObject, eventdata, handles)
% hObject    handle to menu_track_tracking_algorithm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function menu_view_landmark_colors_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
cbkApply = @(varargin)(labeler.hlpApplyCosmetics(varargin{:})) ;
LandmarkColors(labeler,cbkApply);
% AL 20220217: changes now applied immediately
% if ischange
%   cbkApply(savedres.colorSpecs,savedres.markerSpecs,savedres.skelSpecs);
% end

function menu_track_edit_skeleton_Callback(hObject, eventdata, handles)
labeler = handles.labeler;
landmark_specs('labeler',labeler);
%hasSkeleton = ~isempty(labeler.skeletonEdges) ;
%labeler.setShowSkeleton(hasSkeleton) ;

function menu_track_viz_dataaug_Callback(hObject,evtdata,handles)
labeler = handles.labeler;
labeler.retrainAugOnly() ;

function menu_view_showhide_skeleton_Callback(hObject, eventdata, handles)
if strcmpi(get(hObject,'Checked'),'off'),
  hObject.Checked = 'on';
  handles.labeler.setShowSkeleton(true);
else
  hObject.Checked = 'off';
  handles.labeler.setShowSkeleton(false);
end

function menu_view_showhide_maroi_Callback(hObject, eventdata, handles)
if strcmpi(get(hObject,'Checked'),'off'),
  handles.labeler.setShowMaRoi(true);
else
  handles.labeler.setShowMaRoi(false);
end

function menu_view_showhide_maroiaux_Callback(hObject, eventdata, handles)
tf = strcmpi(get(hObject,'Checked'),'off');
handles.labeler.setShowMaRoiAux(tf);

% --- Executes on selection change in popupmenu_prevmode.
function popupmenu_prevmode_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu_prevmode (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

contents = cellstr(get(hObject,'String'));
mode = contents{get(hObject,'Value')};
if strcmpi(mode,'Reference'),
  handles.labeler.setPrevAxesMode(PrevAxesMode.FROZEN,handles.labeler.prevAxesModeInfo);
else
  handles.labeler.setPrevAxesMode(PrevAxesMode.LASTSEEN);
end



function pushbutton_freezetemplate_Callback(hObject, eventdata, handles)
handles.labeler.setPrevAxesMode(PrevAxesMode.FROZEN);



% --- Executes on button press in pushbutton_exitcropmode.
function pushbutton_exitcropmode_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_exitcropmode (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

labeler = handles.labeler;
labeler.cropSetCropMode(false);


% --------------------------------------------------------------------
function menu_view_occluded_points_box_Callback(hObject, eventdata, handles)
% hObject    handle to menu_view_occluded_points_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

labeler = handles.labeler;
labeler.setShowOccludedBox(~labeler.showOccludedBox);
if labeler.showOccludedBox,
  labeler.lblCore.showOcc();
else
  labeler.lblCore.hideOcc();
end

% --- Executes on selection change in pumInfo_labels.
function pumInfo_labels_Callback(hObject, eventdata, handles)
% hObject    handle to pumInfo_labels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns pumInfo_labels contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pumInfo_labels

ipropType = get(hObject,'Value');
% see also InfoTimeline/enforcePropConsistencyWithUI
iprop = get(handles.pumInfo,'Value');
props = handles.labelTLInfo.getPropsDisp(ipropType);
if iprop > numel(props),
  iprop = 1;
end
set(handles.pumInfo,'String',props,'Value',iprop);
handles.labelTLInfo.setCurPropType(ipropType,iprop);



function register_labeler(figure, labeler)

handles = guidata(figure) ;

handles.labeler = labeler ;

handles.labelTLInfo = InfoTimeline(labeler,handles.axes_timeline_manual,...
                                   handles.axes_timeline_islabeled);

set(handles.pumInfo,'String',handles.labelTLInfo.getPropsDisp(),...
  'Value',handles.labelTLInfo.curprop);
set(handles.pumInfo_labels,'String',handles.labelTLInfo.getPropTypesDisp(),...
  'Value',handles.labelTLInfo.curproptype);

% this is currently not used - KB made space here for training status
%set(handles.txProjectName,'String','');

listeners = cell(0,1);
listeners{end+1,1} = addlistener(handles.slider_frame,'ContinuousValueChange',@slider_frame_Callback);
listeners{end+1,1} = addlistener(handles.sldZoom,'ContinuousValueChange',@sldZoom_Callback);
% listeners{end+1,1} = addlistener(handles.axes_curr,'XLim','PostSet',@(s,e)axescurrXLimChanged(s,e,handles));
% listeners{end+1,1} = addlistener(handles.axes_curr,'XDir','PostSet',@(s,e)axescurrXDirChanged(s,e,handles));
% listeners{end+1,1} = addlistener(handles.axes_curr,'YDir','PostSet',@(s,e)axescurrYDirChanged(s,e,handles));
% % listeners{end+1,1} = addlistener(labeler,'didSetProjname',@cbkProjNameChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetCurrTarget',@cbkCurrTargetChanged);
% % listeners{end+1,1} = addlistener(labeler,'didSetLastLabelChangeTS',@cbkLastLabelChangeTS);
% % listeners{end+1,1} = addlistener(labeler,'didSetTrackParams',@cbkParameterChange);
% listeners{end+1,1} = addlistener(labeler,'didSetLabelMode',@cbkLabelModeChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetLabels2Hide',@cbkLabels2HideChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetLabels2ShowCurrTargetOnly',@cbkLabels2ShowCurrTargetOnlyChanged);
% % listeners{end+1,1} = addlistener(labeler,'didSetProjFSInfo',@cbkProjFSInfoChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetShowTrx',@cbkShowTrxChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetShowOccludedBox',@cbkShowOccludedBoxChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetShowTrxCurrTargetOnly',@cbkShowTrxCurrTargetOnlyChanged);
% % listeners{end+1,1} = addlistener(labeler,'didSetTrackersAll',@cbkTrackersAllChanged);
% % listeners{end+1,1} = addlistener(labeler,'didSetCurrTracker',@cbkCurrTrackerChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetTrackModeIdx',@cbkTrackModeIdxChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetTrackNFramesSmall',@cbkTrackerNFramesChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetTrackNFramesLarge',@cbkTrackerNFramesChanged);    
% listeners{end+1,1} = addlistener(labeler,'didSetTrackNFramesNear',@cbkTrackerNFramesChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetMovieCenterOnTarget',@cbkMovieCenterOnTargetChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetMovieRotateTargetUp',@cbkMovieRotateTargetUpChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetMovieForceGrayscale',@cbkMovieForceGrayscaleChanged);
% % listeners{end+1,1} = addlistener(labeler,'didSetMovieInvert',@cbkMovieInvertChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetMovieViewBGsubbed',@cbkMovieViewBGsubbedChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetLblCore',@(src,evt)(handles.controller.didSetLblCore()));
% listeners{end+1,1} = addlistener(labeler,'gtIsGTModeChanged',@cbkGtIsGTModeChanged);
% listeners{end+1,1} = addlistener(labeler,'cropIsCropModeChanged',@cbkCropIsCropModeChanged);
% listeners{end+1,1} = addlistener(labeler,'cropUpdateCropGUITools',@cbkUpdateCropGUITools);
% listeners{end+1,1} = addlistener(labeler,'cropCropsChanged',@cbkCropCropsChanged);
% %listeners{end+1,1} = addlistener(labeler,'newProject',@cbkNewProject);
% listeners{end+1,1} = addlistener(labeler,'newMovie',@cbkNewMovie);
% %listeners{end+1,1} = addlistener(labeler,'projLoaded',@cbkProjLoaded);

listeners{end+1,1} = addlistener(handles.labelTLInfo,'selectOn','PostSet',@cbklabelTLInfoSelectOn);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'props','PostSet',@cbklabelTLInfoPropsUpdated);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'props_tracker','PostSet',@cbklabelTLInfoPropsUpdated);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'props_allframes','PostSet',@cbklabelTLInfoPropsUpdated);
listeners{end+1,1} = addlistener(handles.labelTLInfo,'proptypes','PostSet',@cbklabelTLInfoPropTypesUpdated);

% %listeners{end+1,1} = addlistener(labeler,'startAddMovie',@cbkAddMovie);
% %listeners{end+1,1} = addlistener(labeler,'finishAddMovie',@cbkAddMovie);
% %listeners{end+1,1} = addlistener(labeler,'startSetMovie',@cbkSetMovie);
% listeners{end+1,1} = addlistener(labeler,'dataImported',@cbkDataImported);
% listeners{end+1,1} = addlistener(labeler,'didSetShowSkeleton',@cbkShowSkeletonChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetShowMaRoi',@cbkShowMaRoiChanged);
% listeners{end+1,1} = addlistener(labeler,'didSetShowMaRoiAux',@cbkShowMaRoiAuxChanged);

handles.listeners = listeners;

% Make the debug menu visible, if called for
handles.menu_debug.Visible = onIff(labeler.isInDebugMode) ;

% Stash the guidata
guidata(figure, handles) ;



function register_controller(main_figure, controller)
handles = guidata(main_figure) ;
handles.controller = controller ;
set(handles.tblTrx, 'CellSelectionCallback', @(s,e)(controller.controlActuated('tblTrx', s, e))) ;
set(handles.tblFrames, 'CellSelectionCallback',@(s,e)(controller.controlActuated('tblFrames', s, e))) ;

hZ = zoom(main_figure);  % hZ is a "zoom object"
hZ.ActionPostCallback = @(s,e)(obj.cbkPostZoom(s,e)) ;
hP = pan(main_figure);  % hP is a "pan object"
hP.ActionPostCallback = @(s,e)(obj.cbkPostPan(s,e)) ;

set(main_figure, 'CloseRequestFcn', @(s,e)(controller.figure_CloseRequestFcn())) ;

guidata(main_figure, handles) ;
