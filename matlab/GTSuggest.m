function varargout = GTSuggest(varargin)
% GTSUGGEST MATLAB code for GTSuggest.fig
%      GTSUGGEST, by itself, creates a new GTSUGGEST or raises the existing
%      singleton*.
%
%      H = GTSUGGEST returns the handle to a new GTSUGGEST or the handle to
%      the existing singleton*.
%
%      GTSUGGEST('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GTSUGGEST.M with the given input arguments.
%
%      GTSUGGEST('Property','Value',...) creates a new GTSUGGEST or raises
%      the existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GTSuggest_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GTSuggest_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GTSuggest

% Last Modified by GUIDE v2.5 30-Jun-2018 07:53:39

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GTSuggest_OpeningFcn, ...
                   'gui_OutputFcn',  @GTSuggest_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% GTSuggest(lObj)
function GTSuggest_OpeningFcn(hObject, eventdata, handles, varargin)
lObj = varargin{1};  % Should change things so we pass in the LabelerController, not the Labeler
set(hObject,'MenuBar','None');

centerOnParentFigure(hObject,lObj.hFig);

handles.lObj = lObj;
handles.movMgrCtrler = lObj.controller_.movieManagerController_ ;  
  % subpoptimal to have to touch lObj.controller_, which is deprecated
  % also suboptimal to be touching movieManagerController_, which is
  % private-by-convention
handles.listener = event.listener(handles.movMgrCtrler,'tableClicked',...
  @(s,e)cbkTableClicked(hObject));

setGUIfromRC(handles);
updateNumMoviesSelected(handles,false);

handles.output = [];
guidata(hObject, handles);
uiwait(handles.figure1);

function varargout = GTSuggest_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;
delete(handles.listener);
delete(hObject);

function cbkTableClicked(hObject)
handles = guidata(hObject);
updateNumMoviesSelected(handles);

function updateNumMoviesSelected(handles,tfdowarn)
if handles.movMgrCtrler.gtTabSelected
  mIdxSel = handles.lObj.moviesSelected;
  nSel = numel(mIdxSel);
  handles.rbSelectedMovies.String = sprintf('Selected Movies (currently %d)',nSel);
else
  if tfdowarn
    warningNoTrace('Please select movies in GT movie list.');
  end
  handles.rbSelectedMovies.String = sprintf('Selected Movies');
end

function btngrpMovies_SelectionChangedFcn(hObject, eventdata, handles)
updateNumMoviesSelected(handles,true);

function setGUIfromRC(handles)
tag = RC.getprop('gtsuggest_btngrpMovies_selectedObject');
if ~isempty(tag) && isfield(handles,tag)
  handles.btngrpMovies.SelectedObject = handles.(tag);
else
  handles.btngrpMovies.SelectedObject = handles.rbAllMovies;
  RC.saveprop('gtsuggest_btngrpMovies_selectedObject','rbAllMovies');
end

tag = RC.getprop('gtsuggest_btngrpFrames_selectedObject');
if ~isempty(tag) && isfield(handles,tag)
  handles.btngrpFrames.SelectedObject = handles.(tag);
else
  handles.btngrpFrames.SelectedObject = handles.rbInTotal;
  RC.saveprop('gtsuggest_btngrpFrames_selectedObject','rbInTotal');
end

val = RC.getprop('gtsuggest_numFrames');
if ~isempty(val) 
  handles.etNumFrames.String = num2str(val);
else
  val = 100;
  handles.etNumFrames.String = num2str(val);
  RC.saveprop('gtsuggest_numFrames',val);
end

val = RC.getprop('gtsuggest_minDistTrainingFrames');
if ~isempty(val) 
  handles.etMinDistTrainingFrames.String = num2str(val);
else
  val = 0;
  handles.etMinDistTrainingFrames.String = num2str(val);
  RC.saveprop('gtsuggest_minDistTrainingFrames',val);
end

function setRCfromGUI(handles)
RC.saveprop('gtsuggest_btngrpMovies_selectedObject',...
  handles.btngrpMovies.SelectedObject.Tag);
RC.saveprop('gtsuggest_btngrpFrames_selectedObject',...
  handles.btngrpFrames.SelectedObject.Tag);
RC.saveprop('gtsuggest_numFrames',...
  str2double(handles.etNumFrames.String));
RC.saveprop('gtsuggest_minDistTrainingFrames',...
  str2double(handles.etMinDistTrainingFrames.String));

function etNumFrames_Callback(hObject, eventdata, handles)
x = str2double(hObject.String);
if isnan(x) || x<1 || round(x)~=x
  warningNoTrace('Invalid number of frames.');
  hObject.String = '';
end

function etMinDistTrainingFrames_Callback(hObject, eventdata, handles)
x = str2double(hObject.String);
if isnan(x) || x<0 || round(x)~=x
  warningNoTrace('Minimum distance must be a non-negative integer.');
  hObject.String = '';
end

function pbGenerateSelections_Callback(hObject, eventdata, handles)
gtsg = getCurrentConfig(handles);
setRCfromGUI(handles);
handles.output = gtsg;
guidata(handles.figure1,handles);
close(handles.figure1);

function gtsg = getCurrentConfig(handles)
s = struct();
s.numFrames = str2double(handles.etNumFrames.String);
if isnan(s.numFrames) || round(s.numFrames~=s.numFrames) || s.numFrames<1
  error('The number of frames must be a positive integer.');
end
s.minDistTraining = str2double(handles.etMinDistTrainingFrames.String);
if round(s.minDistTraining~=s.minDistTraining) || s.minDistTraining<0
  error('The minimum distance to training frames must be a non-negative integer.');  
elseif isnan(s.minDistTraining)
  % maybe just silent for now
  s.minDistTraining = 0;
  %warningNoTrace('Any labeled training rows in 
end

movSelObj = handles.btngrpMovies.SelectedObject;
if movSelObj==handles.rbAllMovies
  s.movSet = MovieIndexSetVariable.AllGTMov;
elseif movSelObj==handles.rbSelectedMovies
  s.movSet = MovieIndexSetVariable.SelMov;  
else
  assert(false);
end
s.movIdxs = s.movSet.getMovieIndices(handles.lObj);

frmSelObj = handles.btngrpFrames.SelectedObject;
if frmSelObj==handles.rbInTotal
  s.frmSpecType = GTSetNumFramesType.Total;
elseif frmSelObj==handles.rbPerMovie
  s.frmSpecType = GTSetNumFramesType.PerMovie;
elseif frmSelObj==handles.rbPerAnimal
  s.frmSpecType = GTSetNumFramesType.PerTarget;
else
  assert(false);
end

gtsg = GTSetGenerator(s.numFrames,s.frmSpecType,s.minDistTraining,s.movIdxs);

function pbCancel_Callback(hObject, eventdata, handles)
close(handles.figure1);

function figure1_CloseRequestFcn(hObject, eventdata, handles)
if strcmp(get(hObject,'waitstatus'),'waiting')
  uiresume(hObject);
else
  delete(hObject);
end
