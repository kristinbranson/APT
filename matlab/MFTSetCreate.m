function varargout = MFTSetCreate(varargin)
% MFTSETCREATE MATLAB code for MFTSetCreate.fig
%      MFTSETCREATE, by itself, creates a new MFTSETCREATE or raises the existing
%      singleton*.
%
%      H = MFTSETCREATE returns the handle to a new MFTSETCREATE or the handle to
%      the existing singleton*.
%
%      MFTSETCREATE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MFTSETCREATE.M with the given input arguments.
%
%      MFTSETCREATE('Property','Value',...) creates a new MFTSETCREATE or raises
%      the existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MFTSetCreate_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MFTSetCreate_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MFTSetCreate

% Last Modified by GUIDE v2.5 18-Jan-2018 16:06:28

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @MFTSetCreate_OpeningFcn, ...
                   'gui_OutputFcn',  @MFTSetCreate_OutputFcn, ...
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

% MFTSetCreate(lObj)
function MFTSetCreate_OpeningFcn(hObject, eventdata, handles, varargin)
set(hObject,'MenuBar','None');
lObj = varargin{1};
handles.lObj = lObj;
handles.etFrameRadius.String = num2str(lObj.trackNFramesNear);
handles.etDecimation.String = num2str(lObj.trackNFramesSmall);
handles.output = [];
guidata(hObject, handles);
uiwait(handles.figure1);

function varargout = MFTSetCreate_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;
delete(hObject);

function etFrameRadius_Callback(hObject, eventdata, handles)
x = str2double(hObject.String);
if isnan(x) || x<1 || round(x)~=x
  warningNoTrace('Invalid within-frames value.');
  hObject.String = num2str(handles.lObj.trackNFramesNear);
end

function btngrpFrames_SelectionChangedFcn(hObject, eventdata, handles)
hBG = handles.btngrpFrames;
switch hBG.SelectedObject
  case handles.rbAllFrames
    handles.etFrameRadius.Enable = 'off';
  case handles.rbLabeledFrames
    handles.etFrameRadius.Enable = 'off';
  case handles.rbSelectedFrames
    handles.etFrameRadius.Enable = 'off';
  case handles.rbWithin
    handles.etFrameRadius.Enable = 'on';
  otherwise
    assert(false);
end 
  
function etDecimation_Callback(hObject, eventdata, handles)
x = str2double(hObject.String);
if isnan(x) || x<1 || round(x)~=x
  warningNoTrace('Invalid Decimation value.');
  hObject.String = num2str(handles.lObj.trackNFramesSmall);
end

function pbApply_Callback(hObject, eventdata, handles)
hBG = handles.btngrpMovies;
switch hBG.SelectedObject  
  case handles.rbCurrMovie
    mset = MovieIndexSetVariable.CurrMov;
  case handles.rbAllMovies
    mset = MovieIndexSetVariable.AllMov;
  case handles.rbSelectedMovies
    mset = MovieIndexSetVariable.SelMov;
  otherwise
    assert(false);
end

hBG = handles.btngrpTgts;
switch hBG.SelectedObject
  case handles.rbCurrentTarget
    tset = TargetSetVariable.CurrTgt;
  case handles.rbAllTargets
    tset = TargetSetVariable.AllTgts;
  otherwise
    assert(false);    
end

hBG = handles.btngrpFrames;
switch hBG.SelectedObject
  case handles.rbAllFrames
    fset = FrameSetVariable.AllFrm;
  case handles.rbLabeledFrames
    fset = FrameSetVariable.LabeledFrm;
  case handles.rbSelectedFrames
    fset = FrameSetVariable.SelFrm;
  case handles.rbWithin
    fset = FrameSetVariable.WithinCurrFrm;
    frmRadius = str2double(handles.etFrameRadius.String);
    assert(~isnan(frmRadius));
    handles.lObj.trackNFramesNear = frmRadius;
  otherwise
    assert(false);
end
frmDec = str2double(handles.etDecimation.String);
assert(~isnan(frmDec));
handles.lObj.trackNFramesSmall = frmDec;
frmDec = FrameDecimationVariable.EveryNFrameSmall;

mftset = MFTSet(mset,fset,frmDec,tset);
handles.output = mftset;
guidata(handles.figure1,handles);
close(handles.figure1);

function pbCancel_Callback(hObject, eventdata, handles)
close(handles.figure1);

function figure1_CloseRequestFcn(hObject, eventdata, handles)
if strcmp(get(hObject,'waitstatus'),'waiting')
  uiresume(hObject);
else
  delete(hObject);
end

