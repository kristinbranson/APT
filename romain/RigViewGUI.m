function varargout = RigViewGUI(varargin)
% RIGVIEWGUI MATLAB code for RigViewGUI.fig
%      RIGVIEWGUI, by itself, creates a new RIGVIEWGUI or raises the existing
%      singleton*.
%
%      H = RIGVIEWGUI returns the handle to a new RIGVIEWGUI or the handle to
%      the existing singleton*.
%
%      RIGVIEWGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RIGVIEWGUI.M with the given input arguments.
%
%      RIGVIEWGUI('Property','Value',...) creates a new RIGVIEWGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before RigViewGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to RigViewGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help RigViewGUI

% Last Modified by GUIDE v2.5 15-May-2016 21:13:33

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
if ispc && ~verLessThan('matlab','8.6') % 8.6==R2015b
  gui_Name = 'RigViewGUI_PC_15b';
else
  gui_Name = 'RigViewGUI';
end
gui_State = struct('gui_Name',       gui_Name, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @RigViewGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @RigViewGUI_OutputFcn, ...
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


% --- Executes just before RigViewGUI is made visible.
function RigViewGUI_OpeningFcn(hObject, eventdata, handles, varargin)
handles.rvObj = varargin{1};
handles.axs = [handles.axes1 handles.axes2 handles.axes3];
handles.ims = [ ...
  imagesc(0,'Parent',handles.axes1); ...
  imagesc(0,'Parent',handles.axes2); ...
  imagesc(0,'Parent',handles.axes3); ...
  ];
set(handles.ims,'Hittest','off');
nIm = numel(handles.ims);
for i=1:nIm
  ax = handles.axs(i);
  hold(ax,'on');
  axisoff(ax);
  axis(ax,'image');
%   ax.LineWidth = 8;
%   ax.XColor = [0 0 1];
%   ax.YColor = [0 0 1];
  %ax.Visible = 'off';
end

rvObj = handles.rvObj;
listeners = cell(0,1);
listeners{end+1,1} = addlistener(rvObj,'movDir','PostSet',@cbkMovDirChanged);
listeners{end+1,1} = addlistener(rvObj,'movCurrFrameLR','PostSet',@cbkMovCurrFrameLRChanged);
handles.listeners = listeners;

% handles.TB_BGCOLOR_ORIG = handles.tbAxes1.BackgroundColor;
% handles.TB_BGCOLOR_SEL = [0 0 1];

handles.output = hObject;

colormap(handles.figure,gray);

guidata(hObject, handles);

% UIWAIT makes RigViewGUI wait for user response (see UIRESUME)
% uiwait(handles.figure);

function varargout = RigViewGUI_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

function figure_CloseRequestFcn(hObject, eventdata, handles)
delete(handles.rvObj);
delete(hObject);

% function cbkTFAxSelChanged(~,evt)
% rvObj = evt.AffectedObject;
% tf = rvObj.tfAxSel;
% gd = rvObj.gdata;
% for i = 1:3
%   if tf(i)
%     lw = 8;
%   else
%     lw = 0.5;
%   end    
%   gd.axs(i).LineWidth = lw;
% end

function cbkMovDirChanged(src,evt)
rvObj = evt.AffectedObject;
mdir = rvObj.movDir;
gd = rvObj.gdata;
gd.txMovie.String = mdir;

function cbkMovCurrFrameLRChanged(src,evt)
rvObj = evt.AffectedObject;
frmLR = rvObj.movCurrFrameLR;
frmB = (frmLR-1)/2;
frmLRstr = num2str(frmLR);
frmBstr = num2str(frmB);
gd = rvObj.gdata;
gd.etFrame.String = frmLRstr;
gd.txAxes1Hud.String = frmLRstr;
gd.txAxes2Hud.String = frmLRstr;
gd.txAxes3Hud.String = frmBstr;

function etFrame_Callback(hObject, eventdata, handles)
s = hObject.String;
val = str2double(s);
rvObj = handles.rvObj;
if ~isnan(val)
  rvObj.movSetFrameLR(val);
else
  frmLR = rvObj.movCurrFrameLR;
  hObject.String = num2str(frmLR);
end

function pbFrameDown_Callback(hObject, eventdata, handles)
handles.rvObj.movFrameDown();
function pbFrameUp_Callback(hObject, eventdata, handles)
handles.rvObj.movFrameUp();
