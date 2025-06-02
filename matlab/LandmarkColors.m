function varargout = LandmarkColors(varargin)
% LANDMARKCOLORS MATLAB code for LandmarkColors.fig
%      LANDMARKCOLORS, by itself, creates a new LANDMARKCOLORS or raises the existing
%      singleton*.
%
%      H = LANDMARKCOLORS returns the handle to a new LANDMARKCOLORS or the handle to
%      the existing singleton*.
%
%      LANDMARKCOLORS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in LANDMARKCOLORS.M with the given input arguments.
%
%      LANDMARKCOLORS('Property','Value',...) creates a new LANDMARKCOLORS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before LandmarkColors_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to LandmarkColors_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help LandmarkColors

% Last Modified by GUIDE v2.5 17-Feb-2022 15:18:29

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @LandmarkColors_OpeningFcn, ...
                   'gui_OutputFcn',  @LandmarkColors_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1}) && exist(varargin{1}),
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


function LandmarkColors_OpeningFcn(hObject, eventdata, handles, varargin)
% [tfchanges,savedinfo] = LandmarkColors(lObj,cbk)

% cbk sig: cbk(colorSpecs,markerSpecs,skelSpecs)
%   colorSpecs: array of LandmarkColorSpec objs (could be nonscalar)
%   markerSpecs: [3] struct array nested Marker/TextProps etc
%   skelSpecs: [3] struct array 

handles.output = hObject;
set(hObject,'MenuBar','None');

hObject.CloseRequestFcn = @figure_landmarkcolors_CloseRequestFcn;

lObj = varargin{1};
handles.nlandmarks = lObj.nPhysPoints;
handles.applyCbkFcn = varargin{2}; % sig:

centerOnParentFigure(hObject,lObj.hFig);


% Marker State
% This is stored only in the table. Get and Set using Set/GetMarker*.
lsetTypesCell = num2cell(enumeration('LandmarkSetType'));
ppiAll = {lObj.labelPointsPlotInfo; lObj.predPointsPlotInfo; lObj.impPointsPlotInfo};
FLDS = {'MarkerProps' 'TextProps' 'TextOffset'};
sPropsMrkr = cellfun(@(x)structrestrictflds(x,FLDS),ppiAll);
[sPropsMrkr.landmarkSetType] = deal(lsetTypesCell{:});
% Color State
% This state is maintained in handles.colorSpecs. What is shown in the
% Colors pane is per pumShowing.
handles.colorSpecs = cellfun(@(x1,x2,x3)LandmarkColorSpec(x1,x2,x3),...
  lsetTypesCell,...
  repmat({lObj.nPhysPoints},3,1),ppiAll);
% Skel State
FLDS = {'SkeletonProps'};
sPropsSkel = cellfun(@(x)structrestrictflds(x,FLDS),ppiAll);
[sPropsSkel.landmarkSetType] = deal(lsetTypesCell{:});

handles.sPropsMrkr0 = sPropsMrkr;
handles.colorSpecs0 = handles.colorSpecs.copy();
handles.sPropsSkel0 = sPropsSkel;

handles = initColorsPane(handles);
updateColorsPane(handles);

MARKERS = 'o+*.xsd^v><ph';
MARKERS = num2cell(MARKERS);
tbl = handles.tblProps;
tbl.ColumnFormat{1} = MARKERS;
strcmp(tbl.ColumnName{6},'Label Font Angle');
tbl.ColumnFormat{6} = {'normal' 'italic'};
ncol = numel(tbl.ColumnFormat);
set(tbl,'Data',cell(3,ncol),'RowName',{'Label' 'Prediction' 'Imported'});

MarkerControlsSet(handles,sPropsMrkr);

handles.sldSkeletonLineWidth.Min = -1; %log scale
handles.sldSkeletonLineWidth.Max = 5;
handles.hSldListener = addlistener(handles.sldSkeletonLineWidth,...
  'ContinuousValueChange',@(s,e)sldSkeletonLineWidth_Callback(s,[],guidata(s)));
SkelControlsSet(handles,sPropsSkel);

set(handles.figure_landmarkcolors,'Name','Landmark Cosmetics');
handles.saved = [];
guidata(hObject, handles);

handles.tblProps.CellEditCallback = @tblCellEditCallback;

% UIWAIT makes LandmarkColors wait for user response (see UIRESUME)
uiwait(handles.figure_landmarkcolors);


function tblCellEditCallback(src,evt)
handles = guidata(src);
pbApply_Callback(handles.output,[],handles);


%%%%%%%%%%
% COLORS %
%%%%%%%%%%

function handles = initColorsPane(handles)
pum = handles.pumShowing;
pum.String = { ...
  'Labels'
  'Predictions'
  'Imported'
  };
pum.Value = 1;

% Colormap pulldown options
pum = handles.popupmenu_colormap;
cmapnames = LandmarkColorSpec.CMAPNAMES;
ncmapnames = numel(cmapnames);
cmapname2idx = cell2struct(num2cell(1:ncmapnames),cmapnames,2);
set(pum,'String',LandmarkColorSpec.CMAPNAMES,'Value',1,'UserData',cmapname2idx);

handles.colormapim = imagesc(1:handles.nlandmarks,'Parent',handles.axes_colormap);
axis(handles.axes_colormap,'off');

% create buttons
w = 1/handles.nlandmarks;
borderfrac = .05;
for i = 1:handles.nlandmarks,  
  handles.hbuttons(i) = uicontrol('style','pushbutton',...
    'ForegroundColor','k',...
    'BackgroundColor',[0.5 0.5 0.5],...
    'Units','normalized',...
    'Position',[w*borderfrac+w*(i-1),borderfrac,w*(1-2*borderfrac),1-2*borderfrac],...
    'Parent',handles.uipanel_manual,...
    'Callback',@(hObject,eventdata)LandmarkColors('pushbutton_manual_Callback',hObject,eventdata,guidata(hObject),i),...
    'Tag',sprintf('pushbutton_manual_%d',i),...
    'String',num2str(i));
end

function updateColorsPane(handles)
% Update colors pane uicontrols per current colorSpecs

cs = handles.colorSpecs;
iCS = handles.pumShowing.Value;
cs = cs(iCS);

% mode
btnGrp = handles.radiobutton_manual.Parent;
if cs.tfmanual
  btnGrp.SelectedObject = handles.radiobutton_manual;
else
  btnGrp.SelectedObject = handles.radiobutton_colormap;
end

UpdateColorMode(handles);

if cs.tfmanual
  % manual button colors
  for i = 1:handles.nlandmarks,
    set(handles.hbuttons(i),'BackgroundColor',cs.colors(i,:));
  end
else
  % colormapname
  cmapname = cs.colormapname;
  pumcmap = handles.popupmenu_colormap;
  cmapidx = pumcmap.UserData.(cmapname);
  pumcmap.Value = cmapidx;

  % colormap
  colormap(handles.axes_colormap,cs.colormap);

  % brightness
  set(handles.edit_brightness,'String',num2str(cs.brightness));
  set(handles.slider_brightness,'Value',cs.brightness);
end

function UpdateColorMode(handles)
% enable/disable based on radiobutton setting
v = get(handles.radiobutton_manual,'Value');
hcm = findobj(handles.uipanel_colormap,'-property','Enable');
hm = findobj(handles.uipanel_manual,'-property','Enable');
if v == 1,
  set(hcm,'Enable','off');
  set(hm,'Enable','on');
else
  set(hcm,'Enable','on');
  set(hm,'Enable','off');
end

% function UpdateButtonColors(handles)

function pumShowing_Callback(hObject, eventdata, handles)
updateColorsPane(handles);

function popupmenu_colormap_Callback(hObject, eventdata, handles)
contents = cellstr(get(hObject,'String'));
cmapname = contents{get(hObject,'Value')};

cs = handles.colorSpecs;
iCS = handles.pumShowing.Value;
cs = cs(iCS);
cs.setColormapName(cmapname);
updateColorsPane(handles);
pbApply_Callback(handles.output,[],handles);

function slider_brightness_Callback(hObject, eventdata, handles)
v = get(hObject,'Value');
cs = handles.colorSpecs;
iCS = handles.pumShowing.Value;
cs = cs(iCS);
cs.setBrightness(v);
updateColorsPane(handles);
pbApply_Callback(handles.output,[],handles);

function edit_brightness_Callback(hObject, eventdata, handles)
v = str2double(get(hObject,'String'));
cs = handles.colorSpecs;
iCS = handles.pumShowing.Value;
cs = cs(iCS);

if isnan(v) || v < 0 || v > 1,
  warndlg('Brightness must be a number between 0 and 1','Error setting brightness','modal');
  set(hObject,'String',num2str(cs.brightness));
else
  cs.setBrightness(v);
  %set(handles.slider_brightness,'Value',v);
  updateColorsPane(handles);
  pbApply_Callback(handles.output,[],handles);
end

function radiobutton_colormap_Callback(hObject, eventdata, handles)
cs = handles.colorSpecs;
iCS = handles.pumShowing.Value;
cs = cs(iCS);
cs.setTFManual(false);
updateColorsPane(handles);

function radiobutton_manual_Callback(hObject, eventdata, handles)
cs = handles.colorSpecs;
iCS = handles.pumShowing.Value;
cs = cs(iCS);
cs.setManualColorsToColormap();
cs.setTFManual(true);
updateColorsPane(handles);

function pushbutton_manual_Callback(hObject, eventdata, handles, landmarki)
fprintf('Landmark %d\n',landmarki);
cs = handles.colorSpecs;
iCS = handles.pumShowing.Value;
cs = cs(iCS);
clr = uisetcolor(cs.colors(landmarki,:),sprintf('Landmark %d color',landmarki));
cs.setColorManual(landmarki,clr);
updateColorsPane(handles);
pbApply_Callback(handles.output,[],handles);

function pumShowing_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function slider_brightness_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

function edit_brightness_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function popupmenu_colormap_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%
% MARKERS %
%%%%%%%%%%%

function MarkerControlsSet(handles,s)
% Set tblProps per pvStructs
%
% s: as in structure returned by MarkerControlsGet

uitbl = handles.tblProps;
assert(isequal(uitbl.RowName,{'Label' 'Prediction' 'Imported'}'));

FLDS_MARKER = {'Marker' 'MarkerSize' 'LineWidth'};
tMrkr = struct2table([s.MarkerProps]');
assert(isequal(tMrkr.Properties.VariableNames,FLDS_MARKER));

FLDS_TEXT = {'Visible' 'FontSize' 'FontAngle'};
sTxt = [s.TextProps]';
sTxt = structrestrictflds(sTxt,FLDS_TEXT);
sTxt = orderfields(sTxt,FLDS_TEXT);
tTxt = struct2table(sTxt);
%assert(isequal(tTxt.Properties.VariableNames,FLDS_TEXT));
tTxt.Visible = strcmp(tTxt.Visible,'on');

txtOffset = num2cell([s.TextOffset]');

assert(isequal(uitbl.ColumnName,...
  {'Marker' 'MarkerSize' 'LineWidth' 'Show Text Label' 'Label Font Size' 'Label Font Angle' 'Label Offset'}'));

dat = [table2cell(tMrkr) table2cell(tTxt) txtOffset];
uitbl.Data = dat;

function s = MarkerControlsGet(handles)
% Read handles.tblProps into various PV-structs etc that can be used to set
% HG handles.
%
% s: structarr of PV structs
% 
% s(i).pvLine -- PV struct for line handle
% s(i).pvTxt -- PV struct for text label
% s(i).txtOffset -- text offset (in px)
% Els of s labeled by lbl, pred, imp

uitbl = handles.tblProps;
assert(isequal(uitbl.RowName,{'Label' 'Prediction' 'Imported'}'));

uitblColumnNameSanitized = regexprep(uitbl.ColumnName,' ','_'); % for eg ML19a
t = cell2table(uitbl.Data,'VariableNames',uitblColumnNameSanitized);

FLDS_MARKER = {'Marker' 'MarkerSize' 'LineWidth'};
tLine = t(:,FLDS_MARKER);
sLine = table2struct(tLine);
% s = struct();
% s.pvLine = sLine;

FLDS_TEXT = {'Show_Text_Label','Label_Font_Size' 'Label_Font_Angle'};
tTxt = t(:,FLDS_TEXT);
tTxt.Properties.VariableNames = {'Visible' 'FontSize' 'FontAngle'};
tTxt.Visible = arrayfun(@onIff,tTxt.Visible,'uni',0);
sTxt = table2struct(tTxt);
% s.pvTxt = sTxt;

txtOffset = t{:,'Label_Offset'};

s = struct(...
  'landmarkSetType',num2cell(enumeration('LandmarkSetType')),...
  'MarkerProps',num2cell(sLine),...
  'TextProps',num2cell(sTxt),...
  'TextOffset',num2cell(txtOffset) ...
  );

%%%%%%%%%%%%
% SKELETON
%%%%%%%%%%%%
function SkelControlsSet(handles,s)
s1 = s(1).SkeletonProps;
handles.pbSkeletonColor.BackgroundColor = s1.Color;
handles.sldSkeletonLineWidth.Value = log2(s1.LineWidth);

function s = SkelControlsGet(handles)
s0 = struct();
s0.Color = handles.pbSkeletonColor.BackgroundColor;
s0.LineWidth = 2^handles.sldSkeletonLineWidth.Value;
s = struct(...
  'landmarkSetType',num2cell(enumeration('LandmarkSetType')),...
  'SkeletonProps',s0);


%%%%%%%%%%%%%%%%%%
% APPLY/DONE/ETC %
%%%%%%%%%%%%%%%%%%

function handles = SaveState(handles)

tfApplyAll = handles.cbApplyAll.Value;
iCS = handles.pumShowing.Value;
if tfApplyAll
  colorSpecs = handles.colorSpecs;
  for i=1:numel(colorSpecs)
    if i==iCS
      continue;
    end
    colorSpecs(i).copyColorState(colorSpecs(iCS));
  end
  colorSpecs0 = handles.colorSpecs0;
else
  colorSpecs = handles.colorSpecs(iCS);
  colorSpecs0 = handles.colorSpecs0(iCS);
end
colorSpecs.setManualColorsToColormapIfNec();
sPropsMrkr = MarkerControlsGet(handles);
sPropsSkel = SkelControlsGet(handles);

tfclrchanged = ~arrayfun(@isequal,colorSpecs,colorSpecs0);
% We do something convoluted here as handles.sPropsMrkr0 might have
% properties that are not visible/editable in the UI
tfmkrchanged = ~arrayfun(@(x,y)isequaln(x,structoverlay(x,y)),...
  handles.sPropsMrkr0,sPropsMrkr);
tfskelchanged = ~arrayfun(@(x,y)isequaln(x,structoverlay(x,y)),...
  handles.sPropsSkel0,sPropsSkel);

%~arrayfun(@isequaln,sPropsMrkr,handles.sPropsMrkr0);
handles.saved = struct(...
  'colorSpecs',colorSpecs(tfclrchanged),...  
  'markerSpecs',sPropsMrkr(tfmkrchanged),...
  'skeletonSpecs',sPropsSkel(tfskelchanged) ...
  );
% Note either field of handles.saved could be empty

function varargout = LandmarkColors_OutputFcn(hObject, eventdata, handles) 
% [tfchanges,savedinfo] = LandmarkColors(...)

if isempty(handles),
  warning('State lost');
  varargout{1} = false;
  varargout{2} = [];
else
  varargout{1} = ~isempty(handles.saved);
  varargout{2} = handles.saved;
  delete(handles.figure_landmarkcolors);
end

function figure_landmarkcolors_CloseRequestFcn(hObject, eventdata)
uiresume(hObject);

function pbApply_Callback(hObject, eventdata, handles)
handles = SaveState(handles);
saved = handles.saved;
handles.applyCbkFcn(saved.colorSpecs,saved.markerSpecs,saved.skeletonSpecs);
guidata(hObject,handles);

function pbCancel_Callback(hObject, eventdata, handles)
uiresume(handles.figure_landmarkcolors);

function pbDone_Callback(hObject, eventdata, handles)
handles = SaveState(handles);
guidata(hObject,handles);
uiresume(handles.figure_landmarkcolors);

function sldSkeletonLineWidth_Callback(hObject, eventdata, handles)
pbApply_Callback(handles.output,[],handles);

function sldSkeletonLineWidth_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

function pbSkeletonColor_Callback(hObject, eventdata, handles)
c0 = handles.pbSkeletonColor.BackgroundColor;
handles.pbSkeletonColor.BackgroundColor = uisetcolor(c0);
pbApply_Callback(handles.output,[],handles);
