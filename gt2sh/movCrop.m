function varargout = movCrop(varargin)
% MOVCROP MATLAB code for movCrop.fig
%      MOVCROP, by itself, creates a new MOVCROP or raises the existing
%      singleton*.
%
%      H = MOVCROP returns the handle to a new MOVCROP or the handle to
%      the existing singleton*.
%
%      MOVCROP('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MOVCROP.M with the given input arguments.
%
%      MOVCROP('Property','Value',...) creates a new MOVCROP or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before movCrop_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to movCrop_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help movCrop

% Last Modified by GUIDE v2.5 26-Apr-2018 11:28:19

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @movCrop_OpeningFcn, ...
                   'gui_OutputFcn',  @movCrop_OutputFcn, ...
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

function movCrop_OpeningFcn(hObject, eventdata, handles, varargin)
% movCrop(tblFlyMov,Is,cpts)
% tblFlyMov: table with 'fly' (int), 'mov1', 'mov2' (cellstrs)

tbl = varargin{1};
I1 = varargin{2};
n = height(tbl);
szassert(I1,[n 2]);
if numel(varargin)>=3
  cpts = varargin{3};
else
  cpts = nan(n,2,2); % row,{x,y},vw
end
szassert(cpts,[n 2 2]);

handles.tbl = tbl;
handles.I1 = I1;
handles.n = n;
axs = [handles.axes1 handles.axes2];
handles.axs = axs;
ims = gobjects(1,2);
for iax=1:numel(axs)
  ims(iax) = imagesc(nan,'parent',axs(iax));
  set(ims(iax),'PickableParts','none');
  colormap(axs(iax),'gray');
  hold(axs(iax),'on');
  axis(axs(iax),'image');
  set(axs(iax),'linewidth',2,'XColor',[0.6 0.6 0.6],'YColor',[0.6 0.6 0.6]);
end
handles.ims = ims;
handles.hPt = [ plot(handles.axs(1),nan,nan,'r.','markersize',20,'pickableparts','none') ...
                plot(handles.axs(2),nan,nan,'r.','markersize',20,'pickableparts','none') ];
set(handles.axs,'ButtonDownFcn',@cbkAxBDF);

handles.clickpts = cpts;

handles.txMov1.FontSize = 10;
handles.txMov2.FontSize = 10;
handles.txMov = [handles.txMov1 handles.txMov2];
handles.output = hObject;

guidata(hObject, handles);

lclSetRow(handles,1);

% UIWAIT makes movCrop wait for user response (see UIRESUME)
% uiwait(handles.figure1);

function varargout = movCrop_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;


function lclSetRow(handles,i)
trow = handles.tbl(i,:);
handles.txRow.String = sprintf('%d/%d',i,handles.n);
handles.txFly.String = num2str(trow.fly);
movs = [trow.mov1 trow.mov2];
I1row = handles.I1(i,:);
for ivw=1:2
  set(handles.ims(ivw),'CData',I1row{ivw});
  axis(handles.axs(ivw),'image');
  try    
    [~,~,handles.txMov(ivw).String] = parseSHfullmovie(movs{ivw},'noID',true);
  catch ME
    fprintf(2,'Couldn''t parse movie %s.\n',movs{ivw});
    handles.txMov(ivw).String = movs{ivw};
  end
end

handles.i = i;
guidata(handles.figure1,handles);
lclUpdateHPts(handles);

function cbkAxBDF(src,edata)
handles = guidata(src);
pos = get(src,'CurrentPoint');
pos = pos(1,1:2);
iView = find(src==handles.axs);
fly = handles.tbl.fly(handles.i);
ifly = find(handles.tbl.fly==fly);
if handles.i==ifly(1) && all(isnan(handles.clickpts(handles.i,:,iView)))
  % Labeling first mov for this fly. 
  handles.clickpts(ifly,:,iView) = repmat(pos,numel(ifly),1);  
  fprintf(1,'Wrote %d frames for fly %d\n',numel(ifly),fly);
else
  handles.clickpts(handles.i,:,iView) = pos;
end

guidata(src,handles);
lclUpdateHPts(handles);

function lclUpdateHPts(handles)
xy = squeeze(handles.clickpts(handles.i,:,:)); % 2x2, {x,y},iVw
for iVw=1:2
  set(handles.hPt(iVw),'XData',xy(1,iVw),'YData',xy(2,iVw));
end

function pbLeft_Callback(hObject, eventdata, handles)
i = max(1,handles.i-1);
if i~=handles.i
  lclSetRow(handles,i);
end
function pbRight_Callback(hObject, eventdata, handles)
i = min(handles.n,handles.i+1);
if i~=handles.i
  lclSetRow(handles,i);
end
