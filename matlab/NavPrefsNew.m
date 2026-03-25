function varargout = NavPrefsNew(varargin)
% NAVPREFS_EXPORT MATLAB code for NavPrefsNew.fig
%      NAVPREFS_EXPORT, by itself, creates a new NAVPREFS_EXPORT or raises the existing
%      singleton*.
%
%      H = NAVPREFS_EXPORT returns the handle to a new NAVPREFS_EXPORT or the handle to
%      the existing singleton*.
%
%      NAVPREFS_EXPORT('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in NAVPREFS_EXPORT.M with the given input arguments.
%
%      NAVPREFS_EXPORT('Property','Value',...) creates a new NAVPREFS_EXPORT or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before NavPrefsNew_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to NavPrefsNew_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help NavPrefsNew

% Last Modified by GUIDE v2.5 25-Mar-2026 18:20:26

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @NavPrefsNew_OpeningFcn, ...
                   'gui_OutputFcn',  @NavPrefsNew_OutputFcn, ...
                   'gui_LayoutFcn',  @NavPrefsNew_LayoutFcn, ...
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

function NavPrefsNew_OpeningFcn(hObject, eventdata, handles, varargin)
% NavPrefsNew(lObj)

hObject.Visible = 'off';

lObj = varargin{1};
handles.lObj = lObj;
handles.etFrameSkip.String = num2str(lObj.movieFrameStepBig);
handles.etPlaybackSpeed.String = num2str(lObj.moviePlayFPS);
handles.etPlaybackLoopDiameter.String = num2str(lObj.moviePlaySegRadius);
shiftArrowModes = enumeration('ShiftArrowMovieNavMode');
pum = handles.pumShiftArrow;
pum.String = arrayfun(@(x)x.prettyStr,shiftArrowModes,'uni',0);
pum.Value = find(lObj.movieShiftArrowNavMode==shiftArrowModes);
pum.UserData = shiftArrowModes;

et = handles.etShiftArrowTimelineThresh;
et.String = num2str(lObj.movieShiftArrowNavModeThresh);
pum = handles.pumShiftArrowTimelineThreshCmp;
idx = find(strcmp(pum.String,lObj.movieShiftArrowNavModeThreshCmp));
assert(isscalar(idx));
pum.Value = idx;
itm = lObj.infoTimelineModel;
[~,tlprop] = itm.getCurPropSmart();
handles.txTimelineProp.String = tlprop.name;

guidata(hObject, handles);

updateTimelineStatComparisonEnable(handles);

mainFigure = varargin{2};
centerOnParentFigure(hObject,mainFigure);
hObject.Visible = 'on';

uiwait(handles.figure1);

function varargout = NavPrefsNew_OutputFcn(hObject, eventdata, handles) 
delete(hObject);

function etFrameSkip_Callback(hObject, eventdata, handles)
val = str2double(hObject.String);
if isnan(val)
  hObject.String = num2str(handles.lObj.movieFrameStepBig);
end

function etPlaybackSpeed_Callback(hObject, eventdata, handles)
val = str2double(hObject.String);
if isnan(val) || val<=0
  hObject.String = num2str(handles.lObj.moviePlayFPS);
end

function etPlaybackLoopDiameter_Callback(hObject, eventdata, handles)
val = str2double(hObject.String);
if isnan(val) || val<=0
  hObject.String = num2str(handles.lObj.moviePlaySegRadius);
end

function pumShiftArrow_Callback(hObject, eventdata, handles)
updateTimelineStatComparisonEnable(handles);

function updateTimelineStatComparisonEnable(handles)
pum = handles.pumShiftArrow;
sam = pum.UserData(pum.Value);
tfTLThresh = sam==ShiftArrowMovieNavMode.NEXTTIMELINETHRESH;
onoff = onIff(tfTLThresh);

handles.txTimelineProp.Enable = onoff;
handles.pumShiftArrowTimelineThreshCmp.Enable = onoff;
handles.etShiftArrowTimelineThresh.Enable = onoff;

function etShiftArrowTimelineThresh_Callback(hObject, eventdata, handles)
val = str2double(hObject.String);
if isnan(val)
  hObject.String = num2str(handles.lObj.movieShiftArrowNavModeThresh);
end

function pbApply_Callback(hObject, eventdata, handles)
lObj = handles.lObj;
lObj.movieFrameStepBig = str2double(handles.etFrameSkip.String);
lObj.moviePlayFPS = str2double(handles.etPlaybackSpeed.String);
lObj.moviePlaySegRadius = str2double(handles.etPlaybackLoopDiameter.String);
pum = handles.pumShiftArrow;
sam = pum.UserData(pum.Value);
lObj.movieShiftArrowNavMode = sam;

if sam==ShiftArrowMovieNavMode.NEXTTIMELINETHRESH
  pum = handles.pumShiftArrowTimelineThreshCmp;
  cmp = pum.String{pum.Value};
  thresh = str2double(handles.etShiftArrowTimelineThresh.String);
  lObj.setMovieShiftArrowNavModeThresh(thresh);
  lObj.movieShiftArrowNavModeThreshCmp = cmp;  
end

close(handles.figure1);

function pbCancel_Callback(hObject, eventdata, handles)
delete(handles.figure1);

function figure1_CloseRequestFcn(hObject, eventdata, handles)
if strcmp(get(hObject,'waitstatus'),'waiting')
  uiresume(hObject);
else
  delete(hObject);
end


% --- Creates and returns a handle to the GUI figure. 
function h1 = NavPrefsNew_LayoutFcn(policy)
% policy - create a new figure or use a singleton. 'new' or 'reuse'.

persistent hsingleton;
if strcmpi(policy, 'reuse') & ishandle(hsingleton)
    h1 = hsingleton;
    return;
end
% load('NavPrefsNew.mat', 'mat') ;
mat = NavPrefsNewMat() ;

appdata = [];
appdata.GUIDEOptions = mat{1};
appdata.lastValidTag = 'figure1';
appdata.SavedVisible = mat{2};
appdata.InGUIInitialization = 1;
appdata.UsedByGUIData_m = struct(...
    'figure1', [], ...
    'FigureToolBar', [], ...
    'etShiftArrowTimelineThresh', [], ...
    'pumShiftArrowTimelineThreshCmp', [], ...
    'txTimelineProp', [], ...
    'text6', [], ...
    'etPlaybackLoopDiameter', [], ...
    'text4', [], ...
    'etPlaybackSpeed', [], ...
    'pbCancel', [], ...
    'pbApply', [], ...
    'text3', [], ...
    'text2', [], ...
    'etFrameSkip', [], ...
    'pumShiftArrow', []);
appdata.FileMenuFcnLastExportedAsType = 4;
appdata.GUIDELayoutEditor = mat{3};

h1 = figure(...
'Position',[2916 915 551 248],...
'CloseRequestFcn',@(hObject,eventdata)NavPrefsNew('figure1_CloseRequestFcn',hObject,eventdata,guidata(hObject)),...
'CurrentAxesMode','manual',...
'CurrentObjectMode','manual',...
'CurrentPointMode','manual',...
'SelectionTypeMode','manual',...
'IntegerHandle','off',...
'MenuBar','none',...
'ToolBar','none',...
'Name','Navigation Preferences',...
'NumberTitle','off',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} ,...
'Tag','figure1',...
'Resize','off',...
'ScreenPixelsPerInchMode','manual');

appdata = [];
appdata.lastValidTag = 'pumShiftArrow';

h2 = uicontrol(...
'Parent',h1,...
'FontUnits','pixels',...
'String',{  'Pop-up Menu' },...
'Style','popupmenu',...
'Position',[219 89 317 30],...
'BackgroundColor',[1 1 1],...
'Callback',@(hObject,eventdata)NavPrefsNew('pumShiftArrow_Callback',hObject,eventdata,guidata(hObject)),...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} ,...
'Tag','pumShiftArrow',...
'FontSize',13.3333333333333);

appdata = [];
appdata.lastValidTag = 'etFrameSkip';

h3 = uicontrol(...
'Parent',h1,...
'FontUnits','pixels',...
'HorizontalAlignment','left',...
'String',{  'Edit Text' },...
'Style','edit',...
'Position',[219 127 90 30],...
'BackgroundColor',[1 1 1],...
'Callback',@(hObject,eventdata)NavPrefsNew('etFrameSkip_Callback',hObject,eventdata,guidata(hObject)),...
'Tooltip','Number of frames to skip when using up/down arrows',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} ,...
'Tag','etFrameSkip',...
'FontSize',13.3333333333333);

appdata = [];
appdata.lastValidTag = 'text2';

h4 = uicontrol(...
'Parent',h1,...
'FontUnits','pixels',...
'HorizontalAlignment','right',...
'String',{  'Ctrl-left/right '; 'frame increment:' },...
'Style','text',...
'Position',[14 111 200 45],...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} ,...
'Tag','text2',...
'FontSize',16,...
'FontWeight','bold');

appdata = [];
appdata.lastValidTag = 'text3';

h5 = uicontrol(...
'Parent',h1,...
'FontUnits','pixels',...
'HorizontalAlignment','right',...
'String','Shift-left/right seeks:',...
'Style','text',...
'Position',[14 81 200 30],...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} ,...
'Tag','text3',...
'FontSize',16,...
'FontWeight','bold');

appdata = [];
appdata.lastValidTag = 'pbApply';

h6 = uicontrol(...
'Parent',h1,...
'FontUnits','pixels',...
'String','Apply',...
'Position',[174 3 100 36],...
'Callback',@(hObject,eventdata)NavPrefsNew('pbApply_Callback',hObject,eventdata,guidata(hObject)),...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} ,...
'Tag','pbApply',...
'FontSize',16,...
'FontWeight','bold');

appdata = [];
appdata.lastValidTag = 'pbCancel';

h7 = uicontrol(...
'Parent',h1,...
'FontUnits','pixels',...
'String','Cancel',...
'Position',[277 3 100 36],...
'Callback',@(hObject,eventdata)NavPrefsNew('pbCancel_Callback',hObject,eventdata,guidata(hObject)),...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} ,...
'Tag','pbCancel',...
'FontSize',16,...
'FontWeight','bold');

appdata = [];
appdata.lastValidTag = 'etPlaybackSpeed';

h8 = uicontrol(...
'Parent',h1,...
'FontUnits','pixels',...
'HorizontalAlignment','left',...
'String',{  'Edit Text' },...
'Style','edit',...
'Position',[219 204 90 30],...
'BackgroundColor',[1 1 1],...
'Callback',@(hObject,eventdata)NavPrefsNew('etPlaybackSpeed_Callback',hObject,eventdata,guidata(hObject)),...
'Tooltip','Number of frames to skip when using up/down arrows',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} ,...
'Tag','etPlaybackSpeed',...
'FontSize',13.3333333333333);

appdata = [];
appdata.lastValidTag = 'text4';

h9 = uicontrol(...
'Parent',h1,...
'FontUnits','pixels',...
'HorizontalAlignment','right',...
'String','Playback speed (fps):',...
'Style','text',...
'Position',[14 198 200 30],...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} ,...
'Tag','text4',...
'FontSize',16,...
'FontWeight','bold');

appdata = [];
appdata.lastValidTag = 'etPlaybackLoopDiameter';

h10 = uicontrol(...
'Parent',h1,...
'FontUnits','pixels',...
'HorizontalAlignment','left',...
'String',{  'Edit Text' },...
'Style','edit',...
'Position',[219 171 90 30],...
'BackgroundColor',[1 1 1],...
'Callback',@(hObject,eventdata)NavPrefsNew('etPlaybackLoopDiameter_Callback',hObject,eventdata,guidata(hObject)),...
'Tooltip','Number of frames shown during segment/loop playback',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} ,...
'Tag','etPlaybackLoopDiameter',...
'FontSize',13.3333333333333);

appdata = [];
appdata.lastValidTag = 'text6';

h11 = uicontrol(...
'Parent',h1,...
'FontUnits','pixels',...
'HorizontalAlignment','right',...
'String','Playback loop radius:',...
'Style','text',...
'Position',[14 165 200 30],...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} ,...
'Tag','text6',...
'FontSize',16,...
'FontWeight','bold');

appdata = [];
appdata.lastValidTag = 'txTimelineProp';

h12 = uicontrol(...
'Parent',h1,...
'FontUnits','pixels',...
'HorizontalAlignment','right',...
'String','dx_body_mean_abs',...
'Style','text',...
'Position',[14 48 200 30],...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} ,...
'Tag','txTimelineProp',...
'FontSize',16,...
'FontAngle','italic',...
'FontWeight','bold');

appdata = [];
appdata.lastValidTag = 'pumShiftArrowTimelineThreshCmp';

h13 = uicontrol(...
'Parent',h1,...
'FontUnits','pixels',...
'String',{  '>'; '<'; '>='; '<=' },...
'Style','popupmenu',...
'Position',[221 48 48 30],...
'BackgroundColor',[1 1 1],...
'Tooltip','Comparison operator for timeline statistic',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} ,...
'Tag','pumShiftArrowTimelineThreshCmp',...
'FontSize',16);

appdata = [];
appdata.lastValidTag = 'etShiftArrowTimelineThresh';

h14 = uicontrol(...
'Parent',h1,...
'FontUnits','pixels',...
'HorizontalAlignment','left',...
'Style','edit',...
'Position',[274 50 90 30],...
'BackgroundColor',[1 1 1],...
'Callback',@(hObject,eventdata)NavPrefsNew('etShiftArrowTimelineThresh_Callback',hObject,eventdata,guidata(hObject)),...
'Tooltip','Threshold for timeline statistics during seek',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} ,...
'Tag','etShiftArrowTimelineThresh',...
'FontSize',13.3333333333333);


hsingleton = h1;


% --- Set application data first then calling the CreateFcn. 
function local_CreateFcn(hObject, eventdata, createfcn, appdata)

if ~isempty(appdata)
   names = fieldnames(appdata);
   for i=1:length(names)
       name = char(names(i));
       setappdata(hObject, name, getfield(appdata,name));
   end
end

if ~isempty(createfcn)
   if isa(createfcn,'function_handle')
       createfcn(hObject, eventdata);
   else
       eval(createfcn);
   end
end


% --- Handles default GUIDE GUI creation and callback dispatch
function varargout = gui_mainfcn(gui_State, varargin)

gui_StateFields =  {'gui_Name'
    'gui_Singleton'
    'gui_OpeningFcn'
    'gui_OutputFcn'
    'gui_LayoutFcn'
    'gui_Callback'};
gui_Mfile = '';
for i=1:length(gui_StateFields)
    if ~isfield(gui_State, gui_StateFields{i})
        error(message('MATLAB:guide:StateFieldNotFound', gui_StateFields{ i }, gui_Mfile));
    elseif isequal(gui_StateFields{i}, 'gui_Name')
        gui_Mfile = [gui_State.(gui_StateFields{i}), '.m'];
    end
end

numargin = length(varargin);

if numargin == 0
    % NAVPREFS_EXPORT
    % create the GUI only if we are not in the process of loading it
    % already
    gui_Create = true;
elseif local_isInvokeActiveXCallback(gui_State, varargin{:})
    % NAVPREFS_EXPORT(ACTIVEX,...)
    vin{1} = gui_State.gui_Name;
    vin{2} = [get(varargin{1}.Peer, 'Tag'), '_', varargin{end}];
    vin{3} = varargin{1};
    vin{4} = varargin{end-1};
    vin{5} = guidata(varargin{1}.Peer);
    feval(vin{:});
    return;
elseif local_isInvokeHGCallback(gui_State, varargin{:})
    % NAVPREFS_EXPORT('CALLBACK',hObject,eventData,handles,...)
    gui_Create = false;
else
    % NAVPREFS_EXPORT(...)
    % create the GUI and hand varargin to the openingfcn
    gui_Create = true;
end

if ~gui_Create
    % In design time, we need to mark all components possibly created in
    % the coming callback evaluation as non-serializable. This way, they
    % will not be brought into GUIDE and not be saved in the figure file
    % when running/saving the GUI from GUIDE.
    designEval = false;
    if (numargin>1 && ishghandle(varargin{2}))
        fig = varargin{2};
        while ~isempty(fig) && ~ishghandle(fig,'figure')
            fig = get(fig,'parent');
        end
        
        designEval = isappdata(0,'CreatingGUIDEFigure') || (isscalar(fig)&&isprop(fig,'GUIDEFigure'));
    end
        
    if designEval
        beforeChildren = findall(fig);
    end
    
    % evaluate the callback now
    varargin{1} = gui_State.gui_Callback;
    if nargout
        [varargout{1:nargout}] = feval(varargin{:});
    else       
        feval(varargin{:});
    end
    
    % Set serializable of objects created in the above callback to off in
    % design time. Need to check whether figure handle is still valid in
    % case the figure is deleted during the callback dispatching.
    if designEval && ishghandle(fig)
        set(setdiff(findall(fig),beforeChildren), 'Serializable','off');
    end
else
    if gui_State.gui_Singleton
        gui_SingletonOpt = 'reuse';
    else
        gui_SingletonOpt = 'new';
    end

    % Check user passing 'visible' P/V pair first so that its value can be
    % used by oepnfig to prevent flickering
    gui_Visible = 'auto';
    gui_VisibleInput = '';
    for index=1:2:length(varargin)
        if length(varargin) == index || ~ischar(varargin{index})
            break;
        end

        % Recognize 'visible' P/V pair
        len1 = min(length('visible'),length(varargin{index}));
        len2 = min(length('off'),length(varargin{index+1}));
        if ischar(varargin{index+1}) && strncmpi(varargin{index},'visible',len1) && len2 > 1
            if strncmpi(varargin{index+1},'off',len2)
                gui_Visible = 'invisible';
                gui_VisibleInput = 'off';
            elseif strncmpi(varargin{index+1},'on',len2)
                gui_Visible = 'visible';
                gui_VisibleInput = 'on';
            end
        end
    end
    
    % Open fig file with stored settings.  Note: This executes all component
    % specific CreateFunctions with an empty HANDLES structure.

    
    % Do feval on layout code in m-file if it exists
    gui_Exported = ~isempty(gui_State.gui_LayoutFcn);
    % this application data is used to indicate the running mode of a GUIDE
    % GUI to distinguish it from the design mode of the GUI in GUIDE. it is
    % only used by actxproxy at this time.   
    setappdata(0,genvarname(['OpenGuiWhenRunning_', gui_State.gui_Name]),1);
    if gui_Exported
        gui_hFigure = feval(gui_State.gui_LayoutFcn, gui_SingletonOpt);

        % make figure invisible here so that the visibility of figure is
        % consistent in OpeningFcn in the exported GUI case
        if isempty(gui_VisibleInput)
            gui_VisibleInput = get(gui_hFigure,'Visible');
        end
        set(gui_hFigure,'Visible','off')

        % openfig (called by local_openfig below) does this for guis without
        % the LayoutFcn. Be sure to do it here so guis show up on screen.
        movegui(gui_hFigure,'onscreen');
    else
        gui_hFigure = local_openfig(gui_State.gui_Name, gui_SingletonOpt, gui_Visible);
        % If the figure has InGUIInitialization it was not completely created
        % on the last pass.  Delete this handle and try again.
        if isappdata(gui_hFigure, 'InGUIInitialization')
            delete(gui_hFigure);
            gui_hFigure = local_openfig(gui_State.gui_Name, gui_SingletonOpt, gui_Visible);
        end
    end
    if isappdata(0, genvarname(['OpenGuiWhenRunning_', gui_State.gui_Name]))
        rmappdata(0,genvarname(['OpenGuiWhenRunning_', gui_State.gui_Name]));
    end

    % Set flag to indicate starting GUI initialization
    setappdata(gui_hFigure,'InGUIInitialization',1);

    % Fetch GUIDE Application options
    gui_Options = getappdata(gui_hFigure,'GUIDEOptions');
    % Singleton setting in the GUI MATLAB code file takes priority if different
    gui_Options.singleton = gui_State.gui_Singleton;

    if ~isappdata(gui_hFigure,'GUIOnScreen')
        % Adjust background color
        if gui_Options.syscolorfig
            set(gui_hFigure,'Color', get(0,'DefaultUicontrolBackgroundColor'));
        end

        % Generate HANDLES structure and store with GUIDATA. If there is
        % user set GUI data already, keep that also.
        data = guidata(gui_hFigure);
        handles = guihandles(gui_hFigure);
        if ~isempty(handles)
            if isempty(data)
                data = handles;
            else
                names = fieldnames(handles);
                for k=1:length(names)
                    data.(char(names(k)))=handles.(char(names(k)));
                end
            end
        end
        guidata(gui_hFigure, data);
    end

    % Apply input P/V pairs other than 'visible'
    for index=1:2:length(varargin)
        if length(varargin) == index || ~ischar(varargin{index})
            break;
        end

        len1 = min(length('visible'),length(varargin{index}));
        if ~strncmpi(varargin{index},'visible',len1)
            try set(gui_hFigure, varargin{index}, varargin{index+1}), catch break, end
        end
    end

    % If handle visibility is set to 'callback', turn it on until finished
    % with OpeningFcn
    gui_HandleVisibility = get(gui_hFigure,'HandleVisibility');
    if strcmp(gui_HandleVisibility, 'callback')
        set(gui_hFigure,'HandleVisibility', 'on');
    end

    feval(gui_State.gui_OpeningFcn, gui_hFigure, [], guidata(gui_hFigure), varargin{:});

    if isscalar(gui_hFigure) && ishghandle(gui_hFigure)
        % Handle the default callbacks of predefined toolbar tools in this
        % GUI, if any
        guidemfile('restoreToolbarToolPredefinedCallback',gui_hFigure); 
        
        % Update handle visibility
        set(gui_hFigure,'HandleVisibility', gui_HandleVisibility);

        % Call openfig again to pick up the saved visibility or apply the
        % one passed in from the P/V pairs
        if ~gui_Exported
            gui_hFigure = local_openfig(gui_State.gui_Name, 'reuse',gui_Visible);
        elseif ~isempty(gui_VisibleInput)
            set(gui_hFigure,'Visible',gui_VisibleInput);
        end
        if strcmpi(get(gui_hFigure, 'Visible'), 'on')
            figure(gui_hFigure);
            
            if gui_Options.singleton
                setappdata(gui_hFigure,'GUIOnScreen', 1);
            end
        end

        % Done with GUI initialization
        if isappdata(gui_hFigure,'InGUIInitialization')
            rmappdata(gui_hFigure,'InGUIInitialization');
        end

        % If handle visibility is set to 'callback', turn it on until
        % finished with OutputFcn
        gui_HandleVisibility = get(gui_hFigure,'HandleVisibility');
        if strcmp(gui_HandleVisibility, 'callback')
            set(gui_hFigure,'HandleVisibility', 'on');
        end
        gui_Handles = guidata(gui_hFigure);
    else
        gui_Handles = [];
    end

    if nargout
        [varargout{1:nargout}] = feval(gui_State.gui_OutputFcn, gui_hFigure, [], gui_Handles);
    else
        feval(gui_State.gui_OutputFcn, gui_hFigure, [], gui_Handles);
    end

    if isscalar(gui_hFigure) && ishghandle(gui_hFigure)
        set(gui_hFigure,'HandleVisibility', gui_HandleVisibility);
    end
end

function gui_hFigure = local_openfig(name, singleton, visible)

% openfig with three arguments was new from R13. Try to call that first, if
% failed, try the old openfig.
if nargin('openfig') == 2
    % OPENFIG did not accept 3rd input argument until R13,
    % toggle default figure visible to prevent the figure
    % from showing up too soon.
    gui_OldDefaultVisible = get(0,'defaultFigureVisible');
    set(0,'defaultFigureVisible','off');
    gui_hFigure = matlab.hg.internal.openfigLegacy(name, singleton);
    set(0,'defaultFigureVisible',gui_OldDefaultVisible);
else
    % Call version of openfig that accepts 'auto' option"
    gui_hFigure = matlab.hg.internal.openfigLegacy(name, singleton, visible);  
%     %workaround for CreateFcn not called to create ActiveX
%         peers=findobj(findall(allchild(gui_hFigure)),'type','uicontrol','style','text');    
%         for i=1:length(peers)
%             if isappdata(peers(i),'Control')
%                 actxproxy(peers(i));
%             end            
%         end
end

function result = local_isInvokeActiveXCallback(gui_State, varargin)

try
    result = ispc && iscom(varargin{1}) ...
             && isequal(varargin{1},gcbo);
catch
    result = false;
end

function result = local_isInvokeHGCallback(gui_State, varargin)

try
    fhandle = functions(gui_State.gui_Callback);
    result = ~isempty(findstr(gui_State.gui_Name,fhandle.file)) || ...
             (ischar(varargin{1}) ...
             && isequal(ishghandle(varargin{2}), 1) ...
             && (~isempty(strfind(varargin{1},[get(varargin{2}, 'Tag'), '_'])) || ...
                ~isempty(strfind(varargin{1}, '_CreateFcn'))) );
catch
    result = false;
end


