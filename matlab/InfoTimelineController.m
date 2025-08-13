classdef InfoTimelineController < handle

  properties (Constant)
    TLPROPFILESTR = 'landmark_features.yaml';
  end
   
  properties (SetAccess=private)
    TLPROPS  % struct array, features we can compute. Initted from yaml at construction-time
    TLPROPS_TRACKER  % struct array, features for current tracker. Initted at setTracker time
  end
  
  properties
    parent_  % scalar LabelerController handle
    lObj % scalar Labeler handle
    hAx % scalar handle to manual timeline axis
    hAxL % scalar handle to is-labeled timeline axis
    hCurrFrame % scalar line handle current frame
    hCurrFrameL % scalar line handle current frame
    hStatThresh % scalar line handle, threshold
    hCMenuClearAll % scalar context menu
    hCMenuClearBout % scalar context menu

    hPts % [npts] line handles
    hPtStat % scalar line handle
    npts % number of label points in current movie/timeline
    tldata % [nptsxnfrm] most recent data set/shown in setLabelsFull. this is NOT y-normalized
    hPtsL % [npts] patch handles (non-MA projs), or [1] image handle (MA projs)
    axLmaxntgt = 3 % applies to hAxL for MA projs; number of tgts to display
    custom_data % [1 x nframes] custom data to plot
    
    listeners % [nlistener] col cell array of labeler prop listeners
    
    color = [1,1,1]  % color when there is only one statistic for all landmarks
  end


  properties
    curprop % row index into props, or props_tracker, depending on curproptype.
    curproptype % row index into proptypes
    isdefault = true % whether this has been changed
  end

  
  %% Select
  properties (SetAccess=private)
    hSelIm % scalar image handle for selection
    isinit
  end
  
  %% GT/highlighting
  properties 
    hSegLineGT % scalar SegmentedLine
    hSegLineGTLbled % scalar SegmentedLine
  end
  
  %%
  properties (Dependent)
    prefs % projPrefs.InfoTimelines preferences substruct
    nfrm
  end
  
  events
  end
    
  methods
    function v = get.prefs(obj)
      v = obj.lObj.projPrefs.InfoTimelines;
    end

    function v = get.nfrm(obj)
      lblObj = obj.lObj;
      if lblObj.hasMovie
        v = lblObj.nframes;
      else
        v = 1;
      end
    end
  end
  
  methods
    
    function obj = InfoTimelineController(parent)
      % parent a LabelerController

      axtm = parent.axes_timeline_manual ;
      axti = parent.axes_timeline_islabeled ;

      obj.parent_ = parent ;
      labeler = parent.labeler_ ;
      obj.lObj = labeler ;
      axtm.Color = [0 0 0];
      axtm.ButtonDownFcn = @(src,evt)obj.cbkBDF(src,evt);
      hold(axtm,'on');
      obj.hAx = axtm;
      obj.hCurrFrame = plot(axtm,[nan nan],[0 1],'-','Color',[1 1 1],...
        'hittest','off','Tag','InfoTimeline_CurrFrame');
      obj.hStatThresh = plot(axtm,[nan nan],[0 0],'-','Color',[1 1 1],...
        'hittest','off','visible','off','Tag','InfoTimeline_StatThresh');
      
%       obj.hMarked = plot(ax,[nan nan],[nan nan],'-','Color',[1 1 0],'hittest','off');

      if ~isempty(axti) && ishandle(axti),
        axti.Color = [0 0 0];
        axti.ButtonDownFcn = @(src,evt)obj.cbkBDF(src,evt);
        hold(axti,'on');
      end
      obj.hAxL = axti;
      
      if obj.isL,
        obj.hCurrFrameL = plot(axti,[nan nan],[0 1],'-','Color',[1 1 1],'hittest','off','Tag','InfoTimeline_CurrFrameLabel');
      else
        obj.hCurrFrameL = [];
      end

      % None of this below seemed to actually do anything, presumably b/c the
      % ButtonDownFcn for axtm and atxi are set above, and that overrides them.
      % (Not 100% clear why ButtonDownFcn would override the scroll wheel, but
      % empirically seems to be the case.)  The setAxesZoomMotion() was erroring
      % out on r2025a, which prompted this whole investigation.  -- ALT, 2025-06-19
      %
      % fig = axtm.Parent ;  % the main figure
      % hZ = zoom(fig);
      % setAxesZoomMotion(hZ,axtm,'vertical');  % Make zooming gesture (scroll wheel) only zoom the timeline axes in y
      % hZ.ActionPostCallback = @(src,evt) obj.cbkPostZoom(src,evt);
      % hP = pan(fig);
      % setAxesPanMotion(hP,axtm,'vertical');  % Make pannning gesture (drag) only pan the timeline axes in y
      % hP.ActionPostCallback = @(src,evt) obj.cbkPostZoom(src,evt);
      % 
      % if obj.isL,
      %   setAxesZoomMotion(hZ,axti,'horizontal');
      %   setAxesPanMotion(hP,axti,'horizontal');
      % end
      
      obj.hPts = [];
      obj.hPtStat = [];
      obj.hPtsL = [];
      obj.npts = nan;
      %Rxobj.nfrm = nan;
            
      listeners = cell(0,1);
      listeners{end+1,1} = ...
        addlistener(labeler, 'didSetLabels', @obj.cbkLabelUpdated) ;
      % listeners{end+1,1} = ...
      %   addlistener(labeler, 'gtIsGTModeChanged',@obj.cbkGTIsGTModeUpdated) ;
      % listeners{end+1,1} = ...
      %   addlistener(labeler, 'gtSuggUpdated',@obj.cbkGTSuggUpdated) ;
      listeners{end+1,1} = ...
        addlistener(labeler, 'gtSuggMFTableLbledUpdated',@obj.cbkGTSuggMFTableLbledUpdated) ;
      listeners{end+1,1} = ...
          addlistener(labeler, 'newTrackingResults', @obj.cbkNewTrackingResults) ;      
      obj.listeners = listeners;      
    
      obj.TLPROPS_TRACKER = EmptyLandmarkFeatureArray();
      obj.readTimelinePropsNew();
      obj.initializePropsAllFrames();
            
      obj.updateProps();

      obj.curprop = 1;
      obj.curproptype = 1;
      obj.isdefault = true;
      
      obj.isinit = true;
      obj.hSelIm = [];
      % obj.hSegLineGT = SegmentedLine(axtm,'InfoTimeline_SegLineGT');
      obj.hSegLineGT = line('XData',nan,'YData',nan,'Parent',axtm,'Tag','InfoTimeline_SegLineGT');
      % obj.hSegLineGTLbled = SegmentedLine(axtm,'InfoTimeline_SegLineGTLbled');
      obj.hSegLineGTLbled = line('XData',nan,'YData',nan,'Parent',axtm,'Tag','InfoTimeline_SegLineGTLbled');
      obj.isinit = false;
      
      hCMenu = uicontextmenu('parent',axtm.Parent,...
        'callback',@(src,evt)obj.cbkContextMenu(src,evt),...
        'UserData',struct('bouts',nan(0,2)),...
        'Tag','InfoTimeline_ContextMenu');
      uimenu('Parent',hCMenu,'Label','Set number of frames shown',...
        'Callback',@(src,evt)obj.cbkSetNumFramesShown(src,evt),...
        'Tag','menu_InfoTimeline_SetNumFramesShown');
      obj.hCMenuClearAll = uimenu('Parent',hCMenu,...
        'Label','Clear selection (N bouts)',...
        'UserData',struct('LabelPat','Clear selection (%d bouts)'),...
        'Callback',@(src,evt)obj.selectClearSelection(),...
        'Tag','menu_InfoTimeline_selectClearSelection');
      obj.hCMenuClearBout = uimenu('Parent',hCMenu,...
        'Label','Clear bout (frame M--N)',...
        'UserData',struct('LabelPat','Clear bout (frame %d-%d)','iBout',nan),...
        'Callback',@(src,evt)obj.cbkClearBout(src,evt),...
        'Tag','menu_InfoTimeline_ClearBout');
      uimenu('Parent',hCMenu,'Label','Toggle statistic threshold visibility',...
        'Callback',@(src,evt)obj.cbkToggleThresholdViz(src,evt),...
        'Tag','menu_InfoTimeline_ToggleThresholdViz');      
      axtm.UIContextMenu = hCMenu;
            
      if obj.isL,
%         hCMenuL = uicontextmenu('parent',axl.Parent);
%         uimenu('Parent',hCMenu,'Label','Set number of frames shown',...
%           'Callback',@(src,evt)obj.cbkSetNumFramesShown(src,evt));
%         obj.hCMenuClearAll(end+1) = uimenu('Parent',hCMenuL,...
%           'Label','Clear selection (N bouts)',...
%           'UserData',struct('LabelPat','Clear selection (%d bouts)'),...
%           'Callback',@(src,evt)obj.selectClearSelection());
%         obj.hCMenuClearBout(end+1) = uimenu('Parent',hCMenuL,...
%           'Label','Clear bout (frame M--N)',...
%           'UserData',struct('LabelPat','Clear bout (frame %d-%d)','iBout',nan),...
%           'Callback',@(src,evt)obj.cbkClearBout(src,evt));
%         hq = uimenu('Parent',hCMenuL,'Label','What''s this?');
%         uimenu('Parent',hq,'Label','Timeline showing which frames have been labeled');
        axti.UIContextMenu = hCMenu;
      end
      
    end
    
    function delete(obj)
      deleteValidGraphicsHandles([obj.hCurrFrame,obj.hCurrFrameL,obj.hStatThresh]);
      obj.hCurrFrame = [];
      obj.hCurrFrameL = [];
      obj.hStatThresh = [];
      % if ~isempty(obj.hZoom)
      %   delete(obj.hZoom);
      % end
      % if ~isempty(obj.hPan)
      %   delete(obj.hPan);
      % end
      deleteValidGraphicsHandles(obj.hPts);
      deleteValidGraphicsHandles(obj.hPtStat);
      obj.hPts = [];
      obj.hPtStat = [];
      deleteValidGraphicsHandles(obj.hPtsL);
      obj.hPtsL = [];
      if ~isempty(obj.listeners),
        cellfun(@delete,obj.listeners);
      end
      obj.listeners = [];
      deleteValidGraphicsHandles(obj.hSelIm);
      obj.hSelIm = [];
      deleteValidGraphicsHandles(obj.hSegLineGT);
      obj.hSegLineGT = [];
      deleteValidGraphicsHandles(obj.hSegLineGTLbled);
      obj.hSegLineGTLbled = [];
    end
    
  end  
  
  methods
    
    function readTimelinePropsNew(obj)
      path = fullfile(APT.Root, 'matlab') ;
      tlpropfile = fullfile(path,obj.TLPROPFILESTR);
      assert(logical(exist(tlpropfile,'file')), 'File %s is missing', tlpropfile);      
      obj.TLPROPS = ReadLandmarkFeatureFile(tlpropfile);      
    end
    
    function initializePropsAllFrames(obj)
      
      obj.lObj.infoTimelineModel.props_allframes = struct('name','Add custom...',...
        'code','add_custom',...
        'file','');
      
    end
    
    function initializePropsTracker(obj)
      obj.lObj.infoTimelineModel.props_tracker = cat(1,obj.lObj.infoTimelineModel.props,obj.TLPROPS_TRACKER);      
    end
    
    function initNewProject(obj)
      obj.npts = obj.lObj.nLabelPoints;

      deleteValidGraphicsHandles(obj.hPts);
      deleteValidGraphicsHandles(obj.hPtStat);
      deleteValidGraphicsHandles(obj.hPtsL);
      obj.hPts = gobjects(obj.npts,1);
      obj.hPtStat = gobjects(1);
      colors = obj.lObj.LabelPointColors;
      ax = obj.hAx;
      axl = obj.hAxL;
      for i=1:obj.npts
        obj.hPts(i) = plot(ax,nan,i,'.','linestyle','-','Color',colors(i,:),...
          'hittest','off','Tag',sprintf('InfoTimeline_Pt%d',i));
      end
      isMA = obj.lObj.maIsMA;
      if isMA
        obj.hPtsL = gobjects(1,1);
      else
        obj.hPtsL = gobjects(obj.npts,1);        
      end
      if obj.isL
        if isMA
          obj.hPtsL = image(nan,'parent',axl,'hittest','off','tag','InfoTimeline_Label_ma');
        else
          for i=1:obj.npts
            obj.hPtsL(i) = patch(axl,nan(1,5),i-1+[0,1,1,0,0],colors(i,:),...
              'EdgeColor','none','hittest','off','Tag',sprintf('InfoTimeline_Label_%d',i));
          end
        end
      end
      
      obj.hPtStat = plot(ax,nan,i,'.-','Color',obj.color,'hittest','off','LineWidth',2,'Tag','InfoTimeline_Stat');
      
      prefsTL = obj.prefs;
      ax.XColor = prefsTL.XColor;
      dy = .01;
      ax.YLim = [0-dy 1+dy];
      if ishandle(obj.hSelIm)
        obj.hSelIm.YData = ax.YLim;
      end
      if obj.isL
        if isMA
          axl.YLim = [0-dy obj.axLmaxntgt+dy];
          colormap(axl,[0 0 0;0 0 1]);
          axis(axl,'ij');
        else
          axl.YLim = [0-dy obj.npts+dy];
          axis(axl,'xy');
        end
      end
      
      set(obj.hCurrFrame,'XData',[nan nan],'ZData',[1 1]);
      set(obj.hCurrFrameL,'XData',[nan nan],'YData',[0,obj.npts],'ZData',[1 1]);
      set(obj.hStatThresh,'XData',[nan nan],'ZData',[1 1]);
      linkaxes([obj.hAx,obj.hAxL],'x');
    end
    
    function initNewMovie(obj)
%       if obj.lObj.hasMovie
%         obj.nfrm = obj.lObj.nframes;
%       else
%         obj.nfrm = 1;
%       end
      ax = obj.hAx;
      prefsTL = obj.prefs;
      ax.XTick = 0:prefsTL.dXTick:obj.nfrm;

      obj.selectInit();
      
      xlims = [1 obj.nfrm];
      sPV = struct('LineWidth',5,'Color',AxesHighlightManager.ORANGE);
      sPVLbled = struct('LineWidth',5,'Color',AxesHighlightManager.ORANGE/2);
      initSegmentedLineBang(obj.hSegLineGT,xlims,sPV);
      initSegmentedLineBang(obj.hSegLineGTLbled,xlims,sPVLbled);
      obj.custom_data = [];
      if obj.getCurPropTypeIsAllFrames(),
        obj.setCurPropTypeDefault();
      end
      
      obj.updateProps();
        
      cbkGTSuggUpdated(obj,[],[]);
    end
    
    function updateProps(obj)
      % Set .props, .props_tracker from .TLPROPS, .TLPROPS_TRACKER
      
      % remove body features if no body tracking
      props = obj.TLPROPS;
      if ~obj.lObj.hasTrx,
        idxremove = strcmpi({props.coordsystem},'Body');
        props(idxremove) = [];
      end
      obj.lObj.infoTimelineModel.props = props;      
      obj.initializePropsTracker();
      obj.initializePropsAllFrames();
    end
        
    function didChangeCurrentTracker(obj)
      tracker = obj.lObj.tracker ;
      
      % Set .proptypes, .props_tracker
      if isempty(tracker),
        % AL: Probably obsolete codepath
        obj.lObj.infoTimelineModel.proptypes(strcmpi(obj.lObj.infoTimelineModel.proptypes,'Predictions')) = [];
        obj.lObj.infoTimelineModel.props_tracker = [];
      else
        if ~ismember('Predictions',obj.lObj.infoTimelineModel.proptypes),
          obj.lObj.infoTimelineModel.proptypes{end+1} = 'Predictions';
        end
        obj.TLPROPS_TRACKER = tracker.propList(); %#ok<*PROPLC>
        obj.initializePropsTracker();
      end
      
      obj.enforcePropConsistencyWithUI(false);
      
      obj.setLabelsFull();
    end
    
    function setLabelsFull(obj,runinit)
      % Get data and set .hPts, .hMarked
      
      if nargin < 2,
        runinit = false;
      end
      
      if isnan(obj.npts) || (~runinit && obj.lObj.isinit), return; end
      
      dat = obj.getDataCurrMovTgt(); % [nptsxnfrm]
      dat(isinf(dat)) = nan;
      datnonnan = dat(~isnan(dat));

      obj.tldata = dat;

      %for i=1:obj.npts
        set(obj.hPts,'XData',nan,'YData',nan);
      %end
      set(obj.hPtStat,'XData',nan,'YData',nan);
      
      if ~isempty(datnonnan)
%         set(obj.hMarked,'XData',nan,'YData',nan);
        
        y1 = min(datnonnan(:));
        y2 = max(datnonnan(:));
        if y1 == y2,
          if y1==0
            y1 = -eps;
            y2 = eps;
          else
            % y1, y2 potentially negative
            y1 = y1-abs(y1)*eps;
            y2 = y2+abs(y2)*eps;
          end
        end
        %dy = max(y2-y1,eps);
        %lposNorm = (dat-y1)/dy; % Either nan, or in [0,1]
        x = 1:size(dat,2);
        if ishandle(obj.hSelIm),
          set(obj.hSelIm,'YData',[y1,y2]);
        end
        
        set(obj.hAx,'YLim',[y1,y2]);
        set(obj.hCurrFrame,'YData',[y1,y2]);
        if size(dat,1) == obj.npts,
          for i=1:obj.npts
            set(obj.hPts(i),'XData',x,'YData',dat(i,:));
          end
        elseif size(dat,1) == 1,
          set(obj.hPtStat,'XData',x,'YData',dat(1,:));
        else
          warningNoTrace(sprintf('InfoTimeline: Number of rows in statistics was %d, expected either %d or 1',size(dat,1),obj.npts));
        end
        
        set(obj.hStatThresh,'XData',x([1 end]));
      end
      
      if obj.isL,
        if obj.lObj.maIsMA
          tflbledDisp = obj.getlabeledTgts();
          set(obj.hPtsL,'CData',uint8(tflbledDisp'));          
        else
          islabeled = obj.getIsLabeledCurrMovTgt(); % [nptsxnfrm]
          for i = 1:obj.npts,
            if any(islabeled(i,:)),
              [t0s,t1s] = get_interval_ends(islabeled(i,:));
              nbouts = numel(t0s);
              t0s = t0s(:)'-.5; t1s = t1s(:)'-.5;
              xd = [t0s;t0s;t1s;t1s;t0s];
              yd = i-1+repmat([0;1;1;0;0],[1,nbouts]);
            else
              xd = nan;
              yd = nan;
            end
            set(obj.hPtsL(i),'XData',xd,'YData',yd);
          end
        end
      end
      
%       markedFrms = find(any(obj.getMarkedDataCurrMovTgt(),1));
%       xxm = repmat(markedFrms,[3 1]);
%       xxm = xxm(:)+0.05; 
      % slightly off so that both current frame and labeled frame are both
      % visible.
%       yym = repmat([0 1 nan],[1 size(markedFrms,2)]);
%       set(obj.hMarked,'XData',xxm(:),'YData',yym(:));
    end
    
    function setLabelsFrame(obj,frm) %#ok<INUSD>
      % frm: [n] frame indices. Optional. If not supplied, defaults to
      % labeler.currFrame
      
      % AL20170616: Originally, timeline was not intended to listen
      % directly to Labeler.labeledpos etc; instead, notification of change
      % in labels was done by piggy-backing on Labeler.updateFrameTable*
      % (which explicitly calls this method). However, obj is now listening 
      % directly to lObj.labeledpos so this method is obsolete. Leave stub 
      % here in case need to go back to piggy-backing on
      % .updateFrameTable* eg for performance reasons.
            
%       lpos = obj.getDataCurrMovTgt();
%       for i=1:obj.npts
%         h = obj.hPts(i);
%         set(h,'XData',1:size(lpos,2),'YData',lpos(i,:));
%       end
    end
    
    function newFrame(obj,frm)
      % Respond to new .lObj.currFrame
      % This gets called after the user changes the frame they're looking at.
      
      if isnan(obj.npts), return; end
            
      nominal_xspan = 2*obj.prefs.FrameRadius;
      nominal_dxtick = obj.prefs.dXTick ;
      if nominal_xspan==0 || nominal_xspan > obj.nfrm
        x0 = 1;
        x1 = obj.nfrm;
        xspan = x1-x0 ;
      else
        xspan = nominal_xspan ;
        r = xspan/2 ;
        x0_raw = frm-r;
        x1_raw = frm+r; %min(frm+r,obj.nfrm);
        % Make sure the limits don't run off the end
        if x0_raw<1
          x0 = 1 ;
          x1 = 1 + 2*r ;
        elseif x1_raw>obj.nfrm
          x1 = obj.nfrm ;
          x0 = x1 - 2*r ;
        else
          x0 = x0_raw ;
          x1 = x1_raw ;
        end
      end
      if xspan/nominal_dxtick > 20 ,
        dxtick = apt.heuristic_dxtick_from_xspan(xspan) ;
      else
        dxtick = nominal_dxtick ;
      end
      obj.hAx.XTick = 0 : dxtick : obj.nfrm ;
      obj.hAx.XLim = [x0 x1];
      set(obj.hCurrFrame,'XData',[frm frm]);
      if obj.isL,
        obj.hAxL.XLim = [x0 x1];
        set(obj.hCurrFrameL,'XData',[frm frm]);
      end
      
      if obj.lObj.infoTimelineModel.selectOn
        f0 = obj.lObj.infoTimelineModel.selectOnStartFrm;
        f1 = frm;
        if f1>f0
          idx = f0:f1;
        else
          idx = f1:f0;
        end
        obj.hSelIm.CData(:,idx) = 1;
      end
    end
    
    function newTarget(obj)
      obj.setLabelsFull();
    end
    
    function updateLandmarkColors(obj)
      tflbl = obj.getCurPropTypeIsLabel();
      lblcolors = obj.lObj.LabelPointColors();
      if tflbl
        ptclrs = lblcolors;
      else
        ptclrs = obj.lObj.PredictPointColors();
      end
      for i=1:obj.npts
        set(obj.hPts(i),'Color',ptclrs(i,:));
      end
      if obj.isL && ~obj.lObj.maIsMA
        for i=1:obj.npts
          set(obj.hPtsL(i),'FaceColor',lblcolors(i,:));
        end
      end
    end
    
    function selectInit(obj)
      if obj.lObj.isinit || isnan(obj.nfrm), return; end

      deleteValidGraphicsHandles(obj.hSelIm);
      obj.hSelIm = image(1:obj.nfrm,obj.hAx.YLim,uint8(zeros(1,obj.nfrm)),...
        'parent',obj.hAx,'HitTest','off',...
        'CDataMapping','direct');

      obj.lObj.infoTimelineModel.selectInit();
      colorTBSelect = obj.parent_.tbTLSelectMode.BackgroundColor;
      colormap(obj.hAx,[0 0 0;colorTBSelect]);
      
      obj.setLabelerSelectedFrames();
    end

    function bouts = selectGetSelection(obj)
      % Get currently selected bouts (can be noncontiguous)
      %
      % bouts: [nBout x 2]. col1 is startframe, col2 is one-past-endframe

      cdata = obj.hSelIm.CData;
      [sp,ep] = get_interval_ends(cdata);
      bouts = [sp(:) ep(:)];
    end
    
    function selectClearSelection(obj)
      obj.selectInit();
    end
    
    function setStatThresh(obj,th)
      obj.hStatThresh.YData = [th th];
    end
    
    function setStatThreshViz(obj,tfshow)
      % show stat threshold and y-axis labels/ticks
      onoff = onIff(tfshow);
      obj.hStatThresh.Visible = onoff;
      if tfshow
        obj.hAx.YColor = obj.hAx.XColor;
      else
        obj.hAx.YColor = [0.15 0.15 0.15];
      end
    end
    
    function v = isL(obj)
      % Returns true if obj.hAxL (which hold the is-labeled timeline axes handle)
      % points to a valid graphics object.
      v = ~isempty(obj.hAxL) && ishandle(obj.hAxL);
    end
    
  end
  
  methods %getters setters
    function enforcePropConsistencyWithUI(obj, tfSetLabelsFull)
      % Checks that .curprop is in range for current .props,
      % .props_tracker, .curproptype. 
      %
      % Theoretically this check is necessary whenever .curprop, .props,
      % .props_tracker, .curproptype change.
      %
      % If it is not, it resets .curprop, resets obj.parent_.pumInfo.Value,
      % and optionally calls setLabelsFull (only optional to avoid
      % redundant/dup calls near callsite).

      ptype = obj.lObj.infoTimelineModel.proptypes{obj.curproptype};
      switch ptype
        case 'Predictions'
          tfOOB = obj.curprop > numel(obj.lObj.infoTimelineModel.props_tracker);
        otherwise
          tfOOB = obj.curprop > numel(obj.lObj.infoTimelineModel.props);
      end
      
      if tfOOB
        NEWPROP = 1;
        obj.curprop = NEWPROP;
        obj.parent_.pumInfo.Value = NEWPROP;
      end
      
      if tfSetLabelsFull
        obj.setLabelsFull();
      end
    end
    function props = getPropsDisp(obj,ipropType)
      % Get available properties for given propType (idx)
      if nargin < 2,
        ipropType = obj.curproptype;
      end
      if strcmpi(obj.lObj.infoTimelineModel.proptypes{ipropType},'Predictions'),
        props = {obj.lObj.infoTimelineModel.props_tracker.name};
      elseif strcmpi(obj.lObj.infoTimelineModel.proptypes{ipropType},'All Frames'),
        props = {obj.lObj.infoTimelineModel.props_allframes.name};
      else
        props = {obj.lObj.infoTimelineModel.props.name};
      end
    end
    function proptypes = getPropTypesDisp(obj)
      proptypes = obj.lObj.infoTimelineModel.proptypes;
    end
    function tfSucc = setCurProp(obj,iprop)
      % setLabelsFull will essentially assert that iprop is in range for
      % current proptype.
      %
      % Does not update UI
      tfSucc = true;
      if obj.getCurPropTypeIsAllFrames() && ...
          strcmpi(obj.lObj.infoTimelineModel.props_allframes(iprop).name,'Add custom...'),
        [tfSucc] = obj.addCustomFeature();
        if ~tfSucc,
          return;
        end
      else
        obj.curprop = iprop;
      end
      obj.setLabelsFull();
      obj.isdefault = false;
    end
    function v = getCurProp(obj)
      v = obj.curprop;
    end
    function setCurPropType(obj,iproptype,iprop)
      % iproptype, iprop assumed to be consistent already.
      obj.curproptype = iproptype;
      if nargin >= 3 && iprop ~= obj.curprop,
        obj.curprop = iprop;
      end
      obj.setLabelsFull();
      obj.updateLandmarkColors();
    end
    function tfSucc = addCustomFeature(obj)
      tfSucc = false;
      movfile = obj.lObj.getMovieFilesAllFullMovIdx(obj.lObj.currMovIdx);
      defaultpath = fileparts(movfile{1});
      [f,p] = uigetfile('*.mat','Select .mat file with a feature value for each frame for current movie',defaultpath);
      if ~ischar(f),
        return;
      end
      file = fullfile(p,f);
      try
        d = load(fullfile(p,f),'x');
        obj.custom_data = d.x;
      catch,
        uiwait(errordlg('Custom feature mat file must have a variable x which is 1 x nframes','Error loading custom feature'));
        return;
      end
      
      newprop = struct('name',['Custom: ',f],'code','custom','file',file);
      obj.initializePropsAllFrames();
      obj.lObj.infoTimelineModel.props_allframes = [newprop,obj.lObj.infoTimelineModel.props_allframes];
      obj.curprop = 1;
      tfSucc = true;      
    end
    function [ptype,prop] = getCurPropSmart(obj)
      % Get current proptype, and prop-specification-struct
      
      ptype = obj.lObj.infoTimelineModel.proptypes{obj.curproptype};
      switch ptype
        case 'Predictions'
          prop = obj.lObj.infoTimelineModel.props_tracker(obj.curprop);
        otherwise
          prop = obj.lObj.infoTimelineModel.props(obj.curprop);
      end
    end
    function tf = getCurPropTypeIsLabel(obj)
      v = obj.curproptype;
      tf = strcmp(obj.lObj.infoTimelineModel.proptypes{v},'Labels');
    end
    function tf = getCurPropTypeIsAllFrames(obj)
      v = obj.curproptype;
      tf = strcmpi(obj.lObj.infoTimelineModel.proptypes{v},'All Frames');
    end
    function setCurPropTypeDefault(obj)
      obj.setCurPropType(1,1);
      obj.isdefault = true;
    end
    function updatePropsGUI(obj)
      obj.parent_.pumInfo_labels.Value = obj.curproptype;
      props = obj.getPropsDisp(obj.curproptype);
      obj.parent_.pumInfo.String = props;
      obj.parent_.pumInfo.Value = obj.curprop;
    end
  end
    
  methods  % callbacks
    function cbkBDF(obj,src,evt) 
      % fprintf('InfoTimeline.cbkBDF() called\n') ;
      if ~obj.lObj.isReady || ~(obj.lObj.hasProject && obj.lObj.hasMovie)
        return
      end

      if evt.Button==1
        % Navigate to clicked frame        
        pos = get(src,'CurrentPoint');
        if obj.lObj.hasTrx,
          [sf,ef] = obj.lObj.trxGetFrameLimits();
        else
          sf = 1;
          ef = obj.nfrm;
        end
        frm = round(pos(1,1));
        frm = min(max(frm,sf),ef);
        obj.lObj.setFrameGUI(frm);
      end
    end  % function

%     function cbkLabelMode(obj,src,evt) %#ok<INUSD>
% %       onoff = onIff(obj.lObj.labelMode==LabelMode.ERRORCORRECT);
%       onoff = 'off';
%       set(obj.hMarked,'Visible',onoff);
%     end

    function tf = isDefaultProp(obj)
      tf = obj.isdefault;
    end
    function tf = hasPredictionConfidence(obj)
      tf = ~isempty(obj.TLPROPS_TRACKER);
    end
    function tf = hasPrediction(obj)
      tf = ismember('Predictions',obj.lObj.infoTimelineModel.proptypes) && isvalid(obj.lObj.tracker);
      if tf,
        pcode = obj.lObj.infoTimelineModel.props_tracker(1);
        data = obj.lObj.tracker.getPropValues(pcode);
        tf = ~isempty(data) && any(~isnan(data(:)));
      end
    end
    function setCurPropTypePredictionDefault(obj)
      proptypei =  find(strcmpi(obj.lObj.infoTimelineModel.proptypes,'Predictions'),1);
      if obj.hasPredictionConfidence(),
        propi = numel(obj.lObj.infoTimelineModel.props)+1;
      else
        propi = 1;
      end
      obj.setCurPropType(proptypei,propi);
      obj.updatePropsGUI();
    end

    
    function cbkLabelUpdated(obj, ~, ~)
      if ~obj.lObj.isinit ,
        obj.setLabelsFull() ;
      end
    end
    
    function cbkNewTrackingResults(obj, ~, ~)
      if obj.isDefaultProp() && obj.hasPrediction() ,
        obj.setCurPropTypePredictionDefault() ;
      end
      obj.cbkLabelUpdated() ;
    end
    
    function cbkSetNumFramesShown(obj,src,evt) %#ok<INUSD>
      frmRad = obj.prefs.FrameRadius;
      aswr = inputdlg('Number of frames (0 to show full movie)',...
        'Timeline',1,{num2str(2*frmRad)});
      if ~isempty(aswr)
        nframes = str2double(aswr{1});
        validateattributes(nframes,{'numeric'},{'nonnegative' 'integer'});
        obj.lObj.projPrefs.InfoTimelines.FrameRadius = round(nframes/2);
        obj.newFrame(obj.lObj.currFrame);
      end
    end
    function cbkToggleThresholdViz(obj,src,evt)  %#ok<INUSD> 
      tfviz = strcmp(obj.hStatThresh.Visible,'on');
      obj.setStatThreshViz(~tfviz);
    end
    function cbkContextMenu(obj,src,evt)  %#ok<INUSD>
      bouts = obj.selectGetSelection;
      nBouts = size(bouts,1);
      src.UserData.bouts = bouts;

      % Fill in bout number in "clear all" menu item
      hMnuClearAll = obj.hCMenuClearAll;
      set(hMnuClearAll,'Label',sprintf(hMnuClearAll.UserData.LabelPat,nBouts));
      
      % figure out if user clicked within a bout
      pos = get(obj.hAx,'CurrentPoint');
      frmClick = pos(1);
      tf = bouts(:,1)<=frmClick & frmClick<=bouts(:,2);
      iBout = find(tf);
      tfClickedInBout = ~isempty(iBout);
      hMnuClearBout = obj.hCMenuClearBout;
      set(hMnuClearBout,'Visible',onIff(tfClickedInBout));
      if tfClickedInBout
        assert(isscalar(iBout));
        set(hMnuClearBout,'Label',sprintf(hMnuClearBout.UserData.LabelPat,...
          bouts(iBout,1),bouts(iBout,2)-1));
        for i = 1:numel(hMnuClearBout),
          hMnuClearBout(i).UserData.iBout = iBout;  % store bout that user clicked in
        end
      end
    end
    function cbkClearBout(obj,src,evt) %#ok<INUSD>
      % Prob should have a select* method, for now just do everything here
      iBout = src.UserData.iBout;
      boutsAll = src.Parent.UserData.bouts;
      bout = boutsAll(iBout,:);
      obj.hSelIm.CData(:,bout(1):bout(2)-1) = 0;
      obj.setLabelerSelectedFrames();
    end    
    function cbkGTIsGTModeUpdated(obj,src,evt) %#ok<INUSD>
      lblObj = obj.lObj;
      gt = lblObj.gtIsGTMode;
      if gt
        obj.cbkGTSuggUpdated([],[]);
      end
      onOff = onIff(gt);
      obj.hSegLineGT.Visible = onOff ;
      obj.hSegLineGTLbled.Visible = onOff ;   
      set(obj.hPtsL,'Visible',onIff(~gt));
    end
    function cbkGTSuggUpdated(obj,src,evt) %#ok<INUSD>
      % full update to any change to labeler.gtSuggMFTable*
      
      lblObj = obj.lObj;
      if lblObj.isinit || ~lblObj.hasMovie || ~lblObj.gtIsGTMode
        % segLines are not visible; more importantly, cannot set segLine
        % highlighting based on suggestions in current movie
        return;
      end
      
      % find rows for current movie
      tblLbled = table(lblObj.gtSuggMFTableLbled,'variableNames',{'hasLbl'});
      tbl = [lblObj.gtSuggMFTable tblLbled];
      mIdx = lblObj.currMovIdx;
      tf = mIdx==tbl.mov;
      tblCurrMov = tbl(tf,:); % current mov, various frm/tgts
      
      % for hSegLineGT, we highlight any/all frames (regardless of, or across all, targets)
      frmsOn = tblCurrMov.frm; % could contain repeat frames (across diff targets)
      % obj.hSegLineGT.setOnAtOnly(frmsOn);
      setSegmentedLineOnAtOnlyBang(obj.hSegLineGT, obj.nfrm, frmsOn) ;
      
      % For hSegLineGTLbled, we turn on a given frame only if all
      % targets/rows for that frame are labeled.
      tblRes = rowfun(@(zzHasLbl)all(zzHasLbl),tblCurrMov,...
        'groupingVariables',{'frm'},'inputVariables','hasLbl',...
        'outputVariableNames',{'allTgtsLbled'});
      frmsAllTgtsLbled = tblRes.frm(tblRes.allTgtsLbled);
      % obj.hSegLineGTLbled.setOnAtOnly(frmsAllTgtsLbled);
      setSegmentedLineOnAtOnlyBang(obj.hSegLineGTLbled, obj.nfrm, frmsAllTgtsLbled) ;     
    end
    function cbkGTSuggMFTableLbledUpdated(obj,src,evt) %#ok<INUSD>
      % React to incremental update to labeler.gtSuggMFTableLbled
      
      lblObj = obj.lObj;
      if ~lblObj.gtIsGTMode
        % segLines are not visible,; more importantly, cannot set segLine
        % highlighting based on suggestions in current movie
        return;
      end
      
      % find rows for current movie/frm
      tbl = lblObj.gtSuggMFTable;
      currFrm = lblObj.currFrame;
      tfCurrMovFrm = tbl.mov==lblObj.currMovIdx & tbl.frm==currFrm;
      tfLbled = lblObj.gtSuggMFTableLbled;
      tfLbledCurrMovFrm = tfLbled(tfCurrMovFrm,:);
      tfHiliteOn = numel(tfLbledCurrMovFrm)>0 && all(tfLbledCurrMovFrm);
      %obj.hSegLineGTLbled.setOnOffAt(currFrm,tfHiliteOn);
      setSegmentedLineOnOffAtBang(obj.hSegLineGTLbled, currFrm, tfHiliteOn) ;
    end
    
    % function cbkPostZoom(obj,src,evt) %#ok<INUSD>
    %   if ishandle(obj.hSelIm),
    %     obj.hSelIm.YData = obj.hAx.YLim;
    %   end
    % end
    
  end

  methods (Access=private)
    function setLabelerSelectedFrames(obj)
      % Labeler owns the property-of-record on what frames
      % are set
      selFrames = bouts2frames(obj.selectGetSelection);
      obj.lObj.setSelectedFrames(selFrames);
    end

    function data = getDataCurrMovTgt(obj)
      % lpos: [nptsxnfrm]
      
      [ptype,pcode] = obj.getCurPropSmart();
      labeler = obj.lObj;
      iMov = labeler.currMovie;
      iTgt = labeler.currTarget;
      
      if isempty(iMov) || iMov==0 
        data = nan(obj.npts,1);
      else
        switch ptype
          case {'Labels','Imported'}
            needtrx = obj.lObj.hasTrx && strcmpi(pcode.coordsystem,'Body');
            if needtrx,
              trxFile = obj.lObj.trxFilesAllFullGTaware{iMov,1};
              bodytrx = obj.lObj.getTrx(trxFile,obj.lObj.movieInfoAllGTaware{iMov,1}.nframes);
              bodytrx = bodytrx(iTgt);
            else
              bodytrx = [];
            end
            
            nfrmtot = labeler.nframes;
            if strcmp(ptype,'Labels'),
              s = labeler.labelsGTaware{iMov};
              [tfhasdata,lpos,lposocc,lpost0,lpost1] = Labels.getLabelsT(s,iTgt);
              lpos = reshape(lpos,size(lpos,1)/2,2,[]);
            else
              s = labeler.labels2GTaware{iMov};
              if labeler.maIsMA
                % Use "current Tracklet" for imported data
                if ~isempty(labeler.labeledpos2trkViz)
                  iTgt = labeler.labeledpos2trkViz.currTrklet;
                  if isnan(iTgt)
                    warningNoTrace('No Tracklet currently selected; showing timeline data for first tracklet.');
                    iTgt = 1;
                  end
                else
                  iTgt = 1;
                end
              end  
              [tfhasdata,lpos,lposocc,lpost0,lpost1] = s.getPTrkTgt2(iTgt);
            end
            if tfhasdata
              data = ComputeLandmarkFeatureFromPos(...
                lpos,lposocc,lpost0,lpost1,nfrmtot,bodytrx,pcode);
            else
              data = nan(obj.npts,1); % looks like we don't need 2nd dim to be nfrmtot
            end
          case 'Predictions'
            % AL 20200511 hack, initialization ordering. If the timeline
            % pum has 'Predictions' selected and a new project is loaded,
            % the trackers are not updated (via
            % LabelerGUI/cbkCurrTrackerChanged) until after a movieSetGUI()
            % call which leads here.
            tracker = obj.lObj.tracker ;
            if ~isempty(tracker) && isvalid(tracker)
              data = tracker.getPropValues(pcode);
            else
              data = nan(obj.npts,1);
            end
          case 'All Frames'
            %fprintf('getDataCurrMovTarg -> All Frames, %d\n',obj.curprop);
            if strcmpi(obj.lObj.infoTimelineModel.props_allframes(obj.curprop).name,'Add custom...'),
              data = nan(obj.npts,1);
            else
              data = obj.custom_data;
            end
          otherwise
            error('Unknown data type %s',ptype);
        end
        %szassert(data,[obj.npts obj.nfrm]);
      end
    end
    
    function data = getIsLabeledCurrMovTgt(obj)
      % lpos: [nptsxnfrm]
      
      labeler = obj.lObj;
      iMov = labeler.currMovie;
      iTgt = labeler.currTarget;
      
      if isempty(iMov) || iMov==0 || ~labeler.hasMovie
        data = nan(obj.npts,1);
      else
        s = labeler.labelsGTaware{iMov};       
        [p,~] = Labels.getLabelsT_full(s,iTgt,obj.nfrm);
        xy = reshape(p,obj.npts,2,obj.nfrm);
        data = reshape(all(~isnan(xy),2),obj.npts,obj.nfrm);
      end
    end
    
    function tflbledDisp = getlabeledTgts(obj)
      labeler = obj.lObj;
      iMov = labeler.currMovie;
      if iMov==0
        tflbledDisp = nan;
        return;
      end
      tflbledDisp = labeler.labelPosLabeledTgts(iMov);
      ntgtsmax = size(tflbledDisp,2);
      ntgtDisp = obj.axLmaxntgt;
      if ntgtsmax>=ntgtDisp 
        tflbledDisp = tflbledDisp(:,1:ntgtDisp);
      else
        tflbledDisp(:,ntgtsmax+1:ntgtDisp) = false;
      end
    end
  end

  methods
    function didSetTimelineSelectMode(obj)
      % Handle didSetTimelineSelectMode UI updates
      if ~obj.isinit
        selectOn = obj.lObj.infoTimelineModel.selectOn;
        if selectOn
          obj.hCurrFrame.LineWidth = 3;
          if obj.isL
            obj.hCurrFrameL.LineWidth = 3;
          end
        else
          obj.hCurrFrame.LineWidth = 0.5;
          if obj.isL
            obj.hCurrFrameL.LineWidth = 0.5;
          end
          obj.setLabelerSelectedFrames();
        end
      end
    end
    
  end
  
end
