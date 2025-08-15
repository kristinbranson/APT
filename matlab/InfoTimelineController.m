classdef InfoTimelineController < handle
  % A sub-controller to manage the two timeline axes and their contents.
  % The things shown in here roughly reflect the model state represented in
  % labeler.infoTimelineModel, an InfoTimelineModel object.  But the
  % correspondence is not exact.

  properties (Constant)
    axLmaxntgt = 3  % applies to hAxL for MA projs; number of tgts to display
  end
  
  properties
    parent_  % scalar LabelerController handle
    lObj  % scalar Labeler handle

    hAx  % scalar handle to manual timeline axis
    hAxL  % scalar handle to is-labeled timeline axis
    hCurrFrame  % scalar line handle current frame
    hCurrFrameL  % scalar line handle current frame
    hStatThresh  % scalar line handle, threshold
    % hCMenuClearAll  % scalar context menu
    hCMenuClearBout  % scalar context menu
    hCMenuSetNumFramesShown
    hCMenuToggleThresholdViz

    hPts  % [npts] line handles
    hPtStat  % scalar line handle
    hPtsL  % [npts] patch handles (non-MA projs), or [1] image handle (MA projs)
    
    listeners  % [nlistener] col cell array of labeler prop listeners

    hSelIm  % scalar image handle for selection

    hSegLineGT  % scalar line handle
    hSegLineGTLbled  % scalar line handle
  end
  
  properties (Dependent)
    nfrm_
  end
  
  methods
    function v = get.nfrm_(obj)
      lblObj = obj.lObj;
      if lblObj.hasMovie
        v = lblObj.nframes;
      else
        v = 1;
      end
    end

    function obj = InfoTimelineController(parent)
      % parent a LabelerController

      axtm = parent.axes_timeline_manual ;
      axti = parent.axes_timeline_islabeled ;

      obj.parent_ = parent ;
      labeler = parent.labeler_ ;
      obj.lObj = labeler ;
      axtm.Color = [0 0 0];
      axtm.ButtonDownFcn = @(src,evt)obj.cbkBDF(src,evt);
      %hold(axtm,'on');
      obj.hAx = axtm;
      obj.hCurrFrame = ...
        line('Parent',axtm, ...
             'XData',[nan nan], ...
             'YData',axtm.YLim, ...
             'LineStyle','-', ...
             'Color',[1 1 1],...
             'hittest','off', ...
             'Tag','InfoTimeline_CurrFrame');
      obj.hStatThresh = ...
        line('Parent',axtm, ...
             'XData',[nan nan], ...
             'YData',[0 0], ...
             'LineStyle','-', ...
             'Color',[1 1 1],...
             'hittest','off', ...
             'visible','off', ...
             'Tag','InfoTimeline_StatThresh');      

      if ~isempty(axti) && ishandle(axti),
        axti.Color = [0 0 0];
        axti.ButtonDownFcn = @(src,evt)(obj.cbkBDF(src,evt));
        %hold(axti,'on');
      end
      obj.hAxL = axti;
      
      if obj.isL,
        obj.hCurrFrameL = ...
          line('Parent',axti, ...
               'XData',[nan nan], ...
               'YData',[0 1], ...
               'LineStyle','-', ...
               'Color',[1 1 1], ...
               'hittest','off', ...
               'Tag','InfoTimeline_CurrFrameLabel') ;
      else
        obj.hCurrFrameL = [];
      end

      obj.hPts = [];
      obj.hPtStat = [];
      obj.hPtsL = [];
            
      listeners = cell(0,1);
      listeners{end+1,1} = ...
        addlistener(labeler, 'didSetLabels', @obj.cbkLabelUpdated) ;
      listeners{end+1,1} = ...
        addlistener(labeler, 'gtSuggMFTableLbledUpdated',@obj.cbkGTSuggMFTableLbledUpdated) ;
      listeners{end+1,1} = ...
          addlistener(labeler, 'newTrackingResults', @obj.cbkNewTrackingResults) ;      
      obj.listeners = listeners;      
    
      obj.hSelIm = [];
      obj.hSegLineGT = line('XData',nan,'YData',nan,'Parent',axtm,'Tag','InfoTimeline_SegLineGT');
      obj.hSegLineGTLbled = line('XData',nan,'YData',nan,'Parent',axtm,'Tag','InfoTimeline_SegLineGTLbled');
      
      hCMenu = ...
        uicontextmenu('Parent',axtm.Parent,...
                      'Tag','InfoTimeline_ContextMenu');
      obj.hCMenuSetNumFramesShown = ...
        uimenu('Parent',hCMenu, ...
               'Label','Set number of frames shown',...
               'Callback',@(src,evt)obj.cbkSetNumFramesShown(src,evt),...
               'Tag','menu_InfoTimeline_SetNumFramesShown');
      obj.hCMenuClearBout = ...
        uimenu('Parent',hCMenu,...
               'Label','Clear single bout',...
               'Callback',@(src,evt)obj.cbkClearBout(src,evt),...
               'Tag','menu_InfoTimeline_ClearBout') ;
      obj.hCMenuToggleThresholdViz = ...
        uimenu('Parent',hCMenu, ...
               'Label','Toggle statistic threshold visibility',...
               'Callback',@(src,evt)obj.cbkToggleThresholdViz(src,evt),...
               'Tag','menu_InfoTimeline_ToggleThresholdViz');
      axtm.UIContextMenu = hCMenu;            
    end  % function
    
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
        
    function initNewProject(obj)
      deleteValidGraphicsHandles(obj.hPts);
      deleteValidGraphicsHandles(obj.hPtStat);
      deleteValidGraphicsHandles(obj.hPtsL);
      obj.hPts = gobjects(obj.lObj.nLabelPoints,1);
      obj.hPtStat = gobjects(1);
      colors = obj.lObj.LabelPointColors;
      ax = obj.hAx;
      axl = obj.hAxL;
      for i=1:obj.lObj.nLabelPoints
        obj.hPts(i) = ...
          line('Parent',ax, ...
               'XData',nan, ...
               'YData',i, ...
               'Marker','.', ...
               'LineStyle','-', ...
               'Color',colors(i,:),...
               'hittest','off', ...
               'Tag',sprintf('InfoTimeline_Pt%d',i)) ;
      end
      isMA = obj.lObj.maIsMA;
      if isMA
        obj.hPtsL = gobjects(1,1);
      else
        obj.hPtsL = gobjects(obj.lObj.nLabelPoints,1);        
      end
      if obj.isL
        if isMA
          obj.hPtsL = image('Parent',axl,'CData',nan,'hittest','off','tag','InfoTimeline_Label_ma');
        else
          for i=1:obj.lObj.nLabelPoints
            obj.hPtsL(i) = ...
              patch('Parent',axl, ...
                    'XData',nan(1,5), ...
                    'YData',i-1+[0,1,1,0,0], ...
                    'CData',colors(i,:),...
                    'EdgeColor','none', ...
                    'hittest','off', ...
                    'Tag',sprintf('InfoTimeline_Label_%d',i)) ;
          end
        end
      end
      
      clr = [1 1 1] ;  % color when there is only one statistic for all landmarks     
      obj.hPtStat = line('Parent',ax, ...
                         'XData',nan, ...
                         'YData',i, ...
                         'LineStyle','-.', ...
                         'Color',clr, ...
                         'hittest','off', ...
                         'LineWidth',2, ...
                         'Tag','InfoTimeline_Stat');
      
      prefsTL = obj.lObj.projPrefs.InfoTimelines;
      ax.XColor = prefsTL.XColor;
      dy = .01;
      ax.YLim = [0-dy 1+dy];
      if ishandle(obj.hSelIm)
        obj.hSelIm.YData = ax.YLim;
      end
      if obj.isL
        if isMA
          axl.YLim = [0-dy obj.axLmaxntgt+dy];
          axl.Colormap = [0 0 0 ; 0 0 1] ;
          % colormap(axl,[0 0 0;0 0 1]);
          % axis(axl,'ij');
          axl.YDir = 'reverse' ;
        else
          axl.YLim = [0-dy obj.lObj.nLabelPoints+dy];
          % axis(axl,'xy');
          axl.YDir = 'normal' ;
        end
      end
      
      set(obj.hCurrFrame,'XData',[nan nan],'YData',ax.YLim,'ZData',[1 1]);
      set(obj.hCurrFrameL,'XData',[nan nan],'YData',axl.YLim,'ZData',[1 1]);
      set(obj.hStatThresh,'XData',[nan nan],'ZData',[1 1]);
      linkaxes([obj.hAx,obj.hAxL],'x');
    end
    
    function initNewMovie(obj)
      ax = obj.hAx;
      prefsTL = obj.lObj.projPrefs.InfoTimelines;
      ax.XTick = 0:prefsTL.dXTick:obj.nfrm_;

      if obj.lObj.isinit || isnan(obj.nfrm_)
        return
      end

      deleteValidGraphicsHandles(obj.hSelIm);
      obj.hSelIm = ...
        image('Parent', obj.hAx, ...
              'XData', 1:obj.nfrm_, ...
              'YData', obj.hAx.YLim, ...
              'CData', uint8(zeros(1,obj.nfrm_)), ...
              'HitTest', 'off',...
              'CDataMapping', 'direct') ;

      % itm.setSelectMode(false) ;
      colorTBSelect = obj.parent_.tbTLSelectMode.BackgroundColor;
      obj.hAx.Colormap = [0 0 0;colorTBSelect] ;
      
      % obj.setLabelerSelectedFrames_();
      
      xlims = [1 obj.nfrm_];
      sPV = struct('LineWidth',5,'Color',AxesHighlightManager.ORANGE);
      sPVLbled = struct('LineWidth',5,'Color',AxesHighlightManager.ORANGE/2);
      initSegmentedLineBang(obj.hSegLineGT,xlims,sPV);
      initSegmentedLineBang(obj.hSegLineGTLbled,xlims,sPVLbled);
      if obj.getCurPropTypeIsAllFrames(),
        obj.setCurPropTypeDefault();
      end
      
      % itm.initializePropsEtc(obj.lObj.hasTrx);
        
      cbkGTSuggUpdated(obj,[],[]);
    end
    
        
    function didChangeCurrentTracker(obj)
      %itm.didChangeCurrentTracker() ;  % now done directly in Labeler
      obj.enforcePropConsistencyWithUI(false);      
      obj.setLabelsFull();
    end
    
    function setLabelsFull(obj)
      % Get data and set .hPts, .hMarked
      
      if isempty(obj.lObj.nLabelPoints) || isnan(obj.lObj.nLabelPoints)
        return
      end
      
      dat = obj.lObj.getTimelineDataCurrMovTgt(); % [nptsxnfrm]
      dat(isinf(dat)) = nan;
      datnonnan = dat(~isnan(dat));

      set(obj.hPts,'XData',nan,'YData',nan);
      set(obj.hPtStat,'XData',nan,'YData',nan);
      
      if ~isempty(datnonnan)
        
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
        if size(dat,1) == obj.lObj.nLabelPoints,
          for i=1:obj.lObj.nLabelPoints
            set(obj.hPts(i),'XData',x,'YData',dat(i,:));
          end
        elseif size(dat,1) == 1,
          set(obj.hPtStat,'XData',x,'YData',dat(1,:));
        else
          warningNoTrace(sprintf('InfoTimeline: Number of rows in statistics was %d, expected either %d or 1',size(dat,1),obj.lObj.nLabelPoints));
        end
        
        set(obj.hStatThresh,'XData',x([1 end]));
      end
      
      if obj.isL,
        if obj.lObj.maIsMA
          tflbledDisp = obj.lObj.getLabeledTgts(obj.axLmaxntgt);
          set(obj.hPtsL,'CData',uint8(tflbledDisp'));          
        else
          islabeled = obj.lObj.getIsLabeledCurrMovTgt(); % [nptsxnfrm]
          for i = 1:obj.lObj.nLabelPoints,
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
    end  % function
    
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
%       for i=1:obj.lObj.nLabelPoints
%         h = obj.hPts(i);
%         set(h,'XData',1:size(lpos,2),'YData',lpos(i,:));
%       end
    end
    
    function updateAfterCurrentFrameSet(obj)
      % This gets called after the user changes the frame they're looking at, i.e.
      % after labeler.currFrame is set.      
      if isnan(obj.lObj.nLabelPoints), return; end      
      obj.updateCurrentFrameLineXData_() ;      
      itm = obj.lObj.infoTimelineModel ;
      if itm.selectOn
        obj.updateSelectionImageCData_() ;
      end
      obj.updateContextMenu_() ;
    end  % function

    function updateCurrentFrameLineXData_(obj)
      if isnan(obj.lObj.nLabelPoints), return; end
      if isempty(obj.lObj.projPrefs)
        return
      end
      currFrame = obj.lObj.currFrame ;
      nominal_xspan = 2*obj.lObj.projPrefs.InfoTimelines.FrameRadius;
      nominal_dxtick = obj.lObj.projPrefs.InfoTimelines.dXTick ;
      if nominal_xspan==0 || nominal_xspan > obj.nfrm_
        x0 = 1;
        x1 = obj.nfrm_;
        xspan = x1-x0 ;
      else
        xspan = nominal_xspan ;
        r = xspan/2 ;
        x0_raw = currFrame-r;
        x1_raw = currFrame+r; %min(frm+r,obj.nfrm);
        % Make sure the limits don't run off the end
        if x0_raw<1
          x0 = 1 ;
          x1 = 1 + 2*r ;
        elseif x1_raw>obj.nfrm_
          x1 = obj.nfrm_ ;
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
      obj.hAx.XTick = 0 : dxtick : obj.nfrm_ ;
      obj.hAx.XLim = [x0 x1];
      set(obj.hCurrFrame,'XData',[currFrame currFrame],'YData',obj.hAx.YLim);
      if obj.isL,
        obj.hAxL.XLim = [x0 x1];
        set(obj.hCurrFrameL,'XData',[currFrame currFrame],'YData',obj.hAxL.YLim);
      end
    end  % function
    
    function updateSelectionImageCData_(obj)
      itm = obj.lObj.infoTimelineModel ;
      if ~isempty(obj.hSelIm) && isvalid(obj.hSelIm) 
        obj.hSelIm.CData = itm.isSelectedFromFrameIndex ;
      end
    end  % function   

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
      for i=1:obj.lObj.nLabelPoints
        set(obj.hPts(i),'Color',ptclrs(i,:));
      end
      if obj.isL && ~obj.lObj.maIsMA
        for i=1:obj.lObj.nLabelPoints
          set(obj.hPtsL(i),'FaceColor',lblcolors(i,:));
        end
      end
    end
    
    % function bouts = selectGetSelection(obj)
    %   % Get currently selected bouts (can be noncontiguous)
    %   %
    %   % bouts: [nBout x 2]. col1 is startframe, col2 is one-past-endframe
    % 
    %   cdata = obj.hSelIm.CData;
    %   [sp,ep] = get_interval_ends(cdata);
    %   bouts = [sp(:) ep(:)];
    % end
    
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

      itm = obj.lObj.infoTimelineModel ;
      ptype = itm.proptypes{itm.curproptype};
      switch ptype
        case 'Predictions'
          tfOOB = itm.curprop > numel(itm.props_tracker);
        otherwise
          tfOOB = itm.curprop > numel(itm.props);
      end
      
      if tfOOB
        NEWPROP = 1;
        itm.curprop = NEWPROP;
        obj.parent_.pumInfo.Value = NEWPROP;
      end
      
      if tfSetLabelsFull
        obj.setLabelsFull();
      end
    end

    function tfSucc = setCurProp(obj,iprop)
      % setLabelsFull will essentially assert that iprop is in range for
      % current proptype.
      %
      % Does not update UI
      tfSucc = true;
      itm = obj.lObj.infoTimelineModel ;
      if obj.getCurPropTypeIsAllFrames() && ...
          strcmpi(itm.props_allframes(iprop).name,'Add custom...'),
        [tfSucc] = obj.addCustomFeature_();
        if ~tfSucc,
          return;
        end
      else
        itm.curprop = iprop;
      end
      obj.setLabelsFull();
      itm.isdefault = false;
    end

    function v = getCurProp(obj)
      itm = obj.lObj.infoTimelineModel ;
      v = itm.curprop;
    end

    function setCurPropType(obj, iproptype, iprop)
      % iproptype, iprop assumed to be consistent already.
      obj.lObj.setTimelineCurrentPropertyType(iproptype, iprop) ;
      obj.setLabelsFull();
      obj.updateLandmarkColors();
    end

    function tfSucc = addCustomFeature_(obj)
      tfSucc = false;
      movfile = obj.lObj.getMovieFilesAllFullMovIdx(obj.lObj.currMovIdx);
      defaultpath = fileparts(movfile{1});
      [f,p] = uigetfile('*.mat','Select .mat file with a feature value for each frame for current movie',defaultpath);
      if ~ischar(f),
        return;
      end
      file = fullfile(p,f);
      try
        obj.lObj.addCustomTimelineFeatureGivenFileName(file);
      catch ME
        uiwait(errordlg(ME.message, 'Error loading custom feature'));
        return;
      end
      tfSucc = true;      
    end

    function tf = getCurPropTypeIsLabel(obj)
      itm = obj.lObj.infoTimelineModel ;
      v = itm.curproptype;
      tf = strcmp(itm.proptypes{v},'Labels');
    end

    function tf = getCurPropTypeIsAllFrames(obj)
      itm = obj.lObj.infoTimelineModel ;
      v = itm.curproptype;
      tf = strcmpi(itm.proptypes{v},'All Frames');
    end

    function setCurPropTypeDefault(obj)
      obj.setCurPropType(1,1);
      itm = obj.lObj.infoTimelineModel ;
      itm.isdefault = true;
    end

    function updatePropsGUI(obj)
      itm = obj.lObj.infoTimelineModel ;
      obj.parent_.pumInfo_labels.Value = itm.curproptype;
      props = itm.getPropsDisp(itm.curproptype);
      obj.parent_.pumInfo.String = props;
      obj.parent_.pumInfo.Value = itm.curprop;
    end

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
          ef = obj.nfrm_;
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
      itm = obj.lObj.infoTimelineModel ;
      tf = itm.isdefault;
    end

    function tf = hasPrediction(obj)
      itm = obj.lObj.infoTimelineModel ;
      tf = ismember('Predictions',itm.proptypes) && isvalid(obj.lObj.tracker);
      if tf,
        pcode = itm.props_tracker(1);
        data = obj.lObj.tracker.getPropValues(pcode);
        tf = ~isempty(data) && any(~isnan(data(:)));
      end
    end

    function setCurPropTypePredictionDefault(obj)
      itm = obj.lObj.infoTimelineModel ;
      proptypei =  find(strcmpi(itm.proptypes,'Predictions'),1);
      if itm.hasPredictionConfidence(),
        propi = numel(itm.props)+1;
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
      frmRad = obj.lObj.projPrefs.InfoTimelines.FrameRadius;
      aswr = inputdlg('Number of frames (0 to show full movie)',...
                      'Timeline',1,{num2str(2*frmRad)});
      if ~isempty(aswr)
        nframes = str2double(aswr{1});
        obj.lObj.setTimelineFramesInView(nframes) ;
      end
    end

    function cbkToggleThresholdViz(obj,src,evt)  %#ok<INUSD> 
      tfviz = strcmp(obj.hStatThresh.Visible,'on');
      obj.setStatThreshViz(~tfviz);
    end

    % function cbkContextMenu(obj,src,evt)  %#ok<INUSD>
    %   itm = obj.lObj.infoTimelineModel ;
    %   bouts = itm.selectGetSelectionAsBouts() ;
    %   % nBouts = size(bouts,1);
    %   src.UserData.bouts = bouts;
    % 
    %   % % Fill in bout number in "clear all" menu item
    %   % hMnuClearAll = obj.hCMenuClearAll;
    %   % set(hMnuClearAll,'Label',sprintf(hMnuClearAll.UserData.LabelPat,nBouts));
    % 
    %   % figure out if user clicked within a bout
    %   pos = get(obj.hAx,'CurrentPoint');
    %   frameClicked = pos(1);
    %   bouts = itm.selectGetSelectionAsBouts() ;
    %   tf = bouts(:,1)<=frameClicked & frameClicked<=bouts(:,2);
    %   iBout = find(tf);
    %   tfClickedInBout = ~isempty(iBout);
    %   hMnuClearBout = obj.hCMenuClearBout;
    %   set(hMnuClearBout,'Visible',onIff(tfClickedInBout));
    %   if tfClickedInBout
    %     assert(isscalar(iBout));
    %     set(hMnuClearBout,'Label',sprintf(hMnuClearBout.UserData.LabelPat,...
    %       bouts(iBout,1),bouts(iBout,2)-1));
    %     for i = 1:numel(hMnuClearBout),
    %       hMnuClearBout(i).UserData.iBout = iBout;  % store bout that user clicked in
    %     end
    %   end
    % end

    function cbkClearBout(obj,src,evt)  %#ok<INUSD>
      obj.lObj.clearBoutInTimeline() ;
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
      setSegmentedLineOnAtOnlyBang(obj.hSegLineGT, obj.nfrm_, frmsOn) ;
      
      % For hSegLineGTLbled, we turn on a given frame only if all
      % targets/rows for that frame are labeled.
      tblRes = rowfun(@(zzHasLbl)all(zzHasLbl),tblCurrMov,...
        'groupingVariables',{'frm'},'inputVariables','hasLbl',...
        'outputVariableNames',{'allTgtsLbled'});
      frmsAllTgtsLbled = tblRes.frm(tblRes.allTgtsLbled);
      % obj.hSegLineGTLbled.setOnAtOnly(frmsAllTgtsLbled);
      setSegmentedLineOnAtOnlyBang(obj.hSegLineGTLbled, obj.nfrm_, frmsAllTgtsLbled) ;     
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
    % function setLabelerSelectedFrames_(obj)
    %   % Labeler owns the property-of-record on what frames
    %   % are set
    %   selFrames = bouts2frames(obj.selectGetSelection());
    %   obj.lObj.setSelectedFrames(selFrames);
    % end

  end  % methods (Access=private)

  methods
    function updateCurrentFrameLineWidths_(obj)
      itm = obj.lObj.infoTimelineModel ;
      selectOn = itm.selectOn;
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
        % obj.setLabelerSelectedFrames_();
      end
    end  % function

    function update(obj)
      % Update controls to reflect the model state
      obj.updateCurrentFrameLineWidths_() ;
      obj.updateCurrentFrameLineXData_() ;
      obj.updateSelectionImageCData_() ;
      obj.updateContextMenu_() ;
    end  % function

    function updateContextMenu_(obj)
      lObj = obj.lObj ;
      % Gray menu item if not in a bout
      set(obj.hCMenuClearBout,'Enable',onIff(lObj.isCurrentFrameSelected()));
    end
    
  end  % methods
  
end  % classdef
