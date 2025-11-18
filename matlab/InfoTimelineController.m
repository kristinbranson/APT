classdef InfoTimelineController < handle
  % A sub-controller to manage the two timeline axes and their contents.
  % The things shown in here roughly reflect the model state represented in
  % labeler.infoTimelineModel, an InfoTimelineModel object.  But the
  % correspondence is not exact.  All changes to graphics handles held by this
  % object are made by this object.  (Except for hAx and hAxL, since those two are
  % not 'owned' by this object.  Some changes to it are made elsewhere.)
  %
  % Note that objects of this class do not implement any listeners, and do not
  % directly respond to any UI callbacks.
  %
  % As of Aug 16 2025, the update*() methods are still a hodge-podge of
  % situationally-appropriate updates, which should likely be streamlined
  % at some point.  Ideally the update() method would be the core update method,
  % and would be a general-purpose method to update all the controls that are
  % owned by this object, no matter the situation.  Additional less-general
  % update*() methods would be for use in more specific settings when the
  % performance of update() is inadequate.

  properties (Constant)
    axLmaxntgt = 3  % applies to hAxL for MA projs; number of tgts to display
  end
  
  properties
    lObj  % scalar Labeler handle
    hAx  % scalar handle to manual timeline axis
    hAxL  % scalar handle to is-labeled timeline axis
    hCurrFrame  % scalar line handle current frame
    hCurrFrameL  % scalar line handle current frame
    hStatThresh  % scalar line handle, threshold
    hCMenuClearBout  % scalar context menu
    hCMenuSetNumFramesShown
    hCMenuToggleThresholdViz
    hPts  % [npts] line handles
    hPtStat  % scalar line handle
    hPtsL  % [npts] patch handles (non-MA projs), or [1] image handle (MA projs)    
    hSelIm  % scalar image handle for selection
    hSegLineGT  % scalar line handle
    hSegLineGTLbled  % scalar line handle
  end
  
  methods
    function obj = InfoTimelineController(labeler, axtm, axti)
      obj.lObj = labeler ;
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

      obj.hAxL = axti;
      
      obj.hCurrFrameL = ...
        line('Parent',axti, ...
             'XData',[nan nan], ...
             'YData',[0 1], ...
             'LineStyle','-', ...
             'Color',[1 1 1], ...
             'hittest','off', ...
             'Tag','InfoTimeline_CurrFrameLabel') ;

      obj.hPts = [];
      obj.hPtStat = [];
      obj.hPtsL = [];
            
      obj.hSelIm = [];
      obj.hSegLineGT = line('Parent',axtm,'XData',nan,'YData',nan,'Tag','InfoTimeline_SegLineGT');
      obj.hSegLineGTLbled = line('Parent',axtm,'XData',nan,'YData',nan,'Tag','InfoTimeline_SegLineGTLbled');
      
      hCMenu = ...
        uicontextmenu('Parent',axtm.Parent,...
                      'Tag','InfoTimeline_ContextMenu');
      obj.hCMenuSetNumFramesShown = ...
        uimenu('Parent',hCMenu, ...
               'Label','Set number of frames shown',...
               'Tag','menu_InfoTimeline_SetNumFramesShown');
      obj.hCMenuClearBout = ...
        uimenu('Parent',hCMenu,...
               'Label','Clear single bout',...
               'Tag','menu_InfoTimeline_ClearBout') ;
      obj.hCMenuToggleThresholdViz = ...
        uimenu('Parent',hCMenu, ...
               'Label','Toggle statistic threshold visibility',...
               'Tag','menu_InfoTimeline_ToggleThresholdViz');
      axtm.UIContextMenu = hCMenu;            

      % Make sure the main timeline axes and the is-labeled axes always have the
      % same XLim, even when user uses Matlab built-in zoom/pan features.
      linkaxes([obj.hAx,obj.hAxL],'x');
    end  % function
    
    function delete(obj)
      deleteValidGraphicsHandles([obj.hCurrFrame,obj.hCurrFrameL,obj.hStatThresh]);
      obj.hCurrFrame = [];
      obj.hCurrFrameL = [];
      obj.hStatThresh = [];
      deleteValidGraphicsHandles(obj.hPts);
      deleteValidGraphicsHandles(obj.hPtStat);
      obj.hPts = [];
      obj.hPtStat = [];
      deleteValidGraphicsHandles(obj.hPtsL);
      obj.hPtsL = [];
      deleteValidGraphicsHandles(obj.hSelIm);
      obj.hSelIm = [];
      deleteValidGraphicsHandles(obj.hSegLineGT);
      obj.hSegLineGT = [];
      deleteValidGraphicsHandles(obj.hSegLineGTLbled);
      obj.hSegLineGTLbled = [];
    end
        
    function updateForNewProject(obj)
      % Update the controls in the wake of a new project being created/loaded.

      % Get the core things we need from the labeler
      lObj = obj.lObj ;
      nLabelPoints = lObj.nLabelPoints ;
      isMA = lObj.maIsMA ;
      colors = lObj.LabelPointColors ;
      prefsXColor = lObj.projPrefs.InfoTimelines.XColor ;

      deleteValidGraphicsHandles(obj.hPts);
      deleteValidGraphicsHandles(obj.hPtStat);
      deleteValidGraphicsHandles(obj.hPtsL);
      obj.hPts = gobjects(nLabelPoints,1);
      obj.hPtStat = gobjects(1);
      ax = obj.hAx;
      axl = obj.hAxL;
      for i=1:nLabelPoints
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
      if isMA
        obj.hPtsL = gobjects(1,1);
      else
        obj.hPtsL = gobjects(nLabelPoints,1);        
      end
      if isMA
        obj.hPtsL = image('Parent',axl,'CData',nan,'hittest','off','tag','InfoTimeline_Label_ma');
      else
        for i=1:nLabelPoints
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
      
      clr = [1 1 1] ;  % color when there is only one statistic for all landmarks     
      obj.hPtStat = line('Parent',ax, ...
                         'XData',nan, ...
                         'YData',i, ...
                         'LineStyle','-.', ...
                         'Color',clr, ...
                         'hittest','off', ...
                         'LineWidth',2, ...
                         'Tag','InfoTimeline_Stat');
      
      ax.XColor = prefsXColor;
      dy = .01;
      ax.YLim = [0-dy 1+dy];
      if ishandle(obj.hSelIm)
        obj.hSelIm.YData = ax.YLim;
      end
      if isMA
        axl.YLim = [0-dy obj.axLmaxntgt+dy];
        axl.Colormap = [0 0 0 ; 0 0 1] ;
        axl.YDir = 'reverse' ;
      else
        axl.YLim = [0-dy nLabelPoints+dy];
        axl.YDir = 'normal' ;
      end
      
      set(obj.hCurrFrame,'XData',[nan nan],'YData',ax.YLim,'ZData',[1 1]);
      set(obj.hCurrFrameL,'XData',[nan nan],'YData',axl.YLim,'ZData',[1 1]);
      set(obj.hStatThresh,'XData',[nan nan],'ZData',[1 1]);
    end
    
    function updateForNewMovie(obj, colorTBSelect)
      % Update the controls in the wake of a new movie being made current.

      % Return early if labeler is being initialized
      lObj = obj.lObj ;
      if lObj.isinit, return; end

      % Get the core things we need from the labeler
      nframes = lObj.nframes ;
      dXTick = lObj.projPrefs.InfoTimelines.dXTick ;

      % Return early if nframes is nan (not sure when this might happen...)
      if isnan(nframes), return; end

      % Set control properties
      ax = obj.hAx;
      ax.XTick = 0:dXTick:nframes;
      deleteValidGraphicsHandles(obj.hSelIm);
      obj.hSelIm = ...
        image('Parent', obj.hAx, ...
              'XData', 1:nframes, ...
              'YData', obj.hAx.YLim, ...
              'CData', uint8(zeros(1,nframes)), ...
              'HitTest', 'off',...
              'CDataMapping', 'direct') ;
      obj.hAx.Colormap = [ 0 0 0 ; colorTBSelect ] ;      
      xlims = [1 nframes];
      sPV = struct('LineWidth',5,'Color',AxesHighlightManager.ORANGE);
      sPVLbled = struct('LineWidth',5,'Color',AxesHighlightManager.ORANGE/2);
      initSegmentedLineBang(obj.hSegLineGT,xlims,sPV);
      initSegmentedLineBang(obj.hSegLineGTLbled,xlims,sPVLbled);

      % Call another update method to handle the GT-related controls
      obj.updateGTModeRelatedControls();
    end
            
    function updateLabels(obj)
      % Update .hPts, .hMarked, .hPtStat
      
      lObj = obj.lObj ;
      if lObj.isinit || isempty(lObj.nLabelPoints) || isnan(lObj.nLabelPoints)
        return
      end
      
      dat = lObj.getTimelineDataForCurrentMovieAndTarget();  % [nptsxnfrm]
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
        if size(dat,1) == lObj.nLabelPoints,
          for i=1:lObj.nLabelPoints
            set(obj.hPts(i),'XData',x,'YData',dat(i,:));
          end
        elseif size(dat,1) == 1,
          set(obj.hPtStat,'XData',x,'YData',dat(1,:));
        else
          warningNoTrace(sprintf('InfoTimeline: Number of rows in statistics was %d, expected either %d or 1',size(dat,1),lObj.nLabelPoints));
        end
        
        set(obj.hStatThresh,'XData',x([1 end]));
      end
      
      if lObj.maIsMA
        tflbledDisp = lObj.getLabeledTgts(obj.axLmaxntgt);
        set(obj.hPtsL,'CData',uint8(tflbledDisp'));          
      else
        islabeled = lObj.getIsLabeledCurrMovTgt(); % [nptsxnfrm]
        for i = 1:lObj.nLabelPoints,
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
    end  % function
    
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
      nominal_r = obj.lObj.projPrefs.InfoTimelines.FrameRadius;
      nominal_dxtick = obj.lObj.projPrefs.InfoTimelines.dXTick ;
      % MK says he wants the current frame in the center,
      % even it means the limits run off the end.
      if nominal_r==0 || 2*nominal_r > obj.lObj.nframes
        r = floor(obj.lObj.nframes/2) ;
      else
        r = nominal_r ;
      end
      x0 = currFrame-r ;
      x1 = currFrame+r ;
      if r/nominal_dxtick > 10 ,
        dxtick = apt.heuristic_dxtick_from_xspan(2*r) ;
      else
        dxtick = nominal_dxtick ;
      end
      obj.hAx.XTick = 0 : dxtick : obj.lObj.nframes ;
      obj.hAx.XLim = [x0 x1];
      set(obj.hCurrFrame,'XData',[currFrame currFrame],'YData',obj.hAx.YLim);
      obj.hAxL.XLim = [x0 x1];
      set(obj.hCurrFrameL,'XData',[currFrame currFrame],'YData',obj.hAxL.YLim);
    end  % function
    
    function updateSelectionImageCData_(obj)
      itm = obj.lObj.infoTimelineModel ;
      if ~isempty(obj.hSelIm) && isvalid(obj.hSelIm) 
        obj.hSelIm.CData = itm.isSelectedFromFrameIndex ;
      end
    end  % function   

    function updateLandmarkColors(obj)
      tflbl = obj.lObj.infoTimelineModel.getCurPropTypeIsLabel();
      lblcolors = obj.lObj.LabelPointColors();
      if tflbl
        ptclrs = lblcolors;
      else
        ptclrs = obj.lObj.PredictPointColors();
      end
      for i=1:obj.lObj.nLabelPoints
        set(obj.hPts(i),'Color',ptclrs(i,:));
      end
      if ~obj.lObj.maIsMA
        for i=1:obj.lObj.nLabelPoints
          set(obj.hPtsL(i),'FaceColor',lblcolors(i,:));
        end
      end
    end  % function   
    
    function updateStatThresh(obj)
      % Update the statistic threshold display from the model
      itm = obj.lObj.infoTimelineModel;
      thresh = itm.statThresh;
      tfshow = itm.isStatThreshVisible;
      
      if ~isempty(thresh)
        obj.hStatThresh.YData = [thresh thresh];
      end
      
      % Update visibility and axis colors
      onoff = onIff(tfshow);
      obj.hStatThresh.Visible = onoff;
      if tfshow
        obj.hAx.YColor = obj.hAx.XColor;
      else
        obj.hAx.YColor = [0.15 0.15 0.15];
      end
    end  % function   
    
    function updateGTModeRelatedControls(obj)
      lObj = obj.lObj;
      gt = lObj.gtIsGTMode;
      onOff = onIff(gt);
      obj.hSegLineGT.Visible = onOff ;
      obj.hSegLineGTLbled.Visible = onOff ;   
      set(obj.hPtsL,'Visible',onIff(~gt));
      if gt
        if lObj.isinit || ~lObj.hasMovie || ~lObj.gtIsGTMode
          % segLines are not visible; more importantly, cannot set segLine
          % highlighting based on suggestions in current movie
          return
        end
        
        % find rows for current movie
        tblLbled = table(lObj.gtSuggMFTableLbled,'variableNames',{'hasLbl'});
        tbl = [lObj.gtSuggMFTable tblLbled];
        mIdx = lObj.currMovIdx;
        tf = mIdx==tbl.mov;
        tblCurrMov = tbl(tf,:); % current mov, various frm/tgts
        
        % for hSegLineGT, we highlight any/all frames (regardless of, or across all, targets)
        frmsOn = tblCurrMov.frm; % could contain repeat frames (across diff targets)
        % obj.hSegLineGT.setOnAtOnly(frmsOn);
        setSegmentedLineOnAtOnlyBang(obj.hSegLineGT, obj.lObj.nframes, frmsOn) ;
        
        % For hSegLineGTLbled, we turn on a given frame only if all
        % targets/rows for that frame are labeled.
        tblRes = rowfun(@(zzHasLbl)all(zzHasLbl),tblCurrMov,...
                        'groupingVariables',{'frm'},'inputVariables','hasLbl',...
                        'outputVariableNames',{'allTgtsLbled'});
        frmsAllTgtsLbled = tblRes.frm(tblRes.allTgtsLbled);
        % obj.hSegLineGTLbled.setOnAtOnly(frmsAllTgtsLbled);
        setSegmentedLineOnAtOnlyBang(obj.hSegLineGTLbled, obj.lObj.nframes, frmsAllTgtsLbled) ;     
      end
    end

    function updateGTModeRelatedControlsLight(obj)
      % React to incremental update to labeler.gtSuggMFTableLbled
      
      lObj = obj.lObj;
      if ~lObj.gtIsGTMode
        % segLines are not visible,; more importantly, cannot set segLine
        % highlighting based on suggestions in current movie
        return
      end
      
      % find rows for current movie/frm
      tbl = lObj.gtSuggMFTable;
      currFrm = lObj.currFrame;
      tfCurrMovFrm = tbl.mov==lObj.currMovIdx & tbl.frm==currFrm;
      tfLbled = lObj.gtSuggMFTableLbled;
      tfLbledCurrMovFrm = tfLbled(tfCurrMovFrm,:);
      tfHiliteOn = numel(tfLbledCurrMovFrm)>0 && all(tfLbledCurrMovFrm);
      setSegmentedLineOnOffAtBang(obj.hSegLineGTLbled, currFrm, tfHiliteOn) ;
    end    

    function updateCurrentFrameLineWidths_(obj)
      itm = obj.lObj.infoTimelineModel ;
      selectOn = itm.selectOn;
      if selectOn
        obj.hCurrFrame.LineWidth = 3;
        obj.hCurrFrameL.LineWidth = 3;
      else
        obj.hCurrFrame.LineWidth = 0.5;
        obj.hCurrFrameL.LineWidth = 0.5;
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
    end  % function
  end  % methods  
end  % classdef
