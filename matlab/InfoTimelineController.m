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
  % The update() method is the core update method, and is a general-purpose
  % method to update all the controls that are owned by this object, no
  % matter the situation.  Additional less-general update*() methods are for
  % use in more specific settings when the performance of update() is
  % inadequate.

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
    hUncertainThresh  % scalar line handle, uncertain-frames threshold
    hCMenuClearBout  % scalar context menu
    hCMenuSetNumFramesShown
    hCMenuToggleThresholdViz
    hPts  % [nLabelPoints] line handles 
    hPtStat  % scalar line handle
    hPtsL  % [nLabelPoints] patch handles (non-MA projs), or [1] image handle (MA projs)    
    hSelIm  % scalar image handle for selection
    hSegLineGT  % scalar line handle
    hSegLineGTLbled  % scalar line handle
  end
  
  methods
    function obj = InfoTimelineController(labeler, mainTimelineAxes, isLabeledTimelineAxes)
      obj.lObj = labeler ;
      obj.hAx = mainTimelineAxes;
      obj.hCurrFrame = ...
        line('Parent',mainTimelineAxes, ...
             'XData',[nan nan], ...
             'YData',mainTimelineAxes.YLim, ...
             'LineStyle','-', ...
             'Color',[1 1 1],...
             'hittest','off', ...
             'Tag','InfoTimeline_CurrFrame');
      obj.hStatThresh = ...
        line('Parent',mainTimelineAxes, ...
             'XData',[nan nan], ...
             'YData',[0 0], ...
             'LineStyle','-', ...
             'Color',[1 1 1],...
             'hittest','off', ...
             'visible','off', ...
             'Tag','InfoTimeline_StatThresh');
      obj.hUncertainThresh = ...
        line('Parent',mainTimelineAxes, ...
             'XData',[nan nan], ...
             'YData',[0 0], ...
             'LineStyle','--', ...
             'Color',[1 1 1],...
             'hittest','off', ...
             'visible','off', ...
             'Tag','InfoTimeline_UncertainThresh');

      obj.hAxL = isLabeledTimelineAxes;
      
      obj.hCurrFrameL = ...
        line('Parent',isLabeledTimelineAxes, ...
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
      obj.hSegLineGT = line('Parent',mainTimelineAxes,'XData',nan,'YData',nan,'Tag','InfoTimeline_SegLineGT');
      obj.hSegLineGTLbled = line('Parent',mainTimelineAxes,'XData',nan,'YData',nan,'Tag','InfoTimeline_SegLineGTLbled');
      
      % Build the context menu for the main timeline axes
      hCMenu = ...
        uicontextmenu('Parent',mainTimelineAxes.Parent,...
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
      mainTimelineAxes.UIContextMenu = hCMenu;            

      % Make sure the main timeline axes and the is-labeled axes always have the
      % same XLim, even when user uses Matlab built-in zoom/pan features.
      linkaxes([obj.hAx,obj.hAxL],'x');
    end  % function
    
    function delete(obj)
      deleteValidGraphicsHandles([obj.hCurrFrame,obj.hCurrFrameL,obj.hStatThresh,obj.hUncertainThresh]);
      obj.hCurrFrame = [];
      obj.hCurrFrameL = [];
      obj.hStatThresh = [];
      obj.hUncertainThresh = [];
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
        
    function update(obj)
      % Bring all controls fully into sync with the model, regardless of
      % the current state of the model.  More-specific update*() methods
      % exist for use in more-specific settings when the performance of
      % this method is inadequate.

      lObj = obj.lObj ;

      % If there is no movie loaded, the timeline has nothing to show.
      % (hasMovie implies hasProject.)  Push all widgets to a known-good
      % blank state so they don't show stale data.
      if ~lObj.hasMovie
        % Delete per-project handles that may be stale
        deleteValidGraphicsHandles(obj.hPts) ;
        obj.hPts = [] ;
        deleteValidGraphicsHandles(obj.hPtStat) ;
        obj.hPtStat = [] ;
        deleteValidGraphicsHandles(obj.hPtsL) ;
        obj.hPtsL = [] ;
        % Delete per-movie handles that may be stale
        deleteValidGraphicsHandles(obj.hSelIm) ;
        obj.hSelIm = [] ;
        % Reset persistent line handles to show nothing
        set(obj.hCurrFrame, 'XData', [nan nan]) ;
        set(obj.hCurrFrameL, 'XData', [nan nan]) ;
        set(obj.hStatThresh, 'XData', [nan nan], 'Visible', 'off') ;
        set(obj.hUncertainThresh, 'XData', [nan nan], 'Visible', 'off') ;
        set(obj.hSegLineGT, 'XData', nan, 'YData', nan, 'Visible', 'off') ;
        set(obj.hSegLineGTLbled, 'XData', nan, 'YData', nan, 'Visible', 'off') ;
        % Disable context menu items
        set(obj.hCMenuClearBout, 'Enable', 'off') ;
        return ;
      end

      % From here on, we know a project and movie are loaded.

      % Ensure the per-landmark handle arrays (hPts, hPtStat, hPtsL) are
      % the right size for the current project.  If they're not,
      % updateForNewProject() will delete and recreate them.
      if numel(obj.hPts) ~= lObj.nLabelPoints
        obj.updateForProject_() ;
      end

      % Ensure the per-movie controls (selection image, segmented GT
      % lines) are initialized for the current movie's frame count.  This
      % must happen before updateGTModeRelatedControls(), because that
      % method sets data on the segmented GT lines using the current
      % movie's nframes.  If the lines are still sized for a previous
      % movie, the XData/YData sizes will be mismatched.
      obj.updateForMovie_() ;

      % Update the data traces (hPts, hPtStat, hPtsL)
      obj.updateTraces() ;

      % Update the landmark colors
      obj.updateLandmarkColors() ;

      % Update the current frame line position and axes limits.  Must
      % happen after updateTraces(), because updateTraces() sets YLim on
      % hAx, and updateCurrentFrameLineXData() reads YLim for
      % hCurrFrame's YData.
      obj.updateCurrentFrameLineXData_() ;

      % Update current frame line widths (depends on selection mode)
      obj.updateCurrentFrameLineWidths_() ;

      % Update selection image CData
      obj.updateSelectionImageCData_() ;

      % Update statistic threshold display
      obj.updateStatThresh() ;

      % Update uncertain-frames threshold display
      obj.updateUncertainThresh() ;

      % Update GT mode related controls (segmented line visibility and
      % data).  Note: updateForMovie_() above also calls this, so it
      % is redundant in most cases, but is needed when only the GT mode
      % has changed without a movie change.
      obj.updateGTModeRelatedControls() ;

      % Update context menu
      obj.updateContextMenu_() ;
    end  % function

    function updateForProject_(obj)
      % Update the controls to match the current project.

      % Get the core things we need from the labeler
      lObj = obj.lObj ;
      nLabelPoints = lObj.nLabelPoints ;
      isMA = lObj.maIsMA ;
      colors = lObj.LabelPointColors() ;
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
               'Color',colors(i,:), ...
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
      ax.YColor = prefsXColor;
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
      set(obj.hUncertainThresh,'XData',[nan nan],'ZData',[1 1]);
    end
    
    function updateForMovie_(obj)
      % Update the controls to sync with the current movie.

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
      PURPLE = [80 31 124]/256 ;
      obj.hAx.Colormap = [ 0 0 0 ; PURPLE ] ;      
      xlims = [1 nframes];
      sPV = struct('LineWidth',5,'Color',AxesHighlightManager.ORANGE);
      sPVLbled = struct('LineWidth',5,'Color',AxesHighlightManager.ORANGE/2);
      initSegmentedLineBang(obj.hSegLineGT,xlims,sPV);
      initSegmentedLineBang(obj.hSegLineGTLbled,xlims,sPVLbled);

      % Call another update method to handle the GT-related controls
      obj.updateGTModeRelatedControls();
    end
            
    function updateTraces(obj)
      % Update .hPts, .hMarked, .hPtStat
      
      lObj = obj.lObj ;
      if ~lObj.hasMovie
        set(obj.hPts, 'XData', nan, 'YData', nan) ;
        set(obj.hPtStat, 'XData', nan, 'YData', nan) ;
        set(obj.hPtsL, 'XData', nan, 'YData', nan) ;
        set(obj.hCurrFrame, 'XData', [nan nan]) ;
        set(obj.hStatThresh, 'XData', [nan nan]) ;
        set(obj.hUncertainThresh, 'XData', [nan nan]) ;
        set(obj.hAx, 'YLim', [0 1]) ;
        return
      end

      traceData = lObj.getTimelineDataForCurrentMovieAndTarget();  % [nLabelPoints x nFrames]
      nonnanTraceData = traceData(~isnan(traceData));

      set(obj.hPts,'XData',nan,'YData',nan);
      set(obj.hPtStat,'XData',nan,'YData',nan);
      
      if ~isempty(nonnanTraceData)        
        y1 = min(nonnanTraceData(:));
        y2 = max(nonnanTraceData(:));
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
        x = 1:size(traceData,2);
        if ishandle(obj.hSelIm),
          set(obj.hSelIm,'YData',[y1,y2]);
        end
        
        % Expand the y range to include the uncertain-frames threshold if it
        % would be visible (UFC is visible and timeline shows confidence)
        ufm = lObj.uncertainFramesModel_ ;
        itm = lObj.infoTimelineModel ;
        [ptype, prop] = itm.getCurPropSmart() ;
        isShowingConfidence = strcmp(ptype, 'Predictions') && ...
                              isfield(prop, 'feature') && strcmp(prop.feature, 'confidence') ;
        if ufm.isVisible && isfinite(ufm.absoluteConfidenceThreshold) && isShowingConfidence
          y1 = min(y1, ufm.absoluteConfidenceThreshold) ;
          y2 = max(y2, ufm.absoluteConfidenceThreshold) ;
        end

        set(obj.hAx,'YLim',[y1,y2]);
        set(obj.hCurrFrame,'YData',[y1,y2]);
        if size(traceData,1) == lObj.nLabelPoints,
          for i=1:lObj.nLabelPoints
            set(obj.hPts(i),'XData',x,'YData',traceData(i,:));
          end
        elseif size(traceData,1) == 1,
          set(obj.hPtStat,'XData',x,'YData',traceData(1,:));
        else
          warningNoTrace(sprintf('InfoTimeline: Number of rows in statistics was %d, expected either %d or 1',size(traceData,1),lObj.nLabelPoints));
        end
        
        set(obj.hStatThresh,'XData',x([1 end]));
        set(obj.hUncertainThresh,'XData',x([1 end]));
      end  % if ~isempty(traceData)
      
      if lObj.maIsMA
        tflbledDisp = lObj.getLabeledTgts(obj.axLmaxntgt) ;
        set(obj.hPtsL,'CData',uint8(tflbledDisp')) ;
      else
        islabeled = lObj.getIsLabeledCurrMovTgt() ; % [nLabelPoints x nFrames]
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
    
    % function updateAfterCurrentFrameSet(obj)
    %   % This gets called after the user changes the frame they're looking at, i.e.
    %   % after labeler.currFrame is set.      
    %   if isnan(obj.lObj.nLabelPoints), return; end      
    %   obj.updateCurrentFrameLineXData() ;      
    %   obj.updateSelectionImageCData() ;
    %   obj.updateContextMenu() ;
    % end  % function

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
      if ~isnan(obj.lObj.nframes)
        obj.hAx.XTick = 0 : dxtick : obj.lObj.nframes ;
      end
      obj.hAx.XLim = [x0 x1];
      set(obj.hCurrFrame,'XData',[currFrame currFrame],'YData',obj.hAx.YLim);
      obj.hAxL.XLim = [x0 x1];
      set(obj.hCurrFrameL,'XData',[currFrame currFrame],'YData',obj.hAxL.YLim);
    end  % function
    
    function updateSelectionImageCData_(obj)
      % Update the selection-highlight image from the model state.
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
    end  % function

    function updateUncertainThresh(obj)
      % Update the uncertain-frames threshold display from the model.
      % Only visible when the UFC is visible and the timeline is showing a
      % Predictions confidence feature.
      ufm = obj.lObj.uncertainFramesModel_ ;
      threshold = ufm.absoluteConfidenceThreshold ;
      itm = obj.lObj.infoTimelineModel ;
      [ptype, prop] = itm.getCurPropSmart() ;
      isShowingConfidence = strcmp(ptype, 'Predictions') && ...
                            isfield(prop, 'feature') && strcmp(prop.feature, 'confidence') ;
      isVisible = ufm.isVisible && isShowingConfidence ;
      tidyThreshold = fif(isempty(threshold), nan, threshold) ;
      obj.hUncertainThresh.YData = [tidyThreshold tidyThreshold] ;
      obj.hUncertainThresh.Visible = onIff(isVisible) ;
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

    function updateContextMenu_(obj)
      lObj = obj.lObj ;
      % Gray menu item if not in a bout
      set(obj.hCMenuClearBout,'Enable',onIff(lObj.isCurrentFrameSelected()));
    end  % function

    function updateSelection(obj)
      obj.updateCurrentFrameLineWidths_() ;
      obj.updateCurrentFrameLineXData_() ;
      obj.updateSelectionImageCData_() ;
      obj.updateContextMenu_() ;      
    end  % function
  end  % methods  
end  % classdef
