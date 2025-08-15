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

      axti.Color = [0 0 0];
      axti.ButtonDownFcn = @(src,evt)(obj.cbkBDF(src,evt));
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
            
      listeners = cell(0,1);
      % listeners{end+1,1} = ...
      %   addlistener(labeler, 'didSetLabels', @obj.cbkLabelUpdated) ;
      listeners{end+1,1} = ...
        addlistener(labeler, 'gtSuggMFTableLbledUpdated',@(s,e)(obj.cbkGTSuggMFTableLbledUpdated())) ;
      listeners{end+1,1} = ...
          addlistener(labeler, 'newTrackingResults', @(s,e)(obj.cbkNewTrackingResults())) ;      
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
        
    function updateForNewProject(obj)
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
      if isMA
        axl.YLim = [0-dy obj.axLmaxntgt+dy];
        axl.Colormap = [0 0 0 ; 0 0 1] ;
        axl.YDir = 'reverse' ;
      else
        axl.YLim = [0-dy obj.lObj.nLabelPoints+dy];
        axl.YDir = 'normal' ;
      end
      
      set(obj.hCurrFrame,'XData',[nan nan],'YData',ax.YLim,'ZData',[1 1]);
      set(obj.hCurrFrameL,'XData',[nan nan],'YData',axl.YLim,'ZData',[1 1]);
      set(obj.hStatThresh,'XData',[nan nan],'ZData',[1 1]);
    end
    
    function updateForNewMovie(obj)
      ax = obj.hAx;
      prefsTL = obj.lObj.projPrefs.InfoTimelines;
      ax.XTick = 0:prefsTL.dXTick:obj.nframes_();

      if obj.lObj.isinit || isnan(obj.nframes_())
        return
      end

      deleteValidGraphicsHandles(obj.hSelIm);
      obj.hSelIm = ...
        image('Parent', obj.hAx, ...
              'XData', 1:obj.nframes_(), ...
              'YData', obj.hAx.YLim, ...
              'CData', uint8(zeros(1,obj.nframes_())), ...
              'HitTest', 'off',...
              'CDataMapping', 'direct') ;

      % itm.setSelectMode(false) ;
      colorTBSelect = obj.parent_.tbTLSelectMode.BackgroundColor;
      obj.hAx.Colormap = [0 0 0;colorTBSelect] ;
      
      xlims = [1 obj.nframes_()];
      sPV = struct('LineWidth',5,'Color',AxesHighlightManager.ORANGE);
      sPVLbled = struct('LineWidth',5,'Color',AxesHighlightManager.ORANGE/2);
      initSegmentedLineBang(obj.hSegLineGT,xlims,sPV);
      initSegmentedLineBang(obj.hSegLineGTLbled,xlims,sPVLbled);
      % if obj.lObj.infoTimelineModel.getCurPropTypeIsAllFrames(),
      %   obj.lObj.setTimelineCurrentPropertyTypeToDefault();
      % end      
      cbkGTSuggUpdated(obj,[],[]);
    end
            
    function updateLabels(obj)
      % Get data and set .hPts, .hMarked, hPtStat
      
      if isempty(obj.lObj.nLabelPoints) || isnan(obj.lObj.nLabelPoints)
        return
      end
      
      dat = obj.lObj.getTimelineDataForCurrentMovieAndTarget(); % [nptsxnfrm]
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
      nominal_xspan = 2*obj.lObj.projPrefs.InfoTimelines.FrameRadius;
      nominal_dxtick = obj.lObj.projPrefs.InfoTimelines.dXTick ;
      if nominal_xspan==0 || nominal_xspan > obj.nframes_()
        x0 = 1;
        x1 = obj.nframes_();
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
        elseif x1_raw>obj.nframes_()
          x1 = obj.nframes_() ;
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
      obj.hAx.XTick = 0 : dxtick : obj.nframes_() ;
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
    end
    
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
    end
    
    function setCurProp(obj, iprop)
      itm = obj.lObj.infoTimelineModel ;
      if itm.getCurPropTypeIsAllFrames() && strcmpi(itm.props_allframes(iprop).name,'Add custom...')
        movfile = obj.lObj.getMovieFilesAllFullMovIdx(obj.lObj.currMovIdx);
        defaultpath = fileparts(movfile{1});
        [f,p] = uigetfile('*.mat','Select .mat file with a feature value for each frame for current movie',defaultpath);
        if ~ischar(f)
          return
        end
        file = fullfile(p,f);
        obj.lObj.addCustomTimelineFeatureGivenFileName(file) ;
      else
        obj.lObj.setTimelineCurrentPropertyType(itm.curproptype, iprop) ;
      end
    end

    % function updatePropsGUI(obj)
    %   itm = obj.lObj.infoTimelineModel ;
    %   obj.parent_.pumInfo_labels.Value = itm.curproptype;
    %   props = itm.getPropsDisp(itm.curproptype);
    %   obj.parent_.pumInfo.String = props;
    %   obj.parent_.pumInfo.Value = itm.curprop;
    % end

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
          ef = obj.nframes_();
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

    function cbkLabelUpdated(obj, ~, ~)
      if ~obj.lObj.isinit ,
        obj.updateLabels() ;
      end
    end
    
    function cbkNewTrackingResults(obj)
      obj.lObj.setCurPropTypePredictionDefault();
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
      obj.lObj.toggleTimelineIsStatThreshVisible();
    end

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
      setSegmentedLineOnAtOnlyBang(obj.hSegLineGT, obj.nframes_(), frmsOn) ;
      
      % For hSegLineGTLbled, we turn on a given frame only if all
      % targets/rows for that frame are labeled.
      tblRes = rowfun(@(zzHasLbl)all(zzHasLbl),tblCurrMov,...
        'groupingVariables',{'frm'},'inputVariables','hasLbl',...
        'outputVariableNames',{'allTgtsLbled'});
      frmsAllTgtsLbled = tblRes.frm(tblRes.allTgtsLbled);
      % obj.hSegLineGTLbled.setOnAtOnly(frmsAllTgtsLbled);
      setSegmentedLineOnAtOnlyBang(obj.hSegLineGTLbled, obj.nframes_(), frmsAllTgtsLbled) ;     
    end

    function cbkGTSuggMFTableLbledUpdated(obj)
      % React to incremental update to labeler.gtSuggMFTableLbled
      
      lObj = obj.lObj;
      if ~lObj.gtIsGTMode
        % segLines are not visible,; more importantly, cannot set segLine
        % highlighting based on suggestions in current movie
        return;
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
    end

    function v = nframes_(obj)
      % Reuturns the number of frames in the current movie, or one if there are no
      % movies.  Gets this info from the Labeler.
      lObj = obj.lObj;
      if lObj.hasMovie
        v = lObj.nframes;
      else
        v = 1;
      end
    end  % function
  end  % methods  
end  % classdef
