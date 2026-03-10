classdef LabelCoreHTController < LabelCoreController
% High-throughput labeling controller
%
% Owns all graphics for HT labeling mode. Receives GUI callbacks,
% extracts GUI state (mouse position, modifiers), delegates data logic
% to LabelCoreHTModel, and syncs graphics in response to model events.
%
% HT mode is unusual in several ways:
% - There is no LABEL/ADJUST/ACCEPTED state machine
% - tbAccept is always disabled
% - Only one point (iPoint) is active/clickable at a time
% - The frame auto-advances after each click

  properties
    supportsSingleView = true ;
    supportsMultiView = false ;
    supportsCalibration = false ;
    supportsMultiAnimal = false ;
  end

  properties (Transient)
    unlabeledPointColor_      % [1x3] RGB color for the unlabeled active point
    otherLabeledPointColor_   % [1x3] RGB color for labeled points that are not iPoint
  end

  methods

    function obj = LabelCoreHTController(labelerController, model)
      % Construct a LabelCoreHTController.
      obj = obj@LabelCoreController(labelerController, model) ;
    end  % function

    function initHook(obj)
      % Initialize HT-specific graphics state.

      mdl = obj.model_ ;
      ppi = mdl.ptsPlotInfo_ ;
      htm = ppi.HighThroughputMode ;
      obj.unlabeledPointColor_ = htm.UnlabeledPointColor ;
      obj.otherLabeledPointColor_ = htm.OtherLabeledPointColor ;

      set(obj.hPts_, 'HitTest', 'off') ;
      set(obj.hPtsTxt_, 'PickableParts', 'none') ;
      set(obj.tbAccept_, 'Enable', 'off') ;

      % Register listener for updateIPoint (HT-specific event)
      obj.listeners_(end+1) = ...
        addlistener(mdl, 'updateIPoint', @(s, e)obj.onUpdateIPoint()) ;

      % Trigger initial iPoint UI setup
      obj.onUpdateIPoint() ;
    end  % function

  end  % methods

  %% Model event handlers
  methods

    function onUpdateIPoint(obj)
      % Sync UI when iPoint changes: context menu, HitTest, point visibility.

      mdl = obj.model_ ;
      iPt = mdl.iPoint_ ;

      % Turn off HitTest for all points, then enable for iPoint
      set(obj.hPts_, 'HitTest', 'off') ;
      set(obj.hPts_(iPt), 'HitTest', 'on') ;

      % Clear old context menu and set up new one
      for i = 1:numel(obj.hPts_)
        obj.hPts_(i).UIContextMenu = [] ;
      end
      obj.setupIPointContextMenu_() ;
    end  % function

    function onUpdateLabelCoords(obj)
      % Sync all point graphics for a new frame in HT mode.
      % This overrides the base class to handle HT-specific coloring,
      % marker, and positioning logic.

      mdl = obj.model_ ;
      iPt = mdl.iPoint_ ;
      xy = mdl.xy_ ;
      hPoints = obj.hPts_ ;
      hPointsOcc = obj.hPtsOcc_ ;
      ppi = mdl.ptsPlotInfo_ ;
      colors = ppi.Colors ;

      % Positioning: sync all points to model xy_
      txtOffset = mdl.labeler_.labelPointsPlotInfo.TextOffset ;
      setPositionsOfLabelLinesAndTextsBangBang(hPoints, obj.hPtsTxt_, xy, txtOffset) ;

      % Read frame label metadata from model
      lastFrame = mdl.lastFrameLabeled_ ;
      if isempty(lastFrame)
        return ;
      end
      tfLabeledOrOcc = lastFrame.tfLabeledOrOcc ;
      tfOccTag = lastFrame.tfOccTag ;

      % COLORING
      % - All labeled/occluded points that are not iPoint are dimmed
      % - iPoint is colored if labeled/occluded, otherwise unlabeledPointColor
      % - All other points (unlabeled, unoccluded, not iPoint) are hidden so
      %   coloring is irrelevant
      tfOtherLabeled = tfLabeledOrOcc ;
      tfOtherLabeled(iPt) = false ;
      clrOther = obj.otherLabeledPointColor_ ;
      set(hPoints(tfOtherLabeled), 'Color', clrOther) ;
      if ~isempty(hPointsOcc)
        set(hPointsOcc(tfOtherLabeled), 'Color', clrOther) ;
      end

      if tfLabeledOrOcc(iPt)
        clrIPt = colors(iPt, :) ;
      else
        clrIPt = obj.unlabeledPointColor_ ;
      end
      set(hPoints(iPt), 'Color', clrIPt) ;
      if ~isempty(hPointsOcc)
        set(hPointsOcc(iPt), 'Color', clrIPt) ;
      end

      % MARKER
      % - All labeled or pure-occluded use regular Marker
      % - All labeled and tag-occluded use OccludedMarker
      % - Unlabeled: don't change marker (preserve previous state for iPoint)
      mrkr = ppi.MarkerProps.Marker ;
      mrkrOcc = ppi.OccludedMarker ;

      set(hPoints(tfLabeledOrOcc & ~tfOccTag), 'Marker', mrkr) ;
      set(hPoints(tfLabeledOrOcc & tfOccTag), 'Marker', mrkrOcc) ;
    end  % function

  end  % methods

  %% GUI callback handlers
  methods

    function axBDF(obj, ~, evt)
      % Handle axis button-down: set point position, colorize, write labels,
      % and advance frame.

      mdl = obj.model_ ;
      if ~mdl.labeler_.isReady || evt.Button > 1
        return ;
      end
      if obj.isPanZoom()
        return ;
      end

      mod = obj.hFig_(1).CurrentModifier ;
      tfShift = any(strcmp(mod, 'shift')) ;

      pos = get(obj.hAx_(1), 'CurrentPoint') ;
      pos = pos(1, 1:2) ;
      iPt = mdl.iPoint_ ;

      % Update model coordinates
      mdl.xy_(iPt, :) = pos ;

      % Update point graphics
      ppi = mdl.ptsPlotInfo_ ;
      if ~tfShift
        set(obj.hPts_(iPt), ...
          'Color', ppi.Colors(iPt, :), ...
          'Marker', ppi.MarkerProps.Marker) ;
        mdl.labeler_.labelPosTagClearI(iPt) ;
      else
        set(obj.hPts_(iPt), ...
          'Color', ppi.Colors(iPt, :), ...
          'Marker', ppi.OccludedMarker) ;
        mdl.labeler_.labelPosTagSetI(iPt) ;
      end
      txtOffset = mdl.labeler_.labelPointsPlotInfo.TextOffset ;
      setPositionsOfLabelLinesAndTextsBangBang( ...
        obj.hPts_(iPt), obj.hPtsTxt_(iPt), pos, txtOffset) ;

      % Write label to Labeler
      mdl.labeler_.labelPosSetI(pos, iPt) ;

      % Advance frame
      obj.clickedIncrementFrame_() ;
    end  % function

    function ptBDF(obj, src, evt)
      % Handle point button-down: if clicked point is iPoint, accept it.
      if ~obj.model_.labeler_.isReady || evt.Button > 1
        return ;
      end
      if obj.isPanZoom()
        return ;
      end

      ud = src.UserData ;
      if ud == obj.model_.iPoint_
        obj.acceptCurrentPt_() ;
      end
    end  % function

    function tfKPused = kpf(obj, ~, evt)
      % Handle key press: space accepts, arrows navigate frames.

      mdl = obj.model_ ;
      if ~mdl.labeler_.isReady
        tfKPused = false ;
        return ;
      end

      key = evt.Key ;
      modifier = evt.Modifier ;
      tfCtrl = any(strcmp('control', modifier)) ;

      tfKPused = true ;
      if strcmp(key, 'space')
        obj.acceptCurrentPt_() ;
      elseif any(strcmp(key, {'equal' 'rightarrow' 'd'})) && ~tfCtrl
        obj.labelerController_.frameUpDF(mdl.nFrameSkip_) ;
      elseif any(strcmp(key, {'hyphen' 'leftarrow' 'a'})) && ~tfCtrl
        obj.labelerController_.frameDownDF(mdl.nFrameSkip_) ;
      else
        tfKPused = false ;
      end
    end  % function

    function axOccBDF(obj, ~, ~)
      % Handle occluded-axis button-down: occlude current point and advance.

      mdl = obj.model_ ;
      if ~mdl.labeler_.isReady
        return ;
      end
      if obj.isPanZoom()
        return ;
      end

      iPt = mdl.iPoint_ ;
      mdl.tfOcc_(iPt) = true ;
      set(obj.hPtsOcc_(iPt), 'Color', mdl.ptsPlotInfo_.Colors(iPt, :)) ;
      obj.refreshOccludedPts() ;
      mdl.labeler_.labelPosSetOccludedI(iPt) ;
      tfOcc = mdl.labeler_.labelPosIsOccluded() ;
      assert(isequal(tfOcc, mdl.tfOcc_)) ;

      mdl.labeler_.labelPosTagClearI(iPt) ;

      obj.clickedIncrementFrame_() ;
    end  % function

  end  % methods

  %% Accept helpers (private-in-spirit)
  methods

    function tfEndOfMovie = acceptCurrentPt_(obj)
      % Accept the current point as-is: read position from model xy_, write
      % to Labeler, handle tag, and advance frame.

      mdl = obj.model_ ;
      iPt = mdl.iPoint_ ;
      pos = mdl.xy_(iPt, :) ;
      ppi = mdl.ptsPlotInfo_ ;

      set(obj.hPts_(iPt), 'Color', ppi.Colors(iPt, :)) ;
      lObj = mdl.labeler_ ;
      lObj.labelPosSetI(pos, iPt) ;

      % Determine tag from current marker
      mrkr = obj.hPts_(iPt).Marker ;
      assert(~strcmp(ppi.MarkerProps.Marker, ppi.OccludedMarker), ...
        'Marker and OccludedMarker are identical. Please specify distinguishable markers.') ;
      switch mrkr
        case ppi.MarkerProps.Marker
          lObj.labelPosTagClearI(iPt) ;
        case ppi.OccludedMarker
          lObj.labelPosTagSetI(iPt) ;
        otherwise
          assert(false) ;
      end

      tfEndOfMovie = obj.clickedIncrementFrame_() ;
    end  % function

    function acceptCurrentPtN_(obj, nRepeat)
      % Accept the current point nRepeat times across skip-frames.

      assert(nRepeat > 0) ;

      mdl = obj.model_ ;
      iPt = mdl.iPoint_ ;
      pos = mdl.xy_(iPt, :) ;
      ppi = mdl.ptsPlotInfo_ ;

      set(obj.hPts_(iPt), 'Color', ppi.Colors(iPt, :)) ;

      lObj = mdl.labeler_ ;
      frm0 = lObj.currFrame ;
      frmsMax = min(lObj.nframes, frm0 + (nRepeat - 1) * mdl.nFrameSkip_) ;
      frms = frm0:mdl.nFrameSkip_:frmsMax ;
      nActual = numel(frms) ;
      if nActual ~= nRepeat
        str = sprintf('End of movie reached; %d points labeled (over duration of %d frames)', ...
          nActual, (nActual - 1) * mdl.nFrameSkip_) ;
        msgbox(str, 'End of movie') ;
      end

      lObj.labelPosSetFramesI(frms, pos, iPt) ;

      % Handle tag based on current marker
      mrkr = obj.hPts_(iPt).Marker ;
      assert(~strcmp(ppi.MarkerProps.Marker, ppi.OccludedMarker), ...
        'Marker and OccludedMarker are identical. Please specify distinguishable markers.') ;
      switch mrkr
        case ppi.MarkerProps.Marker
          lObj.labelPosTagClearFramesI(iPt, frms) ;
        case ppi.OccludedMarker
          lObj.labelPosTagSetFramesI(iPt, frms) ;
        otherwise
          assert(false) ;
      end

      dfrm = frms(end) - frms(1) ;
      tfEndOfMovie = obj.clickedIncrementFrame_(dfrm) ;
      if tfEndOfMovie
        warningNoTrace('LabelCoreHTController:EOM', 'End of movie reached.') ;
      end
    end  % function

    function acceptCurrentPtNPrompt_(obj)
      % Prompt user for repeat count, then accept current point that many times.
      resp = inputdlg('Number of times to accept point:', ...
        'Label current point repeatedly', 1, {'1'}) ;
      if isempty(resp)
        return ;
      end
      nRepeat = str2double(resp{1}) ;
      if isnan(nRepeat) || round(nRepeat) ~= nRepeat || nRepeat <= 0
        error('LabelCoreHTController:input', 'Input must be a positive integer.') ;
      end
      obj.acceptCurrentPtN_(nRepeat) ;
    end  % function

    function acceptCurrentPtNFramesPrompt_(obj)
      % Prompt user for frame count, compute repeat count, accept.
      resp = inputdlg('Accept point over next N frames:', ...
        'Label current point repeatedly', 1, {'1'}) ;
      if isempty(resp)
        return ;
      end
      nFrames = str2double(resp{1}) ;
      if isnan(nFrames) || round(nFrames) ~= nFrames || nFrames <= 0
        error('LabelCoreHTController:input', 'Input must be a positive integer.') ;
      end
      nRepeat = ceil(nFrames / obj.model_.nFrameSkip_) ;
      obj.acceptCurrentPtN_(nRepeat) ;
    end  % function

    function acceptCurrentPtEnd_(obj)
      % Accept current point from here to end of movie.
      mdl = obj.model_ ;
      nFrames = mdl.labeler_.nframes - mdl.labeler_.currFrame + 1 ;
      nRepeat = ceil(nFrames / mdl.nFrameSkip_) ;
      obj.acceptCurrentPtN_(nRepeat) ;
    end  % function

  end  % methods

  %% Frame increment (private-in-spirit)
  methods

    function tfEndOfMovie = clickedIncrementFrame_(obj, dfrm)
      % Increment the frame after a click or accept action. Handles
      % end-of-movie by advancing iPoint or showing completion message.

      mdl = obj.model_ ;
      if ~exist('dfrm', 'var')
        dfrm = mdl.nFrameSkip_ ;
      end

      nf = mdl.labeler_.nframes ;
      f = mdl.labeler_.currFrame ;
      iPt = mdl.iPoint_ ;
      nPt = mdl.nPts_ ;
      tfEndOfMovie = (f + dfrm > nf) ;
      if tfEndOfMovie
        if iPt == nPt
          str = sprintf('End of movie reached. Labeling complete for all %d points!', nPt) ;
          msgbox(str, 'Labeling Complete') ;
        else
          iPtNext = iPt + 1 ;
          str = sprintf('End of movie reached. Proceeding to labeling for point %d out of %d.', ...
            iPtNext, nPt) ;
          msgbox(str, 'End of movie reached') ;
          mdl.setIPoint(iPtNext) ;
          mdl.labeler_.setFrameGUI(1) ;
        end
      else
        obj.labelerController_.frameUpDF(dfrm) ;
      end
    end  % function

  end  % methods

  %% Context menu (private-in-spirit)
  methods

    function setupIPointContextMenu_(obj)
      % Create the right-click context menu for the active iPoint.

      mdl = obj.model_ ;
      c = uicontextmenu(obj.labelerController_.mainFigure_) ;
      hPt = obj.hPts_(mdl.iPoint_) ;
      hPt.UIContextMenu = c ;
      uimenu(c, 'Label', 'Accept point for current frame', ...
        'Callback', @(src, evt)obj.acceptCurrentPt_) ;
      uimenu(c, 'Label', ...
        sprintf('Accept point N times (N*%d frames)', mdl.nFrameSkip_), ...
        'Callback', @(src, evt)obj.acceptCurrentPtNPrompt_) ;
      uimenu(c, 'Label', ...
        sprintf('Accept point over N frames (N/%d times)', mdl.nFrameSkip_), ...
        'Callback', @(src, evt)obj.acceptCurrentPtNFramesPrompt_) ;
      uimenu(c, 'Label', 'Accept point until end of movie', ...
        'Callback', @(src, evt)obj.acceptCurrentPtEnd_) ;
    end  % function

  end  % methods

  %% Presentation
  methods

    function h = getLabelingHelp(obj) %#ok<MANU>
      % Return labeling help text for HT mode.
      h = { ...
        '* Left-click labels a point and auto-advances the movie.' ; ...
        '* Right-click labels an estimate/occluded point and auto-advances the movie.' ; ...
        '* Right-click the current point for additional labeling options.' ; ...
        '* A/D, LEFT/RIGHT, or MINUS(-)/EQUAL(=) decrements/increments the frame shown.' ; ...
        '* <space> accepts the point as-is for the current frame.'} ;
    end  % function

  end  % methods

end  % classdef
