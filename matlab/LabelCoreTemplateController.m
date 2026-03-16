classdef LabelCoreTemplateController < LabelCoreController
% Template-based labeling controller
%
% Owns all graphics for template labeling mode. Receives GUI callbacks,
% extracts GUI state (mouse position, modifiers), delegates data logic
% to LabelCoreTemplateModel, and syncs graphics in response to model events.
%
% This is the controller half of the LabelCoreTemplate MVC split. The
% model half is LabelCoreTemplateModel.

  properties
    supportsSingleView = true ;
    supportsMultiView = false ;
    supportsCalibration = false ;
    supportsMultiAnimal = false ;
  end

  properties (Transient)
    % See cosmetics discussion in LabelCoreTemplate.m. Predicted-unadjustedness
    % (or not) toggles Marker Color, Other Marker Props, and Marker Txt Angle.
    hPtsMarkerPVPredUnadjusted_         % struct, HG PV-pairs for unadjusted tracking predictions
    hPtsMarkerPVNotPredUnadjusted_      % struct, HG PV-pairs for not that; reverts the above
    hPtsTxtPVPredUnadjusted_            % struct, HG PV-pairs for unadjusted text
    hPtsTxtPVNotPredUnadjusted_         % struct, HG PV-pairs for not-unadjusted text
  end

  methods

    function obj = LabelCoreTemplateController(labelerController, labeler, model)
      % Construct a LabelCoreTemplateController.
      obj = obj@LabelCoreController(labelerController, labeler, model) ;
    end  % function

    function initHook(obj)
      % Initialize Template-specific graphics state.

      obj.updatePredUnadjustedPVs() ;

      obj.txLblCoreAux_.Visible = 'on' ;
      obj.refreshTxLabelCoreAux() ;

      % Register listener for the Template-specific updateAdjusted event
      mdl = obj.model_ ;
      obj.listeners_ = [ ...
        obj.listeners_ ; ...
        addlistener(mdl, 'updateAdjusted', @(s,e)obj.onUpdateAdjusted()) ; ...
      ] ;

      % LabelCore should probably not talk directly to tracker
      tObj = obj.labeler_.tracker ;
      if ~isempty(tObj) && ~tObj.hideViz
        warningNoTrace('LabelCoreTemplate:viz', ...
          'Enabling View>Hide Predictions. Tracking predictions (when present) are now shown as template points in Template Mode.') ;
        tObj.setHideViz(true) ;
      end
    end  % function

  end  % methods

  %% Model event handlers
  methods

    function onUpdateState(obj)
      % Sync tbAccept appearance to model state.
      mdl = obj.model_ ;
      switch mdl.state
        case LabelState.ADJUST
          set(obj.tbAccept_, 'BackgroundColor', [0.6 0 0], 'String', 'Accept', ...
            'Value', 0, 'Enable', 'on') ;
        case LabelState.ACCEPTED
          set(obj.tbAccept_, 'BackgroundColor', [0 0.4 0], 'String', 'Labeled', ...
            'Value', 1, 'Enable', 'off') ;
        otherwise
          error('LabelCoreTemplateController:unknownState', ...
                'Unknown state %s.', char(mdl.state)) ;
      end
    end  % function

    function onUpdateAdjusted(obj)
      % Sync color/marker cosmetics for adjusted/unadjusted points.
      mdl = obj.model_ ;
      iPt = mdl.lastChangedIPt ;
      if iPt == 0
        % All points changed
        obj.refreshAdjustedCosmeticsAll() ;
      else
        % Single point changed
        obj.refreshAdjustedCosmeticsI(iPt) ;
      end
    end  % function

  end  % methods

  %% GUI callback handlers
  methods

    function axBDF(obj, src, evt) %#ok<INUSL>
      % Handle axis button-down: jump selected point to click location.
      if obj.isPanZoom() || evt.Button > 1
        return
      end
      pos = evt.IntersectionPoint(1:2) ;
      mdl = obj.model_ ;
      mdl.jumpTo(pos) ;
    end  % function

    function ptBDF(obj, src, evt)
      % Handle point button-down: select point or toggle est-occ.
      mdl = obj.model_ ;
      if ~obj.labeler_.isReady || evt.Button > 1
        return ;
      end
      if obj.isPanZoom()
        return ;
      end

      mod = obj.hFig_(1).CurrentModifier ;
      tfShift = any(strcmp(mod, 'shift')) ;
      if ~tfShift
        iPt = get(src, 'UserData') ;
        mdl.toggleSelectPoint(iPt) ;
        % prepare for click-drag of pt
        mdl.iPtMove_ = iPt ;
        mdl.tfMoved_ = false ;
      else
        iPt = get(src, 'UserData') ;
        mdl.toggleEstOccPoint(iPt) ;
      end
    end  % function

    function wbmf(obj, src, evt)  %#ok<INUSD>
      % Handle window button motion: drag selected point.
      mdl = obj.model_ ;
      if ~obj.labeler_.isReady
        return ;
      end

      pos = evt.IntersectionPoint(1:2) ;
      if mdl.state == LabelState.ADJUST || mdl.state == LabelState.ACCEPTED
        iPt = mdl.iPtMove_ ;
        if ~isnan(iPt)
          % tmp = get(obj.hAx_(1), 'CurrentPoint') ;
          % pos = tmp(1, 1:2) ;
          mdl.tfMoved_ = true ;
          mdl.xy(iPt, :) = pos ;
          obj.syncPointGraphicsI(iPt) ;
          mdl.setPointAdjusted(iPt) ;
        end
      end
    end  % function

    function wbuf(obj, ~, ~)
      % Handle window button up: end drag, handle click-without-move.
      mdl = obj.model_ ;
      if ~obj.labeler_.isReady
        return ;
      end

      if mdl.state == LabelState.ADJUST || mdl.state == LabelState.ACCEPTED
        iPt = mdl.iPtMove_ ;
        if ~isnan(iPt) && ~mdl.tfMoved_
          % point was clicked but not moved
          mdl.clearSelected(iPt) ;
          mdl.toggleSelectPoint(iPt) ;
        end

        mdl.iPtMove_ = nan ;
        if mdl.state == LabelState.ACCEPTED && ~isnan(iPt) && mdl.tfMoved_
          mdl.storeLabels() ;
        end
        mdl.tfMoved_ = false ;
      end
    end  % function

    function tfKPused = kpf(obj, src, evt)
      % Handle key press: accept, frame nav, arrow-adjust, point select.
      mdl = obj.model_ ;
      if ~obj.labeler_.isReady
        tfKPused = false ;
        return ;
      end

      key = evt.Key ;
      modifier = evt.Modifier ;
      tfCtrl = any(strcmp('control', modifier)) ;

      tfKPused = true ;

      % Hack iss#58. Ensure focus is not on slider_frame. In practice this
      % callback is called before slider_frame_Callback when slider_frame
      % has focus.
      txStatus = obj.labelerController_.txStatus ;
      if src ~= txStatus  % protect against repeated kpfs (eg scrolling vid)
        uicontrol(txStatus) ;
      end

      if any(strcmp(key, {'s' 'space'})) && ~tfCtrl
        if mdl.state == LabelState.ADJUST
          mdl.acceptLabels() ;
        end
      elseif any(strcmp(key, {'d' 'equal'}))
        obj.labelerController_.frameUp(tfCtrl) ;
      elseif any(strcmp(key, {'a' 'hyphen'}))
        obj.labelerController_.frameDown(tfCtrl) ;
      elseif strcmp(key, 'o') && ~tfCtrl
        [tfSel, iSel] = mdl.anyPointSelected() ;
        if tfSel
          mdl.toggleEstOccPoint(iSel) ;
        end
      elseif any(strcmp(key, {'leftarrow' 'rightarrow' 'uparrow' 'downarrow'}))
        [tfSel, iSel] = mdl.anyPointSelected() ;
        if tfSel && ~mdl.tfOcc(iSel)
          tfShift = any(strcmp('shift', modifier)) ;
          xy = mdl.getLabelCoordsI(iSel) ;
          lc = obj.labelerController_ ;
          switch key
            case 'leftarrow'
              dxdy = -lc.videoCurrentRightVec() ;
            case 'rightarrow'
              dxdy = lc.videoCurrentRightVec() ;
            case 'uparrow'
              dxdy = lc.videoCurrentUpVec() ;
            case 'downarrow'
              dxdy = -lc.videoCurrentUpVec() ;
            otherwise
              error('LabelCoreTemplateController:unknownKey', ...
                    'Unknown arrow key %s.', key) ;
          end
          if tfShift
            xyNew = xy + dxdy * 10 ;
          else
            xyNew = xy + dxdy ;
          end
          xyNew = lc.videoClipToVideo(xyNew) ;
          mdl.setLabelCoordsI(xyNew, iSel) ;
          switch mdl.state
            case LabelState.ADJUST
              mdl.setPointAdjusted(iSel) ;
            case LabelState.ACCEPTED
              mdl.enterAdjust(LabelCoreTemplateResetType.NORESET, false) ;
            otherwise
              error('LabelCoreTemplateController:unknownState', ...
                    'Unknown state %s.', char(mdl.state)) ;
          end
        else
          tfKPused = false ;
        end
      elseif strcmp(key, 'backquote')
        iPt = mdl.kpfIPtFor1Key + 10 ;
        if iPt > mdl.nPts
          iPt = 1 ;
        end
        mdl.kpfIPtFor1Key = iPt ;
        obj.refreshTxLabelCoreAux() ;
      elseif any(strcmp(key, {'0' '1' '2' '3' '4' '5' '6' '7' '8' '9'}))
        iPt = str2double(key) ;
        if iPt == 0
          iPt = 10 ;
        end
        iPt = iPt + mdl.kpfIPtFor1Key - 1 ;
        if iPt > mdl.nPts
          return ;
        end
        mdl.toggleSelectPoint(iPt) ;
      else
        tfKPused = false ;
      end
    end  % function

    function axOccBDF(obj, ~, ~)
      % Handle occluded-axis button-down.
      % Note: currently occluded axis hidden so this should be uncalled.
      if obj.isPanZoom()
        return
      end
      mdl = obj.model_ ;
      mdl.handleOccludedAxisButtonDown() ;
    end  % function

  end  % methods

  %% Cosmetics
  methods

    function updatePredUnadjustedPVs(obj)
      % Update .hPts*PV*Unadjusted_ from model.ptsPlotInfo_.
      % Currently hardcoded but could change in future.

      ppi = obj.model_.ptsPlotInfo ;
      obj.hPtsMarkerPVPredUnadjusted_ = struct( ...
        'MarkerSize', ppi.MarkerProps.MarkerSize * 1.5, ...
        'LineWidth', ppi.MarkerProps.LineWidth / 2) ;
      obj.hPtsMarkerPVNotPredUnadjusted_ = struct( ...
        'MarkerSize', ppi.MarkerProps.MarkerSize, ...
        'LineWidth', ppi.MarkerProps.LineWidth) ;
      obj.hPtsTxtPVPredUnadjusted_ = struct( ...
        'FontAngle', 'italics') ;
      obj.hPtsTxtPVNotPredUnadjusted_ = struct( ...
        'FontAngle', ppi.TextProps.FontAngle) ;
    end  % function

    function refreshMarkerProps(obj)
      % Refresh marker properties (not including .Marker, .Color) on .hPts_
      % based on model.tfAdjusted_, model.lastSetAllUnadjustedResetType_,
      % and .hPtsMarkerPV*_.

      mdl = obj.model_ ;
      resetType = mdl.lastSetAllUnadjustedResetType_ ;
      tfAdj = mdl.tfAdjusted_ ;

      % resetType describes/applies to those els of tfAdj that are false
      tfUnadjPredicted = ~tfAdj & ...
                         resetType == LabelCoreTemplateResetType.RESETPREDICTED ;
      tfNotUnadjPredicted = ~tfUnadjPredicted ;
      set(obj.hPts_(tfNotUnadjPredicted), obj.hPtsMarkerPVNotPredUnadjusted_) ;
      set(obj.hPts_(tfUnadjPredicted), obj.hPtsMarkerPVPredUnadjusted_) ;
    end  % function

    function updateColors(obj)
      % LabelCoreController overload: only color adjusted pts or unadj/predicted.

      mdl = obj.model_ ;
      colors = mdl.ptsPlotInfo.Colors ;

      resetType = mdl.lastSetAllUnadjustedResetType_ ;
      tfSetColor = mdl.tfAdjusted_ | ...
                   resetType == LabelCoreTemplateResetType.RESETPREDICTED ;
      for i = 1:mdl.nPts
        if tfSetColor(i)
          if numel(obj.hPts_) >= i && ishandle(obj.hPts_(i))
            set(obj.hPts_(i), 'Color', colors(i, :)) ;
          end
          if numel(obj.hPtsTxt_) >= i && ishandle(obj.hPtsTxt_(i))
            set(obj.hPtsTxt_(i), 'Color', colors(i, :)) ;
          end
        end
      end
    end  % function

    function updateMarkerCosmetics(obj)
      % LabelCoreController overload: also refresh pred-unadjusted PVs.

      obj.updatePredUnadjustedPVs() ;

      obj.refreshPtMarkers() ;  % updates hPts_.Marker
      obj.refreshMarkerProps() ;  % updates other marker-related props
    end  % function

    function updateTextLabelCosmetics(obj)
      % LabelCoreController overload. Currently, if pvText includes FontAngle
      % this will collide with Unadjusted/Predicted-ness and this will not
      % be handled properly. (However FontAngle currently *not* exposed in
      % cosmetics picker.)

      ppi = obj.model_.ptsPlotInfo ;
      pvText = ppi.TextProps ;
      set(obj.hPtsTxt_, pvText) ;

      obj.redrawTextLabels() ;
    end  % function

  end  % methods

  %% Adjusted cosmetics helpers
  methods

    function refreshAdjustedCosmeticsAll(obj)
      % Refresh cosmetics for all points based on adjustedness.
      mdl = obj.model_ ;
      resetType = mdl.lastSetAllUnadjustedResetType_ ;
      ppi = mdl.ptsPlotInfo ;

      if all(mdl.tfAdjusted_)
        % All adjusted: colored with normal marker props
        clrs = ppi.Colors ;
        pv = obj.hPtsMarkerPVNotPredUnadjusted_ ;
        for i = 1:mdl.nPts
          pv.Color = clrs(i, :) ;
          set(obj.hPts_(i), pv) ;
          if ~isempty(obj.hPtsOcc_)
            set(obj.hPtsOcc_(i), pv) ;
          end
          set(obj.hPtsTxt_(i), 'FontAngle', 'normal') ;
        end
      else
        % All unadjusted
        switch resetType
          case LabelCoreTemplateResetType.RESET
            pv = obj.hPtsMarkerPVNotPredUnadjusted_ ;
            pv.Color = ppi.TemplateMode.TemplatePointColor ;
            set(obj.hPts_, pv) ;
            set(obj.hPtsOcc_, pv) ;
            set(obj.hPtsTxt_, 'FontAngle', 'normal') ;
          case LabelCoreTemplateResetType.RESETPREDICTED
            clrs = ppi.Colors ;
            pv = obj.hPtsMarkerPVPredUnadjusted_ ;
            for i = 1:mdl.nPts
              pv.Color = clrs(i, :) ;
              set(obj.hPts_(i), pv) ;
              if ~isempty(obj.hPtsOcc_)
                set(obj.hPtsOcc_(i), pv) ;
              end
              set(obj.hPtsTxt_(i), 'FontAngle', 'italic') ;
            end
          otherwise
            assert(false) ;
        end
      end
    end  % function

    function refreshAdjustedCosmeticsI(obj, iSel)
      % Refresh cosmetics for a single point that just became adjusted.
      mdl = obj.model_ ;
      if mdl.tfAdjusted_(iSel)
        clr = mdl.ptsPlotInfo.Colors(iSel, :) ;
        pv = obj.hPtsMarkerPVNotPredUnadjusted_ ;
        pv.Color = clr ;
        set(obj.hPts_(iSel), pv) ;
        if ~isempty(obj.hPtsOcc_)
          set(obj.hPtsOcc_(iSel), pv) ;
        end
        set(obj.hPtsTxt_(iSel), 'FontAngle', 'normal') ;
      end
    end  % function

  end  % methods

  %% Presentation
  methods

    function shortcuts = LabelShortcuts(obj)
      % Return shortcut descriptions for Template mode.

      lc = obj.labelerController_ ;
      mdl = obj.model_ ;
      shortcuts = cell(0, 3) ;

      shortcuts{end+1, 1} = 'Accept current labels' ;
      shortcuts{end, 2} = 's or space' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = 'Toggle whether selected kpt is occluded' ;
      shortcuts{end, 2} = 'o' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = 'Toggle whether selected kpt is fully occluded' ;
      shortcuts{end, 2} = 'u' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = 'Forward one frame' ;
      shortcuts{end, 2} = '= or d' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = 'Backward one frame' ;
      shortcuts{end, 2} = '- or a' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = 'Un/Select kpt of current target' ;
      shortcuts{end, 2} = '0-9' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = 'Toggle which kpts 0-9 correspond to' ;
      shortcuts{end, 2} = '`' ;
      shortcuts{end, 3} = {} ;

      rightpx = lc.videoCurrentRightVec() ;
      rightpx = rightpx(1) ;
      uppx = lc.videoCurrentUpVec() ;
      uppx = abs(uppx(2)) ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move right by %.1f px, ow forward one frame', rightpx) ;
      shortcuts{end, 2} = 'Right arrow' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move left by %.1f px, ow back one frame', rightpx) ;
      shortcuts{end, 2} = 'Left arrow' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move up by %.1f px', uppx) ;
      shortcuts{end, 2} = 'Up arrow' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move down by %.1f px', uppx) ;
      shortcuts{end, 2} = 'Down arrow' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = ...
        sprintf('If kpt selected, move left by %.1f px, ow go to next %s', ...
                10*rightpx, obj.labeler_.movieShiftArrowNavMode.prettyStr) ;
      shortcuts{end, 2} = 'Left arrow' ;
      shortcuts{end, 3} = {'Shift'} ;

      shortcuts{end+1, 1} = ...
        sprintf('If kpt selected, move right by %.1f px, ow go to previous %s', ...
                10*rightpx, obj.labeler_.movieShiftArrowNavMode.prettyStr) ;
      shortcuts{end, 2} = 'Right arrow' ;
      shortcuts{end, 3} = {'Shift'} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move up by %.1f px', 10*uppx) ;
      shortcuts{end, 2} = 'Up arrow' ;
      shortcuts{end, 3} = {'Shift'} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move down by %.1f px', 10*uppx) ;
      shortcuts{end, 2} = 'Down arrow' ;
      shortcuts{end, 3} = {'Shift'} ;

      shortcuts{end+1, 1} = 'Zoom in/out' ;
      shortcuts{end, 2} = 'Mouse scroll' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = 'Pan view' ;
      shortcuts{end, 2} = 'Mouse right-click-drag' ;
      shortcuts{end, 3} = {} ;
    end  % function

    function h = getLabelingHelp(obj)
      % Return labeling help text for Template mode.
      h = cell(0, 1) ;
      h{end+1} = 'Adjust all keypoints, then click Accept to store.' ;
      h{end+1} = '' ;
      h{end+1} = ['In Template labeling mode, there is a set of template/"white" points on the ', ...
        'image at all times. To ', ...
        'label a frame, adjust the points as necessary and accept. Adjusted ', ...
        'points are shown in colors (rather than white). '] ;
      h{end+1} = ['Points may also be Selected using hotkeys (0..9). When a point is ', ...
        'selected, the arrow-keys adjust the point as if by mouse. Mouse-clicks ', ...
        'on the image also jump the point immediately to that location. '] ;
      h{end+1} = 'If no point is selected, you can click and drag a point to move it. ' ;
      h{end+1} = '' ;
      h{end+1} = ['Once you have finished adjusting all points, click the Accept button ', ...
        'to store the coordinates. If you change frames before accepting, your work will be lost.'] ;
      h{end+1} = ['You can adjust points once they are accepted. If you change the points, you ', ...
        'must click the Accept button again to store your work.'] ;
      h{end+1} = '' ;
      h1 = obj.getLabelingHelp@LabelCoreController() ;
      h = [h(:) ; h1(:)] ;
    end  % function

  end  % methods

end  % classdef
