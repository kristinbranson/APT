classdef LabelCoreSeqController < LabelCoreController
% Sequential labeling controller
%
% Owns all graphics for sequential labeling mode. Receives GUI callbacks,
% extracts GUI state (mouse position, modifiers), delegates data logic
% to LabelCoreSeqModel, and syncs graphics in response to model events.

  properties
    supportsSingleView = true ;
    supportsMultiView = false ;
    supportsCalibration = false ;
    supportsMultiAnimal = false ;
  end

  methods

    function obj = LabelCoreSeqController(labelerController, model)
      % Construct a LabelCoreSeqController.
      obj = obj@LabelCoreController(labelerController, model) ;
    end  % function

    function initHook(obj)
      % Initialize Seq-specific graphics state.
      obj.txLblCoreAux_.Visible = 'on' ;
      obj.refreshTxLabelCoreAux() ;
      set(obj.tbAccept_, 'Enable', 'off') ;
    end  % function

  end  % methods

  %% Model event handlers
  methods

    function onUpdateState(obj)
      % Sync tbAccept appearance and point HitTest to model state.
      mdl = obj.model_ ;
      switch mdl.state_
        case LabelState.LABEL
          set(obj.tbAccept_, 'BackgroundColor', [0.4 0 0], ...
            'String', 'Unlabeled', 'Enable', 'off', 'Value', 0) ;
          set(obj.hPts_(ishandle(obj.hPts_)), 'HitTest', 'off') ;
        case LabelState.ADJUST
          set(obj.tbAccept_, 'BackgroundColor', [0.6 0 0], 'String', 'Accept', ...
            'Value', 0, 'Enable', 'off') ;
        case LabelState.ACCEPTED
          set(obj.hPts_(ishandle(obj.hPts_)), 'HitTest', 'on') ;
          set(obj.tbAccept_, 'BackgroundColor', [0 0.4 0], 'String', 'Labeled', ...
            'Value', 1, 'Enable', 'off') ;
        otherwise
          error('LabelCoreSeqController:unknownState', ...
                'Unknown state %s.', char(mdl.state_)) ;
      end
    end  % function

    function onUpdateLabelCoordsI(obj)
      % Sync single-point graphics and refresh markers for changed point.
      onUpdateLabelCoordsI@LabelCoreController(obj) ;
      iPt = obj.model_.lastChangedIPt_ ;
      obj.refreshPtMarkers('iPts', iPt) ;
    end  % function

  end  % methods

  %% GUI callback handlers
  methods

    function axBDF(obj, src, evt) %#ok<INUSL>
      % Handle axis button-down: place next point or relocate selected point.
      mdl = obj.model_ ;
      if ~mdl.labeler_.isReady || evt.Button > 1
        return ;
      end
      if obj.isPanZoom()
        return ;
      end

      pos = evt.IntersectionPoint(1:2) ;
      mod = obj.hFig_(1).CurrentModifier ;
      tfShift = any(strcmp(mod, 'shift')) ;
      switch mdl.state_
        case LabelState.LABEL
          % pos = get(obj.hAx_(1), 'CurrentPoint') ;
          % pos = pos(1, 1:2) ;
          mdl.labelNextPoint(pos, false, tfShift) ;
        case {LabelState.ADJUST, LabelState.ACCEPTED}
          [tf, iSel] = mdl.anyPointSelected() ;
          if tf
            % pos = get(obj.hAx_(1), 'CurrentPoint') ;
            % pos = pos(1, 1:2) ;
            mdl.relocatePoint(iSel, pos, tfShift) ;
          end
        otherwise
          error('LabelCoreSeqController:unknownState', ...
                'Unknown state %s.', char(mdl.state_)) ;
      end
    end  % function

    function axOccBDF(obj, ~, ~)
      % Handle occluded-axis button-down: label next point as occluded, or
      % occlude a selected point.
      mdl = obj.model_ ;
      if ~mdl.labeler_.isReady
        return ;
      end
      if obj.isPanZoom()
        return ;
      end

      mod = obj.hFig_(1).CurrentModifier ;
      tfShift = any(strcmp(mod, 'shift')) ;
      switch mdl.state_
        case LabelState.LABEL
          mdl.labelNextPoint([], true, tfShift) ;
        case {LabelState.ADJUST, LabelState.ACCEPTED}
          [tf, iSel] = mdl.anyPointSelected() ;
          if tf
            mdl.occludeSelectedPoint(iSel) ;
          end
        otherwise
          error('LabelCoreSeqController:unknownState', ...
                'Unknown state %s.', char(mdl.state_)) ;
      end
    end  % function

    function ptBDF(obj, src, evt)
      % Handle point button-down: select point and start drag.
      if obj.isPanZoom()
        return ;
      end
      mdl = obj.model_ ;
      if ~mdl.labeler_.isReady || evt.Button > 1
        return ;
      end
      switch mdl.state_
        case {LabelState.ADJUST, LabelState.ACCEPTED}
          if ismember('control', obj.hFig_(1).CurrentModifier)
            return ;
          end
          iPt = get(src, 'UserData') ;
          mdl.toggleSelectPoint(iPt) ;
          mdl.iPtMove_ = iPt ;
        case LabelState.LABEL
          % No point interaction during LABEL state
        otherwise
          error('LabelCoreSeqController:unknownState', ...
                'Unknown state %s.', char(mdl.state_)) ;
      end
    end  % function

    function wbmf(obj, ~, ~)
      % Handle window button motion: drag selected point.
      % Bypasses event system for responsiveness during continuous drag.
      mdl = obj.model_ ;
      if isempty(mdl.state_) || ~mdl.labeler_.isReady
        return ;
      end
      if mdl.state_ == LabelState.ADJUST || mdl.state_ == LabelState.ACCEPTED
        iPt = mdl.iPtMove_ ;
        if ~isnan(iPt)
          tmp = get(obj.hAx_(1), 'CurrentPoint') ;
          pos = tmp(1, 1:2) ;
          mdl.xy_(iPt, :) = pos ;
          obj.syncPointGraphicsI(iPt) ;
        end
      end
    end  % function

    function wbuf(obj, ~, ~)
      % Handle window button up: end drag, persist labels.
      mdl = obj.model_ ;
      if ~mdl.labeler_.isReady
        return ;
      end
      % Don't act if click is on a trx handle
      if ismember(gco, obj.labelerController_.tvTrx_.hTrx)
        return ;
      end
      if mdl.state_ == LabelState.ADJUST || ...
          mdl.state_ == LabelState.ACCEPTED && ...
          ~isempty(mdl.iPtMove_) && ~isnan(mdl.iPtMove_)
        mdl.iPtMove_ = nan ;
        mdl.storeLabels() ;
      end
    end  % function

    function tfKPused = kpf(obj, ~, evt)
      % Handle key press.
      mdl = obj.model_ ;
      if ~mdl.labeler_.isReady
        tfKPused = false ;
        return ;
      end

      key = evt.Key ;
      modifier = evt.Modifier ;
      tfCtrl = ismember('control', modifier) ;

      tfKPused = true ;
      if strcmp(key, 'z') && tfCtrl
        mdl.undoLastLabel() ;
      elseif strcmp(key, 'o') && ~tfCtrl
        [tfSel, iSel] = mdl.anyPointSelected() ;
        if tfSel
          mdl.toggleEstOccPoint(iSel) ;
        end
        if mdl.state_ == LabelState.ACCEPTED
          mdl.storeLabels() ;
        end
      elseif any(strcmp(key, {'d' 'equal'}))
        obj.labelerController_.frameUp(tfCtrl) ;
      elseif any(strcmp(key, {'a' 'hyphen'}))
        obj.labelerController_.frameDown(tfCtrl) ;
      elseif ~tfCtrl && any(strcmp(key, {'leftarrow' 'rightarrow' 'uparrow' 'downarrow'}))
        [tfSel, iSel] = mdl.anyPointSelected() ;
        if tfSel
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
              error('LabelCoreSeqController:unknownKey', ...
                    'Unknown arrow key %s.', key) ;
          end
          if tfShift
            xyNew = xy + dxdy * 10 ;
          else
            xyNew = xy + dxdy ;
          end
          xyNew = lc.videoClipToVideo(xyNew) ;
          mdl.xy_(iSel, :) = xyNew ;
          mdl.lastChangedIPt_ = iSel ;
          mdl.notify('updateLabelCoordsI') ;
          if mdl.state_ == LabelState.ACCEPTED
            mdl.storeLabels() ;
          end
        else
          tfKPused = false ;
        end
      elseif strcmp(key, 'backquote')
        iPt = mdl.kpfIPtFor1Key_ + 10 ;
        if iPt > mdl.nPts_
          iPt = 1 ;
        end
        mdl.kpfIPtFor1Key_ = iPt ;
        obj.refreshTxLabelCoreAux() ;
      elseif any(strcmp(key, {'0' '1' '2' '3' '4' '5' '6' '7' '8' '9'}))
        if mdl.state_ ~= LabelState.LABEL
          iPt = str2double(key) ;
          if iPt == 0
            iPt = 10 ;
          end
          iPt = iPt + mdl.kpfIPtFor1Key_ - 1 ;
          if iPt > mdl.nPts_
            return ;
          end
          mdl.toggleSelectPoint(iPt) ;
        end
      else
        tfKPused = false ;
      end
    end  % function

  end  % methods

  %% Presentation
  methods

    function shortcuts = LabelShortcuts(obj)
      % Return shortcut descriptions for Seq mode.

      lc = obj.labelerController_ ;
      shortcuts = cell(0, 3) ;

      shortcuts{end+1, 1} = 'Undo last label click' ;
      shortcuts{end, 2} = 'z' ;
      shortcuts{end, 3} = {'Ctrl'} ;

      shortcuts{end+1, 1} = 'Toggle whether selected kpt is occluded' ;
      shortcuts{end, 2} = 'o' ;
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
                10*rightpx, obj.model_.labeler_.movieShiftArrowNavMode.prettyStr) ;
      shortcuts{end, 2} = 'Left arrow' ;
      shortcuts{end, 3} = {'Shift'} ;

      shortcuts{end+1, 1} = ...
        sprintf('If kpt selected, move right by %.1f px, ow go to previous %s', ...
                10*rightpx, obj.model_.labeler_.movieShiftArrowNavMode.prettyStr) ;
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
      % Return labeling help text for Seq mode.
      h = cell(0, 1) ;
      h{end+1} = 'Click all keypoints in order to label an animal/frame. ' ;
      h{end+1} = 'After you finish adding all keypoints, you can adjust their positions.' ;
      h{end+1} = ['After entering all keypoints, select a keypoint to modify ', ...
        'by typing the number identifying it. ', ...
        'If you have more than 10 keypoints, the ` (backquote) key lets you ', ...
        'change which set of 10 keypoints the number keys correspond to.'] ;
      h{end+1} = ['Once a keypoint is selected, it can be adjusted ', ...
        'by clicking on the desired location in an image. ', ...
        'Fine adjustments can be made using the arrow keys. '] ;
      h{end+1} = ['Type the keypoint number again to deselect it, or type another ', ...
        'keypoint number to select a different keypoint. '] ;
      h{end+1} = ['If no keypoints are selected, you can adjust any keypoint by ', ...
        'clicking down on it and dragging it to the desired location.'] ;
      h{end+1} = '' ;
      h{end+1} = 'Shift + click will label the point with the "occuded" flag (point will be an o).' ;
      h{end+1} = '' ;
      h1 = obj.getLabelingHelp@LabelCoreController() ;
      h = [h(:) ; h1(:)] ;
    end  % function

  end  % methods

end  % classdef
