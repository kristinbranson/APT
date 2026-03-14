classdef LabelCoreSeqAddController < LabelCoreSeqController
% Sequential adding of new landmarks - controller
%
% Extends LabelCoreSeqController for SeqAdd mode. Old points are
% frozen (HitTest off, no ButtonDownFcn). The tbAccept shows "Adding"
% during LABEL state and "Next" during ACCEPTED state.

  properties
    supportsSingleView = true ;
    supportsMultiView = false ;
    supportsCalibration = false ;
    supportsMultiAnimal = false ;
  end

  methods

    function obj = LabelCoreSeqAddController(labelerController, labeler, model)
      % Construct a LabelCoreSeqAddController.
      obj = obj@LabelCoreSeqController(labelerController, labeler, model) ;
    end  % function

    function initHook(obj)
      % Initialize SeqAdd-specific graphics: freeze old points.
      mdl = obj.model_ ;
      obj.txLblCoreAux_.Visible = 'on' ;
      obj.refreshTxLabelCoreAux() ;
      set(obj.tbAccept_, 'Enable', 'off', 'Style', 'pushbutton') ;

      % Old points are not selectable or movable
      nold = mdl.nold_ ;
      arrayfun(@(x)set(x, 'HitTest', 'off', 'ButtonDownFcn', ''), obj.hPts_(1:nold)) ;
      if ~isempty(obj.hPtsOcc_)
        arrayfun(@(x)set(x, 'HitTest', 'off', 'ButtonDownFcn', ''), obj.hPtsOcc_(1:nold)) ;
      end
    end  % function

  end  % methods

  %% Model event handlers (override parent)
  methods

    function onUpdateState(obj)
      % Sync tbAccept appearance for SeqAdd mode.
      mdl = obj.model_ ;
      switch mdl.state
        case LabelState.LABEL
          set(obj.tbAccept_, 'BackgroundColor', [0 0 0.4], ...
            'String', 'Adding', 'Enable', 'off', 'Value', 0) ;
          set(obj.hPts_(ishandle(obj.hPts_)), 'HitTest', 'off') ;
        case LabelState.ADJUST
          set(obj.tbAccept_, 'BackgroundColor', [0.6 0 0], 'String', 'Accept', ...
            'Value', 0, 'Enable', 'off') ;
        case LabelState.ACCEPTED
          hptsadd = obj.hPts_(mdl.nold_+1:end) ;
          set(hptsadd(ishandle(hptsadd)), 'HitTest', 'on') ;
          set(obj.tbAccept_, 'BackgroundColor', [0 0.4 0], 'String', 'Next', ...
            'Value', 1, 'Enable', 'on') ;
        otherwise
          error('LabelCoreSeqAddController:unknownState', ...
                'Unknown state %s.', char(mdl.state)) ;
      end
    end  % function

  end  % methods

  %% GUI callback overrides
  methods

    function ptBDF(obj, src, evt)
      % Handle point button-down: only allow for new points.
      if ~obj.labeler_.isReady || evt.Button > 1
        return ;
      end
      if obj.isPanZoom()
        return ;
      end
      mdl = obj.model_ ;
      switch mdl.state
        case {LabelState.ADJUST, LabelState.ACCEPTED}
          iPt = get(src, 'UserData') ;
          % Only allow interaction with new points
          if iPt > mdl.nold_
            mdl.toggleSelectPoint(iPt) ;
            mdl.iPtMove_ = iPt ;
          end
        case LabelState.LABEL
          % No point interaction during LABEL state
        otherwise
          error('LabelCoreSeqAddController:unknownState', ...
                'Unknown state %s.', char(mdl.state)) ;
      end
    end  % function

    function tfKPused = kpf(obj, src, evt)  %#ok<INUSD>
      % Handle key press. Override backquote wrapping to respect nold_.
      mdl = obj.model_ ;
      if ~obj.labeler_.isReady
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
        if mdl.state == LabelState.ACCEPTED
          mdl.storeLabels() ;
        end
      elseif any(strcmp(key, {'d' 'equal'}))
        obj.labelerController_.frameUp(tfCtrl) ;
      elseif any(strcmp(key, {'a' 'hyphen'}))
        obj.labelerController_.frameDown(tfCtrl) ;
      elseif any(strcmp(key, {'leftarrow' 'rightarrow' 'uparrow' 'downarrow'}))
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
              error('LabelCoreSeqAddController:unknownKey', ...
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
              % none
            case LabelState.ACCEPTED
              % no-op (removed adjust state)
            case LabelState.LABEL
              % none
            otherwise
              error('LabelCoreSeqAddController:unknownState', ...
                    'Unknown state %s.', char(mdl.state)) ;
          end
        else
          tfKPused = false ;
        end
      elseif strcmp(key, 'backquote')
        iPt = mdl.kpfIPtFor1Key + 10 ;
        if iPt > mdl.nPts
          iPt = mdl.nold_ + 1 ;
        end
        mdl.kpfIPtFor1Key = iPt ;
        obj.refreshTxLabelCoreAux() ;
      elseif any(strcmp(key, {'0' '1' '2' '3' '4' '5' '6' '7' '8' '9'}))
        if mdl.state ~= LabelState.LABEL
          iPt = str2double(key) ;
          if iPt == 0
            iPt = 10 ;
          end
          iPt = iPt + mdl.kpfIPtFor1Key - 1 ;
          if iPt > mdl.nPts || iPt <= mdl.nold_
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

    function h = getLabelingHelp(obj)
      % Return labeling help text for SeqAdd mode.
      h = {} ;
      h{end+1} = 'Click new keypoints to add in order.' ;
      h{end+1} = '' ;
      h{end+1} = 'Navigate to a frame that is partially labeled.' ;
      h{end+1} = 'Click to add the new points in order.' ;
      h{end+1} = 'Adjust as you like, either with the keypoint numbers or click to select.' ;
      h{end+1} = 'Click the Next button to advance to the next partially labeled frame.' ;
      h{end+1} = '' ;
      h1 = obj.getLabelingHelp@LabelCoreController() ;
      h = [h(:) ; h1(:)] ;
    end  % function

  end  % methods

end  % classdef
