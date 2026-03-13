classdef LabelCoreSeqMAModel < LabelCoreModel
% Multi-animal sequential labeling model
%
% Labeling states: Accepted/Browse and Label.
%
% During Browse, existing targets/labels (if any) are shown. Selection of
% a primary/focus target can occur eg via targets table. When this
% occurs, non-primary targets are grayed. Landmarks in the primary tgt
% can be adjusted either by click-drag or by selectin/hotkey-ptBDF/arrows.
%
% During Browse, there may be no primary tgt. In that case no adjustments
% to existing landmarks can be made.
%
% The primary target is always the same as lObj.currTarget.
%
% The Label state as in Seq mode. To enter the Label state, for now one
% must explictly click the New Target button. Once a target is labeled,
% labels are written to lObj and Browse is entered with the just-labeled
% target set as primary.
%
% pbDelTarget can be done either i) during Browse when a primary
% target is set, in which case that target is removed (and no target is
% set to primary); or ii) during Label, in which case the current target
% is canceled and Browse-with-no-primary is entered.
%
% Changing targets or frames in the middle of Label is equivalent to
% hitting pbRemoveTarget first as no new labels are written to lObj.

  properties
    supportsSingleView = true ;
    supportsMultiView = false ;
    supportsCalibration = false ;
    supportsMultiAnimal = true ;
    unsupportedKPFFns = {'tbAccept'} ;
  end

  properties (Transient)
    iPtMove_                % scalar. Either nan, or index of pt being moved
    nPtsLabeled_            % scalar integer. 0..nPts, or inf.
  end

  properties
    maxNumTgts_ = 5 ;
  end

  properties (Constant)
    CLR_NEW_TGT = [0.470588235294118 0.670588235294118 0.188235294117647] ;
    CLR_DEL_TGT = [0.929411764705882 0.690196078431373 0.129411764705882] ;
  end

  properties (Transient)
    tcOn_ = false ;         % scalar logical, true => two-click is on
    tcipt_ = 0 ;            % 0, 1, or 2 depending on current number of two-click pts clicked
    tcShow_ = false ;        % scalar logical. true => leave tc points showing during lbl
    tc_prev_axis_ = [] ;    % to reset to prev view once the 2 click labeling is over
  end

  properties (Dependent)
    tcOn                    % scalar logical, two-click alignment on/off
  end

  methods

    function v = get.tcOn(obj)
      % Return whether two-click alignment is on.
      v = obj.tcOn_ ;
    end  % function

  end  % methods

  events
    updateTwoClickState     % tcOn_/tcipt_ changed
    updateNewFrameTarget    % new frame/target data ready for controller
    updateAccepted          % entering accepted state
    updateAcceptedReset     % entering accepted state with reset
    updateBeginLabel        % entering label state
    restoreVideoAxis        % restore video axis after two-click align
  end

  methods

    function obj = LabelCoreSeqMAModel(labeler)
      % Construct a LabelCoreSeqMAModel.
      obj = obj@LabelCoreModel(labeler) ;
    end  % function

    function initHook(obj)
      % Initialize SeqMA-specific state.
      obj.kpfIPtFor1Key_ = 1 ;
      obj.state_ = LabelState.ACCEPTED ;
    end  % function

  end  % methods

  %% State transition hooks (called by Labeler)
  methods

    function newFrame(obj, iFrm0, iFrm1, iTgt, tfForceUpdate) %#ok<INUSL>
      % Frame has changed, target is the same.
      if ~exist('tfForceUpdate', 'var')
        tfForceUpdate = false ;
      end
      obj.newFrameTarget(iFrm1, iTgt, tfForceUpdate) ;
    end  % function

    function newTarget(obj, iTgt0, iTgt1, iFrm) %#ok<INUSL>
      % Target has changed, frame is the same.
      obj.newFrameTarget(iFrm, iTgt1) ;
    end  % function

    function newFrameAndTarget(obj, ~, iFrm1, ~, iTgt1, tfForceUpdate)
      % Frame and target have both changed.
      if ~exist('tfForceUpdate', 'var')
        tfForceUpdate = false ;
      end
      obj.newFrameTarget(iFrm1, iTgt1, tfForceUpdate) ;
    end  % function

    function clearLabels(obj) %#ok<MANU>
      % Clear current labels. Not supported for SeqMA.
      assert(false, 'Nonproduction codepath') ;
    end  % function

    function unAcceptLabels(obj) %#ok<MANU>
      % Un-accept labels. Not supported for SeqMA.
      assert(false, 'Nonproduction codepath') ;
    end  % function

  end  % methods

  %% Internal state transitions
  methods

    function newFrameTarget(obj, iFrm, iTgt, tfForceUpdate)
      % React to new frame or target which might be labeled or unlabeled.
      %
      % PostCond: Accepted/Browse state

      if ~exist('tfForceUpdate', 'var')
        tfForceUpdate = false ; %#ok<NASGU>
      end

      lObj = obj.labeler_ ;
      [tflabeled, lpos, lpostag] = lObj.labelPosIsLabeled(iFrm, iTgt) ;
      if ~tflabeled
        % iTgt is not labeled, but we set the primary target to a labeled
        % frm if avail
        iTgts = lObj.labelPosIsLabeledFrm(iFrm) ;
        if ~isempty(iTgts)
          iTgt = min(iTgts) ; % TODO could take iTgt closest to existing iTgt
          [~, lpos, lpostag] = lObj.labelPosIsLabeled(iFrm, iTgt) ;
          tflabeled = true ;
          lObj.setTargetMA(iTgt) ;
        end
      end

      if tflabeled
        obj.nPtsLabeled_ = obj.nPts_ ;
        obj.setLabelCoords(lpos, 'lblTags', lpostag) ;
        obj.beginAccepted() ;
      else
        if iTgt ~= 0
          lObj.setTargetMA(0) ;
        end
        obj.beginAcceptedReset() ;
      end

      % Notify controller to update all MA track results and ROI
      obj.notify('updateNewFrameTarget') ;
    end  % function

    function resetState(obj)
      % Reset all label state to blank.
      obj.xy_ = nan(obj.nPts_, 2) ;
      obj.nPtsLabeled_ = 0 ;
      obj.iPtMove_ = nan ;
      obj.tfOcc_(:) = false ;
      obj.tfEstOcc_(:) = false ;
      obj.tfSel_(:) = false ;
      obj.notify('updateLabelCoords') ;
    end  % function

    function acceptLabels(obj)
      % Accept labels for the current target.
      obj.storeLabels() ;
      obj.labeler_.notify('updateTrxTable') ;
      obj.labeler_.restorePrevAxesMode() ;
      obj.beginAccepted() ;
      obj.notify('updateAccepted') ;
    end  % function

    function beginAccepted(obj)
      % Enter accepted state. Preconds:
      % 1. Current primary labeling pts should already be set appropriately.
      % If there is no current label, these should be set to nan.

      obj.iPtMove_ = nan ;
      obj.clearSelected() ;
      obj.tcInit() ;
      if obj.tcOn_ && ~isempty(obj.tc_prev_axis_)
        obj.notify('restoreVideoAxis') ;
        obj.tc_prev_axis_ = [] ;
      end
      obj.state_ = LabelState.ACCEPTED ;
      obj.notify('updateState') ;
    end  % function

    function beginAcceptedReset(obj)
      % Like beginAccepted, but reset first.
      obj.resetState() ;
      obj.tcInit() ;
      if obj.tcOn_ && ~isempty(obj.tc_prev_axis_)
        obj.notify('restoreVideoAxis') ;
        obj.tc_prev_axis_ = [] ;
      end
      obj.state_ = LabelState.ACCEPTED ;
      obj.notify('updateState') ;
    end  % function

    function beginLabel(obj)
      % Enter Label state and clear all label state for current frame/target.
      obj.resetState() ;
      obj.state_ = LabelState.LABEL ;
      obj.notify('updateBeginLabel') ;
      obj.notify('updateState') ;
    end  % function

    function storeLabels(obj)
      % Write current label coordinates and tags to Labeler.
      [xy, tfeo] = obj.getLabelCoords() ;
      obj.labeler_.labelPosSet(xy, tfeo) ;
    end  % function

  end  % methods

  %% Action methods (called by controller after extracting GUI state)
  methods

    function hlpAxBDFLabelState(obj, tfAxOcc, tfShift, pos)
      % Handle a click in LABEL state. pos is [x y].
      % .tfOcc_, .tfEstOcc_, .tfSel_ start off as all false in beginLabel().

      nlbled = obj.nPtsLabeled_ ;
      assert(nlbled < obj.nPts_) ;
      i = nlbled + 1 ;
      if tfAxOcc
        obj.tfOcc_(i) = true ;
        obj.notify('updateOccluded') ;
      else
        obj.xy_(i, :) = pos ;
        if tfShift
          obj.tfEstOcc_(i) = true ;
        end
        obj.lastChangedIPt_ = i ;
        obj.notify('updateLabelCoordsI') ;
      end
      obj.nPtsLabeled_ = i ;
      if i == obj.nPts_
        obj.acceptLabels() ;
      end
    end  % function

    function hlpAxBDFTwoClick(obj, xy)
      % Handle a two-click-align click in LABEL state. Updates model state
      % for tcipt_. Controller handles the graphics.

      switch obj.tcipt_
        case 0
          obj.tcipt_ = 1 ;
          obj.notify('updateTwoClickState') ;
        case 1
          obj.tcipt_ = 2 ;
          obj.notify('updateTwoClickState') ;
        otherwise
          error('LabelCoreSeqMAModel:badTcipt', ...
                'Unexpected tcipt_ value %d.', obj.tcipt_) ;
      end
    end  % function

    function relocatePoint(obj, iPt, pos, tfShift)
      % Move a selected point to a new position and persist to Labeler.
      %
      % iPt: index of point to move
      % pos: [1x2] new position
      % tfShift: logical, set estimated-occluded flag

      obj.xy_(iPt, :) = pos ;
      obj.labeler_.labelPosSetI(pos, iPt) ;
      obj.tfEstOcc_(iPt) = tfShift ;
      obj.lastChangedIPt_ = iPt ;
      obj.notify('updateLabelCoordsI') ;
      obj.toggleSelectPoint(iPt) ;
      if obj.tfOcc_(iPt)
        obj.tfOcc_(iPt) = false ;
        obj.notify('updateOccluded') ;
      end
    end  % function

    function occludeSelectedPoint(obj, iPt)
      % Make a selected point fully occluded.

      obj.xy_(iPt, :) = [nan nan] ;
      obj.labeler_.labelPosSetIFullyOcc(iPt) ;
      if obj.tfEstOcc_(iPt)
        obj.tfEstOcc_(iPt) = false ;
      end
      obj.toggleSelectPoint(iPt) ;
      obj.tfOcc_(iPt) = true ;
      obj.notify('updateOccluded') ;
    end  % function

    function undoLastLabel(obj)
      % Undo the last label placement.

      switch obj.state_
        case LabelState.LABEL
          nlbled = obj.nPtsLabeled_ ;
          if nlbled > 0
            obj.tfSel_(nlbled) = false ;
            obj.tfEstOcc_(nlbled) = false ;
            obj.tfOcc_(nlbled) = false ;
            obj.xy_(nlbled, :) = [nan nan] ;
            obj.nPtsLabeled_ = nlbled - 1 ;
            obj.lastChangedIPt_ = nlbled ;
            obj.notify('updateLabelCoordsI') ;
            obj.notify('updateOccluded') ;
            obj.notify('updateSelected') ;
          end
        case LabelState.ACCEPTED
          % No undo in ACCEPTED state
        otherwise
          error('LabelCoreSeqMAModel:unknownState', ...
                'Unknown state %s.', char(obj.state_)) ;
      end
    end  % function

    function toggleEstOccPoint(obj, iPt)
      % Toggle the estimated-occluded flag for a point.
      obj.tfEstOcc_(iPt) = ~obj.tfEstOcc_(iPt) ;
      assert(~(obj.tfEstOcc_(iPt) && obj.tfOcc_(iPt))) ;
      obj.notify('updateEstOccluded') ;
    end  % function

    function cbkNewTgt(obj)
      % Handle New Target button press (model logic).

      lObj = obj.labeler_ ;
      if obj.state_ == LabelState.LABEL
        % cancel
        obj.beginAcceptedReset() ;
      else % ACCEPTED
        % add a new label
        ntgts = lObj.labelNumLabeledTgts() ;
        lObj.setTargetMA(ntgts + 1) ;
        lObj.notify('updateTrxTable') ;
        obj.beginLabel() ;
      end
    end  % function

    function cbkDelTgt(obj)
      % Handle Delete Target button press (model logic).

      lObj = obj.labeler_ ;
      if obj.state_ == LabelState.ACCEPTED
        ntgts = lObj.labelPosClearWithCompact_New() ;
        iTgt = lObj.currTarget ;
        if iTgt > ntgts
          lObj.setTargetMA(ntgts) ;
        end
      end
      obj.newFrameTarget(lObj.currFrame, lObj.currTarget) ;
    end  % function

    function setTwoClickOn(obj, tfon)
      % Set the two-click-align mode on or off.
      if obj.state_ == LabelState.LABEL
        error('Please finish labeling the current animal.') ;
      end
      obj.tcInit() ;
      obj.tcOn_ = tfon ;
      obj.notify('updateTwoClickState') ;
    end  % function

    function tcInit(obj)
      % Reset two-click state.
      obj.tcipt_ = 0 ;
      % tcShow_ unchanged
    end  % function

  end  % methods

end  % classdef
