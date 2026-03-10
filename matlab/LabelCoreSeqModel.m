classdef LabelCoreSeqModel < LabelCoreModel
% Sequential labeling model
%
% Label mode 1 (Sequential)
%
% There are two labeling states: 'label' and 'accepted'.
%
% During the labeling state, points are being clicked in order. This
% includes the state where there are zero points clicked (fresh image).
%
% Once all points have been clicked, the accepted state is entered.
% This writes to .labeledpos. Points may be adjusted by click-dragging or
% using hotkeys as in Template Mode.
%
% Occluded. In the 'label' state, clicking in the full-occluded subaxis
% sets the current point to be fully occluded.

  properties
    supportsSingleView = true ;
    supportsMultiView = false ;
    supportsCalibration = false ;
    supportsMultiAnimal = false ;
    unsupportedKPFFns = {} ;
  end

  properties (Transient)
    iPtMove_                % scalar. Either nan, or index of pt being moved
    nPtsLabeled_            % scalar integer. 0..nPts
  end

  methods

    function obj = LabelCoreSeqModel(labeler)
      % Construct a LabelCoreSeqModel.
      obj = obj@LabelCoreModel(labeler) ;
    end  % function

    function initHook(obj)
      % Initialize Seq-specific state.
      obj.kpfIPtFor1Key_ = 1 ;
      % AL 20190203 semi-hack. init to something/anything to avoid error
      % with .state_ unset.
      obj.state_ = LabelState.ADJUST ;
    end  % function

  end  % methods

  %% State transition hooks
  methods

    function newFrame(obj, iFrm0, iFrm1, iTgt, tfForceUpdate) %#ok<INUSL>
      % Frame has changed, target is the same.
      if ~exist('tfForceUpdate', 'var')
        tfForceUpdate = false ; %#ok<NASGU>
      end
      obj.newFrameTarget(iFrm1, iTgt) ;
    end  % function

    function newTarget(obj, iTgt0, iTgt1, iFrm) %#ok<INUSL>
      % Target has changed, frame is the same.
      obj.newFrameTarget(iFrm, iTgt1) ;
    end  % function

    function newFrameAndTarget(obj, ~, iFrm1, ~, iTgt1, tfForceUpdate)
      % Frame and target have both changed.
      if ~exist('tfForceUpdate', 'var')
        tfForceUpdate = false ; %#ok<NASGU>
      end
      obj.newFrameTarget(iFrm1, iTgt1) ;
    end  % function

    function clearLabels(obj)
      % Clear current labels and enter initial labeling state.
      obj.beginLabel(true) ;
    end  % function

    function acceptLabels(obj)
      % Accept labels for the current frame/target.
      obj.beginAccepted(true) ;
      obj.labeler_.restorePrevAxesMode() ;
    end  % function

    function unAcceptLabels(obj) %#ok<MANU>
      % Un-accept labels. Currently a no-op for sequential mode.
    end  % function

  end  % methods

  %% Internal state transitions
  methods

    function newFrameTarget(obj, iFrm, iTgt)
      % React to new frame or target. If a frame is labeled, enter Accepted
      % state with saved labels. Otherwise, enter Label state.

      [tflabeled, lpos, lpostag] = obj.labeler_.labelPosIsLabeled(iFrm, iTgt) ;
      if tflabeled
        obj.nPtsLabeled_ = obj.nPts_ ;
        obj.setLabelCoords(lpos, 'lblTags', lpostag) ;
        obj.iPtMove_ = nan ;
        obj.beginAccepted(false) ;
      else
        obj.beginLabel(false) ;
      end
    end  % function

    function beginLabel(obj, tfClearLabels)
      % Enter Label state and clear all label state for current frame/target.

      obj.xy_ = nan(obj.nPts_, 2) ;
      obj.nPtsLabeled_ = 0 ;
      obj.iPtMove_ = nan ;
      obj.tfOcc_(:) = false ;
      obj.tfEstOcc_(:) = false ;
      obj.tfSel_(:) = false ;
      if tfClearLabels
        obj.labeler_.labelPosClear() ;
      end
      obj.state_ = LabelState.LABEL ;
      obj.notify('updateLabelCoords') ;
      obj.notify('updateState') ;
    end  % function

    function adjust2Label(obj)
      % Enter LABEL from ADJUST.
      obj.iPtMove_ = nan ;
      obj.state_ = LabelState.LABEL ;
      obj.notify('updateState') ;
    end  % function

    function beginAdjust(obj)
      % Enter adjustment state for current frame/target.
      assert(obj.nPtsLabeled_ == obj.nPts_) ;
      obj.iPtMove_ = nan ;
      obj.state_ = LabelState.ADJUST ;
      obj.notify('updateState') ;
    end  % function

    function beginAccepted(obj, tfSetLabelPos)
      % Enter accepted state for current frame.

      if tfSetLabelPos
        xy = obj.getLabelCoords() ;
        obj.labeler_.labelPosSet(xy) ;
        obj.setLabelPosTagFromEstOcc() ;
      end
      obj.iPtMove_ = nan ;
      obj.clearSelected() ;
      obj.state_ = LabelState.ACCEPTED ;
      obj.notify('updateState') ;
    end  % function

    function storeLabels(obj)
      % Write current label coordinates and tags to Labeler.
      xy = obj.getLabelCoords() ;
      obj.labeler_.labelPosSet(xy) ;
      obj.setLabelPosTagFromEstOcc() ;
    end  % function

  end  % methods

  %% Action methods (called by controller after extracting GUI state)
  methods

    function labelNextPoint(obj, xy, tfAxOcc, tfShift)
      % Label the next sequential point.
      %
      % xy: [1x2] position (ignored if tfAxOcc)
      % tfAxOcc: logical, true if click was in the occluded box
      % tfShift: logical, true if shift was held

      nlbled = obj.nPtsLabeled_ ;
      assert(nlbled < obj.nPts_) ;
      i = nlbled + 1 ;
      if tfAxOcc
        obj.tfOcc_(i) = true ;
        obj.notify('updateOccluded') ;
      else
        obj.xy_(i, :) = xy ;
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
      %
      % iPt: index of point to occlude

      if obj.tfEstOcc_(iPt)
        obj.tfEstOcc_(iPt) = false ;
      else
        obj.labeler_.labelPosSetIFullyOcc(iPt) ;
      end
      obj.toggleSelectPoint(iPt) ;
      obj.tfOcc_(iPt) = true ;
      obj.notify('updateOccluded') ;
    end  % function

    function undoLastLabel(obj)
      % Undo the last label placement.

      switch obj.state_
        case {LabelState.LABEL, LabelState.ADJUST}
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
          error('LabelCoreSeqModel:unknownState', ...
                'Unknown state %s.', char(obj.state_)) ;
      end
    end  % function

    function toggleEstOccPoint(obj, iPt)
      % Toggle the estimated-occluded flag for a point.
      obj.tfEstOcc_(iPt) = ~obj.tfEstOcc_(iPt) ;
      assert(~(obj.tfEstOcc_(iPt) && obj.tfOcc_(iPt))) ;
      obj.notify('updateEstOccluded') ;
    end  % function

  end  % methods

end  % classdef
