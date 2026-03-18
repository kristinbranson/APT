classdef LabelCoreMultiViewCalibrated2Model < LabelCoreModel
% Multiview calibrated labeling model
%
% Owns all data state for multi-view calibrated labeling: point-to-axis
% mappings, point set mappings, working set, calibration rig, adjusted/
% stored flags, epipolar line visibility state, etc. Communicates state
% changes to the controller via obj.notify() events.
%
% This is the model half of the LabelCoreMultiViewCalibrated2 MVC split.
% The controller half is LabelCoreMultiViewCalibrated2Controller.

  events
    updateAdjusted          % tfAdjusted_ changed
    updateProjection        % projection state changed (iSetWorking_, pjtIPts)
    updateWorkingSet        % working set changed
    updateEpiLineVisibility % showEpiLines_ changed
  end

  properties
    supportsSingleView = false ;
    supportsMultiView = true ;
    supportsCalibration = true ;
    supportsMultiAnimal = false ;
    unsupportedKPFFns = {} ;
  end

  properties (Constant, Hidden)
    DXFAC = 500 ;
    DXFACBIG = 50 ;
  end

  properties (Transient)
    iPt2iAx_                  % [npts]. iPt2iAx_(iPt) gives the axis index for iPt
    iSet2iPt_                 % [nset x nview]. A point 'set' is a nview-tuple of point
                              % indices that represent a single physical (3d) point.
                              % .iSet2iPt_(iSet,:) are the nview pt indices for pointset iSet.
    iPt2iSet_                 % [npts]. set index for each point.
    showCalibration_ = true ; % whether to show calibration-based info

    iSetWorking_              % scalar. Set index of working set. Can be nan for no working set.
    pjtCalRig_                % Scalar calibration rig object
    tfAdjusted_               % [nPts x 1] logical. If true, pt has been adjusted from template.
    tfStored_                 % [nPts x 1] logical. If true, pt has been accepted and stored.
    currFrameAdjust_          % which frame was most recently adjusted
    numHotKeyPtSet_           % scalar positive integer. The pointset that '1' hotkey maps to.
    iPtMove_                  % scalar. Either nan, or index of pt being moved.
    tfMoved_                  % scalar logical. If true, pt being moved was actually moved.
    showEpiLines_ = true ;    % whether epipolar lines are visible
  end

  properties (Dependent)
    nView                     % scalar
    nPointSet                 % scalar, number of point sets
    pjtIPts                   % [nview]. NaN if anchor point not clicked for a view.
    pjtState                  % 0, 1, or 2 for number of defined working pts
    isCalRig                  % scalar logical
    showCalibration           % scalar logical
    showEpiLines              % scalar logical
  end

  methods  % dep prop getters

    function v = get.nView(obj)
      % Return the number of views.
      v = obj.labeler_.nview ;
    end  % function

    function v = get.nPointSet(obj)
      % Return the number of point sets.
      v = size(obj.iSet2iPt_, 1) ;
    end  % function

    function v = get.pjtIPts(obj)
      % Return the projection anchor point indices for each view.
      iSet = obj.iSetWorking_ ;
      v = nan(1, obj.nView) ;
      if isempty(iSet) || isnan(iSet)
        return ;
      end
      for iAx = 1:obj.nView
        iPt = find(obj.iPt2iSet_ == iSet & obj.iPt2iAx_ == iAx) ;
        if obj.tfAdjusted_(iPt)
          v(iAx) = iPt ;
        end
      end
    end  % function

    function v = get.pjtState(obj)
      % Return the number of defined anchor points in the working set.
      v = nnz(~isnan(obj.pjtIPts)) ;
    end  % function

    function v = get.isCalRig(obj)
      % Return whether a calibration rig is set.
      v = ~isempty(obj.pjtCalRig_) ;
    end  % function

    function v = get.showCalibration(obj)
      % Return whether calibration info is shown.
      v = obj.showCalibration_ ;
    end  % function

    function v = get.showEpiLines(obj)
      % Return whether epipolar lines are visible.
      v = obj.showEpiLines_ ;
    end  % function

    function set.showEpiLines(obj, v)
      % Set epipolar line visibility and notify controllers.
      obj.showEpiLines_ = v ;
      obj.notify('updateEpiLineVisibility') ;
    end  % function

  end  % methods

  methods

    function obj = LabelCoreMultiViewCalibrated2Model(labeler)
      % Construct a LabelCoreMultiViewCalibrated2Model.
      obj = obj@LabelCoreModel(labeler) ;
    end  % function

    function initHook(obj)
      % Initialize multiview-specific state from the Labeler.
      obj.iPt2iAx_ = obj.labeler_.labeledposIPt2View ;
      obj.iPt2iSet_ = obj.labeler_.labeledposIPt2Set ;
      obj.iSet2iPt_ = obj.labeler_.labeledposIPtSetMap ;

      obj.setRandomTemplate() ;

      obj.tfAdjusted_ = false(obj.nPts_, 1) ;
      obj.tfStored_ = false(obj.nPts_, 1) ;
      obj.currFrameAdjust_ = nan ;

      obj.numHotKeyPtSet_ = 1 ;
      obj.iSetWorking_ = nan ;

      obj.iPtMove_ = nan ;
      obj.tfMoved_ = false ;

      obj.state_ = LabelState.ADJUST ;
    end  % function

  end  % methods

  %% State transition hooks
  methods

    function newFrame(obj, iFrm0, iFrm1, iTgt, tfForceUpdate)
      % Frame has changed, target is the same.
      if ~exist('tfForceUpdate', 'var')
        tfForceUpdate = false ;
      end
      obj.newFrameAndTarget(iFrm0, iFrm1, iTgt, iTgt, tfForceUpdate) ;
    end  % function

    function newTarget(obj, iTgt0, iTgt1, iFrm)
      % Target has changed, frame is the same.
      obj.newFrameAndTarget(iFrm, iFrm, iTgt0, iTgt1) ;
    end  % function

    function newFrameAndTarget(obj, iFrm0, iFrm1, iTgt0, iTgt1, tfForceUpdate)
      % React to new frame and/or target.
      if ~exist('tfForceUpdate', 'var')
        tfForceUpdate = false ;
      end

      if ~tfForceUpdate && (iFrm0 == iFrm1) && (iTgt0 == iTgt1)
        return ;
      end

      if ~tfForceUpdate && iFrm1 == obj.currFrameAdjust_
        % call is originating from checkAccept failure
        return ;
      end

      res = obj.checkAccept() ;
      if strcmpi(res, 'Yes')
        obj.acceptLabels() ;
        obj.clearSelected() ;
      elseif strcmpi(res, 'Cancel')
        obj.labeler_.setFrameAndTarget(iFrm0, iTgt0) ;
        return ;
      else  % answer = No
        % pass
      end

      [tflabeled, lpos, lpostag] = obj.labeler_.labelPosIsLabeled(iFrm1, iTgt1) ;
      if tflabeled
        obj.setLabelCoords(lpos, 'lblTags', lpostag) ;
        obj.enterAccepted(false) ;
      else
        assert(iTgt0 == iTgt1, 'Multiple targets unsupported.') ;
        assert(~obj.labeler_.hasTrx, 'Targets are unsupported.') ;
        obj.enterAdjust(true, false) ;
      end

      obj.notify('updateProjection') ;
    end  % function

    function clearLabels(obj)
      % Clear current labels and enter adjustment state.
      obj.enterAdjust(true, true) ;
      obj.projectionWorkingSetClear_() ;
      obj.notify('updateProjection') ;
    end  % function

    function acceptLabels(obj)
      % Accept labels for the current frame/target.
      obj.enterAccepted(true) ;
      obj.labeler_.restorePrevAxesMode() ;
    end  % function

    function unAcceptLabels(obj)
      % Un-accept labels for the current frame/target.
      obj.enterAdjust(false, false) ;
    end  % function

  end  % methods

  %% Internal state transitions
  methods

    function enterAdjust(obj, tfResetPts, tfClearLabeledPos)
      % Enter adjustment state for current frame/tgt.
      %
      % if tfResetPts, reset all points to pre-adjustment (white).
      % if tfClearLabeledPos, clear labeled pos.

      if tfResetPts
        obj.tfEstOcc_(:) = 0 ;
        obj.tfAdjusted_(:) = false ;
        obj.tfStored_(:) = false ;
        obj.currFrameAdjust_ = obj.labeler_.currFrame ;
        obj.lastChangedIPt_ = 0 ;  % 0 signals "all points"
        obj.notify('updateAdjusted') ;
      end
      if tfClearLabeledPos
        obj.labeler_.labelPosClear() ;
      end

      obj.iPtMove_ = nan ;
      obj.tfMoved_ = false ;

      obj.currFrameAdjust_ = obj.labeler_.currFrame ;
      obj.state_ = LabelState.ADJUST ;
      obj.notify('updateState') ;
    end  % function

    function enterAccepted(obj, tfSetLabelPos)
      % Enter accepted state for current frame/tgt. All points colored. If
      % tfSetLabelPos, all points/tags written to labelpos/labelpostag.

      obj.tfAdjusted_(:) = true ;
      obj.tfStored_(:) = true ;
      obj.currFrameAdjust_ = nan ;

      if tfSetLabelPos
        xy = obj.getLabelCoords() ;
        obj.labeler_.labelPosSet(xy) ;
        obj.setLabelPosTagFromEstOcc() ;
      end

      obj.currFrameAdjust_ = nan ;
      obj.state_ = LabelState.ACCEPTED ;
      obj.lastChangedIPt_ = 0 ;  % 0 signals "all points"
      obj.notify('updateAdjusted') ;
      obj.notify('updateState') ;
    end  % function

    function setPointAdjusted(obj, iSel)
      % Mark a single point as adjusted.
      obj.tfStored_(iSel) = false ;
      if ~obj.tfAdjusted_(iSel)
        obj.tfAdjusted_(iSel) = true ;
        obj.lastChangedIPt_ = iSel ;
        obj.notify('updateAdjusted') ;
      end
    end  % function

    function toggleEstOccPoint(obj, iPt)
      % Toggle the estimated-occluded flag for a point.
      obj.tfEstOcc_(iPt) = ~obj.tfEstOcc_(iPt) ;
      obj.notify('updateEstOccluded') ;
      if obj.state_ == LabelState.ACCEPTED
        obj.enterAdjust(false, false) ;
      end
    end  % function

  end  % methods

  %% State queries
  methods

    function v = isUnsavedState(obj)
      % Return true if any points have been adjusted but not stored.
      v = any(obj.tfAdjusted_ & ~obj.tfStored_) ;
    end  % function

    function v = isAdjustFrameChange(obj)
      % Return true if the current frame differs from the last adjusted frame.
      v = ~isnan(obj.currFrameAdjust_) && ...
          (obj.currFrameAdjust_ ~= obj.labeler_.currFrame) ;
    end  % function

    function res = checkAccept(obj)
      % Check if unsaved adjustments exist and prompt user.
      res = 'No' ;
      if obj.isUnsavedState() && obj.isAdjustFrameChange()
        buttons = {'Yes', 'No', 'Cancel'} ;
        default = 'Yes' ;
        labeler = obj.labeler_ ;
        labeler.dialogLaunchPad_ = ...
          struct('text', 'Some keypoints have been adjusted but not accepted. Accept before losing this information?', ...
                 'title', 'Accept labels', ...
                 'buttons', {buttons}, ...
                 'default', default) ;
        labeler.dialogLandingPad_ = default ;
        labeler.notify('requestQuestionDialog') ;
        res = labeler.dialogLandingPad_ ;
      end
    end  % function

  end  % methods

  %% Skeleton
  methods

    function edges = skeletonEdges(obj)
      % Get multi-view skeleton edge indices.
      se = obj.labeler_.skeletonEdges ;
      nEdges = size(se, 1) ;
      edges = repmat(se, [obj.nView, 1]) ;
      for ivw = 1:obj.nView
        edges((ivw - 1)*nEdges + 1 : ivw*nEdges, :) = ...
          reshape(obj.iSet2iPt_(se(:), ivw), [nEdges, 2]) ;
      end
    end  % function

  end  % methods

  %% Template
  methods

    function setRandomTemplate(obj)
      % Set random template points around the movie center for each view.
      lbler = obj.labeler_ ;
      movctrs = lbler.movieroictr ;
      movnrs = lbler.movienr ;
      movncs = lbler.movienc ;
      xy = nan(obj.nPts_, 2) ;
      for iPt = 1:obj.nPts_
        iAx = obj.iPt2iAx_(iPt) ;
        nr = movnrs(iAx) ;
        nc = movncs(iAx) ;
        xy(iPt, 1) = movctrs(iAx, 1) + nc/3*2*(rand - 0.5) ;
        xy(iPt, 2) = movctrs(iAx, 2) + nr/3*2*(rand - 0.5) ;
      end
      obj.setLabelCoords(xy) ;
    end  % function

  end  % methods

  %% Projection working set (model state portion)
  methods

    function projectionWorkingSetClear_(obj)
      % Clear the working set (model state only).
      obj.iSetWorking_ = nan ;
      obj.notify('updateWorkingSet') ;
    end  % function

    function projectionWorkingSetSet_(obj, iSet)
      % Set the working set to iSet (model state only).
      obj.iSetWorking_ = iSet ;
      obj.notify('updateWorkingSet') ;
    end  % function

    function projectionWorkingSetToggle(obj, iSet)
      % Toggle the working set.
      if isnan(obj.iSetWorking_)
        obj.projectionWorkingSetSet_(iSet) ;
      else
        tfIsMatch = obj.iSetWorking_ == iSet ;
        obj.projectionWorkingSetClear_() ;
        if ~tfIsMatch
          obj.projectionWorkingSetSet_(iSet) ;
        end
      end
    end  % function

    function tf = projectionWorkingSetPointInWS(obj, iPt)
      % Return true if iPt is in current working set.
      tf = obj.iPt2iSet_(iPt) == obj.iSetWorking_ ;
    end  % function

    function [tfSel, iSelPt, iAx] = projectionPointSelected(obj)
      % Determine the currently selected point based on working set and
      % which figure has focus.
      %
      % TODO: get(0,'CurrentFigure') is a GUI concern. Eventually move
      % this to the controller.
      iAx = find(get(0, 'CurrentFigure') == obj.labeler_.gdata.figs_all) ;
      iWS = obj.iSetWorking_ ;
      tfSel = isscalar(iAx) && ~isnan(iWS) ;
      if tfSel
        iSelPt = obj.iSet2iPt_(iWS, iAx) ;
      else
        iSelPt = nan ;
      end
    end  % function

  end  % methods

  %% Calibration rig
  methods

    function projectionSetCalRig(obj, crig)
      % Set the calibration rig object.
      assert(isa(crig, 'CalRig')) ;

      % 20160923 hack legacy CalRigSH objs and EPlines workaround
      if isa(crig, 'CalRigSH')
        crig.epLineNPts = 2 ;
      end

      obj.pjtCalRig_ = crig ;
    end  % function

    function setShowCalibration(obj, val)
      % Set whether calibration info is shown and notify.
      obj.showCalibration_ = val ;
      if obj.isCalRig
        obj.notify('updateProjection') ;
      end
    end  % function

    function toggleShowCalibration(obj)
      % Toggle the calibration visibility.
      obj.setShowCalibration(~obj.showCalibration_) ;
    end  % function

  end  % methods

  %% 3D position display
  methods

    function [X, xyrp, rpe] = projectionTriangulate(obj)
      % Triangulate the current working set points and return 3D position.
      %
      % Returns X: 3D coordinates, xyrp: reprojected points, rpe: RP error.

      iWS = obj.iSetWorking_ ;
      if isnan(iWS)
        error('No current working pointset.') ;
      end

      ipts = obj.iSet2iPt_(iWS, :) ;
      tfadj = obj.tfAdjusted_(ipts) ;
      if ~all(tfadj)
        error('One or more points in current working pointset have not been set/adjusted.') ;
      end

      nvw = obj.nView ;
      xyAll = obj.xy_(ipts, :) ;
      xs = xyAll(:, 1) ;
      ys = xyAll(:, 2) ;
      xyim = [xs(:) ys(:)]' ;
      xyim = reshape(xyim, [2 1 nvw]) ;

      crig = obj.pjtCalRig_ ;
      [X, xyrp, rpe] = crig.triangulate(xyim) ;

      fprintf('Pointset %d (labelpoints %s)\n', iWS, mat2str(ipts)) ;
      for ivw = 1:nvw
        fprintf(' view %d: clicked posn is %s. RP err is %.2f\n', ivw, ...
          mat2str(round(reshape(xyim(:, 1, ivw), [1 2]))), rpe(ivw)) ;
      end
      fprintf('3D coordinates are %s\n', mat2str(X, 5)) ;
    end  % function

  end  % methods

end  % classdef
