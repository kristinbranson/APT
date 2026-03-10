classdef LabelCoreTemplateModel < LabelCoreModel
% Template-based labeling model
%
% In Template mode, there is a set of template/"white" points on the
% image at all times. (When starting, these points are randomized.) To
% label a frame, adjust the points as necessary and accept. Adjusted
% points are shown in colors (rather than white).
%
% This is the model half of the LabelCoreTemplate MVC split. The
% controller half is LabelCoreTemplateController.

  events
    updateAdjusted          % tfAdjusted_ changed
  end

  properties
    supportsSingleView = true ;
    supportsMultiView = false ;
    supportsCalibration = false ;
    supportsMultiAnimal = false ;
    unsupportedKPFFns = {} ;
  end

  properties (Transient)
    tfAdjusted_             % [nPts x 1] logical. If true, pt has been adjusted
                            % from template or tracking prediction.

    % scalar LabelCoreTemplateResetType. The only way to clear/reset points
    % to unadjusted is to call setAllPointsUnadjusted. At that time, a
    % LabelCoreTemplateResetType is passed indicating whether the frame has
    % existing tracking/predictions or not. We record this so that if at
    % any time an element of .tfAdjusted_ is false, this property indicates
    % whether tracking is present for that unadjusted point.
    lastSetAllUnadjustedResetType_ = LabelCoreTemplateResetType.RESET

    iPtMove_                % scalar. Either nan, or index of pt being moved.
    tfMoved_                % scalar logical. If true, pt being moved was
                            % actually dragged.
  end

  methods

    function obj = LabelCoreTemplateModel(labeler)
      % Construct a LabelCoreTemplateModel.
      obj = obj@LabelCoreModel(labeler) ;
    end  % function

    function initHook(obj)
      % Initialize Template-specific state.
      npts = obj.nPts_ ;
      obj.tfAdjusted_ = false(npts, 1) ;
      obj.kpfIPtFor1Key_ = 1 ;
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
      % React to new frame and/or target. If labeled, enter Accepted. If
      % tracked, show predictions and enter Adjust. Otherwise, transform
      % existing template points and enter Adjust.

      if ~exist('tfForceUpdate', 'var')
        tfForceUpdate = false ; %#ok<NASGU>
      end
      lObj = obj.labeler_ ;

      [tflabeled, lpos, lpostag] = lObj.labelPosIsLabeled(iFrm1, iTgt1) ;
      if tflabeled
        obj.setLabelCoords(lpos, 'lblTags', lpostag) ;
        obj.enterAccepted(false) ;
        return ;
      end

      assert(iFrm1 == lObj.currFrame) ;
      [tftrked, lposTrk, occTrk] = lObj.trackIsCurrMovFrmTracked(iTgt1) ;
      if tftrked
        obj.setLabelCoords(lposTrk, 'lblTags', occTrk) ;
        obj.enterAdjust(LabelCoreTemplateResetType.RESETPREDICTED, false) ;
        return ;
      end

      if iTgt0 == iTgt1  % same target, new frame
        if lObj.hasTrx
          % existing points are aligned onto new frame based on trx at
          % (currTarget,prevFrame) and (currTarget,currFrame)
          xy0 = obj.getLabelCoords() ;
          xy = LabelCoreModel.transformPtsTrx(xy0, ...
            lObj.trx(iTgt0), iFrm0, ...
            lObj.trx(iTgt0), iFrm1) ;
          obj.setLabelCoords(xy) ;
        else
          % none, leave pts as-is
        end
      else  % different target
        assert(lObj.hasTrx, 'Must have trx to change targets.') ;
        [tfneighbor, iFrm0Neighb, lpos0] = ...
          lObj.labelPosLabeledNeighbor(iFrm1, iTgt1) ;
        if tfneighbor
          lpos0 = reshape(lpos0, [], 2) ;
          xy = LabelCoreModel.transformPtsTrx(lpos0, ...
            lObj.trx(iTgt1), iFrm0Neighb, ...
            lObj.trx(iTgt1), iFrm1) ;
        else
          % no neighboring previously labeled points for new target.
          % Just start with current points for previous target/frame.
          xy0 = obj.getLabelCoords() ;
          xy = LabelCoreModel.transformPtsTrx(xy0, ...
            lObj.trx(iTgt0), iFrm0, ...
            lObj.trx(iTgt1), iFrm1) ;
        end
        obj.setLabelCoords(xy) ;
      end
      obj.enterAdjust(LabelCoreTemplateResetType.RESET, false) ;
    end  % function

    function clearLabels(obj)
      % Clear current labels and enter adjustment state.
      obj.clearSelected() ;
      obj.enterAdjust(LabelCoreTemplateResetType.RESET, true) ;
    end  % function

    function acceptLabels(obj)
      % Accept labels for the current frame/target.
      obj.enterAccepted(true) ;
      obj.labeler_.restorePrevAxesMode() ;
    end  % function

    function unAcceptLabels(obj) %#ok<MANU>
      % Un-accept labels. Currently unsupported for template mode.
      assert(false, 'Unsupported') ;
    end  % function

  end  % methods

  %% Internal state transitions
  methods

    function enterAdjust(obj, resetType, tfClearLabeledPos)
      % Enter adjustment state for current frame/tgt.
      %
      % resetType: LabelCoreTemplateResetType
      % tfClearLabeledPos: if true, clear labeled pos.

      if resetType > LabelCoreTemplateResetType.NORESET
        obj.setAllPointsUnadjusted(resetType) ;
      end

      if tfClearLabeledPos
        obj.labeler_.labelPosClear() ;
      end

      obj.iPtMove_ = nan ;
      obj.tfMoved_ = false ;

      obj.state_ = LabelState.ADJUST ;
      obj.notify('updateState') ;
    end  % function

    function enterAccepted(obj, tfSetLabelPos)
      % Enter accepted state for current frame/tgt. All points colored. If
      % tfSetLabelPos, all points/tags written to labelpos/labelpostag.

      obj.setAllPointsAdjusted() ;
      obj.clearSelected() ;

      if tfSetLabelPos
        obj.storeLabels() ;
      end
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

  %% Adjustedness API
  methods

    function setPointAdjusted(obj, iSel)
      % Mark a single point as adjusted.
      if ~obj.tfAdjusted_(iSel)
        obj.tfAdjusted_(iSel) = true ;
        obj.lastChangedIPt_ = iSel ;
        obj.notify('updateAdjusted') ;
      end
    end  % function

    function setAllPointsAdjusted(obj)
      % Mark all points as adjusted.
      obj.tfAdjusted_(:) = true ;
      obj.lastChangedIPt_ = 0 ;  % 0 signals "all points"
      obj.notify('updateAdjusted') ;
    end  % function

    function setAllPointsUnadjusted(obj, resetType)
      % Mark all points as unadjusted with given reset type.
      assert(isa(resetType, 'LabelCoreTemplateResetType')) ;
      obj.tfAdjusted_(:) = false ;
      obj.lastSetAllUnadjustedResetType_ = resetType ;
      obj.lastChangedIPt_ = 0 ;  % 0 signals "all points"
      obj.notify('updateAdjusted') ;
    end  % function

  end  % methods

  %% Action methods (called by controller)
  methods

    function toggleEstOccPoint(obj, iPt)
      % Toggle the estimated-occluded flag for a point.
      obj.tfEstOcc_(iPt) = ~obj.tfEstOcc_(iPt) ;
      obj.notify('updateEstOccluded') ;
      if obj.state_ == LabelState.ACCEPTED
        obj.storeLabels() ;
      end
    end  % function

  end  % methods

  %% Template methods
  methods

    function tt = getTemplate(obj)
      % Create a template struct from current pts.

      tt = struct() ;
      tt.pts = obj.getLabelCoords() ;
      lbler = obj.labeler_ ;
      if lbler.hasTrx
        [x, y, th] = lbler.currentTargetLoc() ;
        tt.loc = [x y] ;
        tt.theta = th ;
      else
        tt.loc = [nan nan] ;
        tt.theta = nan ;
      end
    end  % function

    function setTemplate(obj, tt)
      % Set "white points" to template.

      lbler = obj.labeler_ ;
      tfTemplateHasTarget = ~any(isnan(tt.loc)) ;
      % For some projects (e.g. larva), theta can be nan. So shouldn't test
      % theta. MK 20250728
      tfHasTrx = lbler.hasTrx ; %#ok<NASGU>

      if lbler.hasTrx && ~tfTemplateHasTarget
        warningNoTrace('LabelCoreTemplate:template', ...
          'Using template saved without target coordinates') ;
      elseif ~lbler.hasTrx && tfTemplateHasTarget
        warningNoTrace('LabelCoreTemplate:template', ...
          'Template saved with target coordinates.') ;
      end

      if tfTemplateHasTarget
        [x1, y1, th1] = lbler.currentTargetLoc ;
        if isnan(th1 - tt.theta)
          xys = transformPoints(tt.pts, tt.loc, 0, [x1 y1], 0) ;
        else
          xys = transformPoints(tt.pts, tt.loc, tt.theta, [x1 y1], th1) ;
        end
      else
        xys = tt.pts ;
      end

      obj.setLabelCoords(xys) ;
      obj.enterAdjust(LabelCoreTemplateResetType.RESET, false) ;
    end  % function

    function setRandomTemplate(obj)
      % Set random template points around the current target location.
      lbler = obj.labeler_ ;
      [x0, y0] = lbler.currentTargetLoc('nowarn', true) ;
      nr = lbler.movienr ;
      nc = lbler.movienc ;
      r = round(max(nr, nc) / 6) ;

      n = obj.nPts_ ;
      x = x0 + r * 2 * (rand(n, 1) - 0.5) ;
      y = y0 + r * 2 * (rand(n, 1) - 0.5) ;
      obj.setLabelCoords([x y]) ;
    end  % function

  end  % methods

end  % classdef
