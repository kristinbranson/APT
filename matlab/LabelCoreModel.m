classdef LabelCoreModel < handle
% Labeling model base class
%
% Owns labeling state: current coordinates, occluded/selected flags,
% and the labeling state machine. Communicates state changes to
% controllers via obj.notify() events. Does not own any graphics handles.
%
% This is the model half of the LabelCore MVC split. The controller
% half is LabelCoreController.

  events
    update                  % general full-sync fallback
    updateLabelCoords       % xy_ changed (all or many points)
    updateLabelCoordsI      % single point changed (read model.lastChangedIPt_)
    updateState             % state_ changed
    updateOccluded          % tfOcc_ changed
    updateEstOccluded       % tfEstOcc_ changed
    updateSelected          % tfSel_ changed
    updateHideLabels        % hideLabels_ changed
  end

  properties (Abstract)
    supportsSingleView      % scalar logical
    supportsMultiView       % scalar logical
    supportsCalibration     % scalar logical
    supportsMultiAnimal     % scalar logical
    unsupportedKPFFns       % cell array of char
  end

  properties (Transient)
    labeler_                % scalar Labeler
    nPts_                   % scalar integer
    state_                  % scalar LabelState
    xy_                     % [nPts x 2] current label coords (NaN=unlabeled, Inf=occluded)
    tfOcc_                  % [nPts x 1] logical, fully occluded
    tfEstOcc_               % [nPts x 1] logical, estimated-occluded
    tfSel_                  % [nPts x 1] logical, selected
    kpfIPtFor1Key_          % scalar positive integer
    nexttbl_                % table, frame queue
    nexti_                  % integer, queue index
    hideLabels_             % scalar logical
    panZoomMod_             % char, modifier key for pan-zoom
    ptsPlotInfo_            % struct, points plotting cosmetic info
    lastChangedIPt_         % scratch: index of last changed point (for updateLabelCoordsI)
  end

  properties (Dependent)
    nPts
    state
    hideLabels
  end

  methods

    function result = get.nPts(obj)
      % Return the number of label points.
      result = obj.nPts_ ;
    end

    function result = get.state(obj)
      % Return the current labeling state.
      result = obj.state_ ;
    end

    function result = get.hideLabels(obj)
      % Return whether labels are hidden.
      result = obj.hideLabels_ ;
    end

  end  % methods

  methods (Static)

    function obj = createSafe(labeler, labelMode)
      % Create appropriate LabelCoreModel subclass, adjusting labelMode if needed.
      if labeler.isMultiView && labelMode ~= LabelMode.MULTIVIEWCALIBRATED2
        labelModeOldStr = labelMode.prettyString ;
        labelMode = LabelMode.MULTIVIEWCALIBRATED2 ;
        warningNoTrace('LabelCoreModel:mv', ...
          'Labeling mode ''%s'' does not support multiview projects. Using mode ''%s''.', ...
          labelModeOldStr, labelMode.prettyString) ;
      elseif ~labeler.isMultiView && labelMode == LabelMode.MULTIVIEWCALIBRATED2
        labelModeOldStr = labelMode.prettyString ;
        labelMode = LabelMode.TEMPLATE ;
        warningNoTrace('LabelCoreModel:mv', ...
          'Labeling mode ''%s'' cannot be used for single-view projects. Using mode ''%s''.', ...
          labelModeOldStr, labelMode.prettyString) ;
      end
      obj = LabelCoreModel.create(labeler, labelMode) ;
    end  % function

    function obj = create(labeler, labelMode)
      % Create the LabelCoreModel subclass corresponding to labelMode.
      switch labelMode
        case LabelMode.SEQUENTIAL
          obj = LabelCoreSeqModel(labeler) ;
        case LabelMode.SEQUENTIALADD
          obj = LabelCoreSeqAddModel(labeler) ;
        case LabelMode.TEMPLATE
          obj = LabelCoreTemplateModel(labeler) ;
        case LabelMode.HIGHTHROUGHPUT
          obj = LabelCoreHTModel(labeler) ;
        case LabelMode.MULTIVIEWCALIBRATED2
          obj = LabelCoreMultiViewCalibrated2Model(labeler) ;
        case LabelMode.MULTIANIMAL
          obj = LabelCoreSeqMAModel(labeler) ;
        otherwise
          error('Unknown label mode %s', char(labelMode)) ;
      end
    end  % function

  end  % methods (Static)

  methods (Sealed=true)

    function obj = LabelCoreModel(labeler)
      % Construct a LabelCoreModel, storing a reference to the Labeler.
      if labeler.isMultiView && ~obj.supportsMultiView
        error('LabelCoreModel:MV', ...
              'Multiview labeling not supported by %s.', ...
              class(obj)) ;
      end
      obj.labeler_ = labeler ;
      obj.panZoomMod_ = 'control' ;
      obj.nexttbl_ = [] ;
      obj.nexti_ = 1 ;
    end  % function

    function init(obj, nPts, ptsPlotInfo)
      % Initialize the model with the number of points and plot info.
      obj.nPts_ = nPts ;
      obj.ptsPlotInfo_ = ptsPlotInfo ;
      obj.xy_ = nan(nPts, 2) ;
      obj.tfOcc_ = false(nPts, 1) ;
      obj.tfEstOcc_ = false(nPts, 1) ;
      obj.tfSel_ = false(nPts, 1) ;
      obj.hideLabels_ = false ;
      obj.initHook() ;
    end  % function

  end  % methods (Sealed=true)

  methods

    function initHook(obj) %#ok<MANU>
      % Called from init(). Override in subclasses for custom initialization.
    end  % function

  end  % methods

  %% State transition hooks (called by Labeler)
  methods

    function newFrame(obj, iFrm0, iFrm1, iTgt, tfForceUpdate) %#ok<INUSD>
      % Frame has changed, target is the same.
    end  % function

    function newTarget(obj, iTgt0, iTgt1, iFrm) %#ok<INUSD>
      % Target has changed, frame is the same.
    end  % function

    function newFrameAndTarget(obj, iFrm0, iFrm1, iTgt0, iTgt1, tfForceUpdate) %#ok<INUSD>
      % Frame and target have both changed.
    end  % function

    function clearLabels(obj) %#ok<MANU>
      % Clear current labels and enter initial labeling state.
    end  % function

    function acceptLabels(obj) %#ok<MANU>
      % Accept labels for the current frame/target.
    end  % function

    function unAcceptLabels(obj) %#ok<MANU>
      % Un-accept labels for the current frame/target.
    end  % function

  end  % methods

  %% Coordinate accessors
  methods

    function [xy, tfEO] = getLabelCoords(obj, occval)
      % Get current label coordinates from xy_.
      if ~exist('occval', 'var')
        occval = nan ;
      end
      xy = obj.xy_ ;
      xy(obj.tfOcc_, :) = occval ;
      tfEO = obj.tfEstOcc_ ;
    end  % function

    function xy = getLabelCoordsI(obj, iPt)
      % Get coordinates for specific point(s) from xy_.
      xy = obj.xy_(iPt, :) ;
    end  % function

    function setLabelCoordsI(obj, xy, iPt)
      % Set coordinates for specific point(s) in xy_.
      obj.xy_(iPt, :) = xy ;
      obj.lastChangedIPt_ = iPt ;
      obj.notify('updateLabelCoordsI') ;
    end  % function

    function setLabelCoords(obj, xy, varargin)
      % Set all label coordinates in xy_ and update occlusion flags.
      [lblTags] = myparse(varargin, ...
        'lblTags', []) ;
      obj.xy_ = xy ;
      tfOccld = any(isinf(xy), 2) ;
      obj.tfOcc_ = tfOccld ;
      if ~isempty(lblTags)
        obj.tfEstOcc_ = logical(lblTags) ;
      end
      obj.notify('updateLabelCoords') ;
    end  % function

  end  % methods

  %% Selection
  methods

    function [tf, iSelected] = anyPointSelected(obj)
      % Return whether any point is selected, and its index.
      iSelected = find(obj.tfSel_, 1) ;
      tf = ~isempty(iSelected) ;
    end  % function

    function toggleSelectPoint(obj, iPts)
      % Toggle selection of the specified points.
      tfSl = ~obj.tfSel_(iPts) ;
      obj.tfSel_(:) = false ;
      obj.tfSel_(iPts) = tfSl ;
      obj.notify('updateSelected') ;
    end  % function

    function clearSelected(obj, iExclude)
      % Clear all point selections, optionally excluding some points.
      tf = obj.tfSel_ ;
      if exist('iExclude', 'var')
        tf(iExclude) = false ;
      end
      iSelPts = find(tf) ;
      obj.toggleSelectPoint(iSelPts) ; %#ok<FNDSB>
    end  % function

  end  % methods

  %% Labeler interaction
  methods

    function setLabelPosTagFromEstOcc(obj)
      % Write estimated-occluded tags to Labeler.
      lObj = obj.labeler_ ;
      tfEO = obj.tfEstOcc_ ;
      assert(~any(tfEO & obj.tfOcc_)) ;
      iPtEO = find(tfEO) ;
      iPtNO = find(~tfEO) ;
      lObj.labelPosTagSetI(iPtEO) ; %#ok<FNDSB>
      lObj.labelPosTagClearI(iPtNO) ; %#ok<FNDSB>
    end  % function

    function setNextTable(obj, tbl)
      % Set the table of frames/targets to label next.
      obj.nexttbl_ = tbl ;
      obj.nexti_ = 0 ;
    end  % function

    function clearNextTable(obj)
      % Clear the frame/target queue.
      obj.nexttbl_ = [] ;
      obj.nexti_ = 1 ;
    end  % function

    function setHideLabels(obj, tf)
      % Set label hide state and notify.
      obj.hideLabels_ = tf ;
      obj.notify('updateHideLabels') ;
    end  % function

    function labelsHideToggle(obj)
      % Toggle label visibility.
      obj.setHideLabels(~obj.hideLabels_) ;
    end  % function

  end  % methods

  %% Skeleton and utility
  methods

    function edges = skeletonEdges(obj)
      % Get skeleton edge indices from Labeler.
      edges = obj.labeler_.skeletonEdges ;
    end  % function

  end  % methods

  methods (Static)

    function uv = transformPtsTrx(uv0, trx0, iFrm0, trx1, iFrm1)
      % Transform points between trx coordinate frames.
      %
      % uv0: npts x 2 array of points
      % trx0: scalar trx for source frame
      % iFrm0: absolute frame number for source
      % trx1: scalar trx for destination frame
      % iFrm1: absolute frame number for destination
      %
      % NaN points -> NaN points, Inf points -> Inf points.

      assert(trx0.off == 1 - trx0.firstframe) ;
      assert(trx1.off == 1 - trx1.firstframe) ;

      tfFrmsInBounds = trx0.firstframe <= iFrm0 && iFrm0 <= trx0.endframe && ...
                       trx1.firstframe <= iFrm1 && iFrm1 <= trx1.endframe ;
      if tfFrmsInBounds
        iFrm0Idx = iFrm0 + trx0.off ;
        xy0 = [trx0.x(iFrm0Idx) trx0.y(iFrm0Idx)] ;
        th0 = trx0.theta(iFrm0Idx) ;

        iFrm1Idx = iFrm1 + trx1.off ;
        xy1 = [trx1.x(iFrm1Idx) trx1.y(iFrm1Idx)] ;
        th1 = trx1.theta(iFrm1Idx) ;

        if isnan(th0 - th1)
          th0 = 0 ;
          th1 = 0 ;
        end
        uv = transformPoints(uv0, xy0, th0, xy1, th1) ;
        tfinf = any(isinf(uv0), 2) ;
        uv(tfinf, :) = inf ;
      else
        uv = uv0 ;
      end
    end  % function

  end  % methods (Static)

end  % classdef
