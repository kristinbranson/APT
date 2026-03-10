classdef LabelCoreHTModel < LabelCoreModel
% High-throughput labeling model
%
% Owns the data-side state for HT labeling mode. The user goes through
% a movie, labeling one point at a time (iPoint) in every nFrameSkip-th
% frame. When the end of the movie is reached, iPoint advances to the
% next point and labeling restarts at frame 1.
%
% There is no LABEL/ADJUST/ACCEPTED state machine in HT mode. The
% tbAccept toggle is always disabled. Only one point (iPoint) is
% "active" at a time.
%
% Communicates with the controller via events:
%   updateIPoint       - iPoint_ changed
%   updateLabelCoords  - xy_/frame label data changed (inherited)

  events
    updateIPoint              % iPoint_ changed
  end

  properties
    supportsSingleView = true ;
    supportsMultiView = false ;
    supportsCalibration = false ;
    supportsMultiAnimal = false ;
    unsupportedKPFFns = {} ;
  end

  properties (Transient)
    iPoint_             % scalar. Either nan, or index of pt currently being labeled
    nFrameSkip_         % scalar positive integer. Number of frames to skip per click
  end

  properties (Dependent)
    iPoint
    nFrameSkip
  end

  methods

    function result = get.iPoint(obj)
      % Return the current point index.
      result = obj.iPoint_ ;
    end  % function

    function set.iPoint(obj, val)
      % Set the current point index (delegates to setIPoint for validation).
      obj.setIPoint(val) ;
    end  % function

    function result = get.nFrameSkip(obj)
      % Return the frame skip value.
      result = obj.nFrameSkip_ ;
    end  % function

    function set.nFrameSkip(obj, val)
      % Set the frame skip value with validation.
      validateattributes(val, {'numeric'}, {'positive' 'integer'}) ;
      obj.nFrameSkip_ = val ;
    end  % function

  end  % methods

  methods

    function obj = LabelCoreHTModel(labeler)
      % Construct a LabelCoreHTModel.
      obj = obj@LabelCoreModel(labeler) ;
    end  % function

    function initHook(obj)
      % Initialize HT-specific state from ptsPlotInfo.
      obj.iPoint_ = 1 ;
      obj.xy_ = nan(obj.nPts_, 2) ;

      ppi = obj.ptsPlotInfo_ ;
      htm = ppi.HighThroughputMode ;
      obj.nFrameSkip_ = htm.NFrameSkip ;

      obj.labeler_.currImHud.updateReadoutFields('hasLblPt', true) ;
    end  % function

  end  % methods

  %% State transition hooks
  methods

    function newFrame(obj, iFrm0, iFrm1, iTgt, tfForceUpdate) %#ok<INUSL>
      % Frame has changed. Read labels from Labeler, update xy_, and notify.
      if ~exist('tfForceUpdate', 'var')
        tfForceUpdate = false ; %#ok<NASGU>
      end

      s = obj.labeler_.labelsCurrMovie ;
      [tf, p] = Labels.isLabeledFT(s, iFrm1, iTgt) ;
      xy = reshape(p, [], 2) ;

      iPt = obj.iPoint_ ;

      % If iPoint is unlabeled, preserve its current position in xy_
      tfUnlabeled = isnan(xy(:, 1)) ;
      if tfUnlabeled(iPt)
        xy(iPt, :) = obj.xy_(iPt, :) ;
      end

      obj.xy_ = xy ;

      % Store frame label metadata for the controller to use
      obj.lastFrameLabeled_.tfUnlabeled = tfUnlabeled ;
      obj.lastFrameLabeled_.tfLabeledOrOcc = ~tfUnlabeled ;

      % Read tag data for the controller
      lpostag = obj.labeler_.labeledpostagCurrMovie ;
      obj.lastFrameLabeled_.tfOccTag = lpostag(:, iFrm1, iTgt) ;

      obj.notify('updateLabelCoords') ;
    end  % function

    function newTarget(obj, iTgt0, iTgt1, iFrm) %#ok<INUSD>
      % Target changed. No-op for HT mode (multi-target not supported).
    end  % function

    function newFrameAndTarget(obj, iFrm0, iFrm1, iTgt0, iTgt1, tfForceUpdate) %#ok<INUSL>
      % Frame and target both changed. Delegate to newFrame.
      if ~exist('tfForceUpdate', 'var')
        tfForceUpdate = false ; %#ok<NASGU>
      end
      obj.newFrame([], iFrm1, iTgt1) ;
    end  % function

    function clearLabels(obj)
      % Clear the label for iPoint in the Labeler.
      iPt = obj.iPoint_ ;
      obj.labeler_.labelPosClearI(iPt) ;
      obj.notify('updateLabelCoords') ;
    end  % function

    function acceptLabels(obj) %#ok<MANU>
      % Accept labels. Not applicable in HT mode.
      assert(false) ;
    end  % function

    function unAcceptLabels(obj) %#ok<MANU>
      % Un-accept labels. Not applicable in HT mode.
      assert(false) ;
    end  % function

  end  % methods

  %% Point management
  methods

    function setIPoint(obj, iPt)
      % Set the currently-labeled point index with validation.

      if ~any(iPt == (1:obj.nPts_))
        error('LabelCoreHTModel:setIPoint', ...
              'Invalid value for labeling point iPoint.') ;
      end

      obj.iPoint_ = iPt ;

      lObj = obj.labeler_ ;
      lObj.currImHud.updateLblPoint(iPt, obj.nPts_) ;

      obj.notify('updateIPoint') ;

      if lObj.currMovie > 0
        obj.newFrame([], lObj.currFrame, lObj.currTarget) ;
      end
    end  % function

  end  % methods

  %% Scratch properties for model-controller communication
  properties (Transient)
    lastFrameLabeled_       % struct with fields: tfUnlabeled, tfLabeledOrOcc, tfOccTag
  end

end  % classdef
