function test_ma_label_overlay_survives_tracker_switch()
% Switching the current tracker must not blank the multi-animal label
% overlay.  Regression test for a bug where Labeler.labelingInit_ fired the
% LabelCore model's updateMultiTargetLabelOverlay event before the new LabelCore
% controller existed, so LabelCoreSeqMAController never redrew the other
% targets' labels or the target ROI/pch after a tracker switch -- leaving
% only the current target's editable points on screen.

  linuxProjectFilePath = ...
    '/groups/branson/bransonlab/apt/unittest/htflies-10-with-trks-from-two-trackers.lbl' ;
  [projectFilePath, replacePath] = localize_test_project_path(linuxProjectFilePath) ;
  [labeler, controller] = StartAPT('projfile', projectFilePath, ...
                                   'replace_path', replacePath, ...
                                   'isInDebugMode', true, 'isInYodaMode', true) ;
  cleaner = onCleanup(@()(delete(controller))) ;  %#ok<NASGU>
  labeler.isInBatchMode = true ;

  if ~labeler.maIsMA
    error('Test project is expected to be multi-animal') ;
  end
  if numel(labeler.trackerHistory_) < 2
    error('Test project is expected to have at least two trackers') ;
  end

  % Go to a frame with several labeled targets so the MA overlay is populated.
  frame = 19734 ;
  labeler.movieSet(1) ;
  labeler.setFrameAndTarget(frame, 1) ;

  shownBefore = maOverlayTargetCount_(controller) ;
  if shownBefore < 2
    error(['Expected the MA label overlay to show several targets at frame ' ...
           '%d, but it shows %d'], frame, shownBefore) ;
  end

  % Switch to the other tracker.  The overlay must be unchanged.
  labeler.trackMakeExistingTrackerCurrentGivenIndex(2) ;

  shownAfter = maOverlayTargetCount_(controller) ;
  if shownAfter ~= shownBefore
    error(['Switching trackers changed the MA label overlay from %d targets ' ...
           'shown to %d (the other targets'' labels and the target box ' ...
           'disappeared)'], shownBefore, shownAfter) ;
  end
end  % function


function count = maOverlayTargetCount_(controller)
  % Number of targets currently drawn in the multi-animal label overlay
  % (LabelCoreSeqMAController.tv_).  A target counts as shown if any of its
  % label points has a finite coordinate.
  labelCoreController = controller.lblCoreController_ ;
  trackingVisualizer = labelCoreController.tv_ ;
  pointHandles = trackingVisualizer.hXYPrdRed ;  % [nPts x nTgt]
  [pointCount, targetCount] = size(pointHandles) ;
  count = 0 ;
  for targetIndex = 1 : targetCount
    isShown = false ;
    for pointIndex = 1 : pointCount
      xData = get(pointHandles(pointIndex, targetIndex), 'XData') ;
      if ~isempty(xData) && any(isfinite(xData))
        isShown = true ;
        break
      end
    end
    if isShown
      count = count + 1 ;
    end
  end
end  % function
