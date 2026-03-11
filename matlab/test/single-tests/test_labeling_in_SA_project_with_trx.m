function test_labeling_in_SA_project_with_trx()
% Test programmatic single-animal labeling with trx by simulating user clicks.

% Set RNG to fixed seed for reproducibility.
originalRngState = rng() ;
rng(42) ;
rngCleanupObj = onCleanup(@()(rng(originalRngState))) ;  

% Start APT with an SA project that has trx
projectFile = '/groups/branson/bransonlab/apt/unittest/with-trx-project.lbl' ;
[labeler, controller] = StartAPT('projfile', projectFile, ...
                                  'isInDebugMode', true) ;
cleanupObj = onCleanup(@()(delete(controller))) ; 
drawnow() ;

lblCore = labeler.lblCore ;
ax = controller.axes_curr ;
nPts = lblCore.nPts ;
fprintf('nPts = %d\n', nPts) ;

% Navigate to an unlabeled frame/target to ensure we start in LABEL state
iFrm = labeler.currFrame ;
iTgt = labeler.currTarget ;
[tfIsLabeled, ~, ~] = labeler.labelPosIsLabeled(iFrm, iTgt) ;
if tfIsLabeled
  controller.controlActuated('pbClear') ;
  drawnow() ;
end

assert(lblCore.state == LabelState.LABEL, ...
       'Expected LABEL state on unlabeled frame, got %s', ...
       char(lblCore.state)) ;

% Label nPts points by clicking
xLim = get(ax, 'XLim') ;
yLim = get(ax, 'YLim') ;
nFramesToLabel = 3 ;

for iFrame = 1:nFramesToLabel
  % Navigate to a different target for each iteration (stay on same frame)
  if iFrame > 1
    nextTgt = mod(iTgt - 1 + (iFrame - 1), labeler.nTrx) + 1 ;
    labeler.setFrameAndTargetGUI(iFrm, nextTgt, true) ;
    drawnow() ;
    iTgt = nextTgt ;

    % Clear if already labeled
    [tfIsLabeled, ~, ~] = labeler.labelPosIsLabeled(iFrm, iTgt) ;
    if tfIsLabeled
      controller.controlActuated('pbClear') ;
      drawnow() ;
    end
  end

  assert(lblCore.state == LabelState.LABEL, ...
         'Expected LABEL state before labeling target %d, got %s', ...
         iFrame, char(lblCore.state)) ;

  % Click nPts times to place all landmark points
  for iClick = 1:nPts
    x = xLim(1) + rand() * (xLim(2) - xLim(1)) ;
    y = yLim(1) + rand() * (yLim(2) - yLim(1)) ;
    fakeAxesClick(ax, x, y) ;
    drawnow() ;
    if iClick < nPts
      assert(lblCore.state == LabelState.LABEL, ...
             'Expected LABEL state after click %d for target %d, got %s', ...
             iClick, iFrame, char(lblCore.state)) ;
    end
  end

  % After all points placed, acceptLabels() fires automatically
  assert(lblCore.state == LabelState.ACCEPTED, ...
         'Expected ACCEPTED state after labeling target %d, got %s', ...
         iFrame, char(lblCore.state)) ;

  % Verify the frame/target is now labeled
  [tfIsLabeled, ~, ~] = labeler.labelPosIsLabeled(iFrm, iTgt) ;
  assert(tfIsLabeled, ...
         'Frame %d target %d should be labeled but is not.', iFrm, iTgt) ;
end

fprintf('Labeled %d targets successfully.\n', nFramesToLabel) ;

% Test that clearing labels works: clear current labels and verify LABEL state
controller.controlActuated('pbClear') ;
drawnow() ;
assert(lblCore.state == LabelState.LABEL, ...
       'Expected LABEL state after clearing labels, got %s', ...
       char(lblCore.state)) ;
[tfIsLabeled, ~, ~] = labeler.labelPosIsLabeled(iFrm, iTgt) ;
assert(~tfIsLabeled, ...
       'Frame %d target %d should be unlabeled after clear but is not.', ...
       iFrm, iTgt) ;

fprintf('Clear labels passed.\n') ;

% Test undo: place some points and undo them
for iClick = 1:2
  x = xLim(1) + rand() * (xLim(2) - xLim(1)) ;
  y = yLim(1) + rand() * (yLim(2) - yLim(1)) ;
  fakeAxesClick(ax, x, y) ;
  drawnow() ;
end
assert(lblCore.nPtsLabeled_ == 2, ...
       'Expected 2 points labeled, got %d', lblCore.nPtsLabeled_) ;

lblCore.undoLastLabel() ;
drawnow() ;
assert(lblCore.nPtsLabeled_ == 1, ...
       'Expected 1 point after undo, got %d', lblCore.nPtsLabeled_) ;

lblCore.undoLastLabel() ;
drawnow() ;
assert(lblCore.nPtsLabeled_ == 0, ...
       'Expected 0 points after second undo, got %d', lblCore.nPtsLabeled_) ;

fprintf('Undo passed.\n') ;

% Test that clearing after different numbers of clicks returns to LABEL state
% without persisting any labels.  Covers 0, 1, 2, and 3 clicks.
% First, make sure the current frame/target is unlabeled.
[tfIsLabeled, ~, ~] = labeler.labelPosIsLabeled(iFrm, iTgt) ;
if tfIsLabeled
  controller.controlActuated('pbClear') ;
  drawnow() ;
end

for nClicksBeforeClear = 0:3
  assert(lblCore.state == LabelState.LABEL, ...
         'Expected LABEL state before clear test with %d clicks, got %s', ...
         nClicksBeforeClear, char(lblCore.state)) ;

  % Place some clicks
  xLim = get(ax, 'XLim') ;
  yLim = get(ax, 'YLim') ;
  for iClick = 1:nClicksBeforeClear
    x = xLim(1) + rand() * (xLim(2) - xLim(1)) ;
    y = yLim(1) + rand() * (yLim(2) - yLim(1)) ;
    fakeAxesClick(ax, x, y) ;
    drawnow() ;
  end
  if nClicksBeforeClear > 0
    assert(lblCore.nPtsLabeled_ == nClicksBeforeClear, ...
           'Expected %d points labeled, got %d', ...
           nClicksBeforeClear, lblCore.nPtsLabeled_) ;
  end

  % Clear to cancel
  controller.controlActuated('pbClear') ;
  drawnow() ;

  % Should be back in LABEL state with no labels persisted
  assert(lblCore.state == LabelState.LABEL, ...
         'Expected LABEL state after clear with %d clicks, got %s', ...
         nClicksBeforeClear, char(lblCore.state)) ;
  assert(lblCore.nPtsLabeled_ == 0, ...
         'Expected 0 points after clear with %d clicks, got %d', ...
         nClicksBeforeClear, lblCore.nPtsLabeled_) ;
  [tfIsLabeled, ~, ~] = labeler.labelPosIsLabeled(iFrm, iTgt) ;
  assert(~tfIsLabeled, ...
         'Frame should be unlabeled after clear with %d clicks', ...
         nClicksBeforeClear) ;
end

fprintf('Clear after 0-3 clicks passed.\n') ;
fprintf('test_labeling_in_SA_project_with_trx passed.\n') ;

end  % function
