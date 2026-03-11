function test_template_labeling_in_SA_project_with_trx()
% Test template-mode labeling with trx by simulating user clicks.

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

% Switch to Template labeling mode
labeler.labelingInit('labelMode', LabelMode.TEMPLATE) ;
drawnow() ;

lblCore = labeler.lblCore ;
ax = controller.axes_curr ;
nPts = lblCore.nPts ;
fprintf('nPts = %d\n', nPts) ;

% Template mode starts in ADJUST state (not LABEL like sequential mode)
iFrm = labeler.currFrame ;
iTgt = labeler.currTarget ;

% Clear any existing labels to ensure we start in ADJUST state
[tfIsLabeled, ~, ~] = labeler.labelPosIsLabeled(iFrm, iTgt) ;
if tfIsLabeled
  controller.controlActuated('pbClear') ;
  drawnow() ;
end

assert(lblCore.state == LabelState.ADJUST, ...
       'Expected ADJUST state in template mode, got %s', ...
       char(lblCore.state)) ;

% All points should start unadjusted
assert(~any(lblCore.tfAdjusted_), ...
       'Expected all points unadjusted at start') ;

% Label several targets by selecting each point, clicking to move it, and accepting
xLim = get(ax, 'XLim') ;
yLim = get(ax, 'YLim') ;
nFramesToLabel = 3 ;

for iFrame = 1:nFramesToLabel
  % Navigate to a different target for each iteration (stay on same frame)
  if iFrame > 1
    nextTgt = mod(iTgt - 1 + (iFrame - 1), labeler.nTrx) + 1 ;
    controller.setFrameAndTargetGUI(iFrm, nextTgt, true) ;
    drawnow() ;
    iTgt = nextTgt ;

    % Clear if already labeled
    [tfIsLabeled, ~, ~] = labeler.labelPosIsLabeled(iFrm, iTgt) ;
    if tfIsLabeled
      controller.controlActuated('pbClear') ;
      drawnow() ;
    end
  end

  assert(lblCore.state == LabelState.ADJUST, ...
         'Expected ADJUST state before labeling target %d, got %s', ...
         iFrame, char(lblCore.state)) ;

  % Adjust each point: select it, then click on axes to move it there
  for iPt = 1:nPts
    x = xLim(1) + rand() * (xLim(2) - xLim(1)) ;
    y = yLim(1) + rand() * (yLim(2) - yLim(1)) ;
    lblCore.toggleSelectPoint(iPt) ;
    drawnow() ;
    fakeAxesClick(ax, x, y) ;
    drawnow() ;
  end

  % All points should now be adjusted
  assert(all(lblCore.tfAdjusted_), ...
         'Expected all points adjusted after adjusting target %d', iFrame) ;

  % Accept labels
  lblCore.acceptLabels() ;
  drawnow() ;

  assert(lblCore.state == LabelState.ACCEPTED, ...
         'Expected ACCEPTED state after accepting target %d, got %s', ...
         iFrame, char(lblCore.state)) ;

  % Verify the frame/target is now labeled
  [tfIsLabeled, ~, ~] = labeler.labelPosIsLabeled(iFrm, iTgt) ;
  assert(tfIsLabeled, ...
         'Frame %d target %d should be labeled but is not.', iFrm, iTgt) ;
end

fprintf('Labeled %d targets successfully.\n', nFramesToLabel) ;

% Test that clearing labels works: clear and verify ADJUST state
controller.controlActuated('pbClear') ;
drawnow() ;
assert(lblCore.state == LabelState.ADJUST, ...
       'Expected ADJUST state after clearing labels, got %s', ...
       char(lblCore.state)) ;
[tfIsLabeled, ~, ~] = labeler.labelPosIsLabeled(iFrm, iTgt) ;
assert(~tfIsLabeled, ...
       'Frame %d target %d should be unlabeled after clear but is not.', ...
       iFrm, iTgt) ;
assert(~any(lblCore.tfAdjusted_), ...
       'Expected all points unadjusted after clear') ;

fprintf('Clear labels passed.\n') ;

% Test partial adjustment: adjust only 1 point, verify mixed state
x = xLim(1) + rand() * (xLim(2) - xLim(1)) ;
y = yLim(1) + rand() * (yLim(2) - yLim(1)) ;
lblCore.toggleSelectPoint(1) ;
drawnow() ;
fakeAxesClick(ax, x, y) ;
drawnow() ;

assert(lblCore.tfAdjusted_(1), 'Point 1 should be adjusted') ;
if nPts > 1
  assert(~lblCore.tfAdjusted_(2), 'Point 2 should still be unadjusted') ;
end
assert(lblCore.state == LabelState.ADJUST, ...
       'Expected ADJUST state with partial adjustment, got %s', ...
       char(lblCore.state)) ;

fprintf('Partial adjustment passed.\n') ;

% Test clearing partial adjustment
controller.controlActuated('pbClear') ;
drawnow() ;
assert(~any(lblCore.tfAdjusted_), ...
       'Expected all points unadjusted after clearing partial adjustment') ;
assert(lblCore.state == LabelState.ADJUST, ...
       'Expected ADJUST state after clearing partial adjustment, got %s', ...
       char(lblCore.state)) ;

fprintf('Clear partial adjustment passed.\n') ;

% Test that clearing works after 0..3 adjusted points
for nPointsBeforeClear = 0:min(3, nPts)
  assert(lblCore.state == LabelState.ADJUST, ...
         'Expected ADJUST state before clear test with %d points, got %s', ...
         nPointsBeforeClear, char(lblCore.state)) ;

  % Adjust some points by selecting and clicking
  for iPt = 1:nPointsBeforeClear
    x = xLim(1) + rand() * (xLim(2) - xLim(1)) ;
    y = yLim(1) + rand() * (yLim(2) - yLim(1)) ;
    lblCore.toggleSelectPoint(iPt) ;
    drawnow() ;
    fakeAxesClick(ax, x, y) ;
    drawnow() ;
  end

  % Clear
  controller.controlActuated('pbClear') ;
  drawnow() ;

  % Should be back in ADJUST state with all points unadjusted
  assert(lblCore.state == LabelState.ADJUST, ...
         'Expected ADJUST state after clear with %d points, got %s', ...
         nPointsBeforeClear, char(lblCore.state)) ;
  assert(~any(lblCore.tfAdjusted_), ...
         'Expected all points unadjusted after clear with %d points', ...
         nPointsBeforeClear) ;
  [tfIsLabeled, ~, ~] = labeler.labelPosIsLabeled(iFrm, iTgt) ;
  assert(~tfIsLabeled, ...
         'Frame should be unlabeled after clear with %d points', ...
         nPointsBeforeClear) ;
end

fprintf('Clear after 0-3 adjusted points passed.\n') ;

% Test navigation: label current target, navigate away, navigate back
for iPt = 1:nPts
  x = xLim(1) + rand() * (xLim(2) - xLim(1)) ;
  y = yLim(1) + rand() * (yLim(2) - yLim(1)) ;
  lblCore.toggleSelectPoint(iPt) ;
  drawnow() ;
  fakeAxesClick(ax, x, y) ;
  drawnow() ;
end
lblCore.acceptLabels() ;
drawnow() ;

assert(lblCore.state == LabelState.ACCEPTED, ...
       'Expected ACCEPTED before navigation test, got %s', ...
       char(lblCore.state)) ;

% Navigate to a different target
nextTgt = mod(iTgt, labeler.nTrx) + 1 ;
controller.setFrameAndTargetGUI(iFrm, nextTgt, true) ;
drawnow() ;

assert(lblCore.state == LabelState.ADJUST, ...
       'Expected ADJUST on unlabeled target, got %s', ...
       char(lblCore.state)) ;

% Navigate back to the labeled target
controller.setFrameAndTargetGUI(iFrm, iTgt, true) ;
drawnow() ;

assert(lblCore.state == LabelState.ACCEPTED, ...
       'Expected ACCEPTED on return to labeled target, got %s', ...
       char(lblCore.state)) ;

fprintf('Navigation to labeled target passed.\n') ;
fprintf('test_template_labeling_in_SA_project_with_trx passed.\n') ;

end  % function
