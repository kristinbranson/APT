function test_labeling_in_MA_project()
% Test programmatic multi-animal labeling by simulating user clicks.

% Set RNG to set seed for reproducibility.
originalRngState = rng() ;
rng(42) ;
rngCleanupObj = onCleanup(@()(rng(originalRngState))) ;

% Start APT with a suitable project
projectFile = '/groups/branson/bransonlab/apt/unittest/htflies-10-with-saved-tracks.lbl' ;
[labeler, controller] = StartAPT('projfile', projectFile, ...
                                  'isInDebugMode', true) ;
cleanupObj = onCleanup(@()(delete(controller))) ;
drawnow() ;

lblCore = labeler.lblCore ;
lblCoreController = controller.lblCoreController_ ;
ax = controller.axes_curr ;
nPts = lblCore.nPts ;
newTargetButton = lblCoreController.pbNewTgt_ ;
fprintf('nPts = %d\n', nPts) ;

assert(lblCore.state == LabelState.ACCEPTED, ...
       'Expected ACCEPTED state after loading project, got %s', ...
       char(lblCore.state)) ;

targetCountBefore = labeler.labelNumLabeledTgts() ;
nAnimalsToLabel = 3 ;

for iAnimal = 1:nAnimalsToLabel
  % Should be in ACCEPTED state with button reading "New Target"
  assert(lblCore.state == LabelState.ACCEPTED, ...
         'Expected ACCEPTED state before animal %d, got %s', ...
         iAnimal, char(lblCore.state)) ;
  buttonString = get(newTargetButton, 'String') ;
  assert(strcmp(buttonString, 'New Target'), ...
         'Expected "New Target" but got "%s" before animal %d', ...
         buttonString, iAnimal) ;

  % Click "New Target" to enter LABEL state
  fakePushbuttonClick(newTargetButton) ;
  drawnow() ;

  % Should now be in LABEL state with button reading "Cancel"
  assert(lblCore.state == LabelState.LABEL, ...
         'Expected LABEL state after clicking New Target for animal %d, got %s', ...
         iAnimal, char(lblCore.state)) ;
  buttonString = get(newTargetButton, 'String') ;
  assert(strcmp(buttonString, 'Cancel'), ...
         'Expected "Cancel" but got "%s" after clicking New Target for animal %d', ...
         buttonString, iAnimal) ;

  % First 2 clicks are consumed by two-click alignment, then nPts clicks
  % place the landmark points.
  xLim = get(ax, 'XLim') ;
  yLim = get(ax, 'YLim') ;
  nClicks = 2 + nPts ;
  for iClick = 1:nClicks
    x = xLim(1) + rand() * (xLim(2) - xLim(1)) ;
    y = yLim(1) + rand() * (yLim(2) - yLim(1)) ;
    fakeAxesClick(ax, x, y) ;
    drawnow() ;
    % Should stay in LABEL state until the last label point is placed
    if iClick < nClicks
      assert(lblCore.state == LabelState.LABEL, ...
             'Expected LABEL state after click %d for animal %d, got %s', ...
             iClick, iAnimal, char(lblCore.state)) ;
    end
  end

  % After all points placed, acceptLabels() fires automatically.
  % Should be back in ACCEPTED state.
  assert(lblCore.state == LabelState.ACCEPTED, ...
         'Expected ACCEPTED state after labeling animal %d, got %s', ...
         iAnimal, char(lblCore.state)) ;
  buttonString = get(newTargetButton, 'String') ;
  assert(strcmp(buttonString, 'New Target'), ...
         'Expected "New Target" after labeling animal %d but got "%s"', ...
         iAnimal, buttonString) ;

  % Target count should have increased by one
  targetCountNow = labeler.labelNumLabeledTgts() ;
  assert(targetCountNow == targetCountBefore + iAnimal, ...
         'Expected %d labeled targets after animal %d, got %d', ...
         targetCountBefore + iAnimal, iAnimal, targetCountNow) ;
end

targetCountAfter = labeler.labelNumLabeledTgts() ;
targetCountAdded = targetCountAfter - targetCountBefore ;
assert(targetCountAdded == nAnimalsToLabel, ...
       'Expected %d new targets but got %d', ...
       nAnimalsToLabel, targetCountAdded) ;

fprintf('Labeled %d animals successfully.\n', nAnimalsToLabel) ;

% Test that clicking Cancel during labeling returns to ACCEPTED state
% without adding a target.  Test after 0, 1, 2, and 3 clicks (covering
% no clicks, alignment-only clicks, and alignment + one label click).
targetCountBeforeCancel = labeler.labelNumLabeledTgts() ;
xLim = get(ax, 'XLim') ;
yLim = get(ax, 'YLim') ;

for nClicksBeforeCancel = 0:3
  assert(lblCore.state == LabelState.ACCEPTED, ...
         'Expected ACCEPTED state before cancel test with %d clicks, got %s', ...
         nClicksBeforeCancel, char(lblCore.state)) ;

  % Click "New Target" to enter LABEL state
  fakePushbuttonClick(newTargetButton) ;
  drawnow() ;
  assert(lblCore.state == LabelState.LABEL, ...
         'Expected LABEL state after New Target in cancel test with %d clicks, got %s', ...
         nClicksBeforeCancel, char(lblCore.state)) ;

  % Place some clicks
  for iClick = 1:nClicksBeforeCancel
    x = xLim(1) + rand() * (xLim(2) - xLim(1)) ;
    y = yLim(1) + rand() * (yLim(2) - yLim(1)) ;
    fakeAxesClick(ax, x, y) ;
    drawnow() ;
  end

  % Click Cancel (same button, now in LABEL state)
  buttonString = get(newTargetButton, 'String') ;
  assert(strcmp(buttonString, 'Cancel'), ...
         'Expected "Cancel" before cancel click with %d clicks, got "%s"', ...
         nClicksBeforeCancel, buttonString) ;
  fakePushbuttonClick(newTargetButton) ;
  drawnow() ;

  % Should be back in ACCEPTED state with no new target added
  assert(lblCore.state == LabelState.ACCEPTED, ...
         'Expected ACCEPTED state after cancel with %d clicks, got %s', ...
         nClicksBeforeCancel, char(lblCore.state)) ;
  buttonString = get(newTargetButton, 'String') ;
  assert(strcmp(buttonString, 'New Target'), ...
         'Expected "New Target" after cancel with %d clicks, got "%s"', ...
         nClicksBeforeCancel, buttonString) ;
  targetCountNow = labeler.labelNumLabeledTgts() ;
  assert(targetCountNow == targetCountBeforeCancel, ...
         'Target count changed from %d to %d after cancel with %d clicks', ...
         targetCountBeforeCancel, targetCountNow, nClicksBeforeCancel) ;
end

fprintf('Cancel after 0-3 clicks passed.\n') ;
fprintf('test_labeling_in_MA_project passed.\n') ;

end  % function
