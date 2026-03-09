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
ax = controller.axes_curr ;
nPts = lblCore.nPts ;
newTargetButton = lblCore.pbNewTgt ;
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

fprintf('test_label_animals passed: labeled %d animals successfully.\n', ...
        nAnimalsToLabel) ;

end  % function
