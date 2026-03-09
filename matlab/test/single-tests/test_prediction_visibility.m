function test_prediction_visibility()
% Test the View > Show Predictions menu, cycling through all three modes.

% Set RNG to fixed seed for reproducibility.
originalRngState = rng() ;
rng(42) ;
rngCleanupObj = onCleanup(@()(rng(originalRngState))) ;  %#ok<NASGU>

% Start APT with a project that has saved tracking results.
projectFile = '/groups/branson/bransonlab/apt/unittest/htflies-10-with-saved-tracks.lbl' ;
[labeler, controller] = StartAPT('projfile', projectFile, ...
                                  'isInDebugMode', true) ;
cleanupObj = onCleanup(@()(delete(controller))) ;  %#ok<NASGU>
drawnow() ;

% Verify prerequisites: tracker and prediction visualizer are present.
tracker = labeler.tracker ;
assert(~isempty(tracker), ...
       'Expected a non-empty tracker after loading project') ;
assert(~isempty(controller.tvTrkPred_), ...
       'Expected non-empty tvTrkPred_ (prediction visualizer) after loading project') ;

fprintf('Prerequisites verified: tracker and tvTrkPred_ are present.\n') ;

% -----------------------------------------------------------------------
% Section 1: Cycle through all three modes via controlActuated.
% -----------------------------------------------------------------------

% Mode: All targets
controller.controlActuated('menu_view_showhide_preds_all_targets', [], []) ;
drawnow() ;
assertModeState(tracker, controller, false, false, 'on', 'off', 'off', 'All targets') ;

% Mode: Current target only
controller.controlActuated('menu_view_showhide_preds_curr_target_only', [], []) ;
drawnow() ;
assertModeState(tracker, controller, false, true, 'off', 'on', 'off', 'Current target only') ;

% Mode: None
controller.controlActuated('menu_view_showhide_preds_none', [], []) ;
drawnow() ;
assertModeState(tracker, controller, true, true, 'off', 'off', 'on', 'None') ;

fprintf('Section 1 passed: all three modes cycled correctly.\n') ;

% -----------------------------------------------------------------------
% Section 2: Rapid toggle via menu_view_hide_predictions.
% -----------------------------------------------------------------------

% Start from a known state: show all targets.
controller.controlActuated('menu_view_showhide_preds_all_targets', [], []) ;
drawnow() ;
assert(~tracker.hideViz, 'Expected hideViz=false before rapid toggle') ;

nToggles = 10 ;
for iToggle = 1:nToggles
  tfExpectedHide = (mod(iToggle, 2) == 1) ;
  controller.controlActuated('menu_view_hide_predictions', [], []) ;
  drawnow() ;
  assert(tracker.hideViz == tfExpectedHide, ...
         'Toggle %d: expected hideViz=%d, got %d', ...
         iToggle, tfExpectedHide, tracker.hideViz) ;
end

fprintf('Section 2 passed: %d rapid toggles alternated correctly.\n', nToggles) ;

% -----------------------------------------------------------------------
% Section 3: Switch modes on a frame with many targets.
% -----------------------------------------------------------------------

% All targets
controller.controlActuated('menu_view_showhide_preds_all_targets', [], []) ;
drawnow() ;
assertModeState(tracker, controller, false, false, 'on', 'off', 'off', 'Many-target all') ;

% Current target only
controller.controlActuated('menu_view_showhide_preds_curr_target_only', [], []) ;
drawnow() ;
assertModeState(tracker, controller, false, true, 'off', 'on', 'off', 'Many-target current') ;

% None
controller.controlActuated('menu_view_showhide_preds_none', [], []) ;
drawnow() ;
assertModeState(tracker, controller, true, true, 'off', 'off', 'on', 'Many-target none') ;

% Back to all targets
controller.controlActuated('menu_view_showhide_preds_all_targets', [], []) ;
drawnow() ;
assertModeState(tracker, controller, false, false, 'on', 'off', 'off', 'Many-target all again') ;

fprintf('Section 3 passed: mode switching on frame with targets.\n') ;
fprintf('test_prediction_visibility passed.\n') ;

end  % function


function assertModeState(tracker, controller, ...
                         tfExpectedHideViz, tfExpectedShowCurrOnly, ...
                         expectedAllChecked, expectedCurrChecked, expectedNoneChecked, ...
                         modeLabel)
% Assert that tracker state and menu checked properties match expectations.

assert(tracker.hideViz == tfExpectedHideViz, ...
       '%s: expected hideViz=%d, got %d', ...
       modeLabel, tfExpectedHideViz, tracker.hideViz) ;
assert(tracker.showPredsCurrTargetOnly == tfExpectedShowCurrOnly, ...
       '%s: expected showPredsCurrTargetOnly=%d, got %d', ...
       modeLabel, tfExpectedShowCurrOnly, tracker.showPredsCurrTargetOnly) ;

allChecked = controller.menu_view_showhide_preds_all_targets.Checked ;
assert(strcmp(allChecked, expectedAllChecked), ...
       '%s: expected all_targets Checked=''%s'', got ''%s''', ...
       modeLabel, expectedAllChecked, allChecked) ;

currChecked = controller.menu_view_showhide_preds_curr_target_only.Checked ;
assert(strcmp(currChecked, expectedCurrChecked), ...
       '%s: expected curr_target_only Checked=''%s'', got ''%s''', ...
       modeLabel, expectedCurrChecked, currChecked) ;

noneChecked = controller.menu_view_showhide_preds_none.Checked ;
assert(strcmp(noneChecked, expectedNoneChecked), ...
       '%s: expected none Checked=''%s'', got ''%s''', ...
       modeLabel, expectedNoneChecked, noneChecked) ;

end  % function
