function test_tracklet_selection()
% Test that selecting tracklets updates the HUD with the correct tracklet ID.

% Start APT with a project that has saved tracking results.
projectFile = '/groups/branson/bransonlab/apt/unittest/htflies-10-with-saved-tracks.lbl' ;
[labeler, controller] = StartAPT('projfile', projectFile, ...
                                 'isInDebugMode', true) ;
cleanupObj = onCleanup(@()(delete(controller))) ;
drawnow() ;

% Verify prerequisites: tracker and prediction visualizer are present.
tracker = labeler.tracker ;
assert(~isempty(tracker), ...
       'Expected a non-empty tracker after loading project') ;
assert(~isempty(controller.tvTrkPred_), ...
       'Expected non-empty tvTrkPred_ (prediction visualizer) after loading project') ;

tv = controller.tvTrkPred_ ;
tvm = tracker.trkVizer ;
ptrx = tvm.ptrx ;
assert(~isempty(ptrx), 'Expected non-empty ptrx (tracklets)') ;

hud = controller.currImHud ;
nTrkletTot = numel(ptrx) ;

fprintf('Prerequisites verified: tracker, tvTrkPred_, and tracklets are present.\n') ;

% -----------------------------------------------------------------------
% Section 1: Select tracklets by simulating user clicks on tracklet
% markers (view callback) and verify HUD.
% -----------------------------------------------------------------------

frm = labeler.currFrame ;
iTrxLive = tvm.frm2trx(frm) ;
assert(numel(iTrxLive) >= 2, ...
       'Expected at least 2 live tracklets on frame %d, got %d', ...
       frm, numel(iTrxLive)) ;

for j = 1:numel(iTrxLive)
  iTrklet = iTrxLive(j) ;
  iTrxViz = find(tvm.iTrxViz2iTrx == iTrklet, 1) ;
  assert(~isempty(iTrxViz), ...
         'Expected tracklet %d to have a viz slot on frame %d', iTrklet, frm) ;
  tv.didSelectTrx(iTrxViz) ;
  drawnow() ;

  expectedStr = sprintf('trklet: %d (%d tot)', ptrx(iTrklet).id, nTrkletTot) ;
  actualStr = hud.hTxtTrklet.String ;
  assert(strcmp(actualStr, expectedStr), ...
         'Section 1: expected HUD "%s", got "%s"', expectedStr, actualStr) ;
end

fprintf('Section 1 passed: view-callback selection updated HUD for %d tracklets.\n', ...
        numel(iTrxLive)) ;

% -----------------------------------------------------------------------
% Section 2: Navigate to a different frame via the controller (simulating
% a right-arrow keypress), then select tracklets via view callback.
% -----------------------------------------------------------------------

controller.frameUp(false) ;
drawnow() ;

frm2 = labeler.currFrame ;
assert(frm2 == frm + 1, ...
       'Expected frame %d after frameUp, got %d', frm + 1, frm2) ;

iTrxLive2 = tvm.frm2trx(frm2) ;
assert(~isempty(iTrxLive2), ...
       'Expected at least 1 live tracklet on frame %d', frm2) ;

for j = 1:numel(iTrxLive2)
  iTrklet = iTrxLive2(j) ;
  iTrxViz = find(tvm.iTrxViz2iTrx == iTrklet, 1) ;
  assert(~isempty(iTrxViz), ...
         'Expected tracklet %d to have a viz slot on frame %d', iTrklet, frm2) ;
  tv.didSelectTrx(iTrxViz) ;
  drawnow() ;

  expectedStr = sprintf('trklet: %d (%d tot)', ptrx(iTrklet).id, nTrkletTot) ;
  actualStr = hud.hTxtTrklet.String ;
  assert(strcmp(actualStr, expectedStr), ...
         'Section 2: expected HUD "%s", got "%s"', expectedStr, actualStr) ;
end

fprintf('Section 2 passed: tracklet selection correct after controller frame navigation.\n') ;
fprintf('test_tracklet_selection passed.\n') ;

end  % function
