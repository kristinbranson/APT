function test_project_switch_skeleton_cleanup()
% Regression test: loading a second project must not drive the skeleton
% visualizer from stale state left over from the first project.
%
% When a project is loaded into an already-running Labeler, projLoad restores
% saved properties, including showSkeleton, whose setter notifies the
% 'didSetShowSkeleton' event.  If the second project has more landmarks than the
% first, the skeleton visualizer -- not yet rebuilt for the new project -- still
% holds the first project's skeleton edges / point count, and
% TrackingVisualizerMTFast.updateSkelStc indexes past the new project's data.
% The resulting error is swallowed into a warning by MATLAB's listener
% machinery, so projLoad "succeeds" while the view is briefly driven from stale
% state.  The desired behavior is that switching projects performs this cleanup
% before notifying, so no such listener-callback error occurs.
%
% We load a 2-landmark MA project (htflies) and then a 16-landmark MA project
% (chimpAct) and assert the second load raises no skeleton listener-callback
% error.  The error never propagates as an exception, so we capture projLoad's
% console output and inspect it.

[fewLandmarkLbl, replacePathFew] = localize_test_project_path( ...
  '/groups/branson/bransonlab/apt/unittest/htflies-10-with-saved-tracks-relocated.lbl') ;
% manyLandmarkLbl / replacePathMany are used only inside the evalc'd command
% below, which mlint cannot see, so they need a NASGU suppression -- which has
% to land on the assignment line.
manyLandmarkPath = '/groups/branson/bransonlab/apt/unittest/chimpAct_noTestlabels_detr400k_nocrop.lbl' ;
[manyLandmarkLbl, replacePathMany] = localize_test_project_path(manyLandmarkPath) ; %#ok<ASGLU>

[labeler, controller] = StartAPT('projfile', fewLandmarkLbl, 'replace_path', replacePathFew) ; %#ok<ASGLU>
cleaner = onCleanup(@()(delete(controller))) ;

% Capture everything projLoad prints; the listener-callback error is emitted as
% a warning (with its stack) rather than thrown, so we detect it in the output.
capturedOutput = evalc('labeler.projLoad(manyLandmarkLbl, ''replace_path'', replacePathMany)') ;

tfSkeletonListenerError = ...
  contains(capturedOutput, 'updateSkelStc') || ...
  ( contains(capturedOutput, 'listener callback') && contains(capturedOutput, 'ShowSkeleton') ) ;
assert(~tfSkeletonListenerError, ...
  ['Loading a project with more landmarks fired a skeleton-visualizer ' ...
   'listener-callback error, meaning stale skeleton state from the previous ' ...
   'project was not cleaned up before showSkeleton was notified.\n' ...
   'Captured projLoad output:\n%s'], capturedOutput) ;

fprintf('test_project_switch_skeleton_cleanup: PASSED\n') ;

end  % function
