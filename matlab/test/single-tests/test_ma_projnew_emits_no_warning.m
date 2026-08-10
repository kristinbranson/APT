function test_ma_projnew_emits_no_warning()
% Creating a new multi-animal project should complete without emitting any
% warning.
%
% This guards against regressions in project initialization generally, not just
% one particular error.  The motivating case: projNew() -> movieSetNoMovie_() ->
% labelingInit_() fires the didInitLblCore event, whose LabelerController
% listener calls updateMultiTargetLabelOverlay() -> labelMAGetLabelsFrm(), which
% indexes the label array for the current movie -- but a freshly created project
% has no current movie, so it errors.  Because that error happens inside a
% listener callback, MATLAB downgrades it to a warning rather than letting it
% propagate, so it does not fail the enclosing test on its own.  Rather than look
% for that one warning, this test asserts that projNew emits no warning at all
% (of any identifier), detected via lastwarn().

[labeler, controller] = StartAPT() ;
cleanupObj = onCleanup(@()(delete(controller))) ; 

cfg = simpleMAProjectConfigForTesting() ;

lastwarn('', '') ;
labeler.projNew(cfg) ;
[warningMessage, warningIdentifier] = lastwarn() ;

assert(isempty(warningMessage), ...
       'Creating a new MA project raised a warning (%s): %s', ...
       warningIdentifier, warningMessage) ;
end  % function
