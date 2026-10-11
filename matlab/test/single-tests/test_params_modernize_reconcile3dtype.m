function test_params_modernize_reconcile3dtype()
  % Test that APTParameters.modernize maps the retired 'experimental' value
  % of PostProcess.reconcile3dType onto 'triangulate', and that the
  % parameter tree no longer offers it.  The experimental 3D reconciliation
  % (viewpref3drecon) is gone: it needed the stereo-calibrator rig
  % (CalRigMLStro) that was removed before it, so it errored for every
  % project.  A project that saved the value must still load with a value
  % the Tracking Parameters dropdown can show.

  old = struct ;
  old.ROOT.PostProcess.reconcile3dType = 'experimental' ;
  capturedOutput = evalc('new = APTParameters.modernize(old) ;') ;
  assert(~contains(capturedOutput, 'Ignoring unrecognized field'), ...
         'modernize() left unrecognized fields for structoverlay to warn about:\n%s', ...
         capturedOutput) ;
  assert(strcmp(new.ROOT.PostProcess.reconcile3dType, 'triangulate'), ...
         'reconcile3dType ''experimental'' should migrate to ''triangulate''') ;

  % The values the tree still offers are preserved.
  for value = {'none', 'triangulate'}
    old.ROOT.PostProcess.reconcile3dType = value{1} ;
    evalc('new = APTParameters.modernize(old) ;') ;
    assert(strcmp(new.ROOT.PostProcess.reconcile3dType, value{1}), ...
           'reconcile3dType ''%s'' should be preserved', value{1}) ;
  end

  tree = APTParameters.defaultParamsTree() ;
  node = tree.findnode('ROOT.PostProcess.reconcile3dType') ;
  assert(iscell(node.Data.Type) && ~any(strcmp(node.Data.Type, 'experimental')), ...
         'The parameter tree should no longer offer ''experimental''') ;
end  % function
