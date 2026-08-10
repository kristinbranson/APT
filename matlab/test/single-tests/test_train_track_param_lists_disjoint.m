function test_train_track_param_lists_disjoint()
  % Each deep-learning parameter is meant to appear in EITHER the Training-
  % parameters list or the Tracking-parameters list, never both.  The
  % training list is exactly the AffectsTraining==true params (that pass the
  % project-state condition filter), so "no parameter in both lists" is
  % equivalent to "the Tracking dialog shows no AffectsTraining==true
  % parameter".  This test checks that latter, equivalent, invariant -- which
  % needs only the tracking dialog and so avoids the training dialog's
  % auto-tune machinery (which is unsupported for multiview projects) -- via
  % the real ParameterSetupModalController, across multi-animal vs
  % single-animal projects and several model (net) types.
  %
  % This currently FAILS: ParameterSetupModalController.resetTreeVisible applies
  % the AffectsTraining filter only for the training dialog, so the tracking
  % dialog also shows training-only params (e.g. "N. training iterations",
  % "Training batch size").

  % Net (model) types to exercise for a multi-animal vs a single-animal
  % project.  Each entry is a row of DLNetType: length 1 for single-stage,
  % length 2 for two-stage (detect + pose) top-down.
  maNets = { DLNetType.multi_mdn_joint_torch, ...
             DLNetType.multi_cid, ...
             DLNetType.multi_dekr, ...
             [DLNetType.detect_mmdetect DLNetType.mdn_joint_fpn] } ;
  saNets = { DLNetType.mdn_joint_fpn, ...
             DLNetType.deeplabcut, ...
             DLNetType.unet, ...
             DLNetType.hrnet } ;

  cases = struct( ...
    'name', {'multi-animal', 'single-animal'}, ...
    'linuxPath', ...
      { '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl', ...
        '/groups/branson/bransonlab/apt/unittest/2011_mouse_cam13_updated_movie_paths_20241111_modded.lbl' }, ...
    'expectMA', {true, false}, ...
    'nets', {maNets, saNets} ) ;

  for caseIndex = 1 : numel(cases)
    kase = cases(caseIndex) ;
    [projectFilePath, replacePath] = localize_test_project_path(kase.linuxPath) ;
    [labeler, controller] = StartAPT('projfile', projectFilePath, ...
                                     'replace_path', replacePath, ...
                                     'isInDebugMode', true, 'isInYodaMode', true) ;
    % Reassigning cleaner on the next iteration deletes the previous case's
    % controller; the final one is deleted when the test returns or errors.
    cleaner = onCleanup(@()(delete(controller))) ;
    labeler.isInBatchMode = true ;
    if ~isequal(logical(labeler.maIsMA), kase.expectMA)
      error('%s project %s: maIsMA is %d, expected %d', ...
            kase.name, projectFilePath, labeler.maIsMA, kase.expectMA) ;
    end

    testedNetCount = 0 ;
    for netIndex = 1 : numel(kase.nets)
      netTypes = kase.nets{netIndex} ;
      netLabel = strjoin(arrayfun(@(x)(char(x)), netTypes, 'UniformOutput', false), '+') ;
      % Some nets may not be applicable to a given project; skip those rather
      % than fail, but require that at least one net was actually tested.
      try
        labeler.trackMakeNewTrackerGivenNetTypes(netTypes) ;
      catch causeException
        fprintf('%s: skipping net %s for %s project (%s)\n', ...
                mfilename(), netLabel, kase.name, causeException.message) ;
        continue
      end
      testedNetCount = testedNetCount + 1 ;

      offenders = trainingParamsShownInTrackingDialog_(controller, labeler) ;
      if ~isempty(offenders)
        error(['%s project, net %s: %d training-only parameter(s) ' ...
               '(AffectsTraining==true) shown in the Tracking list: %s'], ...
              kase.name, netLabel, numel(offenders), strjoin(offenders, ', ')) ;
      end
    end

    if testedNetCount == 0
      error('%s project: no net types could be set, so nothing was tested', kase.name) ;
    end
  end
end  % function


function paths = trainingParamsShownInTrackingDialog_(controller, labeler)
  % Return the fully-qualified paths of any AffectsTraining==true parameters
  % that the Tracking dialog would show.  Builds the real (tracking)
  % ParameterSetupModalController and uses its resetTreeVisible() -- the method
  % under test -- to set visibility.
  parameterSetupController = ParameterSetupModalController(controller, labeler, 'istrain', false) ;
  cleaner = onCleanup(@()(delete(parameterSetupController))) ; 
  parameterSetupController.resetTreeVisible() ;
  paths = affectsTrainingVisibleLeafPaths_(parameterSetupController.tree_, '') ;
end  % function


function paths = affectsTrainingVisibleLeafPaths_(nodes, prefix)
  % Recursively collect dotted paths of visible leaf nodes whose
  % AffectsTraining is true, within a TreeNode (or array of TreeNodes).
  paths = {} ;
  for nodeIndex = 1 : numel(nodes)
    node = nodes(nodeIndex) ;
    if ~node.Data.Visible
      continue
    end
    if isempty(prefix)
      thisPath = node.Data.Field ;
    else
      thisPath = [prefix, '.', node.Data.Field] ;
    end
    if isempty(node.Children)
      affectsTraining = node.Data.AffectsTraining ;
      if ~isempty(affectsTraining) && affectsTraining
        paths = [paths, {thisPath}] ;  %#ok<AGROW>
      end
    else
      paths = [paths, affectsTrainingVisibleLeafPaths_(node.Children, thisPath)] ;  %#ok<AGROW>
    end
  end
end  % function
