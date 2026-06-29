function test_project_setup_new_user()
  % Reproduces the "Conversion to double from cell is not possible" error that
  % a brand-new user hit when creating a project (ProjectSetup.patchCfg).
  %
  % A new user has no ~/.apt directory yet, so RC.getprop('lastProjectConfig')
  % returns empty and Labeler.cfgGetLastProjectConfigNoView() falls back to the
  % default config.  In the default config YAML the optional fields ViewNames
  % and LabelPointNames are left blank, and yaml.ReadYaml() represents a blank
  % field as an empty double array ([]), not an empty cell array ({}).
  % ProjectSetup.patchCfg() then tried to assign name strings into those fields
  % with cell-style indexing, which errored because the fields were doubles.
  %
  % The fix lives in apt.readDefaultCfg(), which reads the YAML and coerces the
  % empty-double name fields to empty cell arrays.  This test checks that
  % coercion and that the full new-user config-patching path now succeeds.

  % Show the raw YAML really does produce empty doubles for the name fields, so
  % the condition this test guards against is the actual one a new user hits.
  rawCfg = yaml.ReadYaml(fullfile(APT.Root, 'matlab', 'config.default.yaml')) ;
  if iscell(rawCfg.ViewNames) || iscell(rawCfg.LabelPointNames)
    error('Test precondition failed: raw YAML name fields should be empty doubles, not cells') ;
  end

  % apt.readDefaultCfg() should coerce those empty doubles to empty cells.
  cfg = apt.readDefaultCfg() ;
  if ~iscell(cfg.ViewNames) || ~iscell(cfg.LabelPointNames)
    error('apt.readDefaultCfg() did not coerce name fields to cell arrays') ;
  end

  % This is the call that errored for the new user.  It should now succeed.
  keypointCount = 5 ;
  viewCount = 2 ;
  result = ProjectSetup.patchCfg(cfg, 'newUserProject', keypointCount, viewCount, false, false) ;

  % The patched config should have proper cell arrays of the right length.
  if ~iscell(result.ViewNames) || numel(result.ViewNames) ~= viewCount
    error('ViewNames was not turned into a cell array of length %d', viewCount) ;
  end
  if ~iscell(result.LabelPointNames) || numel(result.LabelPointNames) ~= keypointCount
    error('LabelPointNames was not turned into a cell array of length %d', keypointCount) ;
  end
  if ~strcmp(result.ProjectName, 'newUserProject')
    error('ProjectName was not set correctly') ;
  end
  if result.NumViews ~= viewCount || result.NumLabelPoints ~= keypointCount
    error('NumViews/NumLabelPoints were not set correctly') ;
  end
end  % function
