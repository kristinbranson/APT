function test_parameter_tree_loading()
% Test that the default parameter tree loads from the parameter config
% files and survives a struct round-trip.  This pins down the current
% behavior so that changes to the parameter file format (e.g. a
% YAML-to-JSON conversion) or reorganizations of the tree get caught.

% The default tree should load without error and be a scalar TreeNode
tree = APTParameters.defaultParamsTree() ;
assert(isscalar(tree) && isa(tree, 'TreeNode'), ...
       'defaultParamsTree() did not return a scalar TreeNode') ;

% The tree should contain the expected top-level sections
sPrmDefault = tree.structize() ;
expectedSectionNames = ...
  { 'Track', 'MultiAnimal', 'DeepTrack', 'PostProcess' } ;
sectionNames = fieldnames(sPrmDefault.ROOT) ;
for i = 1 : numel(expectedSectionNames)
  assert(ismember(expectedSectionNames{i}, sectionNames), ...
         'Section %s is missing from the default parameter tree', ...
         expectedSectionNames{i}) ;
end

% Every leaf should be a well-formed PropertiesGUIProp
nodes = tree.flatten() ;
nodeCount = numel(nodes) ;
leafCount = 0 ;
for i = 1 : nodeCount
  node = nodes(i) ;
  isLeaf = isempty(node.Children) ;
  if ~isLeaf
    continue
  end
  leafCount = leafCount + 1 ;
  data = node.Data ;
  assert(isa(data, 'PropertiesGUIProp'), ...
         'Leaf %d has Data of class %s, not PropertiesGUIProp', i, class(data)) ;
  assert(ischar(data.Field) && ~isempty(data.Field), ...
         'Leaf %d has a missing or empty Field', i) ;
  % Type is a char naming the type, or a cellstr of allowed options
  isTypeValid = (ischar(data.Type) && ~isempty(data.Type)) || ...
                (iscellstr(data.Type) && ~isempty(data.Type)) ;  %#ok<ISCLSTR>
  assert(isTypeValid, ...
         'Leaf %s has an invalid Type (class %s)', data.Field, class(data.Type)) ;
  % For an enumerated (cellstr-typed) leaf, the default must be one of
  % the allowed options -- the new-style GUI's uidropdown errors otherwise
  if iscell(data.Type)
    assert(ismember(data.DefaultValue, data.Type), ...
           'Leaf %s has DefaultValue ''%s'' that is not among its Type options', ...
           data.Field, char(string(data.DefaultValue))) ;
  end
  % The PropertiesGUIProp constructor normalizes Level to an enum, so a
  % char Level (the declared property default) should never appear in a
  % loaded tree
  assert(isscalar(data.Level) && isa(data.Level, 'PropertyLevelsEnum'), ...
         'Leaf %s has a Level that is not a scalar PropertyLevelsEnum (class %s)', ...
         data.Field, class(data.Level)) ;
  assert(iscell(data.Requirements), ...
         'Leaf %s has non-cell Requirements', data.Field) ;
end
assert(leafCount > 100, ...
       'Expected well over 100 parameter leaves, found %d', leafCount) ;

% Applying the structized defaults back onto a fresh tree should be the
% identity: this is the overlay operation used when loading a project's
% saved parameters
secondTree = APTParameters.defaultParamsTree() ;
secondTree.structapply(sPrmDefault) ;
sPrmRoundTripped = secondTree.structize() ;
assert(isequaln(sPrmDefault, sPrmRoundTripped), ...
       'Parameter struct did not survive a structapply()/structize() round-trip') ;

% The tracking-parameter subset (AffectsTraining == false) should
% extract cleanly.  The struct form deliberately contains the full
% parameter structure (the tracking-job config relies on that); the
% tracking/training distinction is carried by the tree's Visible flags,
% which is what the tracking-parameters dialog renders from.
sPrmTrack = APTParameters.all2TrackParams(sPrmDefault) ;
assert(structisfield(sPrmTrack, 'ROOT.Track.NFramesNeighborhood'), ...
       'Tracking parameter NFramesNeighborhood is missing from the tracking-parameter subset') ;
assert(structisfield(sPrmTrack, 'ROOT.MultiAnimal.Track.max_n_animals_user'), ...
       'Tracking parameter max_n_animals_user is missing from the tracking-parameter subset') ;
trackTreeFiltered = APTParameters.all2TrackParams(sPrmDefault, 'outputformat', 'tree') ;
assert(trackTreeFiltered.findnode('ROOT.Track.NFramesNeighborhood').Data.Visible, ...
       'Tracking parameter NFramesNeighborhood should be visible in the tracking-parameter tree') ;
assert(~trackTreeFiltered.findnode('ROOT.DeepTrack.GradientDescent.dl_steps').Data.Visible, ...
       'Training parameter dl_steps should not be visible in the tracking-parameter tree') ;
assert(~trackTreeFiltered.findnode('ROOT.DeepTrack.DataAugmentation.rrange').Data.Visible, ...
       'Training parameter rrange should not be visible in the tracking-parameter tree') ;

% The tracking-parameter tree should also load without error
trackTree = APTParameters.defaultTrackParamsTree() ;
assert(isscalar(trackTree) && isa(trackTree, 'TreeNode'), ...
       'defaultTrackParamsTree() did not return a scalar TreeNode') ;

fprintf('test_parameter_tree_loading passed.\n') ;

end  % function
