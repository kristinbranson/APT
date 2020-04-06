function featureNamesNew=featureNamesNewFromOld(featureNamesOld, ...
                                                scoreFeaturesOld, ...
                                                scoreFeaturesNew)
% Figures out the feature names in the new feature lexicon, given the old
% feature names, the old scoreFeatures, and the new
% scoreFeatures.

% figure out which of the old were retained, which of the new are not
% present in the old
[kept,added]= ...
  setDifferencesScoreFeatures(scoreFeaturesOld,scoreFeaturesNew);

% Get a list of the deleted feature names 
scoreFeatureNamesOld={scoreFeaturesOld(:).scorefilename};
scoreFeatureNamesDeleted=scoreFeatureNamesOld(~kept);

% Make a function that computes whether a feature name was kept
featureNameKept=@(featureName)(~ismember(featureName,scoreFeatureNamesDeleted));

% Get the feature names common to both old a new by filtering the old
% list
featureNamesCommon= ...
  cellFilter(featureNameKept,featureNamesOld);

% Get the score feature names that are truly novel
scoreBaseNameListNew={scoreFeaturesNew(:).scorefilename};
featureNamesAdded=scoreBaseNameListNew(added);

% Merge the common and added lists
featureNamesNew = ...
  [featureNamesCommon;featureNamesAdded];

end
