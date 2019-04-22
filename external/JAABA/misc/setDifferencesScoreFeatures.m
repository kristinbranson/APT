function [kept,added]= ...
  setDifferencesScoreFeatures(scoreFeaturesOld,scoreFeaturesNew)
% Given an old and new scoreFeatures, computes a boolean array that tells
% which elements of the old one are retained in the new one (kept), and a
% boolean array indicating which elements of the new one are not present in
% the old one (added).

% Get the list of (absolute) files names for each scoreFeatures array
fileNameListOld={scoreFeaturesOld(:).classifierfile};
fileNameListNew={scoreFeaturesNew(:).classifierfile};

% define a local function that checks whether a file name is in the new file
% name list
inFileNameListNew=@(fileName)(ismember(fileName,fileNameListNew));

% mark the ones in the old list that are not in the new list (these are
% ones that will be deleted)
kept=cellfun(inFileNameListNew,fileNameListOld);

% define a local function that checks whether a file name is in the old file
% name list
notInFileNameListOld=@(fileName)(~ismember(fileName,fileNameListOld));
            
% mark the ones in the new list that were not in the old list (added
% ones)
added=cellfun(notInFileNameListOld,fileNameListNew);

end
