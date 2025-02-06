function result = trimTrackerHistoryAfterLoad(trackerHistory)
% Trim the tracker history, deleting all untrained models, except that we
% always spare the current tracker (the first one).  We do this after loading
% b/c there are legacy projects which include a bunch of untrained trackers.

if isempty(trackerHistory) ,
  result = trackerHistory ;
  return
end
firstTracker = trackerHistory(1) ;  % singleton cell array
restOfTrackers = trackerHistory(2:end) ;
isTrained = cellfun(@(t)(t.hasBeenTrained()), restOfTrackers) ;
newRestOfTrackers = restOfTrackers(isTrained) ;
result = horzcat(firstTracker, newRestOfTrackers) ;


