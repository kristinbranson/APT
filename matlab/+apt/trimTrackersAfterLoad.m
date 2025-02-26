function result = trimTrackersAfterLoad(trackers, isFilePreTrackerHistory, currTrackerThatWas)
  % Trim the trackers, deleting all untrained models, except for the
  % current tracker.  We do this after loading b/c there are legacy projects
  % which include a bunch of untrained trackers.  For newer project files, there
  % shouldn't be any untrained trackers (except for possibly the current one),
  % but we do it just to be sure.

  if isempty(trackers) ,
    result = trackers ;
    return
  end
  if isFilePreTrackerHistory ,
    % For old files, the tracker at index currTrackerThatWas is the current one,
    % and we retain it whether it is trained or not.
    firstTracker = trackers(currTrackerThatWas) ;  % singleton cell array
    restOfTrackers = delete_elements(trackers, currTrackerThatWas) ;    
  else
    % For new files, the tracker at index 1 is the current one, and we retain it
    % whether it is trained or not.
    firstTracker = trackers(1) ;  % singleton cell array
    restOfTrackers = trackers(2:end) ;
  end
  isTrained = cellfun(@(t)(t.hasBeenTrained()), restOfTrackers) ;
  newRestOfTrackers = restOfTrackers(isTrained) ;
  result = horzcat(firstTracker, newRestOfTrackers) ;
end
