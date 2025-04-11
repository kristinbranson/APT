function result = algorithmNamePrettyFromTCI(tci, labeler)
  % Get the algorithm name from a TCI (tracker constructor info) cell array
  tracker = LabelTracker.create(labeler,tci) ;  % Need a labeler to create the tracker
  result = tracker.algorithmNamePretty ;
end
