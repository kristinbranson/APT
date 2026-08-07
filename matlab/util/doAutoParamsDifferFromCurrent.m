function [toDiffer, autoparams, vizdata] = doAutoParamsDifferFromCurrent(labeler)
  % Determine whether the automatically-computed training parameters differ
  % from the ones currently set in the labeler by more than 10%.
  %
  % toDiffer is a logical scalar, true iff at least one auto-computed
  % parameter differs from its current value by more than 10% (or has no
  % current value to compare against).  A flipped logical yields a large
  % relative difference, so the head-tail alignment and flip recommendations
  % are covered by the same test.  autoparams and vizdata are returned as
  % well (as computed by apt.compute_auto_params), so callers that also need
  % the suggestions do not have to recompute them.

  sPrmCurrent = labeler.trackGetTrainingParams() ;
  tPrm = APTParameters.defaultParamsTree() ;
  tPrm.structapply(sPrmCurrent) ;

  [autoparams, vizdata] = apt.compute_auto_params(labeler) ;

  paramPaths = autoparams.keys() ;
  toDiffer = false ;
  for i = 1 : numel(paramPaths)
    paramPath = paramPaths{i} ;
    currentValue = tPrm.findnode(paramPath).Data.Value ;
    autoValue = autoparams(paramPath) ;
    relativeDifference = (autoValue - currentValue) / (currentValue + 0.001) ;
    if numel(relativeDifference) > 1
      relativeDifference = max(relativeDifference) ;
    end
    if isempty(relativeDifference) || abs(relativeDifference) > 0.1  % first clause if e.g. currentValue empty
      toDiffer = true ;
    end
  end
end  % function
