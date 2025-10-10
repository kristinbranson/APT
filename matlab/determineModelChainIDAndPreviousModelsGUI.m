function [modelChainID, prev_models] = determineModelChainIDAndPreviousModelsGUI(modelChainID0, prev_models0, dlTrnType, augOnly, skip_dlgs)
% Determines the modelChainID for the about-to-be-trained model, and the
% previous models to use as a starting point, if any.

% Check input types
assert(isOldSchoolString(modelChainID0)) ;
assert(iscell(prev_models0)) ;
assert(isrow(prev_models0)) ;
assert(all(cellfun(@isOldSchoolString, prev_models0))) ;
assert(isa(dlTrnType, 'DLTrainType') && isscalar(dlTrnType)) ;
assert(islogical(augOnly) && isscalar(augOnly)) ;
assert(islogical(skip_dlgs) && isscalar(skip_dlgs)) ;

% Do the work
prev_models = cell(1,0) ;
switch dlTrnType
  case DLTrainType.New
    modelChainID = datestr(now(),'yyyymmddTHHMMSS');
    if ~isempty(modelChainID0) && ~augOnly
      assert(~strcmp(modelChainID,modelChainID0));
      fprintf('Training new model %s.\n',modelChainID);
      defaultans = 'Yes';
      if ~skip_dlgs,
        res = questdlg(['Previously trained models exist for current tracking algorithm. ' ...
                        'Do you want to use the previous model for initialization?'], ...
                       'Training Initialization', ...
                       'Yes','No','Cancel', ...
                       defaultans);
      else
        res = defaultans;
      end
      if strcmp(res,'No')
        prev_models = cell(1,0) ;
      elseif strcmp(res,'Yes')
        prev_models = prev_models0;
      else
        return
      end
    else
      % do nothing
    end
  case DLTrainType.Restart
    % Pretty sure this path is not used in APT right now.  -- ALT, 2025-10-08
    if isempty(modelChainID0)
      error('Model has not been trained.');
    end
    modelChainID = modelChainID0;
    fprintf('Restarting train on model %s.\n',modelChainID);
  otherwise
    assert(false, 'Internal error');
end

% Check output types
assert(isOldSchoolString(modelChainID)) ;
assert(iscell(prev_models)) ;
assert(isrow(prev_models)) ;
assert(all(cellfun(@isOldSchoolString, prev_models))) ;

end
