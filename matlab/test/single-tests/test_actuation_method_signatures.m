function test_actuation_method_signatures()
  % Every LabelerController actuation method must accept (obj, source, event).
  %
  % controlActuatedCore_() calls obj.(methodName)(source, event, varargin{:})
  % for all of them, so a method declared with obj alone errors with "Too many
  % input arguments" the moment its control is clicked.
  %
  % Regression test: menu_track_backend_config_moreinfo_actuated_ was declared
  % with obj alone, so Track > Backend Configuration > More information...
  % errored.

  metaClass = ?LabelerController ;
  methodList = metaClass.MethodList ;
  offenderNames = {} ;
  for i = 1 : numel(methodList)
    method = methodList(i) ;
    if ~endsWith(method.Name, '_actuated_')
      continue
    end
    inputNames = method.InputNames ;
    doesAcceptSourceAndEvent = numel(inputNames) >= 3 || any(strcmp(inputNames, 'varargin')) ;
    if ~doesAcceptSourceAndEvent
      offenderNames{end+1} = method.Name ;  %#ok<AGROW>
    end
  end
  if ~isempty(offenderNames)
    error('test_actuation_method_signatures:badSignature', ...
          'Actuation method(s) not accepting (source, event): %s', ...
          strjoin(offenderNames, ', ')) ;
  end
end  % function
