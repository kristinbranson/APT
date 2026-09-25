function test_parse_param_viz_spec_rejects_extra_hash()
  % A ParameterVisualization spec with more than one '#' must be rejected with
  % the "Invalid ParameterVisualization specification" error, quoting the spec.
  %
  % Regression test: the error path of ParameterVisualization.parseParamVizSpec()
  % formatted its message with pgp.ParamViz, a variable that does not exist in
  % that scope, so the caller saw "Unrecognized function or variable 'pgp'"
  % instead of the intended message.

  % The well-formed shapes still parse.
  [className, vizId] = ParameterVisualization.parseParamVizSpec('ParameterVisualizationFoo') ;
  assert(strcmp(className, 'ParameterVisualizationFoo') && isempty(vizId), ...
         'One-token spec did not parse') ;
  [className, vizId] = ParameterVisualization.parseParamVizSpec('ParameterVisualizationFoo#bar') ;
  assert(strcmp(className, 'ParameterVisualizationFoo') && strcmp(vizId, 'bar'), ...
         'Two-token spec did not parse') ;

  badSpec = 'ParameterVisualizationFoo#bar#baz' ;
  didThrow = false ;
  try
    ParameterVisualization.parseParamVizSpec(badSpec) ;
  catch me
    didThrow = true ;
    assert(contains(me.message, 'Invalid ParameterVisualization specification') && ...
           contains(me.message, badSpec), ...
           'Wrong error for a three-token spec: %s', me.message) ;
  end
  assert(didThrow, 'A three-token spec was accepted') ;
end  % function
