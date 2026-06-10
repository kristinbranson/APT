function fakePushbuttonClick(buttonHandle)
% Simulate a click on a pushbutton uicontrol by calling its callback.
  callback = get(buttonHandle, 'Callback') ;
  if isa(callback, 'function_handle')
    callback(buttonHandle, []) ;
  elseif iscell(callback)
    callback{1}(buttonHandle, [], callback{2:end}) ;
  end
end  % function
