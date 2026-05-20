function fakeAxesClick(hAx, x, y)
% Simulate a left-click at (x,y) on an axes by calling its ButtonDownFcn.
  callback = get(hAx, 'ButtonDownFcn') ;
  evt = struct('Button', 1, ...
               'IntersectionPoint', [x y 0]) ;
  if isa(callback, 'function_handle')
    callback(hAx, evt) ;
  elseif iscell(callback)
    callback{1}(hAx, evt, callback{2:end}) ;
  end
end  % function
