function sendMaybe(objMaybe, methodName, varargin)
  % Helper function to send a message (i.e. call a method) if objMaybe is
  % nonempty and valid.  Otherwise does nothing.
  if ~isempty(objMaybe) && isvalid(objMaybe) ,
    objMaybe.(methodName)(varargin{:}) ;
  end
end
