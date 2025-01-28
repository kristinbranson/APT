function sendMaybe(objMaybe, methodName, varargin)
  % Helper function to send a message (i.e. call a method) if objMaybe is
  % nonempty.  If objMaybe is empty, does nothing.
  if ~isempty(objMaybe) ,
    objMaybe.(methodName)(varargin{:}) ;
  end
end
