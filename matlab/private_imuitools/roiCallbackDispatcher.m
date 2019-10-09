function api = roiCallbackDispatcher(getPosition)      
%roiCallbackDispatcher creates an API for use in dispatching callback
%  functions.  API = roiCallbackDispatcher(getPosition) creates an API
%  which can be used to register and dispatch different types of
%  callbacks. getPosition is a function handle provided by the client that
%  provides the current position of an object.
%
%  The API returned by roiCallbackDispatcher contains the following methods:    
%    
%   addCallback
%
%       Adds the function handle FCN to the list of callbacks specified by
%       callback_type.
%
%           id = addCallback(fcn,callback_type)
%    
%       The return value, id, is used only with
%       removeCallback.    
%
%       The string callback_type determines when a callback function is
%       triggered.  Valid strings for callback_type are:
%    
%       'newPosition'   - Triggered when position is changed by setPosition,
%                         setConstrainedPosition or by a mouse drag.
%    
%       'translateDrag' - Triggered when rectangle is translated by a mouse drag.
%    
%       'resizeDrag'    - Triggered when rectangle is resized by a mouse drag.     
%    
%       Each callback function is called with the syntax: 
%
%           fcn(position)
%
%   removeCallback
%
%       Removes the corresponding function from the callback list.
%
%           removeCallback(id)
%
%       where id is the identifier returned by
%       api.addCallback.
%    
%   dispatchCallbacks
%    
%       Dispatches all callback_type callbacks
%
%           dispatchCallbacks(callback_type)
%
%       where callback_type is one of the valid callback_type strings used
%       in addCallback.
  
%   Copyright 2007-2011 The MathWorks, Inc.
%   $Revision: 1.1.6.6 $  $Date: 2011/07/19 23:57:45 $

new_position_callback_functions = makeList;
resize_drag_callback_functions = makeList;
translate_drag_callback_functions = makeList;

api.addCallback               = @addCallback;
api.removeCallback            = @removeCallback;
api.addNewPositionCallback    = @addNewPositionCallback;
api.removeNewPositionCallback = @removeNewPositionCallback;
api.dispatchCallbacks         = @dispatchCallbacks;

    %------------------------------------------
	function s = addCallback(fun,callback_type)
		callback_list = getCallbackList(callback_type);
		id = callback_list.appendItem(fun);
		s.removeCallback = @() callback_list.removeItem(id);
	end

    %-------------------------
	function removeCallback(s)
		s.removeCallback();
	end
    
    %----------------------------------------
	function id = addNewPositionCallback(fun)
		id = addCallback(fun,'newPosition');
	end

    %-------------------------------------
	function removeNewPositionCallback(id)
		removeCallback(id);
	end

    %----------------------------------------
	function dispatchCallbacks(callback_type) 

		callback_functions = getCallbackList(callback_type);

		list = callback_functions.getList();
		for i = 1:numel(list)
			fun = list(i).Item;
			fun(getPosition());
		end

	end

    %------------------------------------------------------
	function callback_list = getCallbackList(callback_type)

		switch(lower(callback_type))
			case 'resizedrag'
				callback_list = resize_drag_callback_functions;
			case 'translatedrag'
				callback_list = translate_drag_callback_functions;
			case 'newposition'
				callback_list = new_position_callback_functions;
			otherwise
				error(message('images:roiCallbackDispatcher:invalidCallbackType'))
		end

	end

end
