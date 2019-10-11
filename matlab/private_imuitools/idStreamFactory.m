function id_stream = idStreamFactory(name)
%idStreamFactory Produce and cache idStream instances.
%   S = idStreamFactory(NAME) returns an idStream instance associated with NAME.
%   If such a stream with NAME does not yet exist, idStreamFactory creates it.
%   If it already exists, it returns the existing one.
%
%   idStreamFactory can be useful if you want to reuse the same idStream for 
%   managing unique ids for multiple instances of a tool. 
%
%   See also idStream.

%   Copyright 2005 The MathWorks, Inc.
%   $Revision $  $Date: 2005/05/27 14:07:21 $

id_stream_name = sprintf('%sIdStream',name);

% Get existing idStream or create it if it hasn't yet been created.
if isappdata(0,id_stream_name)
  id_stream = getappdata(0,id_stream_name);
else
  id_stream = idStream;
  setappdata(0,id_stream_name,id_stream);
end
