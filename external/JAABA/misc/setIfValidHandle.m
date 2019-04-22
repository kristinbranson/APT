function setIfValidHandle(h,varargin)
% setIfValidHandle(h,varargin)

hset = h(ishandle(h));
set(hset,varargin{:});