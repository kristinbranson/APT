function result = setprop(obj, varargin)
% Set properties of a classdef object by name, like setfield() for structs.
% Usage: obj = setprop(obj, prop1, val1, prop2, val2, ...)
assert(~isa(obj, 'handle'), 'setprop:handleClass', 'setprop is for value classes only, not handle classes');
assert(mod(numel(varargin), 2) == 0, 'setprop:badArgs', 'Arguments must be property-value pairs');
result = obj ;
for i = 1:2:numel(varargin)
  result.(varargin{i}) = varargin{i+1};
end
end  % function
