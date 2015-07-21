% [path, fname, extension] = myfileparts(name)
% [path, fname] = myfileparts(name)
% Same as fileparts, except that:
% if name ends with a filesep, the fname will be the last part of the path
% and
% if extension output is not specified, then extension will not be removed
% from file name. 
function varargout = myfileparts(name)

% remove trailing \ or /
if nargin >= 1 && numel(name) > 2 && ...
    (name(end) == filesep || name(end) == '/'),
  name = name(1:end-1);
end

[path,fname,extension] = fileparts(name);
varargout{1} = path;
if nargout > 2,
  varargout{2} = fname;
  varargout{3} = extension;
else
  varargout{2} = [fname,extension];
end
