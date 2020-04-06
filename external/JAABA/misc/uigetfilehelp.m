function varargout = uigetfilehelp(varargin)

tmp = find(cellfun(@ischar,varargin));
i = tmp(strmatch('helpmsg',varargin(tmp)));
hhelp = nan;
inputs = varargin;
if ~isempty(i),
  inputs(i) = [];
  if i < length(varargin),
    inputs(i) = [];
    hhelp = creategetfileinfodialog(varargin{i+1});
  end
end

varargout = cell(1,nargout);
[varargout{:}] = uigetfile(inputs{:});

if ~isnan(hhelp),
  deletefileinfodialog(hhelp);
end