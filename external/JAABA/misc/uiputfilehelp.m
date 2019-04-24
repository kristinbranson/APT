function varargout = uiputfilehelp(varargin)

tmp = find(cellfun(@ischar,varargin));
i = tmp(strmatch('helpmsg',varargin(tmp)));
hhelp = nan;
inputs = varargin;
if ~isempty(i),
  inputs(i) = [];
  if i < length(varargin),
    inputs(i) = [];
    hhelp = createputfileinfodialog(varargin{i+1});
  end
end

varargout = cell(1,nargout);
[varargout{:}] = uiputfile(inputs{:});

if ~isnan(hhelp),
  deletefileinfodialog(hhelp);
end