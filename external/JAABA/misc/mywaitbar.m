% slightly more robust version of matlab's waitbar
function varargout = mywaitbar(varargin)

if nargin >= 2 && ischar(varargin{end-1}) && ...
    strcmpi(varargin{end-1},'interpreter') && ...
    ischar(varargin{end}),
  interpreter = varargin{end};
  leftovers = varargin(1:end-2);
else
  interpreter = 0;
  leftovers = varargin;
end

if numel(leftovers) >= 2 && isnumeric(leftovers{2}),
  if ~ishandle(leftovers{2}),
    leftovers(2) = [];
  end
end

if ischar(interpreter),
  if numel(leftovers) >= 2 && isnumeric(leftovers{2}) && ~isempty(leftovers{2}),
    htitle = get(get(leftovers{2},'Children'),'Title');
    set(htitle,'Interpreter',interpreter);
  elseif ischar(leftovers{end}) && strcmpi(interpreter,'none'),
    ti0 = leftovers{end};
    ti = ti0;
    ti = regexprep(ti,'\\','\\\\');
    ti = regexprep(ti,'_','\_');
    leftovers{end} = ti;
    hwait = waitbar(leftovers{:});
    htitle = get(get(hwait,'Children'),'title');
    set(htitle,'Interpreter',interpreter,'String',ti0);
    drawnow;
    if nargout >= 1,
      varargout = {hwait};
      return;
    end
  end
end
varargout = cell(1,nargout);
[varargout{:}] = waitbar(leftovers{:});
