function varargout = mytitle(varargin)

h = title(varargin{:});
set(h,'fontname','times','fontsize',14);
if nargout >= 1,
  varargout{1} = h;
end