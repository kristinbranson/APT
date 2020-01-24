function warningNoTrace(varargin)
warnst = warning('off','backtrace');
warning(varargin{:});
warning(warnst);