function y = stdignorenan(x,varargin)

y = std(x(~isnan(x)),varargin{:});