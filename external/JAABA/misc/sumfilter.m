function y = sumfilter(x,r,varargin)

y = meanfilter(x,r,varargin{:})*r;
