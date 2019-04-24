function X = randpsd(n,varargin)

[maxv] = myparse(varargin,'maxv',1);

x = rand(n);
X = x'*x;
X = X * maxv/n; 