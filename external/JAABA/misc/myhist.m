function [no,xo,idx] = myhist(varargin)
% almost the same as hist, except it can take weights

% Parse possible Axes input
i = find(strcmpi(varargin,'weights'));
if ~isempty(i),
  weights = varargin{i(end)+1};
  varargin([i,i+1]) = [];
end
error(nargchk(1,inf,nargin,'struct'));
[cax,args,nargs] = axescheck(varargin{:});

y = args{1};

if nargs == 1
    x = 10;
else
    x = args{2};
end

if min(size(y))==1, y = y(:); end
if ~ishistnumeric(x) || ~ishistnumeric(y)
    error('MATLAB:hist:InvalidInput', 'Input arguments must be numeric.')
end

if isempty(y),
    if length(x) == 1,
       x = 1:double(x);
    end
    nn = zeros(size(x)); % No elements to count
    %  Set miny, maxy for call to bar below.
    miny = [];
    maxy = [];
else
    %  Ignore NaN when computing miny and maxy.
    ind = ~isnan(y);
    miny = min(y(ind));
    maxy = max(y(ind));
    %  miny, maxy are empty only if all entries in y are NaNs.  In this case,
    %  max and min would return NaN, thus we set miny and maxy accordingly.
    if (isempty(miny))
      miny = NaN;
      maxy = NaN;
    end
    if length(x) == 1
    	  if miny == maxy,
    		  miny = miny - floor(x/2) - 0.5; 
    		  maxy = maxy + ceil(x/2) - 0.5;
     	  end
        binwidth = (maxy - miny) ./ x;
        xx = miny + binwidth*(0:x);
        xx(length(xx)) = maxy;
        x = xx(1:length(xx)-1) + binwidth/2;
    else
        xx = x(:)';
        binwidth = [diff(xx) 0];
        xx = [xx(1)-binwidth(1)/2 xx+binwidth/2];
        xx(1) = min(xx(1),miny);
        xx(end) = max(xx(end),maxy);
    end
    % Shift bins so the interval is ( ] instead of [ ).
    xx = full(real(xx)); y = full(real(y)); % For compatibility
    bins = xx + eps(xx);
    [nn,idx] = histc(y,[-inf bins],1);
    if exist('weights','var'),
      for i = 1:length(nn),
        nn(i) = sum(weights(idx==i));
      end
    end
    
    % Combine first bin with 2nd bin and last bin with next to last bin
    nn(2,:) = nn(2,:)+nn(1,:);
    nn(end-1,:) = nn(end-1,:)+nn(end,:);
    nn = nn(2:end-1,:);
    idx(idx==1) = 2;
    idx = idx - 1;
end

if nargout == 0
  if ~isempty(cax)
    bar(cax,x,nn,[miny maxy],'hist');
  else
    bar(x,nn,[miny maxy],'hist');
  end
else
  if min(size(y))==1, % Return row vectors if possible.
    no = nn';
    xo = x;
  else
    no = nn;
    xo = x';
  end
end

function a = ishistnumeric(b)
% for backward compatibility, logical is allowed in hist.m
a = isnumeric(b) || islogical(b);
