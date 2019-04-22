function [counts,bin,centers] = histn(x,varargin)

% parse inputs
if length(varargin) < 1,
  error('usage: [counts,bin] = histn(x,centers), [counts,bin] = histn(x,''edges'',edges)');
end

if ischar(varargin{1}),
  if strcmpi(varargin{1},'edges') && length(varargin) >= 2,
    edges = varargin{2};
    nd = length(edges);
    iscenter = false;
  else
    error('usage: [counts,bin] = histn(x,centers), [counts,bin] = histn(x,''edges'',edges)');
  end
else
  centers = varargin{1};
  nd = length(centers);
  iscenter = true;
end

% get input sizes
if nd ~= size(x,2) && nd == size(x,1),
    x = x';
elseif nd ~= size(x,2),
  if iscenter
    error('rows of x must match length length of centers');
  else
    error('rows of x must match length length of edges');
  end
end

% construct edges from centers
if iscenter,
  edges = cell(1,nd);
  for d = 1:nd,
    edges{d} = [-inf,(centers{d}(1:end-1)+centers{d}(2:end))/2,inf];
  end
end

% histogram each dimension independently
sub = cell(1,nd);
for d = 1:nd,
  [tmp,sub{d}] = histc(x(:,d),edges{d});
end

sz = zeros(1,nd);
for d = 1:nd,
  sz(d) = length(edges{d});
end

% if center was input, then last entries will all be 0
if iscenter,
  sz = sz - 1;
end

% convert from subscript to index description of bin membership
bin = sub2ind(sz,sub{:});

% histogram together
countsvec = hist(bin,1:prod(sz));

% reshape
counts = reshape(countsvec,sz);

if ~exist('centers','var'),
  centers = cell(1,nd);
  for d = 1:nd,
    centers{d} = [(edges{d}(1:end-1) + edges{d}(2:end))/2,edges{d}(end)];
  end
end