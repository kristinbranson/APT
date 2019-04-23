function [y,yidx] = weighted_prctile(x,p,w,dim,issorted)
% y = weighted_prctile(x,p,w,dim,issorted)
% similar to MATLAB's prctile, but takes weights

if ~isvector(p) || numel(p) == 0
    error('stats:prctile:BadPercents', ...
          'P must be a scalar or a non-empty vector.');
elseif any(p < 0 | p > 100) || ~isreal(p)
    error('stats:prctile:BadPercents', ...
          'P must take real values between 0 and 100');
end

% make sure sizes match
sz = size(x);
szw = size(w);
sz0 = sz;
if numel(x) ~= numel(w),
  error('Number of elements of x and w must match');
end
if length(sz) ~= length(szw) || ~all(sz == szw),
  if ~(isvector(x) && isvector(w)),
    warning('Number of elements of x and w match, but not sizes. Resizing w to size %s (was size %s)',mat2str(sz),mat2str(szw));
  end
end

% Figure out which dimension prctile will work along.
if nargin < 5,
  issorted = false;
end
if nargin < 4 
    dim = find(sz ~= 1,1);
    if isempty(dim)
        dim = 1; 
    end
    dimArgGiven = false;
else
    % Permute the array so that the requested dimension is the first dim.
    nDimsX = ndims(x);
    perm = [dim:max(nDimsX,dim) 1:dim-1];
    x = permute(x,perm);
    w = permute(w,perm);
    % Pad with ones if dim > ndims.
    if dim > nDimsX
        sz = [sz ones(1,dim-nDimsX)];
    end
    sz = sz(perm);
    dim0 = dim;
    dim = 1;
    dimArgGiven = true;
end

% If X is empty, return all NaNs.
if isempty(x)
    if isequal(x,[]) && ~dimArgGiven
        y = nan(size(p),class(x));
    else
        szout = sz; szout(dim) = numel(p);
        y = nan(szout,class(x));
    end

else
    % Drop X's leading singleton dims, and combine its trailing dims.  This
    % leaves a matrix, and we can work along columns.
    nrows = sz(dim);
    ncols = prod(sz) / nrows;
    x = reshape(x, nrows, ncols);
    w = reshape(w, nrows, ncols);
    y = zeros(numel(p), ncols, class(x));
    yidx = nan(size(y));
    
    if ~issorted,
      % sort x
      [x,order] = sort(x,1);
      % apply ordering to w
      w = w(bsxfun(@plus,order,0:nrows:(ncols-1)*nrows));
      %w = w(sub2ind([nrows,ncols],order,repmat((1:ncols),[nrows,1])));
    end
    nans = isnan(x);
    % set weight to 0 for nans
    w(nans) = 0;
    
    % normalize w to sum to 1
    %w = w ./ repmat(sum(w,1),[nrows,1]);
    w = bsxfun(@rdivide,w,sum(w,1));
    
    % compute cumulative sum
    cw = cumsum(w,1);
    % for numerical errors
    cw = cw / cw(end);

    % start of each column
    off = (0:ncols-1)'*nrows;
    
    for i = 1:numel(p),
      
      if p(i) == 0,
        isbigger = [false(1,ncols);cw > 0];
        ind = find(isbigger(1:end-1,:)==false & isbigger(2:end,:)==true);
        %ind = find(cw > 0,1);
      elseif p(i) == 100,
        ind = off + nrows;
        %ind = find(cw > 0,1,'last');
      else      
        % find the frame where cumsum of weight is at least p
        isbigger = [false(1,ncols);cw >= p(i)/100];
        ind = find(isbigger(1:end-1,:)==false & isbigger(2:end,:)==true);
      end
      % check for border
      indrel = ind - off;
      isexact = cw(ind) == p(i)/100 & indrel < nrows;
      indnext = indrel(isexact)+1 + off(isexact);
      y(i,isexact) = (x(ind(isexact)) + x(indnext))/2;
      y(i,~isexact) = x(ind(~isexact));
      yidx(i,:) = order(ind);

    end

end

% undo the DIM permutation
if dimArgGiven
  y = reshape(y,[numel(p),sz(2:end)]);
  y = ipermute(y,perm);
  yidx = reshape(yidx,[numel(p),sz(2:end)]);
  yidx = ipermute(yidx,perm);
end

% If X is a vector, the shape of Y should follow that of P, unless an
% explicit DIM arg was given.
if ~dimArgGiven && isvector(x)
    y = reshape(y,size(p)); 
end
