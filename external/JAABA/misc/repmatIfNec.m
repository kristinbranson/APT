function v = repmatIfNec(v,n,name,varargin)
% v = repmatIfNec(v,n,name,varargin)
%
% v: Some value that is either i) a scalar value or ii) a vector of length n
% n: Number of repeats desired/expected
% name: string used in warning
% optional pvs:
%   - cell. If true, repmat into a cell vector
%

docell = myparse(varargin,'cell',false);

if ~docell && ~isscalar(v) || docell && iscell(v)
  assert(isvector(v)&&numel(v)==n,...
      'Expected value to be a vector of length %d.',n);
else
  if docell % v is not a cell
    v = repmat({v},1,n);
  else % v is a scalar
    v = repmat(v,1,n);
  end
  warningNoTrace('Field/property %s scalar expanded.',name);
end
