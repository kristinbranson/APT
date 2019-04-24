function smat = sprintfmat(s,varargin)

% check argument sizes
if nargin == 1,
  smat = s;
  return;
end;
if size(varargin{1},1) == 1 & size(varargin{1},2) > 1,
  varargin{1} = varargin{1}';
end;
n = size(varargin{1},1);
for i = 2:nargin-1,
  if size(varargin{i},1) ~= n & size(varargin{i},2) == n,
    varargin{i} = varargin{i}';
  end;
end

% check if everything else is numeric
isallnumeric = isnumeric(varargin{1});
for i = 2:nargin-1,
  if size(varargin{i},1) ~= size(varargin{1},1) | ~isnumeric(varargin{i}),
    isallnumeric = 0;
  end;
end;
if nargin > 2 & isallnumeric,
  m = size(varargin{1},1);
  args = cell2mat(varargin);
  args = reshape(args,[m,nargin-1]);
  args = args';
  svec = sprintf(s,args);
else,  
  svec = sprintf(s,varargin{:});
end;
smat = reshape(svec,[length(svec)/n,n])';