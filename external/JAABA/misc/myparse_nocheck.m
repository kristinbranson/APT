function varargout = myparse_nocheck(params,varargin)

nout = nargout;

% set defaults
ncopy = min(nout,floor(length(varargin)/2));
doreturnleftovers = ncopy < nout;

% parameter names
names = varargin(1:2:end);
varargout = varargin(2:2:2*ncopy);

% go through parameters
i = 1;
leftovers = {};

while true,

  if i > length(params),
    break;
  end
  
  % check if it matches names{j}
  if ~ischar(params{i}),
    leftovers{end+1} = params{i};
    i = i + 1;
    continue;
  end

  j = find(strcmpi(params{i},names));
  if isempty(j),
    leftovers = [leftovers,params(i:i+1)];
  else
    for k = 1:length(j),
      varargout{j(k)} = params{i+1};
    end
  end
  i = i + 2;
  
end;

if doreturnleftovers,
  varargout{ncopy+1} = leftovers;
end