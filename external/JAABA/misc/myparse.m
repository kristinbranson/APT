function varargout = myparse(params,varargin)

nin = length(varargin)/2;
nout = nargout;
if mod(2*nin,2)~=0 || mod(length(params),2)~=0,
  error('Odd number of parameters expected');
end;

% set defaults
varargout = varargin(2:2:2*nout);

% parameter names
names = varargin(1:2:end);

% go through parameters
for i = 1:2:length(params),
  
  % check if it matches names{j}
  if ~ischar(params{i}),
    error('Input %d is not a string.',i);
  end

  %BJA:  6x faster than that below
  idx=find(strcmpi(params{i},names));
  if(isempty(idx))
    warning('myparse:unknownParameterName','Unknown parameter name: %s, ignoring\n',params{i});
  else
    varargout{idx} = params{i+1};
  end

%  if ~any(strcmpi(params{i},names)),
%    warning('myparse:unknownParameterName','Unknown parameter name: %s, ignoring\n',params{i});
%  end
%  
%  for j = 1:length(names),
%
%    % set value
%    if strcmpi(params{i},names{j}),
%      varargout{j} = params{i+1};
%    end;
%
%  end;
  
end;
