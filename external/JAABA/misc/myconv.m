% C = myconv(A,B,...)
% One-dimensional correlation/convolution of vector a with filter B
% Options:
% Type of filtering:
% 'corr': perform correlation [default]
% 'conv': perform convolution
% Output size:
% 'same': output vector C is the central part of the correlation that is
% the same size as A [default]
% 'full': output vector C is the length(A) + length(B) - 1 full correlation
% 'valid': output vector C is the length(A) - length(B) + 1 part of the
% correlation computed without the padded edges
% Padding:
% X: pad edges with number X. (default: X = 0).
% 'symmetric': pad edges by mirror-reflecting across the border, e.g.
% [..., A(3), A(2), A(1), A(1), A(2), A(3), ... 
%       A(end-2), A(end-1), A(end), A(end), A(end-1), A(end-2), ...]
% 'replicate': pad edges with the edge values of the array, e.g.
% [..., A(1), A(1), A(1), A(1), A(2), A(3), ... 
%       A(end-2), A(end-1), A(end), A(end), A(end), A(end), ...]
% 'circular': pad edges so that array is periodic, e.g.
% [..., A(end-2), A(end-1), A(end), A(1), A(2), A(3), ... 
%       A(end-2), A(end-1), A(end), A(1), A(2), A(3), ...]
%
function c = myconv(a,b,varargin)

% defaults
outputsize = 'same';
pad = 0;
corr = 'corr';

% parse optional inputs
padopts = {'symmetric','replicate','circular'};
outputsizeopts = {'same','valid','full'};
corropts = {'corr','conv'};
allopts = [padopts,outputsizeopts,corropts];

for i = 1:length(varargin),
  if isnumeric(varargin{i});
    pad = varargin{i};
  elseif ischar(varargin{i}),
    j = strmatch(varargin{i},allopts);
    if isempty(j),
      warning('Unknown option %s passed to myconv',varargin{i});
      continue;
    end
    opt = allopts{j};
    if ismember(opt,padopts),
      pad = opt;
    elseif ismember(opt,outputsizeopts),
      outputsize = opt;
    else
      corr = opt;
    end
  end
end

sza = size(a);
szb = size(b);
if nnz(sza > 1) > 1 || nnz(szb > 1) > 1,
  warning('One-dimensional convolution; inputs are converted to 1-D arrays first');
end
a = a(:)';
b = b(:)';
na = length(a);
nb = length(b);

% corr vs conv
if strcmp(corr,'conv'),
  b = fliplr(b);
end

% don't need to worry about padding
if strcmp(outputsize,'valid')
  c = filter2(b,a,'valid');
  return;
end

% pad a as described by pad
apad = [zeros(1,nb-1),a,zeros(1,nb-1)];
if isnumeric(pad),
  apad(1:nb-1) = pad;
  apad(na+nb:end) = pad;
elseif strcmp(pad,'symmetric'),
  apad(1:nb-1) = a(nb-1:-1:1);
  apad(na+nb:end) = a(end:-1:end-nb+2);
elseif strcmp(pad,'circular'),
  apad(1:nb-1) = a(end-nb+2:end);
  apad(na+nb:end) = a(1:nb-1);
elseif strcmp(pad,'replicate'),
  apad(1:nb-1) = a(1);
  apad(na+nb:end) = a(end);
end
  
% convolve
cpad = filter2(b,apad,'valid');

% extract the requested region
if strcmp(outputsize,'full'),
  c = cpad;
else % same
  off = floor(nb/2);
  c = cpad(off+1:off+na);
end

if sza(1) > sza(2),
  c = c';
end
