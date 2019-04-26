function mychmod(varargin)

if nargin < 1,
  d = '.';
else
  d = varargin{1};
end
  
if nargin < 2,
  args = {'g+rw'};
else
  args = varargin(2:end);
end

if ~exist(d,'file'),
  error('File/directory %s does not exist',d);
end

mychmod1(d,args);

function mychmod1(d,args)

try
  cmd = ['chmod ',sprintf('%s ',args{:}),' -R ',d];
  [stat,res] = unix(cmd);
  if stat == 0,
    fprintf('SUCCESS: %s\n',cmd);
    return;
  else
    %fprintf('FAIL: %s\n',res);
  end
catch 
end

if ~exist(d,'dir'),
  return;
end

cs = dir(d);
for i = 1:numel(cs),
  c = cs(i).name;
  if ismember(c,{'.','..'}),
    continue;
  end
  mychmod1(fullfile(d,c),args);
end
