% [success,errmsg] = csvwrite(filename,struct,[prefix])
% [success,errmsg] = csvwrite(fid,struct,[prefix])
function [success,errmsg] = csvwrite(f,s,prefix)

errmsg = '';
success = false;

isfilename = ischar(f);
if nargin < 2,
  errmsg = 'Usage: csvwrite(filename,struct,[prefix]) or csvwrite(fid,struct,[prefix])';
  return;
end
if nargin < 3,
  prefix = '';
end

if isfilename,
  filename = f;
  fid = fopen(filename,'w');
  if fid < 0,
    errmsg = sprintf('Could not open file %s for writing',filename);
    return;
  end
else
  fid = f;
end

if isstruct(s),
  fns = fieldnames(s);
  for i = 1:length(fns),
    if isempty(prefix),
      prefix1 = fns{i};
    else
      prefix1 = [prefix,'_',fns{i}];
    end
    [success1,errmsg] = csvwrite(fid,s.(fns{i}),prefix1);
    if ~success1,
      return;
    end
  end
elseif iscell(s),
  for i = 1:numel(s),
    if isempty(prefix),
      prefix1 = sprintf('cell%d',i);
    else
      prefix1 = sprintf('%s_%d',prefix,i);
    end
    [success1,errmsg] = csvwrite(fid,s{i},prefix1);
    if ~success1,
      return;
    end
  end
else
  if isempty(prefix),
    prefix1 = 'var';
  else
    prefix1 = prefix;
  end
  fprintf(fid,prefix1);
  fprintf(fid,',%f',s);
  fprintf(fid,'\n');
end

if isfilename,
  fclose(fid);
end
success = true;
