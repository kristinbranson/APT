function [res,info] = mydir(dirname,varargin)

iscond = nargin > 1;
if iscond,
  [name,daterange,isdir,recursive,maxdepth] = myparse(varargin,'name','','daterange',[],'isdir',[],'recursive',false,'maxdepth',inf);
end

[p,f] = myfileparts(dirname);
if ~any(f == '*'),
  p = fullfile(p,f);
end
s = dir(dirname);
res = {};
info = [];
for i = 1:numel(s),
  if ismember(s(i).name,{'.','..'}),
    continue;
  end
  if iscond,
    if ~isempty(name),
      
      m = regexp(s(i).name,name,'once');
      if isempty(m),
        continue;
      end
      
    end
    
    if ~isempty(daterange),
      
      if s(i).datenum < daterange(1),
        continue;
      end
      if numel(daterange)>=2 && s(i).datenum > daterange(2),
        continue;
      end
      
    end
    
    if ~isempty(isdir),
      if s(i).isdir ~= isdir,
        continue;
      end
    end
    
  end
  if exist(p,'dir'),
    res{end+1} = fullfile(p,s(i).name); %#ok<AGROW>
  else
    res{end+1} = p; %#ok<AGROW>
  end
  infocurr = s(i);
  infocurr.path = p;
  info = structappend(info,infocurr);
end

if iscond && recursive && maxdepth > 0,
  i = find(strcmp(varargin,'maxdepth'));
  if ~isempty(i),
    varargin{i+1} = maxdepth-1;
  end
  for i = 1:numel(s),
    if ismember(s(i).name,{'.','..'}),
      continue;
    end
    if s(i).isdir,
      [rescurr,infocurr] = mydir(fullfile(p,s(i).name),varargin{:});
      if ~isempty(rescurr),
        res = [res,rescurr]; %#ok<AGROW>
        info = structappend(info,infocurr);
      end
    end
  end
end