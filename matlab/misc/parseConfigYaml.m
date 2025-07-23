function t = parseConfigYaml(filename)
% t = parseConfigYaml(filename)
% Parse a configuration yaml.

s = yaml.ReadYaml(filename);
t = lclParse(s);

function tagg = lclParse(s,prefix)

if nargin < 2,
  prefix = '';
end

vallen = 9;

fns = fieldnames(s);
tagg = [];
for f=fns(:)',f=f{1}; %#ok<FXSET>
  val = s.(f);
  isLeaf = iscell(val) && ischar(val{1});
  if isempty(prefix),
    fullPath = f;
  else
    fullPath = [prefix,'.',f];
  end
  if isLeaf
    if numel(val) < vallen,
      val = [val,cell(1,vallen-numel(val))];
    elseif numel(val) > vallen,
      warning('Too many values read in for %s',f);
      val = val(1:vallen);
    end
    pgp = PropertiesGUIProp(fullPath,f,val{1:5},val{5:end});
    t = TreeNode(pgp);
  else
    if numel(val{1}) < vallen,
      val{1} = [val{1},cell(1,vallen-numel(val{1}))];
    elseif numel(val{1}) > vallen,
      warning('Too many values read in for %s',f);
      val{1} = val{1}(1:vallen);
    end
    pgp = PropertiesGUIProp(fullPath,f,val{1}{1:5},val{1}{5:end});
    t = TreeNode(pgp);
    % 20170426: cell2mat, cellfun still don't handle obj arrays    
    children = cellfun(@(s) lclParse(s,fullPath),val(2:end),'uni',0);
    t.Children = cat(1,children{:}); 
  end
  tagg = [tagg;t]; %#ok<AGROW>
end
