function t = parseConfigYaml(filename)
% t = parseConfigYaml(filename)
% Parse a configuration yaml.

s = ReadYaml(filename);
t = lclParse(s);

function t = lclParse(s)
fns = fieldnames(s);
for f=fns(:)',f=f{1}; %#ok<FXSET>
  val = s.(f);
  isLeaf = iscell(val) && ischar(val{1});
  if isLeaf
    pgp = PropertiesGUIProp(f,val{:},val{end});
    t = TreeNode(pgp);
  else
    pgp = PropertiesGUIProp(f,val{1}{:},val{1}{end});
    t = TreeNode(pgp);
    % 20170426: cell2mat, cellfun still don't handle obj arrays    
    children = cellfun(@lclParse,val(2:end),'uni',0);
    t.Children = cat(1,children{:}); 
  end
end
