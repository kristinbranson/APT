function ancestors = getpathancestors(p)

ancestors = {};
while true,
  [p,n,e] = fileparts(p);
  if isempty(n) && isempty(e),
    break;
  end
  ancestors{end+1} = p; %#ok<AGROW>
end