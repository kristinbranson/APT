function [files,didchange] = GetLinkSources(files)

didchange = false(size(files));

if ~isunix,
  return;
end

[stat,~] = unix('which readlink');
if stat ~= 0,
  return;
end

for i = 1:numel(files),
  
  while true,
    [stat,res] = unix(sprintf('readlink %s',files{i}));
    if stat ~= 0,
      break;
    end
    didchange(i) = true;
    files{i} = strtrim(res);
  end
  
end

