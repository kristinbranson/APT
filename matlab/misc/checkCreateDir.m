function checkCreateDir(dirlocs,desc)

if nargin < 2 || ~ischar(desc),
  desc = 'dir';
end
for i = 1:numel(dirlocs),
  if exist(dirlocs{i},'dir')==0
    [succ,msg] = mkdir(dirlocs{i});
    if ~succ
      error('Failed to create %s %s: %s',desc,dirlocs{i},msg);
    else
      fprintf(1,'Created %s: %s\n',desc,dirlocs{i});
    end
  end
end
