function checkCreateDir(backend, dirlocs, desc)

if nargin < 3 || ~ischar(desc),
  desc = 'dir';
end
for i = 1:numel(dirlocs),
  dirloc = dirlocs{i} ;
  if ~backend.exist(dirloc,'dir') ,
    [succ,msg] = backend.mkdir(dirloc);
    if ~succ
      error('Failed to create %s %s: %s',desc,dirloc,msg);
    else
      fprintf(1,'Created %s: %s\n',desc,dirloc);
    end
  end
end
