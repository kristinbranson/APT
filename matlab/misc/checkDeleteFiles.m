function checkDeleteFiles(backend, filelocs, desc)
if nargin < 3 || ~ischar(desc),
  desc = 'file';
end
for i = 1:numel(filelocs),
  fileloc = filelocs{i} ;
  if backend.exist(fileloc,'file'),
    fprintf('Deleting %s %s',desc,fileloc);
    backend.deleteFile(fileloc);
  end
  if backend.exist(fileloc,'file'),
    error('Failed to delete %s: file still exists',fileloc);
  end
end
