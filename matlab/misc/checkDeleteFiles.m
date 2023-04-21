function checkDeleteFiles(filelocs,desc)
if nargin < 2 || ~ischar(desc),
  desc = 'file';
end
for i = 1:numel(filelocs),
  if exist(filelocs{i},'file'),
    fprintf('Deleting %s %s',desc,filelocs{i});
    delete(filelocs{i});
  end
  if exist(filelocs{i},'file'),
    error('Failed to delete %s: file still exists',filelocs{i});
  end
end
