function res = recursiveDir(rootdir,n)

res = mydir(fullfile(rootdir,n));

dirs = mydir(rootdir,'isdir',1);
for i = 1:numel(dirs),
  res = [res,recursiveDir(dirs{i},n)];
end
