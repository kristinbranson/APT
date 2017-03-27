%%
dd = dir('private/*.c*');
files = {dd.name}';
nfile = numel(files);
fprintf('Compiling %d files.......................................\n',nfile);

cd private

for i = 1:nfile
  fprintf('Mexing %s...\n',files{i});
  mex(files{i});
end 

cd ..
disp('DONE');