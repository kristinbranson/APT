function saveLbl(old_bundled_lbl,new_bundled_lbl,new_lbl_obj_savestruct)

tname = tempname;
mkdir(tname);
try
  fprintf('Untarring old project %s into %s\n',old_bundled_lbl,tname);
  fnames = untar(old_bundled_lbl,tname);
  % AL20190424: untar() function output in blatant contradiction with 
  % M-help
  ntname = numel(tname);
  for i=1:numel(fnames)
    if startsWith(fnames{i},tname)
      fnames{i} = fnames{i}(ntname+2:end); % skip filesep
    end
  end
  fprintf('... done with untar.\n');
  rawLblFile = fullfile(tname,'label_file.lbl');
catch ME
  if strcmp(ME.identifier,'MATLAB:untar:invalidTarFile')
    error('Cannot lbl file to raw label file');
  else
    rethrow(ME);
  end
end

save(rawLblFile,'-struct','new_lbl_obj_savestruct');
fprintf('Tarring updated project into %s\n',new_bundled_lbl);
tar([new_bundled_lbl '.tar'],fnames,tname);
fprintf('... done with tar.\n');
movefile([new_bundled_lbl '.tar'],new_bundled_lbl); 
[success, message, ~] = rmdir(tname,'s');
if ~success
  warning('Could not clear the temp directory %s\n',message);
else
  fprintf('Cleared out temp directory %s\n',tname);
end
