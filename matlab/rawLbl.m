function rawLbl(lbl_file,out_file)

tname = tempname;
mkdir(tname);
try
  fprintf('Untarring project %s into %s\n',lbl_file,tname);
  untar(lbl_file,tname);
  fprintf('... done with untar.\n');
  rawLblFile = fullfile(tname,'label_file.lbl');
catch ME
  if strcmp(ME.identifier,'MATLAB:untar:invalidTarFile')
    warningNoTrace('Label file %s is not bundled. Using it in raw (mat) format.',lbl_file);
    rawLblFile = lbl_file;
  else
    ME.rethrow();
  end
end

A = load(rawLblFile,'-mat');
save(out_file,'-struct','A','-v7.3');

[success, message, ~] = rmdir(tname,'s');
if ~success
  error('Could not clear the temp directory %s\n',message);
else
  fprintf('Cleared out temp directory %s\n',tname);
end
